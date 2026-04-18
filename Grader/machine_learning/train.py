"""
Pokemon Card PSA Grader — Model Trainer
========================================
Trains an EfficientNetB3-based classifier on your collected training data.

Usage:
    python train.py training_data/
    python train.py training_data/ --epochs 30 --batch 32
    python train.py training_data/ --resume   # continue from last checkpoint

Expected folder structure:
    training_data/
        grade_1/   *.jpg
        grade_2/   *.jpg
        ...
        grade_10/  *.jpg

Output:
    checkpoints/psa_grader.keras   ← used automatically by grader.py
    checkpoints/training_log.json
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Check TensorFlow before anything else ─────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.applications import EfficientNetB3
    log.info("TensorFlow %s detected", tf.__version__)
except ImportError:
    print("\n  ERROR: TensorFlow not installed.")
    print("  Install it with:  pip install tensorflow\n")
    sys.exit(1)

import numpy as np
from sklearn.model_selection import train_test_split

# ── CONSTANTS ─────────────────────────────────────────────────────────────────

IMG_SIZE        = (224, 224)
NUM_CLASSES     = 10
CHECKPOINT_DIR  = Path(__file__).parent / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "psa_grader.keras"
LOG_PATH        = CHECKPOINT_DIR / "training_log.json"


# ── DATASET LOADER ────────────────────────────────────────────────────────────

def load_dataset(data_dir: Path) -> tuple[list, list]:
    """
    Scan grade_1/ through grade_10/ folders and return
    (image_paths, labels) where label is 0-indexed (grade-1).
    """
    image_paths = []
    labels      = []
    counts      = {}

    for grade in range(1, 11):
        folder = data_dir / f"grade_{grade}"
        if not folder.exists():
            log.warning("Missing folder: %s", folder)
            continue

        imgs = (
            list(folder.glob("*.jpg"))  +
            list(folder.glob("*.jpeg")) +
            list(folder.glob("*.png"))  +
            list(folder.glob("*.webp"))
        )

        if not imgs:
            log.warning("No images in %s", folder)
            continue

        image_paths.extend([str(p) for p in imgs])
        labels.extend([grade - 1] * len(imgs))   # 0-indexed for keras
        counts[grade] = len(imgs)

    # Print dataset summary
    print("\n" + "═" * 46)
    print("  DATASET LOADED")
    print("═" * 46)
    total = 0
    for g in range(1, 11):
        n   = counts.get(g, 0)
        bar = "█" * (n // 10)
        ok  = "✓" if n >= 100 else "⚠" if n > 0 else "✗ MISSING"
        print(f"  PSA {g:2d}: {n:4d} imgs  {bar:<20} {ok}")
        total += n
    print("─" * 46)
    print(f"  TOTAL : {total} images\n")

    if total < 100:
        print("  ERROR: Not enough images to train (need at least 100 total).")
        print("  Run the scraper first: python scraper_playwright.py --grades 1 2 3 4 5 6 7 8 9 10\n")
        sys.exit(1)

    low_grades = [g for g, n in counts.items() if n < 50]
    if low_grades:
        log.warning(
            "Low sample count for grades %s — model accuracy will be limited for these grades. "
            "Run the scraper with more --per-grade to improve.",
            low_grades,
        )

    return image_paths, labels


# ── TF DATASET ────────────────────────────────────────────────────────────────

def make_tf_dataset(
    paths:      list[str],
    labels:     list[int],
    batch_size: int,
    augment:    bool = False,
) -> tf.data.Dataset:
    """Build a tf.data pipeline with optional augmentation."""

    def load_and_preprocess(path, label):
        raw  = tf.io.read_file(path)
        img  = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img  = tf.image.resize(img, IMG_SIZE)
        img  = tf.cast(img, tf.float32) / 255.0
        return img, label

    def augment_fn(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.15)
        img = tf.image.random_contrast(img, 0.85, 1.15)
        img = tf.image.random_saturation(img, 0.85, 1.15)
        img = tf.image.random_jpeg_quality(img, 70, 100)
        # Random rotation ±8°
        angle = tf.random.uniform([], -0.14, 0.14)
        img   = tf.keras.preprocessing.image.apply_affine_transform(
            img.numpy(), theta=angle * 57.3
        ) if False else img   # skip for now — tfa dependency
        img = tf.clip_by_value(img, 0.0, 1.0)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(buffer_size=min(len(paths), 2000))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ── CLASS WEIGHTS (handle imbalanced grades) ──────────────────────────────────

def compute_class_weights(labels: list[int]) -> dict:
    """
    Give higher weight to rare grades (PSA 1, 2) so the model doesn't
    just predict 'PSA 9' for everything.
    """
    from collections import Counter
    counts  = Counter(labels)
    total   = len(labels)
    n_cls   = NUM_CLASSES
    weights = {}
    for cls in range(n_cls):
        n = counts.get(cls, 1)
        weights[cls] = (total / (n_cls * n))
    return weights


# ── MODEL BUILDER ─────────────────────────────────────────────────────────────

def build_model(freeze_base: bool = True) -> keras.Model:
    base = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
    )
    base.trainable = not freeze_base

    inputs = keras.Input(shape=(*IMG_SIZE, 3))

    # Light augmentation baked into the model
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.04)(x)
    x = layers.RandomZoom(0.08)(x)
    x = layers.RandomBrightness(0.10)(x)

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.40)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.30)(x)

    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="grade_output")(x)

    model = keras.Model(inputs, outputs, name="PSAGrader")
    return model


def compile_model(model: keras.Model, lr: float, frozen: bool):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2_acc"),
        ],
    )
    status = "frozen base" if frozen else "fine-tuning"
    log.info("Model compiled — %s, lr=%.2e, params=%d",
             status, lr, model.count_params())


def unfreeze_top(model: keras.Model, n_layers: int = 40):
    base = model.get_layer("efficientnetb3")
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    log.info("Unfroze top %d layers of EfficientNetB3", n_layers)


# ── CALLBACKS ─────────────────────────────────────────────────────────────────

def make_callbacks(stage: int) -> list:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.ModelCheckpoint(
            str(CHECKPOINT_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            str(CHECKPOINT_DIR / f"stage{stage}_log.csv"),
            append=True,
        ),
    ]


# ── TRAINING PIPELINE ─────────────────────────────────────────────────────────

def train(
    data_dir:      Path,
    batch_size:    int   = 32,
    epochs_stage1: int   = 20,
    epochs_stage2: int   = 30,
    resume:        bool  = False,
):
    # ── Load data ──────────────────────────────────────────────────────
    image_paths, labels = load_dataset(data_dir)

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        image_paths, labels,
        test_size=0.30,
        stratify=labels,
        random_state=42,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=0.50,
        stratify=y_tmp,
        random_state=42,
    )

    log.info("Split: train=%d  val=%d  test=%d", len(X_train), len(X_val), len(X_test))

    train_ds = make_tf_dataset(X_train, y_train, batch_size, augment=True)
    val_ds   = make_tf_dataset(X_val,   y_val,   batch_size, augment=False)
    test_ds  = make_tf_dataset(X_test,  y_test,  batch_size, augment=False)

    class_weights = compute_class_weights(y_train)
    log.info("Class weights: %s", {f"PSA{k+1}": round(v, 2) for k, v in class_weights.items()})

    # ── Build or resume model ──────────────────────────────────────────
    if resume and CHECKPOINT_PATH.exists():
        log.info("Resuming from checkpoint: %s", CHECKPOINT_PATH)
        model = keras.models.load_model(str(CHECKPOINT_PATH))
        compile_model(model, lr=1e-4, frozen=False)
    else:
        model = build_model(freeze_base=True)
        compile_model(model, lr=1e-3, frozen=True)

    model.summary(line_length=80)

    # ── STAGE 1: Train head only ───────────────────────────────────────
    if not resume:
        print("\n" + "═"*46)
        print("  STAGE 1: Training classification head")
        print("  (EfficientNet base frozen, ImageNet weights)")
        print("═"*46)

        history1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs_stage1,
            class_weight=class_weights,
            callbacks=make_callbacks(stage=1),
        )

        best_val = max(history1.history.get("val_accuracy", [0]))
        log.info("Stage 1 best val_accuracy: %.4f", best_val)

    # ── STAGE 2: Fine-tune top EfficientNet layers ─────────────────────
    print("\n" + "═"*46)
    print("  STAGE 2: Fine-tuning EfficientNetB3 top layers")
    print("  (Lower learning rate, more layers unfrozen)")
    print("═"*46)

    unfreeze_top(model, n_layers=40)
    compile_model(model, lr=1e-5, frozen=False)

    initial_epoch = epochs_stage1 if not resume else 0

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=initial_epoch + epochs_stage2,
        initial_epoch=initial_epoch,
        class_weight=class_weights,
        callbacks=make_callbacks(stage=2),
    )

    # ── Final evaluation ───────────────────────────────────────────────
    print("\n" + "═"*46)
    print("  FINAL TEST SET EVALUATION")
    print("═"*46)
    results = model.evaluate(test_ds, verbose=1)
    metric_names = model.metrics_names
    metrics_dict = dict(zip(metric_names, results))

    print(f"\n  Test Loss     : {metrics_dict.get('loss', 0):.4f}")
    print(f"  Test Accuracy : {metrics_dict.get('accuracy', 0)*100:.2f}%")
    print(f"  Top-2 Accuracy: {metrics_dict.get('top2_acc', 0)*100:.2f}%")

    # ── Per-class accuracy ─────────────────────────────────────────────
    print("\n  Per-grade accuracy:")
    all_preds = []
    all_true  = []
    for batch_imgs, batch_labels in test_ds:
        preds = model.predict(batch_imgs, verbose=0)
        all_preds.extend(np.argmax(preds, axis=1))
        all_true.extend(batch_labels.numpy())

    for grade_idx in range(NUM_CLASSES):
        mask   = [i for i, y in enumerate(all_true) if y == grade_idx]
        if not mask:
            continue
        n_correct = sum(1 for i in mask if all_preds[i] == grade_idx)
        acc       = n_correct / len(mask) * 100
        bar       = "█" * int(acc / 5)
        print(f"    PSA {grade_idx+1:2d}: {acc:5.1f}%  {bar}")

    # ── Save final model ───────────────────────────────────────────────
    model.save(str(CHECKPOINT_PATH))
    print(f"\n  ✓ Model saved → {CHECKPOINT_PATH}")
    print("  ✓ Run grader.py to use it:\n")
    print("      python grader.py your_card.jpg\n")

    # Save training log
    log_data = {
        "test_metrics":  metrics_dict,
        "train_samples": len(X_train),
        "val_samples":   len(X_val),
        "test_samples":  len(X_test),
        "batch_size":    batch_size,
        "epochs_s1":     epochs_stage1,
        "epochs_s2":     epochs_stage2,
        "checkpoint":    str(CHECKPOINT_PATH),
    }
    with open(LOG_PATH, "w") as f:
        json.dump(log_data, f, indent=2, default=str)

    return model


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train PSA card grading model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py training_data/
  python train.py training_data/ --epochs1 15 --epochs2 25 --batch 16
  python train.py training_data/ --resume
        """,
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Path to training_data/ folder containing grade_1/ … grade_10/ subfolders",
    )
    parser.add_argument("--epochs1",  type=int, default=20,  help="Stage 1 epochs (default 20)")
    parser.add_argument("--epochs2",  type=int, default=30,  help="Stage 2 epochs (default 30)")
    parser.add_argument("--batch",    type=int, default=32,  help="Batch size (default 32; use 16 if OOM)")
    parser.add_argument("--resume",   action="store_true",   help="Resume from existing checkpoint")
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"\n  ERROR: Directory not found: {args.data_dir}")
        print("  Make sure the scraper has finished and training_data/ exists.\n")
        sys.exit(1)

    # GPU info
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        log.info("GPU detected: %s", [g.name for g in gpus])
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        log.warning("No GPU detected — training on CPU will be slow.")
        log.warning("Consider using Google Colab (free GPU) if training takes too long.")

    train(
        data_dir      = args.data_dir,
        batch_size    = args.batch,
        epochs_stage1 = args.epochs1,
        epochs_stage2 = args.epochs2,
        resume        = args.resume,
    )


if __name__ == "__main__":
    main()