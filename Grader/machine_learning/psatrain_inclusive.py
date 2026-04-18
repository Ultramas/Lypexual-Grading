# grader/ml/train.py

import os
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from .model import build_grading_model, unfreeze_for_finetuning
from .preprocess import build_tf_dataset

CHECKPOINT_PATH = "grader/ml/checkpoints/psa_grader.keras"
LOG_DIR = "grader/ml/logs/"


def load_dataset_from_directory(data_dir: str):
    """
    Expected folder structure:
    data_dir/
      grade_1/  ← images of PSA 1 cards
      grade_2/
      ...
      grade_10/
    """
    image_paths, grades = [], []

    for grade in range(1, 11):
        folder = os.path.join(data_dir, f"grade_{grade}")
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                image_paths.append(os.path.join(folder, fname))
                grades.append(grade)

    print(f"Loaded {len(image_paths)} images across grades 1-10")
    return image_paths, grades


def train(data_dir: str, epochs_stage1: int = 20, epochs_stage2: int = 30):
    """Full two-stage training pipeline."""
    image_paths, grades = load_dataset_from_directory(data_dir)

    # Train/val/test split (70/15/15)
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, grades, test_size=0.3, stratify=grades, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    train_ds = build_tf_dataset(X_train, y_train, batch_size=32)
    val_ds = build_tf_dataset(X_val, y_val, batch_size=32)
    test_ds = build_tf_dataset(X_test, y_test, batch_size=32)

    model = build_grading_model(mode="classification")
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            CHECKPOINT_PATH, save_best_only=True,
            monitor="val_accuracy", verbose=1
        ),
        keras.callbacks.EarlyStopping(
            patience=7, restore_best_weights=True,
            monitor="val_accuracy"
        ),
        keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=3, min_lr=1e-7,
            monitor="val_loss", verbose=1
        ),
        keras.callbacks.TensorBoard(log_dir=LOG_DIR),
    ]

    # --- STAGE 1: Train head only ---
    print("\n=== Stage 1: Training head layers ===")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_stage1,
        callbacks=callbacks
    )

    # --- STAGE 2: Fine-tune top EfficientNet layers ---
    print("\n=== Stage 2: Fine-tuning EfficientNet top layers ===")
    model = unfreeze_for_finetuning(model, num_layers=30)
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_stage2,
        initial_epoch=epochs_stage1,
        callbacks=callbacks
    )

    # Evaluate on test set
    print("\n=== Final Test Evaluation ===")
    test_loss, test_acc, test_top2 = model.evaluate(test_ds)
    print(f"Test Accuracy: {test_acc:.4f} | Top-2 Accuracy: {test_top2:.4f}")

    model.save(CHECKPOINT_PATH)
    print(f"Model saved to {CHECKPOINT_PATH}")
    return model