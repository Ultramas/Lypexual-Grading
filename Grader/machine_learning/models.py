# grader/machine_learning/model.py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB3

IMG_SIZE = (224, 224)
NUM_CLASSES = 10  # PSA grades 1-10


def build_grading_model(mode: str = "classification") -> keras.Model:
    """
    Build a card grading model.

    mode="classification" → 10-class softmax (discrete PSA grade)
    mode="regression"     → single float output (continuous score)
    mode="hybrid"         → both outputs combined
    """
    # Base model: EfficientNetB3 pretrained on ImageNet
    # Great balance of accuracy vs. size for fine-grained classification
    base = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
        drop_connect_rate=0.2
    )

    # Stage 1: Freeze base, train only head
    base.trainable = False

    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="card_image")

    # Data augmentation (applied during training only)
    x = layers.RandomFlip("horizontal", name="aug_flip")(inputs)
    x = layers.RandomRotation(0.05, name="aug_rotate")(x)
    x = layers.RandomZoom(0.1, name="aug_zoom")(x)
    x = layers.RandomBrightness(0.1, name="aug_brightness")(x)
    x = layers.RandomContrast(0.1, name="aug_contrast")(x)

    # Feature extraction
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = layers.BatchNormalization(name="bn")(x)

    # Shared dense layers
    x = layers.Dense(512, activation="relu", name="dense_1")(x)
    x = layers.Dropout(0.4, name="drop_1")(x)
    x = layers.Dense(256, activation="relu", name="dense_2")(x)
    x = layers.Dropout(0.3, name="drop_2")(x)

    if mode == "classification":
        # 10-class output: each neuron = probability of that PSA grade
        outputs = layers.Dense(
            NUM_CLASSES, activation="softmax", name="grade_class"
        )(x)
        model = keras.Model(inputs, outputs, name="PSAGrader_Classifier")
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy", keras.metrics.SparseTopKCategoricalAccuracy(k=2)]
        )

    elif mode == "regression":
        # Single continuous output [1, 10]
        raw = layers.Dense(1, activation="sigmoid", name="grade_raw")(x)
        # Scale sigmoid [0,1] → PSA [1,10]
        outputs = layers.Lambda(
            lambda t: t * 9.0 + 1.0, name="grade_score"
        )(raw)
        model = keras.Model(inputs, outputs, name="PSAGrader_Regressor")
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="mean_squared_error",
            metrics=["mae"]
        )

    else:  # hybrid
        # Classification head
        class_out = layers.Dense(
            NUM_CLASSES, activation="softmax", name="grade_class"
        )(x)
        # Regression head
        reg_raw = layers.Dense(1, activation="sigmoid", name="grade_raw")(x)
        reg_out = layers.Lambda(
            lambda t: t * 9.0 + 1.0, name="grade_score"
        )(reg_raw)

        model = keras.Model(
            inputs, [class_out, reg_out], name="PSAGrader_Hybrid"
        )
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss={
                "grade_class": "sparse_categorical_crossentropy",
                "grade_score": "mean_squared_error"
            },
            loss_weights={"grade_class": 1.0, "grade_score": 0.5},
            metrics={
                "grade_class": "accuracy",
                "grade_score": "mae"
            }
        )

    return model


def unfreeze_for_finetuning(model: keras.Model, num_layers: int = 30):
    """
    Stage 2: Unfreeze top layers of EfficientNet for fine-tuning.
    Call this after initial training converges.
    """
    base = model.get_layer("efficientnetb3")
    base.trainable = True

    # Freeze all except last `num_layers`
    for layer in base.layers[:-num_layers]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),  # Much lower LR for fine-tuning
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model