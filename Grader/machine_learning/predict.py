# grader/machine_learning/predict.py

import numpy as np
import tensorflow as tf
from .preprocess import load_and_preprocess, analyze_centering

MODEL_PATH = "grader/machine_learning/checkpoints/psa_grader.keras"
_model = None  # Singleton — loaded once


def get_model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def grade_card(image_path: str) -> dict:
    """
    Full grading pipeline for a single card image.
    Returns grade 1-10 with sub-scores and confidence.
    """
    model = get_model()

    # Preprocess
    img = load_and_preprocess(image_path)
    centering = analyze_centering(img)

    # Run model
    img_batch = np.expand_dims(img, axis=0)
    probs = model.predict(img_batch, verbose=0)[0]  # shape: (10,)

    # PSA grade = argmax + 1 (0-indexed → 1-10)
    predicted_class = int(np.argmax(probs)) + 1
    confidence = float(probs[predicted_class - 1]) * 100

    # Weighted average for continuous score
    grades = np.arange(1, 11)
    continuous_score = float(np.sum(probs * grades))

    # Centering penalty (adjust grade down if heavily off-center)
    centering_penalty = max(0, (0.7 - centering["score"]) * 3)
    adjusted_score = max(1.0, continuous_score - centering_penalty)

    return {
        "predicted_grade": round(continuous_score, 2),
        "adjusted_grade": round(adjusted_score, 2),
        "rounded_grade": round(adjusted_score),
        "confidence_pct": round(confidence, 1),
        "grade_probabilities": {
            f"PSA {i+1}": round(float(p) * 100, 2)
            for i, p in enumerate(probs)
        },
        "sub_scores": {
            "centering": round(centering["score"] * 10, 2),
            "centering_lr_ratio": round(centering["lr_ratio"], 3),
            "centering_tb_ratio": round(centering["tb_ratio"], 3),
        }
    }