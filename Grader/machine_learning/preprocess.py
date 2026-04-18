# grader/machine_learning/preprocess.py

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

IMG_SIZE = (224, 224)  # EfficientNet/ResNet standard input

def load_and_preprocess(image_path: str) -> np.ndarray:
    """Full preprocessing pipeline for a card image."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 1. Auto-crop to card boundaries using contour detection
    img = auto_crop_card(img)

    # 2. Resize to model input size
    img = cv2.resize(img, IMG_SIZE)

    # 3. Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0

    return img


def auto_crop_card(img: np.ndarray) -> np.ndarray:
    """Detect and crop to card edges using contour detection."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        # Find the largest rectangular contour (the card)
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        # Only crop if it looks like a card (reasonable aspect ratio)
        aspect = w / h
        if 0.6 < aspect < 0.8:  # Pokemon cards are ~63x88mm
            padding = 5
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img.shape[1] - x, w + 2 * padding)
            h = min(img.shape[0] - y, h + 2 * padding)
            return img[y:y+h, x:x+w]

    return img  # Return original if no card detected


def analyze_centering(img: np.ndarray) -> dict:
    """
    Estimate centering score by detecting card border widths.
    PSA grades 10/9 require near 55/45 or better centering.
    """
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # Sample border pixels along center rows/columns
    left_border = detect_border_width(gray[h//2, :w//3])
    right_border = detect_border_width(gray[h//2, 2*w//3:][::-1])
    top_border = detect_border_width(gray[:h//3, w//2])
    bottom_border = detect_border_width(gray[2*h//3:, w//2][::-1])

    def ratio(a, b):
        total = a + b
        return (min(a, b) / max(a, b)) if total > 0 else 1.0

    lr_ratio = ratio(left_border, right_border)
    tb_ratio = ratio(top_border, bottom_border)

    # Score: 1.0 = perfect centering
    centering_score = (lr_ratio + tb_ratio) / 2
    return {
        "left": left_border, "right": right_border,
        "top": top_border, "bottom": bottom_border,
        "lr_ratio": lr_ratio, "tb_ratio": tb_ratio,
        "score": centering_score
    }


def detect_border_width(pixel_strip: np.ndarray) -> int:
    """Count border pixels from the edge of a strip."""
    threshold = 200  # Bright border pixels
    count = 0
    for px in pixel_strip:
        if px > threshold:
            count += 1
        else:
            break
    return count


def build_tf_dataset(image_paths: list, grades: list, batch_size: int = 32):
    """Build a tf.data.Dataset for training."""
    # Convert grades to 0-indexed (PSA 1-10 → 0-9)
    labels = [g - 1 for g in grades]

    def load_img(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )
    dataset = dataset.map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset