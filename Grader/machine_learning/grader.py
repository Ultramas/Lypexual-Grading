"""
Pokemon Card PSA Grader — Test & 4-Criteria Analysis
=====================================================
Usage:
    # Test a single card image:
    python grader.py path/to/card.jpg

    # Grade multiple cards:
    python grader.py card1.jpg card2.jpg card3.jpg

    # Run accuracy test against your training_data/ folder:
    python grader.py --test-dataset training_data/

    # Output as JSON (for API integration):
    python grader.py card.jpg --json
"""

import re
import sys
import json
import argparse
import logging
import random
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Try to load the trained Keras model ───────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "checkpoints" / "psa_grader.keras"
_model     = None

def get_model():
    global _model
    if _model is not None:
        return _model
    if not MODEL_PATH.exists():
        log.warning(
            "No trained model found at %s\n"
            "  → Running in DEMO MODE (pixel analysis only, no ML)\n"
            "  → Train the model first: python train.py training_data/",
            MODEL_PATH,
        )
        return None
    import tensorflow as tf
    _model = tf.keras.models.load_model(str(MODEL_PATH))
    log.info("Model loaded from %s", MODEL_PATH)
    return _model


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

IMG_SIZE = (224, 224)

def load_image(path: str) -> np.ndarray | None:
    """Load and resize image to model input size. Returns float32 [0,1]."""
    img = cv2.imread(str(path))
    if img is None:
        log.error("Could not read image: %s", path)
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    return img.astype(np.float32) / 255.0


def load_image_full(path: str) -> np.ndarray | None:
    """Load full-resolution image for detailed pixel analysis."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────────────────────────────────────
# THE 4 PSA CRITERIA — PIXEL-LEVEL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def snap_to_half(v: float) -> float:
    """Round a float to the nearest 0.5 increment, clamped [1, 10]."""
    snapped = round(v * 2) / 2
    return max(1.0, min(10.0, snapped))


# ── 1. CENTERING ──────────────────────────────────────────────────────────────

def analyze_centering(img: np.ndarray) -> dict:
    """
    Measures border width ratios on left/right and top/bottom edges.
    PSA 10 requires ≤55/45 L/R and T/B centering.
    Returns score 1-10 in 0.5 steps.
    """
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) \
           if img.dtype == np.float32 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    def border_width(strip):
        """Count leading bright pixels (the white card border)."""
        count = 0
        for px in strip:
            if px > 200:
                count += 1
            else:
                break
        return max(count, 1)

    mid_y, mid_x = h // 2, w // 2
    span = min(w, h) // 5

    L = border_width(gray[mid_y, :span])
    R = border_width(gray[mid_y, w-span:][::-1])
    T = border_width(gray[:span, mid_x])
    B = border_width(gray[h-span:, mid_x][::-1])

    def ratio(a, b):
        total = a + b
        return min(a, b) / max(a, b) if total > 0 else 0.5

    lr = ratio(L, R)
    tb = ratio(T, B)
    combined = (lr + tb) / 2   # 0.0 (terrible) → 1.0 (perfect)

    # Convert ratio to PSA grade
    if combined >= 0.97:  raw = 10.0
    elif combined >= 0.93: raw = 9.5
    elif combined >= 0.88: raw = 9.0
    elif combined >= 0.82: raw = 8.5
    elif combined >= 0.75: raw = 8.0
    elif combined >= 0.67: raw = 7.0
    elif combined >= 0.58: raw = 6.0
    elif combined >= 0.48: raw = 5.0
    elif combined >= 0.38: raw = 4.0
    elif combined >= 0.28: raw = 3.0
    elif combined >= 0.18: raw = 2.0
    else:                  raw = 1.0

    # L/R percentage strings for description
    lr_pct = f"{round(L/(L+R)*100)}/{round(R/(L+R)*100)}" if (L+R) > 0 else "?/?"
    tb_pct = f"{round(T/(T+B)*100)}/{round(B/(T+B)*100)}" if (T+B) > 0 else "?/?"

    return {
        "score":      snap_to_half(raw),
        "lr_ratio":   round(lr, 3),
        "tb_ratio":   round(tb, 3),
        "lr_pct":     lr_pct,
        "tb_pct":     tb_pct,
        "L": L, "R": R, "T": T, "B": B,
    }


def centering_description(data: dict, score: float) -> str:
    lr  = data["lr_pct"]
    tb  = data["tb_pct"]
    s   = score

    if s >= 9.5:
        return (
            f"Centering is virtually perfect, measuring approximately {lr} left-to-right "
            f"and {tb} top-to-bottom. The borders appear completely uniform to the naked eye "
            f"with no perceptible shift in any direction. Cards at this level meet PSA's gem-mint "
            f"centering threshold of 55/45 or better on both axes."
        )
    elif s >= 8.5:
        return (
            f"Centering is strong, measuring approximately {lr} left-to-right and {tb} "
            f"top-to-bottom. Minor asymmetry is present but requires close inspection to notice. "
            f"The borders are well-proportioned and the card presents cleanly from a normal viewing "
            f"distance. This level is consistent with PSA 9 grading standards."
        )
    elif s >= 7.0:
        return (
            f"Centering is above average, approximately {lr} left-to-right and {tb} "
            f"top-to-bottom. The offset is visible but not distracting — one border is noticeably "
            f"thinner than the opposite side. The card still presents well overall but would not "
            f"meet PSA Mint or Gem Mint centering requirements."
        )
    elif s >= 5.0:
        return (
            f"Centering is noticeably off, measuring roughly {lr} left-to-right and {tb} "
            f"top-to-bottom. The misalignment is immediately apparent, with one or more borders "
            f"significantly thinner than the opposite. This level of centering typically caps the "
            f"overall grade at PSA 5 or 6 regardless of other factors."
        )
    else:
        return (
            f"Centering is severely misaligned at approximately {lr} left-to-right and {tb} "
            f"top-to-bottom. One or more borders may appear nearly absent while the opposite side "
            f"is disproportionately wide. This degree of miscentering is a primary grade-limiter "
            f"and is consistent with PSA 1–4 centering standards."
        )


# ── 2. CORNERS ────────────────────────────────────────────────────────────────

def analyze_corners(img: np.ndarray) -> dict:
    """
    Examines the four card corners for wear, fraying, or blunting.
    Corners are the first place wear appears on a card.
    """
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) \
           if img.dtype == np.float32 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    sz   = max(h, w) // 10   # corner sample size

    corners = {
        "TL": gray[:sz,     :sz],
        "TR": gray[:sz,     w-sz:],
        "BL": gray[h-sz:,   :sz],
        "BR": gray[h-sz:,   w-sz:],
    }

    sharpness_scores = {}
    for name, patch in corners.items():
        # Laplacian variance — sharp corners (good) have high variance
        lap = cv2.Laplacian(patch, cv2.CV_64F).var()
        # Gradient magnitude — worn/blunted corners have low gradient
        gx  = cv2.Sobel(patch, cv2.CV_64F, 1, 0)
        gy  = cv2.Sobel(patch, cv2.CV_64F, 0, 1)
        mag = np.sqrt(gx**2 + gy**2).mean()
        sharpness_scores[name] = (lap * 0.6 + mag * 0.4)

    avg_sharp = np.mean(list(sharpness_scores.values()))
    min_sharp = np.min(list(sharpness_scores.values()))
    weakest   = min(sharpness_scores, key=sharpness_scores.get)

    # Map sharpness to grade (calibrated to card image typical ranges)
    if avg_sharp > 800 and min_sharp > 600:     raw = 10.0
    elif avg_sharp > 600 and min_sharp > 450:   raw = 9.5
    elif avg_sharp > 450 and min_sharp > 300:   raw = 9.0
    elif avg_sharp > 320 and min_sharp > 200:   raw = 8.5
    elif avg_sharp > 220 and min_sharp > 130:   raw = 8.0
    elif avg_sharp > 150 and min_sharp > 80:    raw = 7.0
    elif avg_sharp > 90  and min_sharp > 40:    raw = 6.0
    elif avg_sharp > 50  and min_sharp > 20:    raw = 5.0
    elif avg_sharp > 25:                         raw = 4.0
    elif avg_sharp > 12:                         raw = 3.0
    elif avg_sharp > 5:                          raw = 2.0
    else:                                         raw = 1.0

    return {
        "score":         snap_to_half(raw),
        "avg_sharpness": round(avg_sharp, 1),
        "min_sharpness": round(min_sharp, 1),
        "weakest_corner": weakest,
        "per_corner":    {k: round(v, 1) for k, v in sharpness_scores.items()},
    }


def corners_description(data: dict, score: float) -> str:
    weak = {"TL": "top-left", "TR": "top-right", "BL": "bottom-left", "BR": "bottom-right"}
    weakest = weak.get(data["weakest_corner"], data["weakest_corner"])

    if score >= 9.5:
        return (
            "All four corners appear crisp, sharp, and pointed with no visible wear or fraying. "
            "Under close inspection, the corners maintain their original factory edge without any "
            "softening, blunting, or white stress marks. Corners at this level are consistent with "
            "a pack-fresh or lightly handled card meeting PSA Gem Mint standards."
        )
    elif score >= 8.5:
        return (
            f"Corners are sharp overall with only the slightest imperfection detectable under "
            f"close inspection, most notably at the {weakest} corner. There are no significant "
            f"frays, folds, or stress marks visible under normal lighting. The corners present "
            f"excellently and are consistent with PSA Mint to Near Mint-Mint standards."
        )
    elif score >= 7.0:
        return (
            f"Corners show light wear with some minor softening, particularly at the {weakest} "
            f"corner. Slight blunting or very minor fraying may be visible under direct light, "
            f"though the corners are not severely damaged. This level of corner wear is typical "
            f"of a well-handled card and is consistent with PSA Near Mint standards."
        )
    elif score >= 5.0:
        return (
            f"Moderate corner wear is present across the card, with the {weakest} corner showing "
            f"the most significant damage. Visible blunting, fraying, or minor creasing can be "
            f"observed without magnification. This level of wear indicates the card has seen "
            f"regular handling and is consistent with PSA Excellent to Very Good grades."
        )
    else:
        return (
            f"Corners show heavy wear with significant blunting, fraying, or creasing throughout, "
            f"especially at the {weakest} corner. The original corner points are largely lost and "
            f"the damage is immediately apparent at normal viewing distance. This degree of corner "
            f"wear typically limits the grade to PSA 1–4."
        )


# ── 3. EDGES ──────────────────────────────────────────────────────────────────

def analyze_edges(img: np.ndarray) -> dict:
    """
    Examines the four card edges for nicks, chips, and roughness.
    Uses gradient analysis along the card perimeter.
    """
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) \
           if img.dtype == np.float32 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    band = max(8, h // 20)   # edge band width to sample

    # Sample strips along each edge
    strips = {
        "top":    gray[:band,        :],
        "bottom": gray[h-band:,      :],
        "left":   gray[:,            :band],
        "right":  gray[:,            w-band:],
    }

    edge_scores = {}
    for name, strip in strips.items():
        lap = cv2.Laplacian(strip, cv2.CV_64F)
        # High variance = rough/nicked edges; we want SMOOTH edges (low variance = good)
        variance = lap.var()
        # Also measure std dev of the strip brightness (uniform = clean edge)
        std      = strip.std()
        # Invert: lower variance → cleaner edge → higher score
        edge_scores[name] = variance + std * 0.5

    avg_roughness = np.mean(list(edge_scores.values()))
    worst_edge    = max(edge_scores, key=edge_scores.get)

    # Map roughness to grade (lower roughness = better grade)
    if avg_roughness < 50:        raw = 10.0
    elif avg_roughness < 100:     raw = 9.5
    elif avg_roughness < 180:     raw = 9.0
    elif avg_roughness < 280:     raw = 8.5
    elif avg_roughness < 420:     raw = 8.0
    elif avg_roughness < 600:     raw = 7.0
    elif avg_roughness < 850:     raw = 6.0
    elif avg_roughness < 1200:    raw = 5.0
    elif avg_roughness < 1800:    raw = 4.0
    elif avg_roughness < 2600:    raw = 3.0
    elif avg_roughness < 3800:    raw = 2.0
    else:                          raw = 1.0

    return {
        "score":         snap_to_half(raw),
        "avg_roughness": round(avg_roughness, 1),
        "worst_edge":    worst_edge,
        "per_edge":      {k: round(v, 1) for k, v in edge_scores.items()},
    }


def edges_description(data: dict, score: float) -> str:
    worst = data["worst_edge"]

    if score >= 9.5:
        return (
            "All four edges appear clean, smooth, and free of nicks or chips under close inspection. "
            "The edge lines are straight and uniform with no visible roughness or irregularity. "
            "The card's borders maintain their original factory-cut integrity on all sides, "
            "consistent with PSA Gem Mint edge standards."
        )
    elif score >= 8.5:
        return (
            f"Edges are clean and well-preserved overall, with only minor imperfections visible "
            f"on the {worst} edge under close examination. No significant nicking or chipping "
            f"is present, and the edges are straight from a normal viewing distance. This level "
            f"of edge quality is consistent with PSA Mint to Near Mint-Mint standards."
        )
    elif score >= 7.0:
        return (
            f"Edges show light wear with minor roughness or small nicks visible, most notably "
            f"along the {worst} edge. The imperfections are detectable under direct lighting but "
            f"do not significantly detract from the card's overall presentation. Edge quality "
            f"at this level is consistent with PSA Near Mint grading."
        )
    elif score >= 5.0:
        return (
            f"Moderate edge wear is present with visible nicking and roughness, particularly "
            f"along the {worst} edge. Chipping or fraying of the cardboard may be detectable "
            f"without magnification. This level of edge damage is consistent with PSA Excellent "
            f"to Very Good grading and reflects regular handling."
        )
    else:
        return (
            f"Edges show significant damage including heavy nicking, chipping, or roughness, "
            f"most severely along the {worst} edge. The cardboard edge structure may appear "
            f"frayed or partially delaminated in areas. This degree of edge wear is consistent "
            f"with PSA 1–4 and will substantially limit the overall grade."
        )


# ── 4. SURFACE ────────────────────────────────────────────────────────────────

def analyze_surface(img: np.ndarray) -> dict:
    """
    Detects scratches, print lines, staining, and surface-level defects.
    Uses scratch detection via line detection and color uniformity analysis.
    """
    img_uint8 = (img * 255).astype(np.uint8) if img.dtype == np.float32 else img

    # Convert to LAB color space for better defect detection
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    L   = lab[:, :, 0]

    # ── Scratch detection via Canny + Hough lines ─────────────────────────
    edges     = cv2.Canny(L, 30, 90)
    lines     = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=15, maxLineGap=5)
    n_lines   = len(lines) if lines is not None else 0

    # ── Surface uniformity (std dev of luminance in central region) ───────
    h, w = L.shape
    cx1, cy1 = w // 5, h // 5
    cx2, cy2 = 4 * w // 5, 4 * h // 5
    center = L[cy1:cy2, cx1:cx2]
    lum_std = center.std()

    # ── Color blotchiness (staining indicator) ────────────────────────────
    A_channel = lab[:, :, 1]
    B_channel = lab[:, :, 2]
    color_std = (A_channel.std() + B_channel.std()) / 2

    # Composite defect score (higher = worse)
    defect_score = (n_lines * 3) + (lum_std * 1.5) + (color_std * 0.8)

    # Map to grade
    if defect_score < 15:       raw = 10.0
    elif defect_score < 30:     raw = 9.5
    elif defect_score < 55:     raw = 9.0
    elif defect_score < 90:     raw = 8.5
    elif defect_score < 140:    raw = 8.0
    elif defect_score < 210:    raw = 7.0
    elif defect_score < 310:    raw = 6.0
    elif defect_score < 450:    raw = 5.0
    elif defect_score < 650:    raw = 4.0
    elif defect_score < 950:    raw = 3.0
    elif defect_score < 1400:   raw = 2.0
    else:                        raw = 1.0

    return {
        "score":        snap_to_half(raw),
        "defect_score": round(defect_score, 1),
        "scratch_lines": n_lines,
        "lum_std":      round(lum_std, 2),
        "color_std":    round(color_std, 2),
    }


def surface_description(data: dict, score: float) -> str:
    lines = data["scratch_lines"]
    lum   = data["lum_std"]

    if score >= 9.5:
        return (
            "The surface appears pristine with no visible scratches, print lines, or staining "
            "under normal lighting conditions. Gloss retention is excellent and the holo or foil "
            "elements (if present) show no cloudiness or loss of reflectivity. The card surface "
            "is consistent with pack-fresh condition meeting PSA Gem Mint standards."
        )
    elif score >= 8.5:
        return (
            "Surface quality is very strong with only the faintest trace of wear detectable "
            "under angled lighting. Very minor print lines or light handling marks may be present "
            "but do not detract from the card's overall appearance. Gloss and reflectivity are "
            f"well-preserved, consistent with PSA Mint standards ({lines} minor surface marks detected)."
        )
    elif score >= 7.0:
        return (
            f"The surface shows light scratching or print lines visible under direct or angled "
            f"light ({lines} surface marks detected). Minor haze or cloudiness may be present on "
            f"holo surfaces, and slight handling wear is evident on the card face. Surface quality "
            f"at this level is consistent with PSA Near Mint grading."
        )
    elif score >= 5.0:
        return (
            f"Moderate surface wear is present with visible scratches, scuffs, or print lines "
            f"({lines} marks detected, luminance variance {lum:.1f}). Holo or foil elements may "
            f"show noticeable cloudiness or loss of reflectivity. Staining or ink transfer marks "
            f"may be visible, consistent with PSA Excellent to Very Good surface standards."
        )
    else:
        return (
            f"The surface shows significant damage including heavy scratching, major print lines, "
            f"staining, or creasing ({lines} surface defects detected). Holo elements may be "
            f"heavily damaged or completely clouded. Surface-level damage of this severity "
            f"substantially impacts the overall grade, consistent with PSA 1–4."
        )


# ─────────────────────────────────────────────────────────────────────────────
# FINAL GRADE CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────

# PSA's formula: the overall grade is approximately the average of the 4
# sub-grades, but surface and centering are slightly more weighted,
# and the minimum sub-grade acts as a hard cap.

WEIGHTS = {
    "centering": 0.20,
    "corners":   0.30,
    "edges":     0.25,
    "surface":   0.25,
}

PSA_GRADE_NAMES = {
    10:  "Gem Mint",
    9:   "Mint",
    8:   "Near Mint-Mint",
    7:   "Near Mint",
    6:   "Excellent-Mint",
    5:   "Excellent",
    4:   "Very Good-Excellent",
    3:   "Very Good",
    2:   "Good",
    1:   "Poor",
}

def compute_final_grade(scores: dict) -> dict:
    """
    Compute overall PSA grade from the 4 sub-scores.
    Weighted average + hard cap at lowest sub-grade + 1.
    """
    weighted = sum(scores[k] * WEIGHTS[k] for k in WEIGHTS)
    min_sub  = min(scores.values())

    # Hard cap: overall can't exceed min_sub + 1.5
    capped   = min(weighted, min_sub + 1.5)
    final    = snap_to_half(capped)

    # Map to integer PSA grade (0.5 grades are displayed but PSA uses integers)
    psa_int  = round(final)
    psa_name = PSA_GRADE_NAMES.get(psa_int, f"PSA {psa_int}")

    return {
        "final_grade":      final,
        "psa_integer_grade": psa_int,
        "psa_grade_name":   psa_name,
        "weighted_raw":     round(weighted, 2),
        "capped_at":        round(min_sub + 1.5, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ML MODEL PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def predict_ml(img_normalized: np.ndarray) -> dict | None:
    """Run the trained Keras model and return probabilities + grade."""
    model = get_model()
    if model is None:
        return None

    import tensorflow as tf
    batch = np.expand_dims(img_normalized, axis=0)
    probs = model.predict(batch, verbose=0)[0]   # shape (10,)
    grades = np.arange(1, 11)

    predicted_class = int(np.argmax(probs)) + 1
    confidence      = float(probs[predicted_class - 1]) * 100
    weighted_avg    = float(np.sum(probs * grades))

    return {
        "ml_grade":      snap_to_half(weighted_avg),
        "ml_confidence": round(confidence, 1),
        "probabilities": {
            f"PSA {i+1}": round(float(p) * 100, 2)
            for i, p in enumerate(probs)
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# FULL GRADING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def grade_card(image_path: str) -> dict:
    """
    Full grading pipeline for a single card image.
    Returns all 4 criteria with scores and descriptions, plus final grade.
    """
    path = Path(image_path)
    if not path.exists():
        return {"error": f"File not found: {image_path}"}

    img_norm = load_image(str(path))
    img_full = load_image_full(str(path))

    if img_norm is None or img_full is None:
        return {"error": f"Could not load image: {image_path}"}

    # ── Run all 4 criteria ────────────────────────────────────────────────
    cen_data = analyze_centering(img_norm)
    cor_data = analyze_corners(img_full)
    edg_data = analyze_edges(img_full)
    sur_data = analyze_surface(img_full)

    scores = {
        "centering": cen_data["score"],
        "corners":   cor_data["score"],
        "edges":     edg_data["score"],
        "surface":   sur_data["score"],
    }

    # ── Final grade calculation ────────────────────────────────────────────
    final = compute_final_grade(scores)

    # ── ML model prediction (if model is trained) ─────────────────────────
    ml_result = predict_ml(img_norm)

    # ── Blend ML + pixel analysis if model available ──────────────────────
    if ml_result:
        blended = snap_to_half(
            ml_result["ml_grade"] * 0.60 + final["final_grade"] * 0.40
        )
    else:
        blended = final["final_grade"]

    blended_int  = round(blended)
    blended_name = PSA_GRADE_NAMES.get(blended_int, f"PSA {blended_int}")

    return {
        "file":          path.name,
        "overall_grade": blended,
        "grade_name":    blended_name,

        "criteria": {
            "centering": {
                "score":       cen_data["score"],
                "description": centering_description(cen_data, cen_data["score"]),
                "data":        cen_data,
            },
            "corners": {
                "score":       cor_data["score"],
                "description": corners_description(cor_data, cor_data["score"]),
                "data":        cor_data,
            },
            "edges": {
                "score":       edg_data["score"],
                "description": edges_description(edg_data, edg_data["score"]),
                "data":        edg_data,
            },
            "surface": {
                "score":       sur_data["score"],
                "description": surface_description(sur_data, sur_data["score"]),
                "data":        sur_data,
            },
        },

        "grade_breakdown": final,
        "ml_result":       ml_result,
        "mode":            "ml+pixel" if ml_result else "pixel-only",
    }


# ─────────────────────────────────────────────────────────────────────────────
# PRETTY PRINTER
# ─────────────────────────────────────────────────────────────────────────────

STARS = {10: "★★★★★", 9: "★★★★½", 8: "★★★★☆", 7: "★★★½☆",
         6: "★★★☆☆", 5: "★★½☆☆", 4: "★★☆☆☆", 3: "★½☆☆☆",
         2: "★☆☆☆☆", 1: "☆☆☆☆☆"}

GRADE_COLORS = {
    10: "\033[92m", 9: "\033[92m", 8: "\033[96m", 7: "\033[96m",
    6:  "\033[93m", 5: "\033[93m", 4: "\033[91m", 3: "\033[91m",
    2:  "\033[91m", 1: "\033[91m",
}
RESET = "\033[0m"
BOLD  = "\033[1m"

def grade_bar(score: float, width: int = 20) -> str:
    filled = round((score / 10) * width)
    color  = GRADE_COLORS.get(round(score), "")
    return color + "█" * filled + "░" * (width - filled) + RESET

def print_result(result: dict):
    if "error" in result:
        print(f"\n  ✗ ERROR: {result['error']}\n")
        return

    g     = result["overall_grade"]
    gi    = round(g)
    gname = result["grade_name"]
    mode  = result["mode"]
    stars = STARS.get(gi, "")
    col   = GRADE_COLORS.get(gi, "")

    print("\n" + "═" * 60)
    print(f"  {BOLD}CARD GRADE REPORT{RESET}  [{mode}]")
    print(f"  File: {result['file']}")
    print("═" * 60)
    print(f"\n  {col}{BOLD}PSA {g}  —  {gname}{RESET}  {stars}")
    print(f"  {grade_bar(g, 30)}\n")

    print(f"  {'CRITERIA':<14} {'SCORE':>6}   {'BAR'}")
    print("  " + "─" * 50)

    for name, crit in result["criteria"].items():
        sc  = crit["score"]
        bar = grade_bar(sc, 20)
        print(f"  {name.capitalize():<14} {sc:>5.1f}   {bar}")

    print("\n" + "─" * 60)
    print(f"  {BOLD}DETAILED ASSESSMENT{RESET}")
    print("─" * 60)

    labels = {"centering": "CENTERING", "corners": "CORNERS",
              "edges": "EDGES",    "surface": "SURFACE"}

    for key, crit in result["criteria"].items():
        sc  = crit["score"]
        col = GRADE_COLORS.get(round(sc), "")
        print(f"\n  {BOLD}{labels[key]}{RESET}  {col}{sc}/10{RESET}")

        # Wrap description to 56 chars
        desc  = crit["description"]
        words = desc.split()
        line  = "  "
        for word in words:
            if len(line) + len(word) + 1 > 58:
                print(line)
                line = "  " + word + " "
            else:
                line += word + " "
        if line.strip():
            print(line)

    if result.get("ml_result"):
        ml = result["ml_result"]
        print(f"\n  {BOLD}ML MODEL{RESET}  grade {ml['ml_grade']} "
              f"(confidence {ml['ml_confidence']}%)")

    print("\n" + "═" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# DATASET ACCURACY TEST
# ─────────────────────────────────────────────────────────────────────────────

def test_dataset(data_dir: str, sample: int = 50):
    """
    Test grader accuracy against labeled training_data/ folder.
    Samples `sample` images per grade and reports per-grade accuracy.
    """
    base   = Path(data_dir)
    errors = []
    total  = 0
    exact  = 0
    within1 = 0

    print(f"\n{'═'*60}")
    print(f"  DATASET ACCURACY TEST — {data_dir}")
    print(f"  Sampling up to {sample} images per grade")
    print(f"{'═'*60}\n")

    for true_grade in range(1, 11):
        folder = base / f"grade_{true_grade}"
        if not folder.exists():
            print(f"  PSA {true_grade:2d}: folder not found — skipping")
            continue

        images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        if not images:
            print(f"  PSA {true_grade:2d}: no images found")
            continue

        sample_imgs = random.sample(images, min(sample, len(images)))
        grade_errors = []

        for img_path in sample_imgs:
            result = grade_card(str(img_path))
            if "error" in result:
                continue
            pred = result["overall_grade"]
            err  = abs(pred - true_grade)
            grade_errors.append(err)
            errors.append(err)
            total += 1
            if err < 0.6:  exact   += 1
            if err < 1.6:  within1 += 1

        if grade_errors:
            mae = np.mean(grade_errors)
            acc = sum(1 for e in grade_errors if e < 0.6) / len(grade_errors) * 100
            bar = "█" * int(acc / 5)
            print(f"  PSA {true_grade:2d}: MAE={mae:.2f}  Exact={acc:5.1f}%  {bar}")

    if total > 0:
        overall_mae = np.mean(errors)
        overall_acc = exact / total * 100
        w1_acc      = within1 / total * 100
        print(f"\n{'─'*60}")
        print(f"  OVERALL  n={total}  MAE={overall_mae:.2f}  "
              f"Exact={overall_acc:.1f}%  Within±1={w1_acc:.1f}%")
        print(f"{'═'*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Pokemon Card PSA Grader + Price Comparator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
GRADING MODE:
  python grader.py card.jpg
  python grader.py card1.jpg card2.jpg --json
  python grader.py --test-dataset training_data/

COMPARISON MODE:
  python grader.py --compare "PSA 10 Charizard Base Set"
  python grader.py --compare "https://www.ebay.com/itm/123456"
  python grader.py --compare "Pikachu 1st Edition" --min-price 10 --max-price 500
  python grader.py --compare "PSA 9 Blastoise" --platforms ebay tcgplayer mercari

COMBINED (grade a card AND find cheapest price):
  python grader.py card.jpg --compare "PSA 10 Charizard"
        """,
    )

    # ── Grading args ──────────────────────────────────────────────────────
    parser.add_argument("images",         nargs="*",
                        help="Image file(s) to grade")
    parser.add_argument("--test-dataset", metavar="DIR",
                        help="Run accuracy test against training_data/ folder")
    parser.add_argument("--json",         action="store_true",
                        help="Output raw JSON")
    parser.add_argument("--sample",       type=int, default=30,
                        help="Images to sample per grade during testing")

    # ── Comparison args ───────────────────────────────────────────────────
    parser.add_argument("--compare",      metavar="QUERY_OR_URL",
                        help="Compare prices across platforms for this card name or URL")
    parser.add_argument("--platforms",    nargs="+",
                        choices=["ebay","amazon","tcgplayer","mercari",
                                 "facebook","craigslist","whatnot"],
                        default=["ebay","amazon","tcgplayer","mercari",
                                 "facebook","craigslist","whatnot"],
                        help="Platforms to search (default: all)")
    parser.add_argument("--min-price",    type=float, default=0.0)
    parser.add_argument("--max-price",    type=float, default=999999.0)
    parser.add_argument("--max-results",  type=int,   default=8,
                        help="Max results per platform")

    args = parser.parse_args()

    # ── Dataset test mode ─────────────────────────────────────────────────
    if args.test_dataset:
        test_dataset(args.test_dataset, sample=args.sample)
        return

    # ── Comparison mode ───────────────────────────────────────────────────
    if args.compare:
        from compare import normalize_query, compare_listings, print_comparison
        resolved = normalize_query(args.compare)
        query    = resolved["query"]
        if resolved["source_url"]:
            print(f"\n  Source URL → extracted query: '{query}'")
        results = compare_listings(
            query       = query,
            platforms   = args.platforms,
            min_price   = args.min_price,
            max_price   = args.max_price,
            max_results = args.max_results,
        )
        if args.json:
            import json as _json
            output = {
                "query":    results["query"],
                "cheapest": results["cheapest"].to_dict() if results["cheapest"] else None,
                "total_found": results["total_found"],
                "all_listings": [l.to_dict() for l in results["all_listings"]],
            }
            print(_json.dumps(output, indent=2, default=str))
        else:
            print_comparison(results)

        # If images were also provided, fall through to grade them too
        if not args.images:
            return

    # ── Grading mode ──────────────────────────────────────────────────────
    if not args.images:
        parser.print_help()
        print("\nExamples:")
        print("  python grader.py my_card.jpg")
        print("  python grader.py --compare 'PSA 10 Charizard Base Set'")
        return

    results = []
    for img_path in args.images:
        result = grade_card(img_path)
        results.append(result)
        if args.json:
            import json as _json
            print(_json.dumps(result, indent=2, default=str))
        else:
            print_result(result)

    if len(results) > 1 and not args.json:
        print(f"\n  Graded {len(results)} cards:")
        for r in results:
            if "error" not in r:
                print(f"    {r['file']:<40} PSA {r['overall_grade']}  {r['grade_name']}")



if __name__ == "__main__":
    main()