"""ML steganalysis: a trained SVM classifier that detects falsified images
(images that contain hidden data).

Two uses:
  1. Detection (the "Detect" page) — given any image, decide clean vs
     falsified and report a suspicion score.
  2. Evaluation (Objective iii) — if the attacker cannot separate our stego
     images from clean covers (accuracy near 50% = coin-flip), the hiding
     method is statistically undetectable.

SVM is the classic steganalysis classifier in the literature. Features are
LSB-plane statistics, dominated by the Pairs-of-Values chi-square test which
collapses toward zero when a bit-plane has been overwritten.
"""

import numpy as np
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "steganalysis_svm.joblib"

_model_cache = None


def load_model():
    global _model_cache
    if _model_cache is None and MODEL_PATH.exists():
        _model_cache = joblib.load(MODEL_PATH)
    return _model_cache


def _chi_square_pov(v: np.ndarray) -> float:
    """Pairs-of-Values chi-square (classic LSB detector). Clean images have
    asymmetric (2i, 2i+1) histogram pairs -> high value; LSB embedding
    equalizes the pairs -> value collapses toward zero."""
    h = np.bincount(v.reshape(-1), minlength=256).astype(np.float64)
    even, odd = h[0::2], h[1::2]
    exp = (even + odd) / 2
    m = exp > 0
    return float((((even[m] - exp[m]) ** 2) / exp[m]).sum() / max(1.0, exp[m].sum()))


def features(img: np.ndarray) -> np.ndarray:
    """Statistical features sensitive to LSB-plane embedding, per channel:
    LSB-plane bias, Pairs-of-Values chi-square, PoV histogram asymmetry,
    LSB spatial correlation (H + V), and correlation between bit-plane 0 and 1."""
    if img.ndim == 2:
        img = img[:, :, None]
    feats = []
    for ch in range(img.shape[2]):
        v = img[:, :, ch]
        lsb = (v & 1).astype(np.float64)
        b1 = ((v >> 1) & 1).astype(np.float64)

        feats.append(lsb.mean())
        feats.append(_chi_square_pov(v))

        hist = np.bincount(v.reshape(-1), minlength=256).astype(np.float64)
        pairs = np.abs(hist[0::2] - hist[1::2]).sum() / max(1.0, hist.sum())
        feats.append(pairs)

        feats.append(float((lsb[:, :-1] == lsb[:, 1:]).mean()))
        feats.append(float((lsb[:-1, :] == lsb[1:, :]).mean()))
        feats.append(float((lsb == b1).mean()))
    return np.array(feats, dtype=np.float64)


def predict_stego_probability(img: np.ndarray) -> float | None:
    """Probability that an image contains hidden data, per the trained model.
    Returns None if the model has not been trained yet."""
    model = load_model()
    if model is None:
        return None
    return float(model.predict_proba(features(img)[None, :])[0, 1])


def verdict(prob: float) -> dict:
    """Human-facing detection verdict from a suspicion probability."""
    pct = prob * 100
    if prob >= 0.65:
        return {"label": "Falsified — hidden data detected", "level": "high",
                "detail": "Strong statistical evidence of embedded data."}
    if prob >= 0.5:
        return {"label": "Suspicious — likely contains hidden data", "level": "medium",
                "detail": "LSB-plane statistics deviate from a clean image."}
    if prob >= 0.35:
        return {"label": "Probably clean", "level": "low",
                "detail": "Minor anomalies, but within the normal range."}
    return {"label": "Clean — no hidden data detected", "level": "clean",
            "detail": "Pixel statistics match an untouched image."}
