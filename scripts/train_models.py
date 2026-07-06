"""Train both ML models on the project dataset.

1. Random Forest region model - learns to classify 8x8 blocks as
   busy/medium/smooth from texture features (std + edge strength) of every
   image (passport photos + benchmarks); used to rank embedding regions.
2. SVM steganalysis classifier - trained to distinguish clean covers from
   stego images produced by (a) our 2's complement method and (b) the
   traditional sequential LSB baseline, so the evaluation can compare
   detectability of both.
"""

import sys
import numpy as np
import cv2
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stegoapp import core, regions, steganalysis

rng = np.random.default_rng(42)

PICS = sorted((ROOT / "PICS").glob("*.png"))
BENCH = sorted((ROOT / "benchmarks").glob("*.png"))
IMAGES = PICS + BENCH
print(f"dataset: {len(PICS)} passport photos + {len(BENCH)} benchmarks")


def load(p):
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"unreadable image: {p}")
    return img


# ------------------------------------------------- 1. Random Forest regions
print("\n[1/2] training Random Forest region model on block texture features...")
feats, labels = [], []
for p in IMAGES:
    f, _, _ = regions.block_features(load(p))
    feats.append(f)
    labels.append(regions.texture_labels(f))
X = np.concatenate(feats).astype(np.float64)
y = np.concatenate(labels)
print(f"  {X.shape[0]} blocks x {X.shape[1]} features "
      f"(smooth {np.mean(y==0)*100:.0f}% / medium {np.mean(y==1)*100:.0f}% / busy {np.mean(y==2)*100:.0f}%)")

rf_regions = RandomForestClassifier(n_estimators=100, max_depth=12,
                                    random_state=42, n_jobs=-1)
acc = cross_val_score(rf_regions, X, y, cv=5)
print(f"  5-fold CV accuracy (block texture class): {acc.mean()*100:.1f}%")
rf_regions.fit(X, y)
joblib.dump(rf_regions, regions.MODEL_PATH)
print(f"  saved -> {regions.MODEL_PATH}")
regions._model_cache = None  # reload with the freshly trained model

# ------------------------------------------------------ 2. SVM steganalysis
print("\n[2/2] training SVM steganalysis classifier (clean vs falsified)...")


def lsb_fill(cover, frac):
    stego = cover.copy(); flat = stego.reshape(-1)
    n = int(flat.size * frac)
    flat[:n] = (flat[:n] & 0xFE) | rng.integers(0, 2, n, dtype=np.uint8)
    return stego


def ours_fill(cover, frac):
    cap = core.capacity_bytes(cover, 1)
    payload = rng.integers(0, 256, size=max(1, int(cap * frac)), dtype=np.uint8).tobytes()
    return core.embed(cover, payload, core.TYPE_TEXT, regions.slot_order(cover), 1)


def make_svm():
    return make_pipeline(StandardScaler(),
                         SVC(kernel="rbf", C=10, gamma="scale", probability=True,
                             random_state=42))


# --- balanced detector: 1 clean + 1 falsified per image (mixed method & rate) ---
X_det, y_det = [], []
for i, p in enumerate(IMAGES):
    cover = load(p)
    X_det.append(steganalysis.features(cover)); y_det.append(0)
    frac = rng.choice([0.4, 0.6, 0.8, 1.0])
    stego = ours_fill(cover, frac) if i % 2 == 0 else lsb_fill(cover, frac)
    X_det.append(steganalysis.features(stego)); y_det.append(1)
X_det = np.array(X_det); y_det = np.array(y_det)
print(f"  detector training: {len(y_det)} balanced samples "
      f"({(y_det==0).sum()} clean / {(y_det==1).sum()} falsified)")
det_cv = cross_val_score(make_svm(), X_det, y_det, cv=5)
print(f"  5-fold detection accuracy (mixed rates 40-100%): {det_cv.mean()*100:.1f}% "
      f"+/- {det_cv.std()*100:.1f}")

svm = make_svm(); svm.fit(X_det, y_det)
joblib.dump(svm, steganalysis.MODEL_PATH)
print(f"  saved -> {steganalysis.MODEL_PATH}")
steganalysis._model_cache = None

# --- per-method detectability at 50% capacity (balanced cover vs method) ---
report = {}
for tag, fn in (("ours", ours_fill), ("lsb", lsb_fill)):
    Xa, ya = [], []
    for p in IMAGES:
        cover = load(p)
        Xa.append(steganalysis.features(cover)); ya.append(0)
        Xa.append(steganalysis.features(fn(cover, 0.5))); ya.append(1)
    acc = cross_val_score(make_svm(), np.array(Xa), np.array(ya), cv=5)
    report[tag] = float(acc.mean())
    label = "2's complement + ML (ours)" if tag == "ours" else "traditional sequential LSB"
    print(f"  detectability of {label} @50%: {acc.mean()*100:.1f}% (50% = undetectable)")

import json
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "steganalysis_report.json").write_text(json.dumps({
    "classifier": "SVM (RBF kernel)",
    "region_model": "Random Forest",
    "detection_accuracy": float(det_cv.mean()),
    "detectability_ours": report["ours"],
    "detectability_lsb": report["lsb"],
    "n_samples": int(len(y_det)),
}, indent=2))
print("\ndone.")
