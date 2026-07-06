"""ML-guided embedding region selection (Random Forest).

A Random Forest classifier (trained on texture features of the project
dataset) ranks 8x8 blocks of the cover image: busy/textured blocks are used
first, smooth blocks last, so the embedded bits stay imperceptible.

Determinism requirement: the decoder must derive the SAME slot order from the
stego image alone. Features are therefore computed on the image with its low
2 bits zeroed out ("masked") — embedding only writes those bits, so cover and
stego produce identical features and identical orderings.
"""

import numpy as np
import cv2
import joblib
from pathlib import Path

from .core import HEADER_SLOTS

BLOCK = 8
N_CLASSES = 3  # 0 = smooth, 1 = medium, 2 = busy
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "rf_regions.joblib"

_model_cache = None


def load_model():
    global _model_cache
    if _model_cache is None and MODEL_PATH.exists():
        _model_cache = joblib.load(MODEL_PATH)
    return _model_cache


def block_features(img: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Texture features for every 8x8 block of the LSB-masked image.

    Returns (features [n_blocks, 2], blocks_y, blocks_x). Features are the
    grayscale standard deviation and mean Sobel gradient magnitude.
    """
    masked = (img & 0xFC).astype(np.uint8)
    gray = masked.mean(axis=2).astype(np.float32) if masked.ndim == 3 else masked.astype(np.float32)
    h, w = gray.shape
    by, bx = h // BLOCK, w // BLOCK

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)

    g = gray[:by * BLOCK, :bx * BLOCK].reshape(by, BLOCK, bx, BLOCK)
    m = mag[:by * BLOCK, :bx * BLOCK].reshape(by, BLOCK, bx, BLOCK)
    std = g.std(axis=(1, 3)).reshape(-1)
    edge = m.mean(axis=(1, 3)).reshape(-1)
    return np.stack([std, edge], axis=1), by, bx


def texture_labels(feats: np.ndarray) -> np.ndarray:
    """Teacher labels used to train the Random Forest: per-image texture
    terciles of block standard deviation (0 smooth / 1 medium / 2 busy)."""
    std = feats[:, 0]
    lo, hi = np.quantile(std, [1 / 3, 2 / 3])
    return np.where(std >= hi, 2, np.where(std >= lo, 1, 0)).astype(np.int64)


def slot_order(img: np.ndarray) -> np.ndarray:
    """Deterministic embedding order of channel-slots, best blocks first.

    The trained Random Forest classifies each block as busy/medium/smooth;
    busy blocks are used first, tie-broken by per-block std then block index.
    Falls back to a pure std ranking when no model file exists. Slots covered
    by the fixed header, and pixels outside complete 8x8 blocks, are appended
    last so the full image capacity stays usable.
    """
    h, w = img.shape[:2]
    c = img.shape[2] if img.ndim == 3 else 1
    feats, by, bx = block_features(img)

    model = load_model()
    if model is not None and len(feats):
        busy_class = model.predict(feats.astype(np.float64))
        priority = (N_CLASSES - 1 - busy_class).astype(np.float64)  # busy first
    else:
        priority = np.zeros(len(feats))

    order = np.lexsort((np.arange(len(feats)), -feats[:, 0], priority))

    # channel-slot indices for each block, emitted in ranked block order
    ys = (np.arange(len(feats)) // bx) * BLOCK
    xs = (np.arange(len(feats)) % bx) * BLOCK
    dy, dx = np.meshgrid(np.arange(BLOCK), np.arange(BLOCK), indexing="ij")
    base = (dy.reshape(-1)[:, None] * w + dx.reshape(-1)[:, None]) * c + np.arange(c)[None, :]
    base = base.reshape(-1)  # offsets within a block, all channels

    starts = (ys[order] * w + xs[order]) * c
    slots = (starts[:, None] + base[None, :]).reshape(-1)

    # anything not covered by complete blocks (right/bottom edges)
    all_slots = np.arange(h * w * c)
    in_blocks = np.zeros(h * w * c, dtype=bool)
    in_blocks[slots] = True
    leftovers = all_slots[~in_blocks]

    full = np.concatenate([slots, leftovers])
    return full[full >= HEADER_SLOTS]


def sequential_order(img: np.ndarray) -> np.ndarray:
    """Baseline (traditional LSB): plain raster order, no ML ranking."""
    return np.arange(img.size)[HEADER_SLOTS:]
