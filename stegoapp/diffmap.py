"""Difference-map visualization: shows exactly where (and how strongly) a
stego image differs from its cover. The raw difference is only 0-3 pixel
levels, so it is amplified and colour-mapped to be visible."""

import numpy as np
import cv2


def difference_map(cover: np.ndarray, stego: np.ndarray, amp: int = 40):
    """Return (heatmap_bgr, stats).

    heatmap: amplified |cover - stego| run through a perceptual colour map.
    stats:   dict with changed-pixel %, max per-channel delta, mean delta.
    """
    if cover.shape != stego.shape:
        stego = cv2.resize(stego, (cover.shape[1], cover.shape[0]))
    diff = np.abs(cover.astype(np.int16) - stego.astype(np.int16)).astype(np.float32)

    changed = np.any(diff > 0, axis=2) if diff.ndim == 3 else diff > 0
    stats = {
        "changed_pct": float(changed.mean() * 100),
        "max_delta": int(diff.max()),
        "mean_delta": float(diff.mean()),
    }

    gray = diff.mean(axis=2) if diff.ndim == 3 else diff
    vis = np.clip(gray * amp, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
    return heat, stats
