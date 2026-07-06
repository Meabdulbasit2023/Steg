"""Image-quality metrics for Objective iii: MSE and PSNR."""

import numpy as np


def mse(original: np.ndarray, stego: np.ndarray) -> float:
    diff = original.astype(np.float64) - stego.astype(np.float64)
    return float(np.mean(diff * diff))


def psnr(original: np.ndarray, stego: np.ndarray) -> float:
    m = mse(original, stego)
    if m == 0:
        return float("inf")
    return float(10.0 * np.log10((255.0 ** 2) / m))
