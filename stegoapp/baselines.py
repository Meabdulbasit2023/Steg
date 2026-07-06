"""Traditional steganography baselines for comparison (Objective iii:
'compare results with traditional techniques, e.g. LSB, DCT').

Both operate on a known-length bit payload so the caller (which controls the
payload) can measure extraction accuracy directly — no header needed.

  - lsb_*  : classic spatial least-significant-bit in raster order.
  - dct_*  : frequency-domain embedding via quantization-index modulation
             on a mid-frequency DCT coefficient of each 8x8 block. This is
             lower capacity and lower PSNR than spatial LSB, but far more
             robust to JPEG compression — the classic LSB-vs-DCT trade-off.
"""

import numpy as np
import cv2

DCT_Q = 20            # QIM quantization step (larger = more robust, lower PSNR)
DCT_COEFF = (4, 3)    # mid-frequency coefficient used to carry one bit


# ------------------------------------------------------------- spatial LSB
def lsb_capacity_bits(img: np.ndarray) -> int:
    return int(img.size)


def lsb_embed(cover: np.ndarray, bits: np.ndarray) -> np.ndarray:
    stego = cover.copy()
    flat = stego.reshape(-1)
    n = len(bits)
    flat[:n] = (flat[:n] & 0xFE) | bits.astype(np.uint8)
    return stego


def lsb_extract(stego: np.ndarray, n_bits: int) -> np.ndarray:
    return (stego.reshape(-1)[:n_bits] & 1).astype(np.uint8)


# --------------------------------------------------------------- DCT (QIM)
def dct_capacity_bits(img: np.ndarray) -> int:
    h, w = img.shape[:2]
    c = img.shape[2] if img.ndim == 3 else 1
    return (h // 8) * (w // 8) * c


def dct_embed(cover: np.ndarray, bits: np.ndarray,
              Q: int = DCT_Q, coeff=DCT_COEFF) -> np.ndarray:
    out = cover.astype(np.float32).copy()
    c = out.shape[2]
    n = len(bits)
    idx = 0
    for ch in range(c):
        h, w = out[:, :, ch].shape
        for by in range(0, (h // 8) * 8, 8):
            for bx in range(0, (w // 8) * 8, 8):
                if idx >= n:
                    return np.clip(np.round(out), 0, 255).astype(np.uint8)
                block = out[by:by + 8, bx:bx + 8, ch].copy()
                d = cv2.dct(block)
                step = int(round(d[coeff] / Q))
                step = (step & ~1) | int(bits[idx])
                d[coeff] = step * Q
                out[by:by + 8, bx:bx + 8, ch] = cv2.idct(d)
                idx += 1
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def dct_extract(stego: np.ndarray, n_bits: int,
                Q: int = DCT_Q, coeff=DCT_COEFF) -> np.ndarray:
    s = stego.astype(np.float32)
    c = s.shape[2]
    bits = np.zeros(n_bits, dtype=np.uint8)
    idx = 0
    for ch in range(c):
        h, w = s[:, :, ch].shape
        for by in range(0, (h // 8) * 8, 8):
            for bx in range(0, (w // 8) * 8, 8):
                if idx >= n_bits:
                    return bits
                d = cv2.dct(s[by:by + 8, bx:bx + 8, ch].copy())
                bits[idx] = int(round(d[coeff] / Q)) & 1
                idx += 1
    return bits
