"""Robustness testing (Objective iv: 'evaluate robustness — apply compression,
cropping, format conversion, and test message recovery rate').

A payload is embedded with our 2's complement + ML method, the stego image is
subjected to each attack, and the byte-level recovery rate is measured. High-
capacity spatial methods (ours and LSB) are, by design, fragile to lossy
operations — PNG re-save is lossless (100%) while JPEG/crop/resize degrade
recovery. Demonstrating this is exactly what the objective asks for.
"""

import numpy as np
import cv2

from . import core, regions


def _recovery_rate(payload: bytes, attacked: np.ndarray) -> float:
    """Best-effort byte recovery from an attacked stego image."""
    try:
        order = regions.slot_order(attacked)
        ptype, rec = core.extract(attacked, order)
    except Exception:
        # header destroyed: fall back to reading the raw payload region
        try:
            n_bits = len(payload) * 8
            bits = core._read_bits(attacked.reshape(-1),
                                   regions.slot_order(attacked)[:int(np.ceil(n_bits))], 1)[:n_bits]
            rec = core.bits_to_bytes(bits)
        except Exception:
            return 0.0
    a = np.frombuffer(payload, np.uint8)
    b = np.frombuffer(rec[:len(payload)].ljust(len(payload), b"\0"), np.uint8)
    return float((a == b).mean() * 100)


def _jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _crop(img: np.ndarray, frac: float) -> np.ndarray:
    h, w = img.shape[:2]
    dy, dx = int(h * frac), int(w * frac)
    cropped = img[dy:h - dy, dx:w - dx]
    return cv2.copyMakeBorder(cropped, dy, dy, dx, dx, cv2.BORDER_CONSTANT)


def _resize_roundtrip(img: np.ndarray, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
    return cv2.resize(small, (w, h))


def run_suite(cover: np.ndarray, payload: bytes) -> dict:
    """Embed the payload, apply every attack, and report recovery rates."""
    stego = core.embed(cover, payload, core.TYPE_TEXT, regions.slot_order(cover), 1)

    tests = [
        ("PNG re-save (lossless)", stego.copy(), "control — should be 100%"),
        ("JPEG quality 95", _jpeg(stego, 95), "light compression"),
        ("JPEG quality 75", _jpeg(stego, 75), "moderate compression"),
        ("JPEG quality 50", _jpeg(stego, 50), "heavy compression"),
        ("Crop 5% border", _crop(stego, 0.05), "geometric edit"),
        ("Resize 50%→100%", _resize_roundtrip(stego, 0.5), "scale round-trip"),
    ]
    results = []
    for name, attacked, desc in tests:
        results.append({
            "attack": name, "desc": desc,
            "recovery": round(_recovery_rate(payload, attacked), 1),
        })
    return {"payload_bytes": len(payload), "results": results}
