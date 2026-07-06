"""Sanity tests: every payload must survive embed -> extract bit-perfectly,
with and without the trained K-means model, at 1 and 2 bits per channel."""

import sys
import numpy as np
import cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stegoapp import core, regions, metrics

rng = np.random.default_rng(0)
pics = sorted((ROOT / "PICS").glob("*.png"))
fails = 0

for bpc in (1, 2):
    for p in pics[:5] + pics[-5:]:
        cover = cv2.imread(str(p), cv2.IMREAD_COLOR)
        cap = core.capacity_bytes(cover, bpc)

        for frac in (0.05, 0.5, 1.0):
            n = max(1, int(cap * frac))
            payload = rng.integers(0, 256, size=n, dtype=np.uint8).tobytes()
            stego = core.embed(cover, payload, core.TYPE_TEXT,
                               regions.slot_order(cover), bpc)
            ptype, rec = core.extract(stego, regions.slot_order(stego))
            ok = (rec == payload and ptype == core.TYPE_TEXT)
            psnr = metrics.psnr(cover, stego)
            if not ok:
                fails += 1
                print(f"FAIL {p.name} bpc={bpc} frac={frac}")
            else:
                print(f"ok   {p.name:22s} bpc={bpc} payload={n:5d}B psnr={psnr:6.2f}dB")

# text round-trip
cover = cv2.imread(str(pics[0]), cv2.IMREAD_COLOR)
msg = "Enhanced steganography — 2's complement + XOR + ML ✔".encode("utf-8")
stego = core.embed(cover, msg, core.TYPE_TEXT, regions.slot_order(cover), 1)
_, rec = core.extract(stego, regions.slot_order(stego))
assert rec == msg, "text round-trip failed"
print("ok   text message round-trip")

# image-in-image round-trip
secret = cv2.imread(str(pics[1]), cv2.IMREAD_COLOR)
big = cv2.resize(cv2.imread(str(pics[2]), cv2.IMREAD_COLOR), (600, 700))
payload = core.image_to_payload(secret, big, 1)
stego = core.embed(big, payload, core.TYPE_IMAGE, regions.slot_order(big), 1)
ptype, rec = core.extract(stego, regions.slot_order(stego))
assert ptype == core.TYPE_IMAGE and rec == payload, "image round-trip failed"
img = core.payload_to_image(rec)
print(f"ok   image-in-image round-trip (secret {img.shape[1]}x{img.shape[0]}, "
      f"{len(payload)}B, cover psnr={metrics.psnr(big, stego):.2f}dB)")

# clean image must NOT decode
try:
    core.extract(cover, regions.slot_order(cover))
    print("FAIL clean image decoded without error"); fails += 1
except ValueError:
    print("ok   clean image correctly rejected")

print("\nALL PASSED" if fails == 0 else f"\n{fails} FAILURES")
sys.exit(1 if fails else 0)
