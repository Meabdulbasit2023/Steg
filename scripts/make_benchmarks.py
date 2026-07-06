"""Build the varied-image-type benchmark set (Objective iii: 'repeat tests
with varying image types') from scikit-image's bundled standard test images.
No downloads required."""

import cv2
import numpy as np
from pathlib import Path
from skimage import data

OUT = Path(__file__).resolve().parent.parent / "benchmarks"
OUT.mkdir(exist_ok=True)

SOURCES = {
    "person_astronaut": (data.astronaut, "person"),
    "animal_cat": (data.chelsea, "animal"),
    "scene_coffee": (data.coffee, "scene"),
    "landscape_moon": (data.moon, "landscape"),
    "texture_brick": (data.brick, "texture"),
    "texture_grass": (data.grass, "texture"),
    "texture_gravel": (data.gravel, "texture"),
    "object_camera": (data.camera, "object"),
    "object_coins": (data.coins, "object"),
    "scene_rocket": (data.rocket, "scene"),
}

manifest = []
for name, (fn, itype) in SOURCES.items():
    img = fn()
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    path = OUT / f"{name}.png"
    cv2.imwrite(str(path), img)
    manifest.append(f"{name}.png,{itype},{img.shape[1]}x{img.shape[0]}")
    print(f"wrote {path.name:24s} {itype:10s} {img.shape[1]}x{img.shape[0]}")

(OUT / "manifest.csv").write_text("file,type,size\n" + "\n".join(manifest) + "\n")
print(f"\n{len(manifest)} benchmark images in {OUT}")
