"""Full evaluation over the dataset (Objective iii).

For every image (198 passport photos + benchmarks), hides a payload at 50%
of capacity with (a) our 2's complement + XOR + ML-region method and (b) the
traditional sequential LSB baseline, then measures MSE, PSNR, extraction
accuracy, and embed/extract time. Also runs an image-in-image demo and the
trained steganalysis attacker. Results go to results/results.json and charts
to webapp/static/.
"""

import sys, json, time
import numpy as np
import cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stegoapp import core, regions, metrics, steganalysis

rng = np.random.default_rng(7)

PICS = sorted((ROOT / "PICS").glob("*.png"))
BENCH = sorted((ROOT / "benchmarks").glob("*.png"))


def img_type(p: Path) -> str:
    if p.parent.name == "PICS":
        return "passport"
    return p.stem.split("_")[0]


rows = []
for p in PICS + BENCH:
    cover = cv2.imread(str(p), cv2.IMREAD_COLOR)
    cap = core.capacity_bytes(cover, 1)
    payload = rng.integers(0, 256, size=max(1, cap // 2), dtype=np.uint8).tobytes()

    t0 = time.perf_counter()
    order = regions.slot_order(cover)
    stego = core.embed(cover, payload, core.TYPE_TEXT, order, 1)
    t_embed = time.perf_counter() - t0

    t0 = time.perf_counter()
    _, recovered = core.extract(stego, regions.slot_order(stego))
    t_extract = time.perf_counter() - t0

    acc = 100.0 * (np.frombuffer(recovered, np.uint8) ==
                   np.frombuffer(payload, np.uint8)).mean()

    # traditional LSB baseline (same payload, raster order, no 2C/XOR)
    lsb = cover.copy()
    flat = lsb.reshape(-1)
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    flat[np.arange(len(bits))] = (flat[np.arange(len(bits))] & 0xFE) | bits

    p_ours = steganalysis.predict_stego_probability(stego)
    rows.append({
        "file": p.name, "type": img_type(p),
        "w": cover.shape[1], "h": cover.shape[0],
        "capacity_bytes": cap, "payload_bytes": len(payload),
        "mse": metrics.mse(cover, stego), "psnr": metrics.psnr(cover, stego),
        "mse_lsb": metrics.mse(cover, lsb), "psnr_lsb": metrics.psnr(cover, lsb),
        "extraction_accuracy": acc,
        "embed_ms": t_embed * 1000, "extract_ms": t_extract * 1000,
        "stego_prob": p_ours,
    })

# ------------------------------------------------------- image-in-image demo
demo = {}
cover_p = ROOT / "benchmarks" / "person_astronaut.png"
secret_p = PICS[0]
if cover_p.exists():
    cover = cv2.imread(str(cover_p), cv2.IMREAD_COLOR)
    secret = cv2.imread(str(secret_p), cv2.IMREAD_COLOR)
    payload = core.image_to_payload(secret, cover, 1)
    stego = core.embed(cover, payload, core.TYPE_IMAGE, regions.slot_order(cover), 1)
    ptype, rec = core.extract(stego, regions.slot_order(stego))
    rec_img = core.payload_to_image(rec)
    demo = {
        "cover": cover_p.name, "secret": secret_p.name,
        "secret_recovered_identical": bool(payload == rec),
        "psnr": metrics.psnr(cover, stego), "mse": metrics.mse(cover, stego),
        "payload_bytes": len(payload),
    }
    out = ROOT / "webapp" / "static"
    cv2.imwrite(str(out / "demo_cover.png"), cover)
    cv2.imwrite(str(out / "demo_stego.png"), stego)
    cv2.imwrite(str(out / "demo_secret.png"), secret)
    cv2.imwrite(str(out / "demo_recovered.png"), rec_img)

# ------------------------------------------------------------------ summary
psnrs = np.array([r["psnr"] for r in rows])
accs = np.array([r["extraction_accuracy"] for r in rows])
by_type = {}
for r in rows:
    by_type.setdefault(r["type"], []).append(r["psnr"])

stegan_report = {}
rep_path = ROOT / "results" / "steganalysis_report.json"
if rep_path.exists():
    stegan_report = json.loads(rep_path.read_text())

summary = {
    "n_images": len(rows),
    "psnr_mean": float(psnrs.mean()), "psnr_min": float(psnrs.min()),
    "psnr_max": float(psnrs.max()),
    "pct_above_40db": float((psnrs > 40).mean() * 100),
    "mse_mean": float(np.mean([r["mse"] for r in rows])),
    "extraction_accuracy_mean": float(accs.mean()),
    "embed_ms_mean": float(np.mean([r["embed_ms"] for r in rows])),
    "extract_ms_mean": float(np.mean([r["extract_ms"] for r in rows])),
    "psnr_by_type": {k: {"mean": float(np.mean(v)), "n": len(v)} for k, v in by_type.items()},
    "steganalysis": stegan_report,
    "image_in_image_demo": demo,
}

(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "results.json").write_text(json.dumps(
    {"summary": summary, "rows": rows}, indent=2))

# ------------------------------------------------------------------- charts
# Dark-theme palette (validated for the dark surface, dataviz skill):
#   surface #0e152e · ink #eef1fb · muted #9aa5c4 · grid rgba(white,.08)
#   series blue #3987e5 · aqua #22d3ee · target/red #e66767
static = ROOT / "webapp" / "static"
static.mkdir(parents=True, exist_ok=True)

SURFACE = "#0b1024"; INK = "#eef1fb"; MUTED = "#9aa5c4"
GRID = (1, 1, 1, 0.08); BLUE = "#3987e5"; AQUA = "#22d3ee"; RED = "#e66767"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": (1, 1, 1, 0.14), "axes.titlecolor": INK,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _finish(ax, fig, path, legend=True):
    if legend:
        lg = ax.legend(frameon=False)
        for t in lg.get_texts():
            t.set_color(MUTED)
    ax.margins(x=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


fig, ax = plt.subplots(figsize=(7.4, 4.2))
ax.hist(psnrs, bins=30, color=BLUE, edgecolor=SURFACE, linewidth=1.2)
ax.axvline(40, color=RED, linestyle="--", linewidth=2, label="40 dB target")
ax.set_xlabel("PSNR (dB)"); ax.set_ylabel("images")
ax.set_title("PSNR across dataset  ·  ours, 50% payload")
_finish(ax, fig, static / "chart_psnr_hist.png")

fig, ax = plt.subplots(figsize=(7.4, 4.2))
types = list(by_type.keys())
means = [np.mean(by_type[t]) for t in types]
bars = ax.bar(types, means, color=AQUA, width=0.66, zorder=3)
for b, m in zip(bars, means):
    ax.text(b.get_x() + b.get_width() / 2, m + 0.4, f"{m:.1f}",
            ha="center", va="bottom", color=INK, fontsize=9, fontweight="bold")
ax.axhline(40, color=RED, linestyle="--", linewidth=2, label="40 dB target", zorder=2)
ax.set_ylabel("mean PSNR (dB)"); ax.set_title("PSNR by image type")
ax.set_ylim(0, max(means) * 1.15)
_finish(ax, fig, static / "chart_psnr_by_type.png")

fig, ax = plt.subplots(figsize=(7.4, 4.2))
ours = [r["psnr"] for r in rows]; base = [r["psnr_lsb"] for r in rows]
lims = [min(min(base), min(ours)) - 1, max(max(base), max(ours)) + 1]
ax.plot(lims, lims, color=MUTED, linestyle=":", linewidth=1.5, label="equal quality")
ax.scatter(base, ours, s=26, alpha=0.75, color=BLUE, edgecolor=SURFACE,
           linewidth=0.6, zorder=3, label="images")
ax.set_xlabel("traditional LSB — PSNR (dB)"); ax.set_ylabel("ours — PSNR (dB)")
ax.set_title("Ours vs traditional LSB  ·  same payload")
_finish(ax, fig, static / "chart_ours_vs_lsb.png")

print(json.dumps(summary, indent=2))
