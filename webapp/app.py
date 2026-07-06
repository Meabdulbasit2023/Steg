"""Flask web application — Enhanced Image Steganography using Two's
Complement and Machine Learning.

Pages:
  /            home / landing page
  /encode      hide a secret image (or text) inside a cover image
  /decode      recover the hidden secret from a stego image
  /compare     our method vs traditional LSB and DCT
  /robustness  recovery under compression / cropping / resize
  /evaluate    dataset evaluation dashboard (PSNR/MSE/accuracy/steganalysis)
  /about       algorithm explanation
"""

import sys, json, uuid
from pathlib import Path

import numpy as np
import cv2
from flask import Flask, render_template, request, url_for

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stegoapp import core, regions, metrics, steganalysis, baselines, diffmap, robustness


def load_summary():
    p = ROOT / "results" / "results.json"
    if p.exists():
        return json.loads(p.read_text()).get("summary")
    return None

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

OUT = Path(__file__).resolve().parent / "static" / "out"
OUT.mkdir(parents=True, exist_ok=True)


def pipeline_trace(payload: bytes, cover, stego, order, bpc: int, is_text: bool,
                   n_bytes: int = 4):
    """Worked example of the actual arithmetic applied to the first few
    payload bytes: binary -> 1's complement (NOT) -> +1 -> 2's complement,
    then the XOR-with-pixel-MSB embedding of the first 8 bits."""
    bytes_rows = []
    for i, b in enumerate(payload[:n_bytes]):
        inv = (~b) & 0xFF
        twos = (inv + 1) & 0xFF
        bytes_rows.append({
            "i": i,
            "char": chr(b) if is_text and 32 <= b < 127 else "",
            "dec": b, "bin": format(b, "08b"),
            "inv": format(inv, "08b"),
            "twos": format(twos, "08b"), "twos_dec": twos,
        })

    bit_rows = []
    if payload:
        twos0 = ((~payload[0]) + 1) & 0xFF
        bits = [int(x) for x in format(twos0, "08b")]
        flat_c = cover.reshape(-1)
        flat_s = stego.reshape(-1)
        for k in range(8):
            slot = int(order[k // bpc])
            before = int(flat_c[slot])
            msb = (before >> 7) & 1
            bit_rows.append({
                "k": k, "bit": bits[k],
                "slot": slot,
                "before": before, "before_bin": format(before, "08b"),
                "msb": msb, "xored": bits[k] ^ msb,
                "after": int(flat_s[slot]), "after_bin": format(int(flat_s[slot]), "08b"),
            })
    return {"bytes": bytes_rows, "bits": bit_rows, "total": len(payload)}


def read_upload(file_storage):
    data = np.frombuffer(file_storage.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"'{file_storage.filename}' is not a readable image.")
    return img


@app.route("/")
def home():
    return render_template("home.html", summary=load_summary())


@app.route("/encode", methods=["GET", "POST"])
def encode():
    if request.method == "GET":
        return render_template("encode.html")
    try:
        cover = read_upload(request.files["cover"])
        bpc = int(request.form.get("bpc", "1"))
        mode = request.form.get("mode", "image")

        if mode == "image":
            secret_file = request.files.get("secret")
            if not secret_file or not secret_file.filename:
                raise ValueError("Please choose a secret image to hide.")
            secret = read_upload(secret_file)
            payload = core.image_to_payload(secret, cover, bpc)
            ptype = core.TYPE_IMAGE
        else:
            text = request.form.get("message", "").strip()
            if not text:
                raise ValueError("Please type a secret message.")
            payload = text.encode("utf-8")
            ptype = core.TYPE_TEXT

        order = regions.slot_order(cover)
        stego = core.embed(cover, payload, ptype, order, bpc)
        trace = pipeline_trace(payload, cover, stego, order, bpc,
                               is_text=(ptype == core.TYPE_TEXT))

        token = uuid.uuid4().hex[:12]
        stego_name = f"stego_{token}.png"
        cv2.imwrite(str(OUT / stego_name), stego)
        cv2.imwrite(str(OUT / f"cover_{token}.png"), cover)

        heat, dstats = diffmap.difference_map(cover, stego)
        diff_name = f"diff_{token}.png"
        cv2.imwrite(str(OUT / diff_name), heat)

        prob = steganalysis.predict_stego_probability(stego)
        result = {
            "stego_url": url_for("static", filename=f"out/{stego_name}"),
            "cover_url": url_for("static", filename=f"out/cover_{token}.png"),
            "diff_url": url_for("static", filename=f"out/{diff_name}"),
            "changed_pct": f"{dstats['changed_pct']:.2f}",
            "max_delta": dstats["max_delta"],
            "psnr": f"{metrics.psnr(cover, stego):.2f}",
            "mse": f"{metrics.mse(cover, stego):.4f}",
            "payload_bytes": len(payload),
            "capacity_bytes": core.capacity_bytes(cover, bpc),
            "bpc": bpc,
            "stego_prob": None if prob is None else f"{prob*100:.1f}",
        }
        return render_template("encode.html", result=result, trace=trace)
    except Exception as e:
        return render_template("encode.html", error=str(e))


@app.route("/decode", methods=["GET", "POST"])
def decode():
    if request.method == "GET":
        return render_template("decode.html")
    try:
        stego = read_upload(request.files["stego"])
        ptype, payload = core.extract(stego, regions.slot_order(stego))
        if ptype == core.TYPE_IMAGE:
            secret = core.payload_to_image(payload)
            name = f"recovered_{uuid.uuid4().hex[:12]}.png"
            cv2.imwrite(str(OUT / name), secret)
            return render_template(
                "decode.html",
                result={"kind": "image",
                        "url": url_for("static", filename=f"out/{name}"),
                        "size": f"{secret.shape[1]}x{secret.shape[0]}",
                        "bytes": len(payload)})
        return render_template(
            "decode.html",
            result={"kind": "text",
                    "text": payload.decode("utf-8", errors="replace"),
                    "bytes": len(payload)})
    except Exception as e:
        return render_template("decode.html", error=str(e))


@app.route("/capacity", methods=["POST"])
def capacity():
    """AJAX helper: report capacity of an uploaded cover before embedding."""
    try:
        cover = read_upload(request.files["cover"])
        return {"ok": True,
                "w": cover.shape[1], "h": cover.shape[0],
                "capacity_1": core.capacity_bytes(cover, 1),
                "capacity_2": core.capacity_bytes(cover, 2)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.route("/detect", methods=["GET", "POST"])
def detect():
    if request.method == "GET":
        return render_template("detect.html", summary=load_summary())
    try:
        img = read_upload(request.files["image"])
        prob = steganalysis.predict_stego_probability(img)
        if prob is None:
            raise ValueError("Detector model not trained yet. Run scripts/train_models.py.")
        token = uuid.uuid4().hex[:12]
        cv2.imwrite(str(OUT / f"det_{token}.png"), img)
        v = steganalysis.verdict(prob)
        result = {
            "img_url": url_for("static", filename=f"out/det_{token}.png"),
            "prob": f"{prob*100:.1f}",
            "prob_val": prob,
            "verdict": v["label"], "level": v["level"], "detail": v["detail"],
            "w": img.shape[1], "h": img.shape[0],
        }
        return render_template("detect.html", result=result, summary=load_summary())
    except Exception as e:
        return render_template("detect.html", error=str(e), summary=load_summary())


@app.route("/compare", methods=["GET", "POST"])
def compare():
    if request.method == "GET":
        return render_template("compare.html")
    try:
        cover = read_upload(request.files["cover"])
        msg = request.form.get("message", "").strip() or "Enhanced steganography 2's complement + ML"
        payload = msg.encode("utf-8")

        # bound the payload to what all three methods can carry (DCT is smallest)
        dct_cap_bytes = baselines.dct_capacity_bits(cover) // 8
        if dct_cap_bytes < 1:
            raise ValueError("Cover image is too small for a DCT comparison (needs ≥ 8×8 blocks).")
        if len(payload) > dct_cap_bytes:
            payload = payload[:dct_cap_bytes]
        bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
        n_bits = len(bits)

        token = uuid.uuid4().hex[:12]
        cv2.imwrite(str(OUT / f"cmp_cover_{token}.png"), cover)
        methods = []

        # --- ours: 2's complement + XOR + ML regions ---
        s_ours = core.embed(cover, payload, core.TYPE_TEXT, regions.slot_order(cover), 1)
        _, rec = core.extract(s_ours, regions.slot_order(s_ours))
        acc_ours = 100.0 * (np.frombuffer(rec, np.uint8) == np.frombuffer(payload, np.uint8)).mean()
        cv2.imwrite(str(OUT / f"cmp_ours_{token}.png"), s_ours)
        h_ours, _ = diffmap.difference_map(cover, s_ours)
        cv2.imwrite(str(OUT / f"cmp_ours_diff_{token}.png"), h_ours)

        # --- traditional spatial LSB ---
        s_lsb = baselines.lsb_embed(cover, bits)
        acc_lsb = 100.0 * (baselines.lsb_extract(s_lsb, n_bits) == bits).mean()
        cv2.imwrite(str(OUT / f"cmp_lsb_{token}.png"), s_lsb)
        h_lsb, _ = diffmap.difference_map(cover, s_lsb)
        cv2.imwrite(str(OUT / f"cmp_lsb_diff_{token}.png"), h_lsb)

        # --- traditional DCT (QIM) ---
        s_dct = baselines.dct_embed(cover, bits)
        acc_dct = 100.0 * (baselines.dct_extract(s_dct, n_bits) == bits).mean()
        cv2.imwrite(str(OUT / f"cmp_dct_{token}.png"), s_dct)
        h_dct, _ = diffmap.difference_map(cover, s_dct)
        cv2.imwrite(str(OUT / f"cmp_dct_diff_{token}.png"), h_dct)

        def row(name, tag, s, acc, cap_bits, note):
            return {
                "name": name,
                "img": url_for("static", filename=f"out/cmp_{tag}_{token}.png"),
                "diff": url_for("static", filename=f"out/cmp_{tag}_diff_{token}.png"),
                "psnr": f"{metrics.psnr(cover, s):.2f}",
                "mse": f"{metrics.mse(cover, s):.4f}",
                "accuracy": f"{acc:.1f}",
                "capacity": f"{cap_bits // 8:,}",
                "note": note,
            }

        methods = [
            row("Ours — 2's complement + ML", "ours", s_ours, acc_ours,
                core.capacity_bytes(cover, 1) * 8, "High capacity, highest PSNR, invisible"),
            row("Traditional LSB", "lsb", s_lsb, acc_lsb,
                baselines.lsb_capacity_bits(cover), "High capacity, but easiest to detect"),
            row("Traditional DCT", "dct", s_dct, acc_dct,
                baselines.dct_capacity_bits(cover), "Low capacity & PSNR, robust to JPEG"),
        ]
        ctx = {
            "cover_url": url_for("static", filename=f"out/cmp_cover_{token}.png"),
            "payload_bytes": len(payload), "message": msg[:len(payload)],
            "methods": methods,
        }
        return render_template("compare.html", result=ctx)
    except Exception as e:
        return render_template("compare.html", error=str(e))


@app.route("/robustness", methods=["GET", "POST"])
def robustness_page():
    if request.method == "GET":
        return render_template("robustness.html")
    try:
        cover = read_upload(request.files["cover"])
        msg = request.form.get("message", "").strip() or "Robustness recovery test 2026"
        payload = msg.encode("utf-8")
        cap = core.capacity_bytes(cover, 1)
        if len(payload) > cap:
            payload = payload[:cap]
        report = robustness.run_suite(cover, payload)
        report["message"] = msg[:len(payload)]
        return render_template("robustness.html", result=report)
    except Exception as e:
        return render_template("robustness.html", error=str(e))


@app.route("/evaluate")
def evaluate():
    results_path = ROOT / "results" / "results.json"
    if not results_path.exists():
        return render_template("evaluate.html", missing=True)
    data = json.loads(results_path.read_text())
    return render_template("evaluate.html", summary=data["summary"],
                           rows=data["rows"][:30], total=len(data["rows"]))


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True, port=5001)
