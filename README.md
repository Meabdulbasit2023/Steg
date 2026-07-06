# Enhanced Image Steganography using Two's Complement and Machine Learning

A web application that hides a **secret image or text message inside an ordinary
cover image** so that the result looks identical to the original, yet the hidden
data can be recovered perfectly. It enhances classical steganography by combining
a **two's complement + XOR** transform with **machine learning** — a Random Forest
that chooses the least-visible places to hide data, and a Support Vector Machine
that acts as an attacker to prove the hidden data is undetectable.

The project is built entirely in **Python** with a **Flask** web interface.

---

## 1. What the project is about

**Steganography** is the practice of hiding information inside another medium so
that no one suspects the information is even there. Unlike encryption (which makes
a message unreadable but obvious), steganography makes the message *invisible*.

This project hides data inside the pixels of an image. Every pixel is made of
numbers (its red, green and blue values from 0–255). Each number, in binary, ends
in a **least-significant bit (LSB)** — the smallest bit, worth just 1 out of 255.
Changing that bit alters the colour so slightly that the human eye cannot tell.
By writing our secret into these tiny bits across thousands of pixels, we hide a
whole message or image in plain sight.

The **"enhanced"** part — and the core of this project — is *how* the bits are
prepared and placed:

1. **Two's complement transform.** Before hiding, every byte of the secret is run
   through the two's complement operation, `S₂comp = NOT(S) + 1`. This scrambles
   the raw values so that reading the bits directly yields nonsense.
2. **XOR against the cover.** Each transformed bit is XOR-ed with a bit taken from
   the pixel it will live in, tying the payload to that specific image.
3. **Machine-learning placement.** A Random Forest model looks at the texture of
   the cover image and hides bits in the busiest, most detailed regions first —
   where changes are hardest to notice — instead of blindly filling pixels in
   order.
4. **Machine-learning verification.** A Support Vector Machine is trained as a
   "steganalysis attacker" that tries to detect hidden data. If it cannot tell a
   stego image from a clean one, the method is proven statistically invisible.

Decoding simply reverses every step to reconstruct the secret **bit-for-bit**.

### The aim and objectives

> **Aim:** Design an enhanced image steganography framework that uses two's
> complement and machine learning for image hiding.

| # | Objective | How the project meets it |
|---|-----------|--------------------------|
| i | **Design** the two's complement steganography framework | `stegoapp/core.py` implements the binary → 2's complement → XOR → embed pipeline; `stegoapp/regions.py` adds the ML placement |
| ii | **Implement** it in Python | The full `stegoapp/` library plus the Flask app in `webapp/` |
| iii | **Evaluate** performance with PSNR/MSE and compare with traditional techniques (LSB, DCT) | `scripts/evaluate.py` measures every image; `stegoapp/baselines.py` provides LSB & DCT; the *Compare* and *Evaluation* pages show results |
| iv | **Deploy** and recommend the best method | Delivered as a web app; the *Robustness* page and steganalysis results support the recommendation |

---

## 2. How it works, end to end

```
  ENCODING (hiding)                              DECODING (revealing)
  ─────────────────                              ────────────────────
  Secret (image or text)                         Stego image (PNG)
        │ to bytes                                     │
        ▼                                              ▼
  Binary  10110101…                              Read low bits of ML-chosen pixels
        │ two's complement  (NOT + 1)                  │ reverse XOR (⊕ pixel MSB)
        ▼                                              ▼
  Transformed bits                               Transformed bits
        │ XOR with pixel MSB                           │ reverse two's complement (NOT + 1)
        ▼                                              ▼
  Write into low bits of pixels                  Original bytes
  chosen by the Random Forest                          │
        ▼                                              ▼
  Stego image (saved as PNG)                     Secret recovered exactly
```

Because two's complement is its own inverse and XOR is reversible, and because the
Random Forest derives the *same* pixel ordering from the stego image that it used
on the cover (the features are computed from bits it never modifies), decoding
needs no password or side-channel — the stego PNG alone is enough.

### Key results (measured on 208 images: 198 passport photos + 10 benchmarks)

| Metric | Result | Target |
|--------|--------|--------|
| Mean PSNR (image quality) | **54.15 dB** | > 40 dB |
| Images above the 40 dB target | **100 %** | — |
| Mean MSE (distortion) | **0.25** | low |
| Message extraction accuracy | **100 %** | ≈ 100 % |
| SVM detectability of our method | **58 %** (≈ 50 % is invisible) | low |
| SVM detectability of traditional LSB | **65 %** | — |
| Falsified-image detector accuracy | **70 %** | — |

*(PSNR = Peak Signal-to-Noise Ratio, higher is better; MSE = Mean Squared Error,
lower is better.)*

---

## 3. Project structure

```
AB Project/
│
├── README.md                     ← this file
├── requirements.txt              ← Python packages needed to run the project
├── ML_APPROACH_OPTIONS.txt       ← notes: the ML design options considered
│
├── objectives /                  ← the original project brief (scanned photos)
│   └── WhatsApp Image ….jpeg     ← pages 3–4 of the objectives document
│
├── PICS-1.rar                    ← the raw dataset archive (as supplied)
├── PICS/                         ← 198 passport-photo cover images (extracted)
│   └── HNDAEM16003.png …         ← one .png per student, used as cover images
│
├── benchmarks/                   ← 10 varied-type test images (generated)
│   ├── person_astronaut.png      ← person
│   ├── animal_cat.png            ← animal
│   ├── landscape_moon.png        ← landscape
│   ├── scene_coffee.png,  scene_rocket.png            ← scenes
│   ├── object_camera.png, object_coins.png           ← objects
│   ├── texture_brick.png, texture_grass.png, texture_gravel.png  ← textures
│   └── manifest.csv              ← list of benchmark files, their type and size
│
├── stegoapp/                     ← THE CORE LIBRARY (all the algorithms)
│   ├── __init__.py               ← marks the folder as a Python package
│   ├── core.py                   ← two's complement + XOR embed / extract engine
│   ├── regions.py                ← Random Forest that picks where to hide bits
│   ├── steganalysis.py           ← SVM attacker: detects falsified images
│   ├── baselines.py              ← traditional LSB and DCT methods (for comparison)
│   ├── metrics.py                ← PSNR and MSE image-quality measures
│   ├── diffmap.py                ← amplified "difference map" between two images
│   └── robustness.py             ← tests recovery after JPEG / crop / resize
│
├── scripts/                      ← COMMAND-LINE TOOLS (run once, in order)
│   ├── make_benchmarks.py        ← creates the benchmarks/ images
│   ├── train_models.py           ← trains the Random Forest + SVM models
│   ├── test_roundtrip.py         ← sanity tests: every secret must recover exactly
│   └── evaluate.py               ← full evaluation → results/ + charts
│
├── models/                       ← TRAINED MODELS (generated by train_models.py)
│   ├── rf_regions.joblib         ← the Random Forest region-selection model
│   └── steganalysis_svm.joblib   ← the SVM falsified-image detector
│
├── results/                      ← EVALUATION OUTPUT (generated by evaluate.py)
│   ├── results.json              ← per-image PSNR/MSE/accuracy + summary
│   └── steganalysis_report.json  ← detector accuracy & per-method detectability
│
└── webapp/                       ← THE WEB APPLICATION (Flask)
    ├── app.py                    ← the server: all pages and form handling
    ├── templates/                ← the HTML pages (Jinja2 templates)
    │   ├── base.html             ← shared layout: nav, dropdowns, styling, background
    │   ├── home.html             ← landing page
    │   ├── encode.html           ← "Hide" page (+ live 2's complement trace)
    │   ├── decode.html           ← "Reveal" page
    │   ├── detect.html           ← "Detect" page (falsified-image detector)
    │   ├── compare.html          ← "Compare" page (ours vs LSB vs DCT)
    │   ├── robustness.html       ← "Robustness" page
    │   ├── evaluate.html         ← "Evaluation" dashboard
    │   └── about.html            ← "How it works" explanation
    └── static/                   ← images served to the browser
        ├── chart_*.png           ← evaluation charts (generated)
        ├── demo_*.png            ← image-in-image demo pictures (generated)
        └── out/                  ← stego/decoded images produced as you use the app
```

---

## 4. What each file does (in detail)

### The core library — `stegoapp/`

This is the heart of the project. It has **no web code** — it is pure algorithms,
so it can be tested on its own and reused anywhere.

- **`core.py` — the steganography engine.**
  Implements Objective i's Algorithm 1. Converts a secret to bits, applies the
  **two's complement** transform (`twos_complement`), **XOR**s each bit against the
  most-significant bit of its target pixel, and writes it into the pixel's lowest
  1–2 bits (`embed`). `extract` reverses everything. It also stores a small
  **header** (a marker, the payload type, bits-per-channel and length) so the
  decoder knows what and how much to read, reports the **capacity** of a cover
  image, and packs/unpacks a secret image as lossless PNG bytes, shrinking it to
  fit if necessary.

- **`regions.py` — machine-learning placement (Random Forest).**
  Splits the cover image into 8×8 blocks, measures each block's **texture**
  (standard deviation + edge strength), and uses a trained **Random Forest** to
  rank blocks as *busy*, *medium* or *smooth*. Bits are hidden in the busiest
  blocks first, where the eye can't see changes. Crucially, the features are
  computed from bits that embedding never touches, so the decoder rebuilds the
  **identical** ordering — no key needed.

- **`steganalysis.py` — the falsified-image detector (SVM).**
  Extracts statistical "tell-tale" features from an image's bit-planes — led by the
  **Pairs-of-Values chi-square test**, which collapses toward zero when a bit-plane
  has been overwritten. A trained **Support Vector Machine** turns those features
  into a probability that the image contains hidden data, and `verdict()` converts
  that into a plain-English result ("Clean" … "Falsified").

- **`baselines.py` — traditional methods for comparison.**
  Implements classic **spatial LSB** (hide bits straight into pixel LSBs in order)
  and **DCT** steganography (hide bits in frequency-domain coefficients using
  quantization-index modulation). These let the project compare "ours vs the
  traditional techniques" that Objective iii asks for.

- **`metrics.py` — quality measurement.**
  Two small functions: **`mse`** (mean squared error between two images) and
  **`psnr`** (peak signal-to-noise ratio, in decibels). These are the numbers used
  to prove the stego image looks like the original.

- **`diffmap.py` — difference visualization.**
  Produces an **amplified, colour-mapped picture** of exactly which pixels changed
  between the cover and the stego image, plus statistics (percent of pixels
  changed, maximum change). Used on the Hide and Compare pages as visual proof of
  invisibility.

- **`robustness.py` — attack resistance testing.**
  Embeds a message, then subjects the stego image to **JPEG compression, cropping
  and resizing**, and reports how much of the message still survives each attack
  (Objective iv's robustness requirement).

- **`__init__.py`** — empty file that tells Python `stegoapp` is an importable
  package.

### The scripts — `scripts/` (run from the command line)

These prepare the project. Run them **in this order** the first time:

1. **`make_benchmarks.py`** — creates the 10 varied benchmark images (person,
   animal, textures, landscape, objects) from scikit-image's built-in test images,
   so the evaluation covers "varying image types." No internet needed.
2. **`train_models.py`** — trains and saves **both** machine-learning models: the
   Random Forest region selector and the SVM detector (on balanced clean-vs-
   falsified data). Prints their accuracies and writes `steganalysis_report.json`.
3. **`test_roundtrip.py`** — automated sanity check: hides many payloads (text and
   images, at 1 and 2 bits per channel) and confirms every one comes back
   **exactly**, and that a clean image is correctly rejected.
4. **`evaluate.py`** — the big evaluation: embeds into all 208 images, measures
   PSNR/MSE/accuracy/timing, runs the steganalysis attacker, compares against LSB,
   renders the charts, and writes everything to `results/`.

### The trained models — `models/`

Binary files produced by `train_models.py` and loaded by the app at runtime:

- **`rf_regions.joblib`** — the Random Forest that chooses hiding regions.
- **`steganalysis_svm.joblib`** — the SVM that detects falsified images.

### The results — `results/`

- **`results.json`** — per-image measurements plus an overall summary; read by the
  Evaluation page.
- **`steganalysis_report.json`** — detector accuracy and how detectable each method
  (ours vs LSB) is.

### The web application — `webapp/`

- **`app.py` — the Flask server.**
  Defines every page (route) and handles the uploaded files: `/` home,
  `/encode` (hide), `/decode` (reveal), `/detect`, `/compare`, `/robustness`,
  `/evaluate`, `/about`, plus a `/capacity` helper. It ties the whole `stegoapp`
  library together — receiving images, calling the engine, saving the outputs into
  `static/out/`, and passing results to the templates. It also builds the live
  "binary → two's complement" calculation trace shown on the Hide page.

- **`templates/` — the pages** (HTML with Jinja2 placeholders):
  - **`base.html`** — the master layout every page extends: the top navigation with
    the **Tools** and **Analysis** dropdown menus, the colour scheme, the animated
    background, and the shared JavaScript.
  - **`home.html`** — landing page with headline stats, the pipeline overview and
    links to every tool.
  - **`encode.html`** — the Hide form; after embedding it shows the stego image, the
    difference map, quality metrics, and the step-by-step two's complement table.
  - **`decode.html`** — the Reveal form and the recovered secret.
  - **`detect.html`** — upload any image and get a clean/falsified verdict with a
    suspicion gauge.
  - **`compare.html`** — runs our method, LSB and DCT side by side.
  - **`robustness.html`** — shows recovery rates under each attack as coloured bars.
  - **`evaluate.html`** — the full dataset dashboard with charts and a results table.
  - **`about.html`** — a written explanation of the algorithm, the models and the
    metrics.

- **`static/`** — images the browser loads: the generated evaluation **charts**, the
  image-in-image **demo** pictures, and **`out/`**, where every stego and recovered
  image you create through the app is saved.

### Data and reference files

- **`PICS/`** — the 198 supplied passport photographs used as cover images.
- **`PICS-1.rar`** — the original archive they came in.
- **`benchmarks/`** — the 10 generated varied-type images and their `manifest.csv`.
- **`objectives /`** — scanned photos of the original project brief.
- **`ML_APPROACH_OPTIONS.txt`** — the notes weighing which ML approach to use.
- **`requirements.txt`** — the list of Python libraries to install.

---

## 5. Setup and running

### Install (one time)

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

### Build the models and results (one time, in order)

```bash
./venv/bin/python scripts/make_benchmarks.py    # 1. create benchmark images
./venv/bin/python scripts/train_models.py       # 2. train Random Forest + SVM
./venv/bin/python scripts/test_roundtrip.py     # 3. sanity tests (should all pass)
./venv/bin/python scripts/evaluate.py           # 4. full evaluation + charts
```

### Run the web app

```bash
./venv/bin/python webapp/app.py
# then open http://127.0.0.1:5001 in a browser
```

### Using the app

1. **Hide** — upload a cover image, choose a secret image or type a message, and
   download the resulting stego PNG. Keep it as **PNG** (JPEG would destroy the
   hidden bits).
2. **Reveal** — upload that stego PNG to get the secret back.
3. **Detect** — upload any image to check whether it contains hidden data.
4. **Compare / Robustness / Evaluation** — see the analysis behind the method.

---

## 6. Technology used

- **Python 3** — the whole project.
- **Flask** — the web server and pages.
- **NumPy** — fast array maths for the bit-level operations.
- **OpenCV** — reading, writing and transforming images (including DCT).
- **scikit-image** — the benchmark test images.
- **scikit-learn** — the Random Forest and SVM machine-learning models.
- **Matplotlib** — the evaluation charts.
- **HTML / CSS / JavaScript** — the web interface (no external frameworks).
