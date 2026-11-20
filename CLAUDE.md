# Project Brief: Hand-Drawn Symbol Recognition CNN

**Handoff document for Claude Code.** Save this file as `CLAUDE.md` in the root of the new GitHub repo. Claude Code reads `CLAUDE.md` automatically at the start of every session, so this file is the project's memory.

- **Course:** CS471 Artificial Intelligence, Final Project (250 pts)
- **Parent project:** PSYOP-VISR capstone (SOCOM Ignite 2026), an Android app on a Pixel 9 Pro XL that drives Even Realities G2 smart glasses. Existing app repo: https://github.com/CarsonTitus/SOCOMIgnite2026
- **This repo's job:** train, evaluate and export the CNN that recognizes hand-drawn symbols. Later, the exported model plugs into the app.

---

## 0. Instructions for Claude Code (read first)

1. **The team must be able to explain every line.** The course requires students to defend all submitted code. Write plain, well-commented Python. Prefer clear over clever. At the top of each script, add a short "what this does and why" comment block. After writing each script, explain it in plain English in the chat.
2. **Don't commit data or model weights to git.** Datasets, checkpoints and exported models go in folders listed in `.gitignore` (see §9). Commit only code, configs, small result plots and markdown.
3. **The tower PC runs Ubuntu Linux** (confirmed 2026-11-18; system Python is 3.10, so the project uses a Python 3.12 virtual environment created with `uv`). Teammates' laptops may run Windows, so write cross-platform Python:
   - Use `pathlib` for paths.
   - Put every training entry point under `if __name__ == "__main__":`. PyTorch data loaders with `num_workers > 0` crash on Windows without it.
   - Provide bash commands (tower, Linux/macOS) and PowerShell commands (Windows laptops) in the README.
4. **Ask before changing the scope in §2.** It was revised once (2026-11-20) with the course's permission. Everything must deliver what it promises.
5. **Work in the milestone order in §11.** Get the Lesson 36 checkpoint working end to end before adding extras.
6. **Make every run reproducible:**
   - Fixed random seeds.
   - Every run writes its config and metrics into its own `runs/<timestamp>_<name>/` folder.
7. **Explain acronyms the first time you use them**, in code comments and docs. The team asked for plain language.

---

## 1. The problem in plain language

Operators in unfamiliar places may not recognize local symbols (gang signs, cultural or religious markings) that tell them something about the area. The capstone app points the phone camera at the environment and shows a short explanation in the G2 glasses when it recognizes a symbol.

For this class we use **basic shapes as stand-ins** ("proxies") for real symbols. That keeps the project unclassified and doable in an academic lab. The proof of concept: name the shape (**triangle, quadrilateral or ellipse**) of a hand-drawn mark on a notecard **or** of a real object (a ball, a sign, a window).

### Why the current app fails at this

The app's object detector sees the notecard and calls it "paper," but never sees the triangle drawn on it. Reasons:

- **Too few pixels.** The app captures 640×480 frames (about 0.3 megapixels from a 50 MP camera), and the detector shrinks them further to 416 px. A pen line ends up thinner than one pixel, so the drawing basically disappears.
- **Wrong kind of model for the job.** General object detectors (like YOLO, "You Only Look Once") are trained on everyday objects such as TVs and keyboards. They have no "triangle" category, and they're built to box whole objects, not markings *on* objects.
- **Thin lines are hard for image AI.** Pretrained image networks lean on texture and solid areas. A thin pen outline looks like surface texture of the card.

### The fix, and where this repo fits

Use two steps. **Step 1** finds "there is a drawn mark here." **Step 2** cuts out a sharp, high-resolution close-up of that spot and classifies it.

**This repo builds Step 2: the classifier CNN.** Step 1 and the phone integration are capstone work (§12).

---

## 2. Scope (revised 2026-11-20; ask before changing)

The team first turned in a 3-output scope (no symbol / unknown / triangle). On 2026-11-20 the team revised it, using the course's option to revise the scope doc, to name **three shapes**, to cover **real objects as well as drawings**, and to fold "no symbol" into "unknown" (simpler; the separate class added complexity the assignment doesn't need): the original problem was that YOLO sees objects but not drawn shapes, so the classifier must handle both. The wording to submit is in `docs/scope_revision.md`. All deliverables must satisfy the revised scope. **Ask before changing it again.**

| Field | Scope (summary) |
|---|---|
| **Operational problem** | Alert operators to symbology (e.g., gang signs) in their surroundings. Basic shapes are the proof of concept. |
| **States** | Each image (a close-up of a drawn mark or a real-world object) is a state. |
| **Actions** | The agent labels each image as one of four outputs: **"symbol detected: triangle"**, **"symbol detected: quadrilateral"**, **"symbol detected: ellipse"**, or **"unknown / cannot identify"** (anything else, including images with no symbol at all). |
| **Reward** | Incorrect (including naming the wrong shape, or naming a shape when there is none) → **+0**. Unknown / cannot identify → **+0.5**. Shape detected and correctly identified → **+1**. |
| **Algorithm** | **CNN** (convolutional neural network) |
| **Lesson 36 checkpoint** | Show training output and the trained model recognizing symbols correctly **~90%** of the time across different dataset sizes. |
| **Lesson 41 evaluation** | **Confusion matrix** and **reward convergence graph**, showing **90% accuracy on test data**. |

### How we deliver the "reward" items with a CNN

A CNN doesn't learn from rewards; it learns from labeled examples by reducing a "loss" (a number measuring how wrong its guesses are). We still honor the scope exactly:

- The reward table becomes a **scoring rubric**. After every training round (an **epoch**, one full pass over the training images), score the model on the validation images with the rubric and record the **average reward per image**.
- Plotting average reward against epoch gives the **reward convergence graph**: it should rise and level off as the model learns.
- The **"different dataset sizes"** part of the checkpoint becomes a **learning-curve experiment**: train on 10%, 25%, 50% and 100% of the training data and report test accuracy for each.

### Reward matrix (implement exactly; keep it in the config file)

Rows are the **true** label and columns are the model's **prediction**.

| True \ Predicted | unknown_symbol | triangle | quadrilateral | ellipse |
|---|---|---|---|---|
| **unknown_symbol** | **0.5** | 0.0 | 0.0 | 0.0 |
| **triangle** | **0.5** | **1.0** | 0.0 | 0.0 |
| **quadrilateral** | **0.5** | 0.0 | **1.0** | 0.0 |
| **ellipse** | **0.5** | 0.0 | 0.0 | **1.0** |

Notes:

- **True shape, predicted unknown = 0.5.** The model saw a symbol but couldn't name it, which matches the rubric literally.
- **Wrong shape name = 0.** Calling a triangle an ellipse is incorrect.
- **True unknown, predicted unknown = 0.5** also follows the rubric literally ("cannot identify" is worth +0.5). This includes blank cards and walls: there is no separate "no symbol" output. The best possible average reward is therefore below 1.0 whenever unknown symbols are in the test set. Always report **reward as a percentage of the maximum possible** for that dataset, alongside the raw average.
- **Team decision (2026-11-20):** a correct "unknown" stays at **0.5** (the literal rubric).

---

## 3. Classes (the four outputs)

**The labeling rule:** the class is the **outline of the main thing in the cropped image**, whether it is drawn, painted, printed or a real object.

| Class folder name | Meaning | Examples |
|---|---|---|
| `unknown_symbol` | Anything that is not one of the three named shapes, **including no shape at all** | Stars, X's, arrows, hexagons, zigzags, squiggles; blank notecards, walls, desks, floors, carpet, clutter, **plants**, irregular objects |
| `triangle` | A triangle | Any drawn triangle (equilateral, skinny, rotated, sloppy), yield/warning signs |
| `quadrilateral` | Any 4-sided shape | Squares, rectangles, diamonds, kites, parallelograms, trapezoids; cards, windows, books, a toy kite |
| `ellipse` | A circle or oval | Drawn circles/ovals (nobody draws a perfect circle), balls, plates, clocks, coins |

**Team decision (2026-11-20):** four outputs, real objects included (supersedes the 2026-11-18 three-shape decision). An earlier draft the same day had a separate `no_symbol` class; the team folded it into `unknown_symbol` to keep the scope simple.
- `unknown_symbol` is trained on other clear shapes: stars, hexagons, zigzags, squiggles, lines (option 1 of three discussed; the alternatives were no unknown class, or unknown only from the confidence threshold).
- **Held back** for the never-seen-shapes test, so they appear nowhere in training: hearts, the letter "A", lightning bolts.
- **Left out** because their labels are unclear: octagons and smiley faces (their outlines are basically ellipses), Quick Draw "diamond" (about 1 in 4 is a triangle-shaped gem), triangle-like objects (mountain, tent, pizza).
- Kites are quadrilaterals and are included.
- Set in `configs/default.yaml` (`classes`, `reward_matrix`, `public_data` class maps).
- **Still to do:** public datasets of real-object photos (balls, signs, windows...) plus our own Pixel photos of objects.

**How "unknown" works (design decision):** `unknown_symbol` is its own trained category, learned from many shapes that are not triangles, quadrilaterals or ellipses. As a backup safety net, if the model's top confidence is below a threshold (default 0.6, set in config), the app reports `unknown_symbol` instead of guessing. Report results both with and without the threshold.

---

## 4. Hardware: the lab tower PC

| Part | Reported | Notes |
|---|---|---|
| CPU | AMD EPYC 7443P | 24 cores / 48 threads (confirmed: 48). Plenty for loading data; use `num_workers` 8–16. |
| GPU | 2× NVIDIA RTX 4000 Ada Generation, 20 GB VRAM each | Confirmed 2026-11-18 (driver 580.178.04, PyTorch 2.15.0+cu132). **Far more than enough.** `cuda:1` also drives the monitor (desktop, browser), so use `cuda:0` for the main run and `cuda:1` for the second experiment. |
| RAM | 251 GiB usable (256 GB installed) | Confirmed with `free -h` on 2026-11-18. More than enough. |
| OS | Ubuntu Linux | System Python is 3.10 (too old); use `uv` to get Python 3.12 (see setup below). |

**Is this enough to train?** Yes, easily. We are **not** training a model from scratch. We **fine-tune** a small pretrained model (§6). Expect minutes to under an hour per run on one GPU.

**Using both GPUs:** the simplest plan is to **run two separate experiments at once, one per GPU**. (Because the tower runs Linux, PyTorch's multi-GPU training backend, NCCL, NVIDIA's GPU-to-GPU communication library, would also work, but this small model doesn't need it.) Run two experiments, e.g. two random seeds, or the learning-curve runs split across cards. Select a GPU per run with `--device cuda:0` / `--device cuda:1`.

### Setup commands (bash, on the tower)

```bash
nvidia-smi                                          # confirm GPU model, memory (VRAM), driver version
curl -LsSf https://astral.sh/uv/install.sh | sh     # uv: installs Python 3.12 without sudo
export PATH="$HOME/.local/bin:$PATH"           # make uv findable in this terminal
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch torchvision                    # Linux default build includes CUDA (NVIDIA's GPU platform)
uv pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

The last line should print `True 2 <GPU name>`.

---

## 5. Data plan

### 5a. Public datasets (head start)

Check each license before use and record it in `data/SOURCES.md`.

| Dataset | Where | What's in it | Use for | Gotchas |
|---|---|---|---|---|
| **Hand-Drawn Shapes (HDS)** | github.com/frobertpixto/hand-drawn-shapes-dataset | ~27k grayscale 70×70 drawings: rectangle, ellipse, triangle, other | `triangle` ← triangle; `unknown_symbol` ← rectangle, ellipse, other | Small, centered and clean. Closest match to our classes. |
| **Quick, Draw!** (Google) | Hugging Face `google/quickdraw`, or `https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/<category>.npy` | 50M doodles in 345 categories, as 28×28 bitmaps | `triangle` ← `triangle`. `unknown_symbol` ← e.g. `circle`, `square`, `hexagon`, `octagon`, `star`, `zigzag`, `line`, `lightning`, `diamond`, `arrow`, `smiley face`, `squiggle` | **Very small images (28×28).** Strokes are **white on black**: invert them to dark-on-light. Download only the categories needed (each file can be 100+ MB). **Leave out triangle-like categories** (e.g. `mountain`, `tent`, `pizza` slices) from `unknown_symbol`, or the labels will conflict. Cap each category at a few thousand samples. |
| **Geometric Shapes Mathematics** | Kaggle `reevald/geometric-shapes-mathematics` | 20k hand-drawn shapes, 8 classes incl. triangle | `triangle` ← triangle; `unknown_symbol` ← others | Kite and trapezoid can look triangle-ish. Inspect samples. |
| **Four Shapes** | Kaggle "four-shapes" | 16k camera-captured 200×200 images: square, star, circle, triangle | `triangle` ← triangle; `unknown_symbol` ← others | Real camera photos, so closest to the phone's view. |
| **Geometric Shapes (text background)** | Hugging Face `0-ma/geometric-shapes` | Synthetic shapes; label "1" = text on a colored background, no shape | `unknown_symbol` ← label 1 (blank background) | Synthetic; a minority of `unknown_symbol`. |

Kaggle downloads need a free Kaggle account and an API token: Kaggle → Settings → API → Create New Token, which gives `kaggle.json`. Put it in `%USERPROFILE%\.kaggle\kaggle.json`. **Never commit it.**

### 5b. Our own photos (most important data)

Public data teaches "shapes in general." Our own Pixel photos teach "shapes on notecards in the real world." **The final test set must be our own photos only.**

**Capture plan (target ≥ 300 photos per class; more is better):**

- **Drawers:** every team member plus friends. At least 5 different people, so the model doesn't learn one person's handwriting.
- **Pens:** ballpoint, Sharpie (thin and thick), pencil, highlighter, colored marker, chalk if available.
- **Cards and surfaces:** white notecards, lined index cards, colored cards, printer paper, cardboard, whiteboard.
- **Shapes:**
  - `triangle`: all sizes, rotated, equilateral and skinny, slightly open corners, double-traced lines.
  - `unknown_symbol`: circle, square, star, X, arrow, smiley, squiggle, letters, hearts.
  - `unknown_symbol` also covers "nothing there": blank cards, cards with only a few words, desks, walls, floors, plants.
- **Camera (Pixel 9 Pro XL):**
  - distances 0.3 m, 1 m, 2 m, 3 m
  - angles straight-on, 30°, 60°
  - lighting: room lights, window daylight, dim, flash, harsh shadow
  - with some motion blur and partial cover (finger or another object over part of the card)
- **Cropping:** crop each photo so the card or mark fills most of the frame. This mimics what Step 1 hands to Step 2 in the real app. Keep the original uncropped photos too (`data/raw_own/originals/`) for future Step 1 work.
- **Metadata:** record in `data/raw_own/metadata.csv`, one row per photo: `filename, class, drawer_id, pen, surface, distance_m, angle_deg, lighting, notes`.

### 5c. Splitting the data (prevents fake-high scores)

- **Train ~70% / validation ~15% / test ~15%.** Training data is what the model learns from. Validation data is checked during training to pick the best version. Test data is locked away and used once at the end for the honest score.
- **Split our own photos by drawer**, never randomly. All photos from one person's cards go in only one split. Otherwise the model can memorize a person's handwriting or a specific card, and the score looks better than it really is.
- **Public datasets are training/validation only.** Test = our own photos.
- **Held-out shapes test:** pick 2–3 shapes that never appear anywhere in training (e.g. hand-drawn hearts, lightning bolts, letter "A"). Put them only in a special test set, `test_novel_shapes/`. A good model should call these `unknown_symbol`, **not** `triangle`. This is a great tactical-implications slide: it shows whether the system raises false alarms on things it has never seen.

### 5d. Preparing images

- Convert everything to **3-channel RGB at 224×224**. Grayscale gets copied into all 3 channels, because pretrained models expect color input.
- Invert Quick, Draw! images to dark strokes on a light background.
- Pad to a square before resizing so shapes don't get stretched.
- Final layout:

```
data/processed/
  train/{unknown_symbol,triangle,quadrilateral,ellipse}/
  val/{unknown_symbol,triangle,quadrilateral,ellipse}/
  test/{unknown_symbol,triangle,quadrilateral,ellipse}/   # own photos only
  test_novel_shapes/unknown_symbol/                  # never-seen shapes
```

- **Balance the classes.** Public data will hugely outnumber our photos and the classes will be uneven. Use a weighted sampler (it picks rarer classes more often during training) or cap samples per class.

---

## 6. Model

- **Base model:** `mobilenetv3_large_100` from the `timm` library, with weights pretrained on ImageNet, downloaded automatically from Hugging Face.
  - It's a CNN designed for phones, with about 5.5M parameters, so it will run on the Pixel later.
  - Create it with `timm.create_model("mobilenetv3_large_100", pretrained=True, num_classes=3)`. This replaces the final 1,000-category layer with a 3-category one.
  - Use the model's own preprocessing settings (`timm.data.resolve_data_config`) for input size and color normalization.
- **Comparison model (optional, good for the presentation):** a tiny CNN written from scratch (3–4 convolution layers). It shows *why* starting from a pretrained model matters, and helps the team explain how a CNN works layer by layer.
- **Why fine-tuning works:** the pretrained model already detects edges, corners and curves from millions of photos. We only teach it to combine those features into our 3 categories. That's why a few thousand images and one GPU are enough.

---

## 7. Training recipe

Use two stages (training phases).

**Stage A (pretrain on public data plus our training photos):**

- Freeze the backbone (don't update the pretrained layers) and train only the new final layer for 3 epochs, learning rate 1e-3.
- Then unfreeze everything and train 15–25 epochs, learning rate 1e-4. Use a cosine schedule, which gradually lowers the learning rate.
- Optimizer AdamW, weight decay 0.05, batch size 64 (lower it if GPU memory runs out).
- Loss: cross-entropy with label smoothing 0.1.

**Stage B (fine-tune on our own photos):**

- Start from the best Stage A checkpoint.
- 10–15 epochs, learning rate 3e-5, training mostly on our own photos mixed with about 20% public data, so the model doesn't forget general shapes.

**Both stages:**

- Mixed precision (`torch.autocast`, roughly 2× faster and less memory).
- Seed everything.
- Save the **best checkpoint by validation reward** and the last checkpoint.
- Early stopping: stop if validation reward hasn't improved for 5 epochs.

**Augmentation:** randomly altered copies of training images so the model copes with messy real conditions. Use `albumentations` or `torchvision.transforms.v2`, **training only**:

- random rotation ±180° (triangles can point any way)
- perspective tilt (viewing a card at an angle)
- random resized crop (scale 0.6–1.0)
- brightness/contrast/gamma changes, shadows
- motion blur and Gaussian blur
- camera noise and JPEG compression artifacts (quality 30–95)
- coarse dropout (random blocked patches = partial cover)
- downscale-then-upscale (simulates distance)
- random line thickness: light erosion/dilation, i.e. thinning or thickening strokes

**Do not** use horizontal-flip-only tricks that change meaning for future asymmetric symbols. Flips are fine for triangles but should be a config option.

**Logging per epoch** to `runs/<run>/metrics.csv` and TensorBoard: train loss, val loss, train accuracy, val accuracy, **val average reward**, **val reward % of max**, learning rate, epoch time.

---

## 8. Evaluation (what goes in the presentation)

`evaluate.py` runs the chosen checkpoint on `test/` and `test_novel_shapes/` and saves everything into `runs/<run>/eval/`:

1. **Confusion matrix** (required): a 3×3 grid of true vs predicted, as raw counts and as row percentages. PNG plus CSV.
2. **Reward convergence graph** (required): average validation reward per epoch, with a horizontal line at the maximum possible reward. Show Stage A and Stage B on one plot with a divider line.
3. **Accuracy:** overall test accuracy. Target **≥ 90%**. Also report per-class accuracy.
4. **Training curves:** loss and accuracy per epoch for train vs validation. A big gap between them means overfitting (memorizing instead of learning).
5. **Precision, recall, F1 per class, and precision-recall curves**, especially for `triangle`:
   - **Precision:** when it says triangle, how often is it right? Low precision means false alarms.
   - **Recall:** of all real triangles, how many did it catch? Low recall means missed symbols.
   - **F1:** a single number that balances the two.
   - **The tradeoff:** raising the confidence threshold gives fewer false alarms but more misses. Plot both against the threshold and mark the chosen operating point.
6. **Learning-curve experiment** (checkpoint item "variously sized data sets"): test accuracy and reward when training on 10/25/50/100% of the training data.
7. **Novel-shape test:** percentage of never-seen shapes correctly called `unknown_symbol` vs wrongly called `triangle`.
8. **Before/after own data:** test accuracy after Stage A only vs after Stage B. This proves our collected data mattered.
9. **Error gallery:** a grid image of the worst mistakes with true/predicted labels and confidence. It drives the "tactical implications of failures" discussion (e.g., "it misses faint pencil triangles in dim light").
10. **Speed:** average milliseconds per image on GPU and on CPU, as a first look at phone feasibility.

---

## 9. Repository layout

```
symbol-cnn/
├── CLAUDE.md                 # this file
├── README.md                 # setup + how to run each step (bash + PowerShell)
├── requirements.txt
├── .gitignore                # data/, runs/, exports/, *.pt, *.pth, *.tflite, *.onnx, kaggle.json, .venv/
├── configs/
│   ├── default.yaml          # classes, image size, reward matrix, thresholds, hyperparameters, paths
│   ├── stage_a.yaml
│   └── stage_b.yaml
├── data/                     # gitignored except SOURCES.md and README.md
│   ├── SOURCES.md            # dataset links + licenses
│   ├── raw_public/
│   ├── raw_own/              # our photos + metadata.csv
│   └── processed/
├── src/symbolcnn/
│   ├── __init__.py
│   ├── config.py             # load/validate YAML config
│   ├── datasets.py           # Dataset class, transforms, weighted sampler
│   ├── models.py             # timm MobileNetV3 + tiny from-scratch CNN
│   ├── reward.py             # reward matrix scoring (unit-tested)
│   ├── metrics.py            # confusion matrix, PR curves, per-class stats
│   ├── plots.py              # all matplotlib figures
│   └── utils.py              # seeding, device selection, run-folder creation
├── scripts/
│   ├── download_public.py    # fetch HDS, Quick Draw categories, Kaggle, HF sets
│   ├── prepare_data.py       # map to 3 classes, invert/resize, drawer-based split, balance
│   ├── train.py              # --config --device cuda:0 --data-fraction 1.0 --seed 0
│   ├── evaluate.py           # all §8 outputs
│   ├── learning_curve.py     # runs train.py at 10/25/50/100% (can split across 2 GPUs)
│   ├── predict.py            # classify a single image or folder from the terminal (checkpoint demo)
│   └── export_tflite.py      # stretch: convert to TFLite for the Pixel (§12)
├── tests/
│   ├── test_reward.py        # reward matrix math is correct
│   └── test_datasets.py      # no drawer appears in two splits; class folders correct
├── runs/                     # gitignored, except selected final plots copied to results/
└── results/                  # small committed PNG/CSV results for the presentation
```

**`requirements.txt` starting point:** torch, torchvision, timm, albumentations, opencv-python, numpy, pandas, scikit-learn, matplotlib, pyyaml, tqdm, tensorboard, kaggle, datasets, huggingface_hub, pytest. Pin versions once a working set is confirmed.

---

## 10. Terminal demo for the Lesson 36 checkpoint

The checkpoint only needs terminal output; no GUI (graphical interface) required.

```powershell
python scripts/train.py --config configs/stage_a.yaml --device cuda:0
python scripts/evaluate.py --run runs/<latest> --split test
python scripts/predict.py --run runs/<latest> --image path\to\card_photo.jpg
```

`predict.py` should print something like:

```
card_photo.jpg -> triangle (92.0%)   [unknown_symbol 2.6% | triangle 92.0% | quadrilateral 2.9% | ellipse 2.5%]
```

---

## 11. Milestones

### Lesson 36: Checkpoint ("ugly but functional", 50 pts)

- [ ] Repo created with the structure above, plus README and requirements; environment verified on the tower (§4)
- [ ] `download_public.py` and `prepare_data.py` working; `data/SOURCES.md` filled in
- [ ] At least 100 own photos per class collected, with metadata and drawer-based split
- [ ] `reward.py` with passing unit tests
- [ ] `train.py` Stage A runs to completion; metrics CSV written each epoch
- [ ] `evaluate.py` prints accuracy, the confusion matrix, and the average reward
- [ ] `learning_curve.py` results at 10/25/50/100%
- [ ] `predict.py` terminal demo on a live photo

### Lesson 41: Final (175 pts)

- [ ] ≥ 300 own photos per class, plus the novel-shapes test set
- [ ] Stage B fine-tune; before/after comparison
- [ ] All §8 plots generated and copied to `results/`
- [ ] Threshold chosen from the precision-recall analysis, with justification
- [ ] Error gallery and tactical-implications write-up (what failures mean for an operator)
- [ ] Every team member can explain `datasets.py`, `models.py`, `train.py` and `reward.py`
- [ ] Stretch: TFLite export plus a phone speed test (§12)

---

## 12. Later: plugging into the capstone app (not needed for the grade)

What the existing app repo (`SOCOMIgnite2026`) does today, and what to change:

| Today | Problem | Change |
|---|---|---|
| `YoloV8Detector.kt` sends frames to the Roboflow website at most once every 2 s | Needs internet; useless down range; slow | Run the detector **on the phone** |
| Camera analysis at 640×480 (`MainActivity.kt`) | Drawings vanish | Add a higher-resolution stream; **crop the symbol area from the high-res frame** before classifying |
| MobileCLIP prompts use hand-typed token codes; several prompts have identical codes | Can't tell those labels apart | Regenerate the codes with the real tokenizer, or replace this step with our CNN |
| Symbol meanings hardcoded in Kotlin maps; `CulturalContextEngine` is an in-memory list, not a real database | Adding a symbol means rebuilding the app | Build a **SymbolDB** (SQLite file): symbol id, short HUD title, 1–2 line meaning, alert level, detail pages |
| Every frame runs OCR + CLIP + detection on the CPU only | Slow, drains battery | Use the GPU delegate (TFLite's GPU mode); run heavy steps only on new detections |

**Export path for this model:** PyTorch → TFLite. TFLite is now called **LiteRT**: Google's format for running AI on phones. Options: Google's `ai-edge-torch` converter, or PyTorch → ONNX (Open Neural Network Exchange, a universal model format) → `onnx2tf`. Verify the converted model gives the same predictions as PyTorch on 100 test images before trusting it. Use FP16 (16-bit numbers; half the size, nearly no accuracy loss) for the phone's GPU.

**G2 display constraint:** 576×288 pixels per eye, green monochrome, text pushed over Bluetooth. Pop-ups must be about 3 short lines: `[!] TRIANGLE` / short meaning / direction (LEFT / CENTER / RIGHT). Only alert after the same symbol is seen in 3 of 5 frames, and don't repeat the same alert within 30–60 s.

**Future upgrade (capstone):** switch from a fixed 3-class classifier to an **embedding model with a symbol database**. The CNN turns each crop into a list of numbers (an "embedding"), and the app finds the closest match among reference images stored in the database. New symbols are added by inserting reference photos, with no retraining. The fine-tuned MobileNetV3 from this repo is a natural starting point for that.

---

## 13. Glossary

| Term | Meaning |
|---|---|
| **AI model / network** | A program that learned a task from examples instead of hand-written rules |
| **CNN** (Convolutional Neural Network) | A neural network that scans images with small filters to detect edges, corners and shapes, then combines them to recognize objects |
| **Pretrained / fine-tuning / transfer learning** | Starting from a model already trained on millions of images, then adjusting it to our task with our data |
| **Backbone** | The main feature-detecting layers of a pretrained model |
| **Epoch** | One full pass through all training images |
| **Batch size** | How many images the model looks at before each small update |
| **Learning rate** | How big each update step is; too high is unstable, too low is slow |
| **Loss** | A number measuring how wrong the model's guesses are; training pushes it down |
| **Cross-entropy / label smoothing** | The standard loss for picking a category / a tweak that stops the model being overconfident |
| **Overfitting** | Memorizing the training images instead of learning general patterns; shows as high train accuracy but low validation accuracy |
| **Augmentation** | Randomly altering training images (blur, rotate, darken) so the model handles real-world mess |
| **Train / validation / test split** | Learn from / check during training / final honest score |
| **Confusion matrix** | Grid of true vs predicted labels showing exactly which mistakes happen |
| **Precision / recall / F1** | Correctness of alarms / share of real symbols caught / balance of both |
| **PR curve** (precision-recall curve) | How precision and recall trade off as the confidence threshold changes |
| **Confidence threshold** | Minimum certainty required before the model commits to an answer |
| **Checkpoint** | A saved copy of the model at some point in training |
| **GPU / VRAM** | Graphics card that does training math fast / its memory |
| **CUDA** | NVIDIA's software platform that lets PyTorch use the GPU |
| **Mixed precision** | Doing some math in 16-bit numbers for speed with no real accuracy loss |
| **TFLite / LiteRT** | Google's format and runtime for AI models on phones |
| **ONNX** | A universal file format for moving models between tools |
| **Quantization (FP16/INT8)** | Storing the model's numbers in smaller formats so it runs faster on a phone |
| **Embedding** | A list of numbers that summarizes an image; similar images get similar lists |
| **YOLO** | "You Only Look Once," a popular fast object detector |
| **OCR** | Optical Character Recognition: reading text in images |
| **CLIP / MobileCLIP** | Models that match images to text descriptions |
| **BLE** | Bluetooth Low Energy, how the phone talks to the G2 glasses |
| **HUD** | Heads-Up Display: the text shown in the glasses |

---

## 14. Open questions for the team

1. ~~Exact GPU model and VRAM?~~ 2× RTX 4000 Ada, 20 GB each (2026-11-18). RAM: 251 GiB.
2. ~~Is Python already installed on the tower?~~ Ubuntu with Python 3.10; Python 3.12 comes from `uv`. Still open: is the team allowed to install system software (`sudo apt`) on it?
3. ~~Reward for a correctly predicted `unknown_symbol`?~~ 0.5, the literal rubric (2026-11-20).
4. ~~GitHub repo and collaborators?~~ CarsonTitus/Symbol-Recognition- (owner CarsonTitus), partner SabrinaMcG (2026-11-20).
5. Which teammate owns photo collection, training and the presentation?
