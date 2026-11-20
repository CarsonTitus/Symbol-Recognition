# Symbol Recognition CNN

CS471 Artificial Intelligence final project. This repo trains a CNN (convolutional neural network) that looks at a close-up image and names the **outline shape of the main thing in it**, whether that thing is drawn, painted, printed or a real object (a ball is an ellipse; a kite or a window is a quadrilateral). Each image gets one of four labels:

| Output | Meaning | Examples |
|---|---|---|
| `unknown_symbol` | anything that is not one of the three shapes, including no shape at all | star, arrow, hexagon, zigzag; blank card, wall, carpet, a plant |
| `triangle` | a triangle | drawn triangle, yield sign, any triangle type |
| `quadrilateral` | any 4-sided shape | square, rectangle, diamond, kite, window, card |
| `ellipse` | a circle or oval | drawn circle, ball, plate, clock |

The classes were revised from three (no symbol / unknown / triangle) to these four on 2026-11-20, with the course's permission to revise the scope. There is no separate "no symbol" output: a blank card is simply "not one of the three shapes".

It is "Step 2" (the classifier) of the PSYOP-VISR capstone app. The full project brief is in [`CLAUDE.md`](CLAUDE.md).

## Setup

You need Python **3.11 or 3.12** and `git`. PyTorch is installed **separately** from `requirements.txt` because the right build depends on whether the computer has an NVIDIA GPU. Data and trained models are **not** in git: the scripts below download and build them.

### Linux or macOS (bash)

Install `uv` (a fast Python package tool; it can download Python 3.12 without `sudo`), one time:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Make `uv` findable, now and in every future terminal:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
```

Get the code and create the virtual environment (a private folder of packages for this project):

```bash
git clone https://github.com/CarsonTitus/Symbol-Recognition-.git
```

```bash
cd Symbol-Recognition-
```

```bash
uv venv --python 3.12 .venv
```

```bash
source .venv/bin/activate
```

Install PyTorch. **With an NVIDIA GPU (Linux):** the default package includes CUDA (NVIDIA's GPU platform) support:

```bash
uv pip install torch torchvision
```

**Without an NVIDIA GPU:** use the smaller CPU-only build (on a Mac, the plain command above also works and includes Apple-GPU support):

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Then the rest of the packages, and a check of the machine:

```bash
uv pip install -r requirements.txt
```

```bash
python scripts/check_env.py --download-weights
```

It lists the Python and PyTorch versions and any GPUs, downloads the pretrained MobileNetV3 weights, and ends with `All checks passed.`

**Every new terminal:** `cd Symbol-Recognition-` then `source .venv/bin/activate` before running anything. The prompt starts with `(.venv)` when it's active.

### Windows (PowerShell)

Install `uv` (one time):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen PowerShell, then:

```powershell
git clone https://github.com/CarsonTitus/Symbol-Recognition-.git
```

```powershell
cd Symbol-Recognition-
```

```powershell
uv venv --python 3.12 .venv
```

```powershell
.\.venv\Scripts\Activate.ps1
```

If `Activate.ps1` is blocked ("running scripts is disabled"), run this once, then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Install PyTorch. Without an NVIDIA GPU, use the CPU build:

```powershell
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

With an NVIDIA GPU, get the command from https://pytorch.org/get-started/locally/ (*Stable → Windows → Pip → newest CUDA*) and put `uv` in front of `pip`.

```powershell
uv pip install -r requirements.txt
```

```powershell
python scripts/check_env.py --download-weights
```

The commands in the rest of this README are the same in PowerShell; use `\` instead of `/` in file paths if you prefer. Without an NVIDIA GPU, add `--device cpu` to the training and evaluation commands (training is much slower on a CPU, so use `--limit-per-class` and `--data-fraction` for trial runs).

## Run the tests

```bash
pytest
```

`tests/test_reward.py` proves the reward math matches the submitted rubric; `tests/test_config.py` checks the settings files; `tests/test_download_public.py` checks the download script without using the internet.

## Download the public datasets

```bash
python scripts/download_public.py
```

This downloads about 320 MB into `data/raw_public/` (which git ignores) and takes about a minute:

| Folder | Dataset | What |
|---|---|---|
| `hds/` | Hand-Drawn Shapes | 27,292 drawings by 37 people: triangle, rectangle, ellipse, other |
| `quickdraw/` | Google Quick, Draw! | the first 3,000 doodles of 12 shape categories (only that part of each file is downloaded) |
| `hf_geometric_shapes/` | Hugging Face `0-ma/geometric-shapes` | synthetic images; the "text only, no shape" ones become blank-background examples of `unknown_symbol` |
| `kaggle/` | two Kaggle datasets | needs a Kaggle API token (below); skipped without one |

Running it again skips anything already downloaded. Use `--force` to download again, and `--sources hds quickdraw` to choose specific datasets. Which datasets and categories are used, and which of our classes each maps to, is set under `public_data:` in `configs/default.yaml`.

### Kaggle API token

1. Sign in at https://www.kaggle.com (free account).
2. Go to https://www.kaggle.com/settings/api, then **Generate New Token**, and copy the token.
3. Save it (paste the token between the quotes):

```bash
mkdir -p ~/.kaggle
```

```bash
echo "PASTE_TOKEN_HERE" > ~/.kaggle/access_token
```

```bash
chmod 600 ~/.kaggle/access_token
```

`chmod 600` makes the file readable only by you. **Never commit the token or paste it into chat.** Then:

```bash
python scripts/download_public.py --sources kaggle
```

## Build the training folders

```bash
python scripts/prepare_data.py
```

This sorts every downloaded image (and our own photos, once there are some in `data/raw_own/`) into `data/processed/{train,val,test}/<class>/` (one folder per output above) as 224×224 RGB images, and writes `data/processed/manifest.csv`. It takes about a minute on our training computer and about 1 GB of disk.

- Our photos are split **by drawer** (70/15/15), so one person's drawings never appear in two splits.
- Public data only goes into train/val. **Test = our own photos only.**
- Photos marked `novel_shape` in `metadata.csv` go only to `test_novel_shapes/`.
- It rebuilds `data/processed/` from scratch each time, and the same seed gives the same split.
- `--limit-per-class 200` makes a small version for quick trial runs.

How to add our photos and fill in `metadata.csv`: see [`data/README.md`](data/README.md).

## Train the model

```bash
python scripts/train.py --config configs/stage_a.yaml --device cuda:0
```

This trains the pretrained MobileNetV3 on `data/processed/train` and checks it on `data/processed/val` after every epoch. Each run gets its own folder, `runs/<date-time>_<name>/`, containing:

| File | What |
|---|---|
| `config.yaml` | the exact settings used |
| `metrics.csv` | one row per epoch: losses, accuracies, **validation reward** and **reward % of max**, learning rate, time |
| `best.pt` / `last.pt` | the model with the best validation reward so far / the latest model |
| `what_the_model_sees.png` | the first training batch after augmentation |
| `tensorboard/` | data for the live charts (below) |

Training stops early if the validation reward hasn't improved for 5 epochs. Press **Ctrl+C** to stop at any time; `best.pt` is kept.

Useful variations:

- `--device cuda:1`: use a second GPU, if the computer has one.
- `--data-fraction 0.25`: train on 25% of the training data (learning-curve runs).
- `--model tiny_cnn`: the small from-scratch CNN, for comparison.
- `--device cpu --max-batches 5`: a quick check that everything runs, on any machine.

### Watch it train (TensorBoard)

Open a **second terminal** while training runs:

```bash
cd Symbol-Recognition-
```

```bash
source .venv/bin/activate
```

```bash
tensorboard --logdir runs
```

Then open **http://localhost:6006** in a web browser on the same computer. It refreshes every 30 seconds.

- **Scalars** (or **Time Series**): `loss` and `accuracy` (train vs val), `reward/val_average` (the reward convergence graph), `reward/val_percent_of_max`, `learning_rate`.
- **Images**: `train/what_the_model_sees` is a snapshot of the exact batch being trained on, taken every 200 batches. `val/most_confident_mistakes` shows the validation images the model gets most confidently wrong, after every epoch.

Press Ctrl+C in that terminal to stop TensorBoard; training is unaffected.

## Evaluate the model

```bash
python scripts/evaluate.py --run runs/<run folder> --split val --device cuda:0
```

This prints the accuracy, the confusion matrix, the average reward and precision/recall/F1 per class, both without and with the confidence threshold. It saves everything (confusion matrix PNG/CSV, per-class table, every prediction, an error gallery of the worst mistakes) in `runs/<run>/eval/<split>/`, plus the reward convergence graph and training curves in `runs/<run>/eval/`.

- `--split test`: the honest score, on our own photos only. It stays empty until we add photos, so use `--split val` until then.
- `--split test_novel_shapes`: shapes never seen in training. It reports how many were (correctly) called `unknown_symbol`.

## Learning curve (different dataset sizes)

```bash
python scripts/learning_curve.py --devices cuda:0 cuda:1
```

This trains the model four times, on 10%, 25%, 50% and 100% of the training data, sharing the runs between the listed GPUs (give one, e.g. `--devices cuda:0`, if there is only one), then scores each one. The results go in `runs/<date-time>_learning_curve/learning_curve_val.csv` and `.png`. It takes about 30 minutes on our training computer (below) and needs nobody at the keyboard. To re-score finished runs on our own photos later, without retraining:

```bash
python scripts/learning_curve.py --from runs/<date-time>_learning_curve --split test
```

## Classify a photo (the demo)

```bash
python scripts/predict.py --run latest --image path/to/card_photo.jpg
```

`--run latest` uses the newest finished training run in `runs/`; you can also give a run folder. `--image` takes one or more files and/or folders. Each photo gets one line:

```
card_photo.jpg -> triangle (92.0%)   [unknown_symbol 2.6% | triangle 92.0% | quadrilateral 2.9% | ellipse 2.5%]
```

- **Crop the photo first** so the card or object fills most of the frame. The model names the outline of the *main* thing in the image.
- If the top probability is below the confidence threshold (60%), the answer becomes `unknown_symbol` and the line says `unsure`. `--threshold 0` switches this off.
- Confidence tops out around 92–94%, never 100%. That is on purpose: training uses "label smoothing" so the model never becomes over-confident.
- Add `--device cpu` on a laptop without an NVIDIA GPU. It takes well under a second per photo.

These commands are the same in PowerShell; just use `\` instead of `/` in file paths if you prefer.

## Settings

All settings live in [`configs/`](configs/):

- `default.yaml`: classes, reward matrix, confidence threshold (0.6), image size and training hyperparameters.
- `stage_a.yaml` / `stage_b.yaml`: change only what differs for each training stage.

Change a setting in the YAML file, not in the Python code. Each training run saves a copy of its exact settings in `runs/<timestamp>_<name>/config.yaml`.

### Reward matrix (from the revised scope)

| True \ Predicted | unknown_symbol | triangle | quadrilateral | ellipse |
|---|---|---|---|---|
| unknown_symbol | **0.5** | 0.0 | 0.0 | 0.0 |
| triangle | 0.5 | **1.0** | 0.0 | 0.0 |
| quadrilateral | 0.5 | 0.0 | **1.0** | 0.0 |
| ellipse | 0.5 | 0.0 | 0.0 | **1.0** |

In words: naming the right shape earns +1; "unknown / cannot identify" earns +0.5; anything wrong (naming the wrong shape, or naming a shape when there isn't one) earns 0. Results report the average reward and the **reward as a percentage of the best possible**, because a correct `unknown_symbol` can only earn 0.5.

## Hardware used for training

All results in this repo were produced on the lab's tower PC:

| Part | Details |
|---|---|
| CPU | AMD EPYC 7443P, 24 cores / 48 threads |
| GPU | 2× NVIDIA RTX 4000 Ada Generation, 20 GB VRAM (GPU memory) each |
| RAM | 256 GB |
| Software | Ubuntu Linux, Python 3.12 (via `uv`), PyTorch 2.15 with CUDA |

On it, one full Stage A training run takes about 17 minutes on one GPU, the learning curve about 30 minutes using both GPUs, and the model classifies an image in a few milliseconds.

None of this is required: a laptop with any NVIDIA GPU trains the model in under an hour, and `predict.py` runs on any CPU.

## Repository layout

```
configs/            settings files (YAML)
data/               datasets: NOT in git except README.md and SOURCES.md
src/symbolcnn/      project code: config, reward, utils, imageprep, datasets, models, metrics, plots
docs/               scope revision draft and other write-ups
scripts/            command-line tools: check_env, download_public, prepare_data, train, evaluate,
                    learning_curve, predict
tests/              pytest unit tests
results/            small final plots/tables for the presentation
runs/               one folder per training run: NOT in git
```

## Progress (Lesson 36 checkpoint)

The written progress summary is in [`docs/progress_lesson36.md`](docs/progress_lesson36.md); its plots are in `results/lesson36/`.


- [x] Repo structure, README, requirements
- [x] Environment verified on the training computer (see [Hardware used for training](#hardware-used-for-training))
- [x] `reward.py` with passing unit tests
- [x] `download_public.py` (HDS, Quick, Draw!, Hugging Face; Kaggle once a token is added)
- [x] `prepare_data.py` (drawer-based split, manifest, 224×224 images)
- [x] Kaggle datasets downloaded, mapped and licenses verified in `data/SOURCES.md` (both CC0)
- [ ] At least 100 own photos per class, with metadata
- [x] `train.py` written and tested on CPU
- [x] `train.py` Stage A full run on the GPU (4 classes)
- [x] `evaluate.py`
- [x] `learning_curve.py` (10/25/50/100%)
- [x] `predict.py` demo

## Data credits

- Hand-drawn Shapes (HDS) Dataset © 2022 by Francois Robert, licensed under CC BY 4.0.
- Quick, Draw! data made available by Google, Inc. under CC BY 4.0.
- Geometric Shapes Dataset (`0-ma/geometric-shapes`), MIT license.
- Geometric Shapes Mathematics (Kaggle, Moch. Galang Rivaldo) and Four Shapes (Kaggle, smeschke), both CC0 1.0.
- Other dataset licenses: see [`data/SOURCES.md`](data/SOURCES.md).
