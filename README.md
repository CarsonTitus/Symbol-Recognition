# Hand-Drawn Symbol Recognition CNN

CS471 Artificial Intelligence final project. This repo trains a CNN (convolutional neural network) that labels each image as one of:

| Output | Meaning |
|---|---|
| `no_symbol` | no drawn symbol |
| `unknown_symbol` | a drawn symbol that is not a triangle |
| `triangle` | a hand-drawn triangle |

It is "Step 2" (the classifier) of the PSYOP-VISR capstone app. The full project brief is in [`CLAUDE.md`](CLAUDE.md).

## Where things run

| Machine | Used for | PyTorch build |
|---|---|---|
| **Tower PC** (Windows, 2 NVIDIA GPUs) | real training and evaluation | CUDA (NVIDIA GPU) build |
| **Laptop** | editing, tests, small trial runs, `predict.py` demos | CPU build (or CUDA if it has an NVIDIA GPU) |
| **Claude Code cloud sessions** | writing code, running tests | CPU build, installed automatically |

All machines share code through **GitHub**: `git pull` before working, `git push` after. Data and trained models are **not** in git; copy them by USB drive or shared drive (see [`data/README.md`](data/README.md)).

## Setup

Python **3.11 or 3.12** is required. PyTorch is installed **separately** from `requirements.txt` because each machine needs a different build.

### Tower PC or Windows laptop (PowerShell)

```powershell
git clone https://github.com/CarsonTitus/Symbol-Recognition-.git
cd Symbol-Recognition-
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If `Activate.ps1` is blocked ("running scripts is disabled"), run this once, then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Install PyTorch. **Tower (NVIDIA GPUs):** get the exact command from https://pytorch.org/get-started/locally/ by choosing *Stable → Windows → Pip → Python → the newest CUDA version*. It looks like this, with the CUDA number filled in:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cuXXX
```

**Laptop without an NVIDIA GPU:**

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Then install everything else and check the machine:

```powershell
pip install -r requirements.txt
python scripts/check_env.py --download-weights
```

On the tower, `check_env.py` should list **two** GPUs (`cuda:0` and `cuda:1`) and end with `All checks passed.`
Run `nvidia-smi` too and record the exact GPU model and VRAM (GPU memory) in `CLAUDE.md` §14.

### macOS or Linux laptop (bash)

```bash
git clone https://github.com/CarsonTitus/Symbol-Recognition-.git
cd Symbol-Recognition-
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

(On a Mac, use plain `pip install torch torchvision` instead; it includes Apple-GPU support.)

```bash
pip install -r requirements.txt
python scripts/check_env.py --download-weights
```

## Run the tests

```bash
pytest
```

`tests/test_reward.py` proves the reward math matches the submitted rubric; `tests/test_config.py` checks the settings files.

## Settings

All settings live in [`configs/`](configs/):

- `default.yaml`: classes, reward matrix, confidence threshold (0.6), image size and training hyperparameters.
- `stage_a.yaml` / `stage_b.yaml`: change only what differs for each training stage.

Change a setting in the YAML file, not in the Python code. Each training run saves a copy of its exact settings in `runs/<timestamp>_<name>/config.yaml`.

### Reward matrix (from the submitted scope)

| True \ Predicted | no_symbol | unknown_symbol | triangle |
|---|---|---|---|
| no_symbol | 1.0 | 0.0 | 0.0 |
| unknown_symbol | 0.0 | 0.5 | 0.0 |
| triangle | 0.0 | 0.5 | 1.0 |

Results report the average reward and the **reward as a percentage of the best possible**, because a correct `unknown_symbol` can only earn 0.5.

## Repository layout

```
configs/            settings files (YAML)
data/               datasets: NOT in git except README.md and SOURCES.md
src/symbolcnn/      project code: config.py, reward.py, utils.py (more coming)
scripts/            command-line tools: check_env.py (more coming)
tests/              pytest unit tests
results/            small final plots/tables for the presentation
runs/               one folder per training run: NOT in git
```

## Progress (Lesson 36 checkpoint)

- [x] Repo structure, README, requirements
- [ ] Environment verified on the tower (`python scripts/check_env.py --download-weights`)
- [x] `reward.py` with passing unit tests
- [ ] `download_public.py` and `prepare_data.py`; `data/SOURCES.md` licenses verified
- [ ] At least 100 own photos per class, with metadata
- [ ] `train.py` Stage A
- [ ] `evaluate.py`
- [ ] `learning_curve.py` (10/25/50/100%)
- [ ] `predict.py` demo

## Data credits

- Hand-drawn Shapes (HDS) Dataset © 2022 by Francois Robert, licensed under CC BY 4.0.
- Other dataset licenses: see [`data/SOURCES.md`](data/SOURCES.md).
