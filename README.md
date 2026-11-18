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
| **Tower PC** (Ubuntu Linux, 2 NVIDIA GPUs) | real training and evaluation | CUDA (NVIDIA GPU) build |
| **Laptop** | editing, tests, small trial runs, `predict.py` demos | CPU build (or CUDA if it has an NVIDIA GPU) |
| **Claude Code cloud sessions** | writing code, running tests | CPU build, installed automatically |

All machines share code through **GitHub**: `git pull` before working, `git push` after. Data and trained models are **not** in git; copy them by USB drive or shared drive (see [`data/README.md`](data/README.md)).

## Setup

Python **3.11 or 3.12** is required. PyTorch is installed **separately** from `requirements.txt` because each machine needs a different build.

### Tower PC (Ubuntu Linux, 2 NVIDIA GPUs)

The tower's system Python is 3.10, which is too old. `uv` (a fast Python package tool) downloads Python 3.12 for this project only. It needs no `sudo` and doesn't touch the system Python.

Install `uv` (one time per user account):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Tell the terminal where `uv` was installed, now and in every future terminal:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
```

```bash
uv --version
```

Get the code and create the Python 3.12 virtual environment (a private folder of packages for this project):

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

Install PyTorch. On Linux the default package already includes CUDA (NVIDIA's GPU platform) support:

```bash
uv pip install torch torchvision
```

```bash
uv pip install -r requirements.txt
```

Check the machine:

```bash
nvidia-smi
```

```bash
python scripts/check_env.py --download-weights
```

`check_env.py` should list **two** GPUs (`cuda:0` and `cuda:1`) and end with `All checks passed.` Record the exact GPU model and VRAM (GPU memory) from `nvidia-smi` in `CLAUDE.md` §14.

**Every new terminal:** `cd Symbol-Recognition-` then `source .venv/bin/activate` before running anything. The prompt starts with `(.venv)` when it's active.

### Linux or macOS laptop (bash)

Same steps as the tower, with one difference when installing PyTorch. A laptop **without an NVIDIA GPU** should use the smaller CPU-only build:

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

On a Mac, use plain `uv pip install torch torchvision`; it includes Apple-GPU support.

### Windows laptop (PowerShell)

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
- [x] Environment verified on the tower (2× RTX 4000 Ada, Python 3.12, all checks and tests pass)
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
