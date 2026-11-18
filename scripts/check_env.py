"""
check_env.py - check that a machine is ready to run this project.

What this does:
    Prints the Python version, the PyTorch version, whether PyTorch can see an
    NVIDIA GPU (and each GPU's name and memory), the CPU core count, and whether
    the project's own code and the other packages import. With --download-weights
    it also downloads the pretrained MobileNetV3 and runs one fake image through it.

Why:
    The first Lesson 36 checklist item is "environment verified on the tower"
    (CLAUDE.md §11). Running this on the tower, a laptop or the cloud shows in
    one screen whether that machine is set up correctly.

Usage (PowerShell or bash):
    python scripts/check_env.py
    python scripts/check_env.py --download-weights
"""
from __future__ import annotations

import argparse
import importlib
import os
import platform
import shutil
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Check this machine's setup for the symbol CNN project.")
    parser.add_argument("--download-weights", action="store_true",
                        help="also download the pretrained MobileNetV3 and run one test image")
    args = parser.parse_args()

    problems = []
    print(f"Python      : {sys.version.split()[0]}  ({platform.system()} {platform.release()})")
    if sys.version_info[:2] not in [(3, 11), (3, 12)]:
        print("              note: the project is tested on Python 3.11/3.12")
    print(f"CPU cores   : {os.cpu_count()}")

    try:
        import torch
    except ImportError:
        print("PyTorch     : NOT INSTALLED - see README step 2")
        return 1

    print(f"PyTorch     : {torch.__version__}")
    if torch.cuda.is_available():
        # CUDA = NVIDIA's GPU computing platform; VRAM = the GPU's own memory.
        print(f"CUDA GPUs   : {torch.cuda.device_count()} (CUDA {torch.version.cuda})")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  cuda:{i}    : {props.name}, {props.total_memory / 1024**3:.1f} GB VRAM")
    else:
        print("CUDA GPUs   : none (training will use the CPU - fine for tests, slow for real training)")
        # If the NVIDIA driver's nvidia-smi tool exists, this machine HAS a GPU that PyTorch can't use.
        if shutil.which("nvidia-smi"):
            problems.append("an NVIDIA driver is installed but PyTorch can't use the GPU: "
                            "reinstall PyTorch with GPU support (README, PyTorch install step)")

    for module in ["symbolcnn", "timm", "albumentations", "cv2", "numpy", "pandas",
                   "sklearn", "matplotlib", "yaml", "tqdm"]:
        try:
            importlib.import_module(module)
        except ImportError as error:
            problems.append(f"cannot import {module}: {error}")

    if args.download_weights:
        import timm

        model = timm.create_model("mobilenetv3_large_100", pretrained=True, num_classes=3).eval()
        with torch.no_grad():
            output = model(torch.randn(1, 3, 224, 224))
        print(f"MobileNetV3 : weights downloaded, test output shape {tuple(output.shape)} (expect (1, 3))")

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
