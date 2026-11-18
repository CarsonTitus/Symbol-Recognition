"""
datasets.py - feed images from data/processed to the model during training.

What this does:
    load_split()          reads data/processed/manifest.csv and returns the rows
                          for one split (train, val, test, test_novel_shapes),
                          optionally only a fraction of them (learning-curve runs).
    SymbolDataset         a PyTorch "Dataset": given an index, it loads that image,
                          applies the transforms, and returns (image tensor, label number).
    train_transform()     random changes ("augmentation") for TRAINING images only,
                          so the model copes with messy real photos (CLAUDE.md §7).
    eval_transform()      no random changes; used for validation, test and predict.
    balanced_sampler()    picks rarer classes more often so each class is seen
                          about equally (we have ~13x more unknown_symbol than no_symbol).

Why:
    PyTorch trains on batches of tensors (multi-dimensional number arrays).
    This file is the bridge from image files on disk to those batches.
"""
from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, WeightedRandomSampler


# ----------------------------------------------------------------------------
# Reading the manifest
# ----------------------------------------------------------------------------
def load_split(processed: Path, split: str, fraction: float = 1.0, seed: int = 0) -> pd.DataFrame:
    """Rows of manifest.csv for one split.

    fraction < 1 keeps that share of EACH class (a "stratified" sample), so a
    10% learning-curve run still has all three classes in the same proportions.
    """
    manifest = pd.read_csv(Path(processed) / "manifest.csv", keep_default_na=False)
    rows = manifest[manifest["split"] == split]
    if fraction < 1.0:
        rows = rows.groupby("label", group_keys=False).sample(frac=fraction, random_state=seed)
    return rows.reset_index(drop=True)


# ----------------------------------------------------------------------------
# The Dataset
# ----------------------------------------------------------------------------
class SymbolDataset(Dataset):
    """Loads processed images listed in a manifest table."""

    def __init__(self, rows: pd.DataFrame, processed: Path, classes: list[str], transform):
        self.files = [Path(processed) / f for f in rows["file"]]
        self.labels = [classes.index(label) for label in rows["label"]]   # name -> 0, 1 or 2
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        # OpenCV loads colors in BGR order; convert to the usual RGB.
        image = cv2.cvtColor(cv2.imread(str(self.files[index]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return self.transform(image=image)["image"], self.labels[index]


# ----------------------------------------------------------------------------
# Transforms
# ----------------------------------------------------------------------------
def _to_tensor(mean, std) -> list:
    """Last two steps of every transform: scale pixels like the pretrained model expects, make a tensor."""
    return [A.Normalize(mean=mean, std=std), ToTensorV2()]


def train_transform(cfg: dict, mean, std) -> A.Compose:
    """Random augmentation from CLAUDE.md §7. Each step happens with probability p."""
    size = cfg["image"]["size"]
    replicate = cv2.BORDER_REPLICATE   # new corners after rotating/tilting copy the card's edge color
    steps = [
        # Rotate any direction AND shrink/shift in one step. The public drawings fill their whole
        # image, so rotating or cropping them alone cuts corners off; shrinking first keeps the shape whole.
        A.Affine(rotate=(-180, 180), scale=(0.6, 1.0), translate_percent=(-0.1, 0.1),
                 border_mode=replicate, p=1.0),
        A.Perspective(scale=(0.02, 0.08), border_mode=replicate, p=0.5),          # card seen at an angle
        # CLAUDE.md §7 suggests crop scale 0.6-1.0; 0.75 is used because at 0.6 crops of these
        # edge-to-edge drawings often removed most of the shape (seen in what_the_model_sees.png).
        A.RandomResizedCrop(size=(size, size), scale=(0.75, 1.0), ratio=(0.8, 1.25), p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.RandomGamma(gamma_limit=(70, 150), p=0.3),
        # No A.RandomShadow: it draws thin, dark, pointed polygons that look like drawn
        # triangles, which would teach the model wrong labels. Real shadows come from our photos.
        A.OneOf([A.MotionBlur(blur_limit=(3, 9)), A.GaussianBlur(blur_limit=(3, 7))], p=0.3),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.3),                              # camera noise
        A.ImageCompression(quality_range=(30, 95), p=0.5),                        # JPEG artifacts
        A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(0.1, 0.25),    # finger over card
                        hole_width_range=(0.1, 0.25), fill="random_uniform", p=0.3),
        A.Downscale(scale_range=(0.3, 0.8), p=0.3),                               # far away = fewer pixels
        # On dark-on-light drawings, "erosion" thickens the pen lines and "dilation" thins them.
        A.OneOf([A.Morphological(scale=(2, 3), operation="erosion"),
                 A.Morphological(scale=(2, 3), operation="dilation")], p=0.3),
    ]
    if cfg["image"]["allow_flips"]:   # fine for triangles; switch off for asymmetric symbols
        steps += [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)]
    return A.Compose(steps + _to_tensor(mean, std))


def eval_transform(mean, std) -> A.Compose:
    """No randomness: validation/test/predict images only get normalized."""
    return A.Compose(_to_tensor(mean, std))


def denormalize(images: torch.Tensor, mean, std) -> np.ndarray:
    """Undo Normalize so a batch can be shown as pictures. Returns (N, H, W, 3) uint8."""
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    pixels = (images.detach().float().cpu() * std + mean).clamp(0, 1)
    return (pixels.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# Class balancing
# ----------------------------------------------------------------------------
def balanced_sampler(labels: list[int], num_classes: int, seed: int) -> WeightedRandomSampler:
    """Sample each image with weight 1 / (images in its class).

    Every class then gets about the same number of picks per epoch, even though
    unknown_symbol has far more images. Rare-class images are simply seen more often
    (each time with different random augmentation).
    """
    counts = np.bincount(labels, minlength=num_classes)
    weights = [1.0 / counts[label] for label in labels]
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(labels), replacement=True, generator=generator)
