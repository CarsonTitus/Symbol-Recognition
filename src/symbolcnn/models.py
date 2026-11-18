"""
models.py - build the CNN (convolutional neural network).

What this does:
    create_model(cfg) returns one of two models, chosen by model.name in the config:
      "mobilenetv3_large_100"  the main model: a phone-sized CNN from the timm
                               library, pretrained on ImageNet (1.2M photos), with
                               its final 1000-category layer replaced by a 3-category one.
      "tiny_cnn"               a small CNN written from scratch (CLAUDE.md §6). Used to
                               show why pretraining matters, and to explain a CNN
                               layer by layer. Also used by the fast unit tests.
    freeze_backbone() / unfreeze_all() switch which layers are allowed to learn.
    normalization(model) gives the mean/std the model's input pixels must be scaled by.

Why:
    Stage A first trains only the new final layer (backbone frozen), then the
    whole network (CLAUDE.md §7). The "backbone" = the feature-detecting layers.
"""
from __future__ import annotations

import timm
import torch
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)   # standard pixel scaling for ImageNet-trained models
IMAGENET_STD = (0.229, 0.224, 0.225)


class TinyCNN(nn.Module):
    """A 4-block CNN written from scratch.

    Each block: convolution (slides 3x3 filters over the image to detect patterns)
    -> batch normalization (keeps numbers in a stable range)
    -> ReLU (keeps positive values, zeroes negatives)
    -> max pooling (halves the width and height, keeping the strongest signals).
    Early blocks find edges; later blocks combine them into corners and shapes.
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()

        def block(channels_in: int, channels_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        # 224x224x3 -> 112x112x16 -> 56x56x32 -> 28x28x64 -> 14x14x128
        self.features = nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 128))
        self.pool = nn.AdaptiveAvgPool2d(1)            # average each of the 128 feature maps to 1 number
        self.classifier = nn.Linear(128, num_classes)  # 128 numbers -> one score per class

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.features(x)).flatten(1)
        return self.classifier(x)

    def get_classifier(self) -> nn.Module:
        return self.classifier


def create_model(cfg: dict, pretrained: bool | None = None) -> nn.Module:
    """Build the model named in cfg["model"]["name"] with one output per class."""
    name = cfg["model"]["name"]
    num_classes = len(cfg["classes"])
    if name == "tiny_cnn":
        return TinyCNN(num_classes)
    if pretrained is None:
        pretrained = cfg["model"]["pretrained"]
    # pretrained=True downloads ImageNet weights from Hugging Face (cached after the first time).
    # num_classes=3 swaps the final 1000-category layer for a new, untrained 3-category layer.
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)


def normalization(model: nn.Module) -> tuple[tuple, tuple]:
    """(mean, std) for scaling input pixels, taken from the model's own settings when it has them."""
    if isinstance(model, TinyCNN):
        return IMAGENET_MEAN, IMAGENET_STD
    data_config = timm.data.resolve_data_config({}, model=model)
    return tuple(data_config["mean"]), tuple(data_config["std"])


def freeze_backbone(model: nn.Module) -> None:
    """Only the final classification layer may learn; the pretrained layers stay fixed."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.get_classifier().parameters():
        parameter.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    """Every layer may learn."""
    for parameter in model.parameters():
        parameter.requires_grad = True


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
