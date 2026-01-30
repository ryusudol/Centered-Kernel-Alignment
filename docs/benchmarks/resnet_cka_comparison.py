#!/usr/bin/env python3
"""Generate CKA heatmaps comparing ResNet models using CIFAR-10.

This script generates two CKA heatmap visualizations:
1. ResNet-18 vs ResNet-18 (self-comparison)
2. ResNet-18 vs ResNet-50 (cross-model comparison)

Both comparisons use 18 representational layers and the CIFAR-10 dataset.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50

from cka import compute_cka, plot_cka_heatmap

# 18 representational layers for ResNet-18
# (conv1 + 16 conv layers in BasicBlocks + fc = 18)
RESNET18_LAYERS = [
    "conv1",
    "layer1.0.conv1",
    "layer1.0.conv2",
    "layer1.1.conv1",
    "layer1.1.conv2",
    "layer2.0.conv1",
    "layer2.0.conv2",
    "layer2.1.conv1",
    "layer2.1.conv2",
    "layer3.0.conv1",
    "layer3.0.conv2",
    "layer3.1.conv1",
    "layer3.1.conv2",
    "layer4.0.conv1",
    "layer4.0.conv2",
    "layer4.1.conv1",
    "layer4.1.conv2",
    "fc",
]

# 18 representative layers for ResNet-50
# (conv1 + output conv3 from selected bottleneck blocks + avgpool + fc = 18)
RESNET50_LAYERS = [
    "conv1",
    "layer1.0.conv3",
    "layer1.1.conv3",
    "layer1.2.conv3",
    "layer2.0.conv3",
    "layer2.1.conv3",
    "layer2.2.conv3",
    "layer2.3.conv3",
    "layer3.0.conv3",
    "layer3.1.conv3",
    "layer3.3.conv3",
    "layer3.4.conv3",
    "layer3.5.conv3",
    "layer4.0.conv3",
    "layer4.1.conv3",
    "layer4.2.conv3",
    "avgpool",
    "fc",
]


def get_cifar10_dataloader(batch_size: int = 64, num_workers: int = 4) -> DataLoader:
    """Load CIFAR-10 test set with ImageNet-compatible transforms."""
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    dataset = CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def main():
    # Setup device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(__file__).parent / "plots"
    output_dir.mkdir(exist_ok=True)

    # Load CIFAR-10 dataloader
    print("Loading CIFAR-10 dataset...")
    dataloader = get_cifar10_dataloader()

    # Load pretrained models
    print("Loading pretrained ResNet models...")
    model_resnet18 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model_resnet50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # =========================================================================
    # 1. ResNet-18 vs ResNet-18 (self-comparison)
    # =========================================================================
    print("\nComputing CKA: ResNet-18 vs ResNet-18...")
    cka_matrix_self = compute_cka(
        model_resnet18,
        model_resnet18,
        dataloader,
        layers=RESNET18_LAYERS,
        model2_layers=RESNET18_LAYERS,
        device=device,
        progress=True,
    )[0]

    print(f"Diagonal values (should be ~1.0): {torch.diag(cka_matrix_self)[:5]}...")

    # Generate heatmap
    fig, ax = plot_cka_heatmap(
        cka_matrix_self,
        layers1=RESNET18_LAYERS,
        layers2=RESNET18_LAYERS,
        model1_name="ResNet-18",
        model2_name="ResNet-18",
        title="CKA: ResNet-18 vs ResNet-18 (CIFAR-10)",
        layer_name_depth=2,
    )
    output_path = output_dir / "resnet18_self_cka.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")

    # =========================================================================
    # 2. ResNet-18 vs ResNet-50 (cross-model comparison)
    # =========================================================================
    print("\nComputing CKA: ResNet-18 vs ResNet-50...")
    cka_matrix_cross = compute_cka(
        model_resnet18,
        model_resnet50,
        dataloader,
        layers=RESNET18_LAYERS,
        model2_layers=RESNET50_LAYERS,
        device=device,
        progress=True,
    )[0]

    # Generate heatmap
    fig, ax = plot_cka_heatmap(
        cka_matrix_cross,
        layers1=RESNET18_LAYERS,
        layers2=RESNET50_LAYERS,
        model1_name="ResNet-18",
        model2_name="ResNet-50",
        title="CKA: ResNet-18 vs ResNet-50 (CIFAR-10)",
        layer_name_depth=2,
    )
    output_path = output_dir / "resnet18_resnet50_cka.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
