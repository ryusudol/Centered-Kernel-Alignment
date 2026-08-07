<div align="center">

# pytorch-cka

[![PyPI](https://img.shields.io/pypi/v/pytorch-cka.svg)](https://pypi.org/project/pytorch-cka/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/pytorch-cka/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/pytorch-cka?period=total&units=INTERNATIONAL_SYSTEM&left_color=GREY&right_color=RED&left_text=downloads)](https://pepy.tech/projects/pytorch-cka)

**The Fastest, Memory-efficient Python Library for computing layer-wise similarity between neural network models**

</div>

<p align="center">
    <picture align="center">
        <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/d915e2c9-4d76-4a57-95f2-7b94955d9ffd">
        <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/69afbbdf-a86e-43b6-afd4-a0b8e39037f3">
        <img alt="A bar chart with benchmark results" src="https://github.com/user-attachments/assets/69afbbdf-a86e-43b6-afd4-a0b8e39037f3" width="100%" />
    </picture>
</p>

<p align="center">
  <i><b>44x</b> faster CKA computation across 18 representational layers of ResNet-18 models on CIFAR-10 using NVIDIA H100 GPUs</i>
</p>

- ⚡️ Fastest among CKA libraries thanks to **vectorized ops** & **GPU acceleration**
- 📦 Efficient memory management with explicit deallocation
- 🧠 Supports HuggingFace models, DataParallel, and DDP
- 🎨 Customizable visualizations: heatmaps and line charts

## 📦 Installation

Requires `Python 3.10+`

```bash
# Using pip
pip install pytorch-cka

# Using uv
uv add pytorch-cka
```

## 👟 Quick Start

### Basic Usage

```python
from cka import compute_cka
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnet34

resnet_18 = resnet18(pretrained=True)
resnet_34 = resnet34(pretrained=True)

dataloader1 = Dataloader(your_dataset1, batch_size=bach_size, shuffle=False, num_workers=4)
dataloader2 = Dataloader(your_dataset2, batch_size=bach_size, shuffle=False, num_workers=4)
dataloader3 = Dataloader(your_dataset3, batch_size=bach_size, shuffle=False, num_workers=4)
dataloaders = [dataloader1, dataloader2, dataloader3]

layers = [
    'conv1',
    'layer1.0.conv1',
    'layer2.0.conv1',
    'layer3.0.conv1',
    'layer4.0.conv1',
    'fc',
]

cka_matrices = compute_cka(
    resnet_18,
    resnet_34,
    dataloaders,
    layers=layers,
    device=device,
)

for cka_matrix in cka_matrices:
    print(cka_matrix)
```

### From Pre-extracted Features

If you already have feature matrices, compute CKA without models or dataloaders:

```python
from cka import cka_from_features

# Single layer: (n_samples, feature_dim)
cka_matrix = cka_from_features(features_x, features_y)

# Multi-layer: (n_layers, n_samples, feature_dim)
cka_matrix = cka_from_features(multi_layer_x, multi_layer_y)

# Varying feature dims: list of 2D tensors
cka_matrix = cka_from_features(
    [layer1_x, layer2_x],
    [layer1_y, layer2_y, layer3_y],
)
```

### Visualization

**Heatmap**

```python
from cka import plot_cka_heatmap

fig, ax = plot_cka_heatmap(
    cka_matrix,
    layers1=layers,
    layers2=layers,
    model1_name="ResNet-18 (pretrained)",
    model2_name="ResNet-18 (random init)",
    annot=False,          # Show values in cells
    cmap="inferno",       # Colormap
)
```

<table width="100%">
    <tr>
      <td width="63.5%" style="padding:0; vertical-align:top; line-height:0;">
          <picture>
              <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/987edc7d-cdaf-43a2-bc97-36ca058a0c26">
              <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/173a6fcd-b271-4ad9-aadf-933809a63bd9">
              <img alt="Self-comparison heatmap" src="https://github.com/user-attachments/assets/173a6fcd-b271-4ad9-aadf-933809a63bd9" style="display:block; width:100%;" />
          </picture>
      </td>
      <td width="36.5%" style="padding:0; vertical-align:top; line-height:0;">
          <picture>
              <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/e15a2f7c-7f3d-4233-872b-f72ddca5078e">
              <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/1263198f-da4f-469b-aa50-a64befb670b6">
              <img alt="Cross-model comparison heatmap" src="https://github.com/user-attachments/assets/1263198f-da4f-469b-aa50-a64befb670b6" style="display:block; width:100%;" />
          </picture>
      </td>
    </tr>
    <tr>
      <td align="center">Self-comparison</td>
      <td align="center">Cross-model</td>
    </tr>
</table>

**Trend Plot**

```python
from cka import plot_cka_trend

# Plot diagonal (self-similarity across layers)
diagonal = torch.diag(matrix)

fig, ax = plot_cka_trend(
    layer_trends,
    x_values=epochs,
    labels=RESNET18_LAYERS,
    markers=['o'],
    xlabel='Epoch',
    ylabel='CKA Score',
    title='Pretrained vs. Fine-tuned Across Epochs (ResNet-18)',
    legend=True,
)

fig, ax = plot_cka_layer_trend(
    cka_matrices,
    layers=RESNET18_LAYERS,
    labels=cka_loader_names,
    ylabel='CKA Score',
    title='Pretrained vs. Fine-tuned Across Layers (ResNet-18)',
    legend=True,
)
```

<table>
    <tr>
      <td>
        <picture>
          <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/f89c22a0-a823-4da8-82c4-0bd91985c0f7">
          <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/33df35b3-49b6-47a7-986e-9252f3bb61bc">
          <img src="https://github.com/user-attachments/assets/33df35b3-49b6-47a7-986e-9252f3bb61bc" alt="CKA Score Trend Across Epochs" width="100%"/>
        </picture>
      </td>
      <td>
        <picture>
          <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/c8729b04-8a34-40c6-a5a6-e39322960468">
          <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/a57b5820-57d9-4f64-9df6-d8eaf1720bde">
          <img src="https://github.com/user-attachments/assets/a57b5820-57d9-4f64-9df6-d8eaf1720bde" alt="CKA Score Trend Across Layers" width="100%"/>
        </picture>
      </td>
    </tr>
    <tr>
      <td align="center">CKA Score Trend Across Epochs</td>
      <td align="center">CKA Score Trend Across Layers</td>
    </tr>
</table>

## 📚 References

Kornblith, Simon, et al. ["Similarity of Neural Network Representations Revisited."](https://arxiv.org/abs/1905.00414) _ICML 2019._
