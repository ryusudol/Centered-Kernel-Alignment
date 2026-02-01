<div align="center">

# pytorch-cka

[![PyPI](https://img.shields.io/pypi/v/pytorch-cka.svg)](https://pypi.org/project/pytorch-cka/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/pytorch-cka/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/pytorch-cka?period=total&units=INTERNATIONAL_SYSTEM&left_color=GREY&right_color=RED&left_text=downloads)](https://pepy.tech/projects/pytorch-cka)

**The Fastest, Memory-efficient Python Library for computing layer-wise similarity between neural network models**

</div>

<p align="center">
    <picture align="center">
        <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/a20bb8fb-9485-4259-8239-51ba66fcd49c">
        <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/60a6dc50-6368-4eb9-9145-02a7d6b98961">
        <img alt="A bar chart with benchmark results in light mode" src="https://github.com/user-attachments/assets/a20bb8fb-9485-4259-8239-51ba66fcd49c" width="100%" />
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
          <img
              src="https://github.com/user-attachments/assets/48bfd811-629e-483a-bc63-2a493872289c"
              alt="Self-comparison heatmap"
              style="display:block; width:100%;"
          />
      </td>
      <td width="36.5%" style="padding:0; vertical-align:top; line-height:0;">
          <img
              src="https://github.com/user-attachments/assets/38ad963a-e8ce-4dc1-ab62-73d4f072ceca"
              alt="Cross-model comparison heatmap"
              style="display:block; width:100%;"
          />
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
    diagonal,
    labels=["Self-similarity"],
    xlabel="Layer",
    ylabel="CKA Score",
)
```

<table>
    <tr>
      <td><img src="https://github.com/user-attachments/assets/70643ff3-33a2-4733-bde4-1faefbb9b741" alt="CKA Score Trend Across Epochs" width="100%"/></td>
      <td><img src="https://github.com/user-attachments/assets/72e199ee-fa9f-40e3-ad65-10966ee31ebe" alt="CKA Score Trend Across Layers" width="100%"/></td>
    </tr>
    <tr>
      <td align="center">CKA Score Trend Across Epochs</td>
      <td align="center">CKA Score Trend Across Layers</td>
    </tr>
</table>

## 📚 References

Kornblith, Simon, et al. ["Similarity of Neural Network Representations Revisited."](https://arxiv.org/abs/1905.00414) _ICML 2019._
