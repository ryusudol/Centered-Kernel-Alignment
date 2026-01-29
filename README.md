<div align="center">

# pytorch-cka

[![PyPI](https://img.shields.io/pypi/v/pytorch-cka.svg)](https://pypi.org/project/pytorch-cka/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/pytorch-cka/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/pytorch-cka?period=total&units=INTERNATIONAL_SYSTEM&left_color=GREY&right_color=RED&left_text=downloads)](https://pepy.tech/projects/pytorch-cka)

**The Fastest, Memory-efficient Python Library for CKA with Built-in Visualization**

</div>

<p align="center">
    <picture align="center">
        <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/e5f5c51b-4298-424c-81dd-02657af60247">
        <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/2e665d9d-5645-4726-996f-3a1817136eaf">
        <img alt="A bar chart with benchmark results in dark mode" src="https://github.com/user-attachments/assets/2e665d9d-5645-4726-996f-3a1817136eaf" width="100%" />
    </picture>
</p>

<p align="center">
  <i><b>3500%</b> faster CKA computation across all layers of two distinct ResNet-18 models on CIFAR-10 using NVIDIA H100 GPUs</i>
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

cka_matrices = compute_cka(
    model1,
    model2,
    [dataloader1, dataloader2, dataloader3],
    layers=["layer1", "layer2", "layer3", "fc"],
    device=device,
)

# compute_cka returns one matrix per dataloader, in order
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

<table>
    <tr>
      <td><img src="https://github.com/user-attachments/assets/f74a322d-1a19-4c4f-b1a2-07f238651512" alt="Self-comparison heatmap" width="100%"/></td>
      <td><img src="https://github.com/user-attachments/assets/2121976c-c230-40b0-92d5-c48b5bf876c9" alt="Cross-model comparison heatmap" width="100%"/></td>
      <!-- <td><img src="plots/heatmap_masked.png" alt="Masked upper triangle heatmap" width="100%"/></td> -->
    </tr>
    <tr>
      <td align="center">Self-comparison</td>
      <td align="center">Cross-model</td>
      <!-- <td align="center">Masked Upper</td> -->
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
      <td><img src="https://github.com/user-attachments/assets/d5a42b85-36a4-4778-a13d-1a3a76f55b10" alt="Cross model CKA scores trends" width="100%"/></td>
      <td><img src="https://github.com/user-attachments/assets/6af56561-1e1d-45a9-8af7-796ae5e434d0" alt="Multiple trends comparison" width="100%"/></td>
    </tr>
    <tr>
      <td align="center">Cross Model CKA Scores Trends</td>
      <td align="center">Multiple Trends</td>
    </tr>
</table>

## 📚 References

Kornblith, Simon, et al. ["Similarity of Neural Network Representations Revisited."](https://arxiv.org/abs/1905.00414) _ICML 2019._
