"""pytorch-cka: Centered Kernel Alignment for PyTorch models.

A numerically stable, memory-safe library for comparing neural network
representations using Centered Kernel Alignment (CKA).

Example:
    >>> from cka import compute_cka
    >>>
    >>> matrices = compute_cka(
    ...     model1,
    ...     model2,
    ...     dataloader,
    ...     layers=["layer1", "layer2"],
    ... )
    >>> fig, ax = plot_cka_heatmap(
    ...     matrices[0],
    ...     model1_layers=["layer1", "layer2"],
    ... )

References:
    - Kornblith et al., 2019: "Similarity of Neural Network Representations Revisited"
    - Nguyen et al., 2020: "Do Wide and Deep Networks Learn the Same Things?"
"""

__version__ = "1.0.0"

from .cka import compute_cka, CKA
from .hsic import hsic, hsic_cross
from .viz import (
    plot_cka_comparison,
    plot_cka_heatmap,
    plot_cka_trend,
    plot_cka_trend_with_range,
    save_figure,
)

__all__ = [
    "__version__",
    "compute_cka",
    "CKA",
    "hsic",
    "hsic_cross",
    "plot_cka_heatmap",
    "plot_cka_trend",
    "plot_cka_trend_with_range",
    "plot_cka_comparison",
    "save_figure",
]
