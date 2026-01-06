"""torchcka: Centered Kernel Alignment for PyTorch.

A numerically stable, memory-safe library for comparing neural network
representations using Centered Kernel Alignment (CKA).

Example:
    >>> from torchcka import CKA, CKAConfig
    >>>
    >>> config = CKAConfig(kernel="linear", unbiased=True)
    >>> with CKA(model1, model2, layers1=["layer1", "layer2"], config=config) as cka:
    ...     matrix = cka.compare(dataloader)
    ...     fig, ax = plot_cka_heatmap(matrix, layers1=["layer1", "layer2"])

References:
    - Kornblith et al., 2019: "Similarity of Neural Network Representations Revisited"
    - Nguyen et al., 2020: "Do Wide and Deep Networks Learn the Same Things?"
"""

__version__ = "1.0.0"

from .cka import CKA, ModelInfo
from .config import CKAConfig
from .core import (
    center_gram_matrix,
    cka,
    cka_from_gram,
    compute_gram_matrix,
    hsic,
    hsic_biased,
    hsic_unbiased,
    linear_kernel,
    rbf_kernel,
)
from .utils import (
    FeatureCache,
    eval_mode,
    get_all_layer_names,
    get_device,
    unwrap_model,
    validate_batch_size,
    validate_layers,
)
from .viz import (
    plot_cka_comparison,
    plot_cka_heatmap,
    plot_cka_trend,
    save_figure,
)

__all__ = [
    # Version
    "__version__",
    # Config
    "CKAConfig",
    # Main class
    "CKA",
    "ModelInfo",
    # Core functions
    "linear_kernel",
    "rbf_kernel",
    "compute_gram_matrix",
    "center_gram_matrix",
    "hsic",
    "hsic_biased",
    "hsic_unbiased",
    "cka",
    "cka_from_gram",
    # Utilities
    "validate_batch_size",
    "validate_layers",
    "get_all_layer_names",
    "get_device",
    "unwrap_model",
    "FeatureCache",
    "eval_mode",
    # Visualization
    "plot_cka_heatmap",
    "plot_cka_trend",
    "plot_cka_comparison",
    "save_figure",
]
