"""Configuration dataclasses for CKA computation."""

from dataclasses import dataclass
from typing import Literal, Optional

import torch


@dataclass
class CKAConfig:
    """Configuration for CKA computation.

    Attributes:
        kernel: Kernel type for gram matrix computation ("linear" or "rbf").
        sigma: RBF kernel bandwidth. If None, uses median heuristic.
        unbiased: Use unbiased HSIC estimator (required for minibatch CKA).
        epsilon: Small constant for numerical stability.
        dtype: Computation dtype. float64 recommended for precision.
        device: Target device for computation. If None, auto-detects from model.
    """

    kernel: Literal["linear", "rbf"] = "linear"
    sigma: Optional[float] = None
    unbiased: bool = True
    epsilon: float = 1e-6
    dtype: torch.dtype = torch.float64
    device: Optional[torch.device] = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.kernel not in ("linear", "rbf"):
            raise ValueError(f"kernel must be 'linear' or 'rbf', got '{self.kernel}'")
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")
        if self.sigma is not None and self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        if self.device is not None and not isinstance(self.device, torch.device):
            self.device = torch.device(self.device)
