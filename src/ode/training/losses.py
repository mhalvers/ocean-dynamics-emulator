from __future__ import annotations

import torch


def mse_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.mean((predictions - targets) ** 2)
