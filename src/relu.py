"""ReLU activation implemented with tensor operations."""

import torch


class ReLU:
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x > 0, x, torch.zeros_like(x))

    __call__ = forward
