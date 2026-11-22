"""Flatten layer implemented with tensor operations."""

import torch


class Flatten:
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], -1)

    __call__ = forward
