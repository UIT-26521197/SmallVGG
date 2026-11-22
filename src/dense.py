"""Fully connected layer implemented with tensors and autograd."""

import math
import random

import torch


class Dense:
    def __init__(self, InSize: int, OutSize: int) -> None:
        scale = math.sqrt(2.0 / InSize)
        values = [random.gauss(0.0, scale) for _ in range(InSize * OutSize)]
        self.Weight = torch.tensor(
            values,
            dtype=torch.float32,
        ).reshape(InSize, OutSize).detach()
        self.Weight.requires_grad_()
        self.Bias = torch.tensor(
            [0.0] * OutSize,
            dtype=torch.float32,
            requires_grad=True,
        )

    def parameters(self):
        return [self.Weight, self.Bias]

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            if parameter.grad is not None:
                parameter.grad.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.Weight + self.Bias

    __call__ = forward
