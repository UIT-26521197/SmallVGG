"""2-D convolution implemented with tensors and PyTorch autograd only."""

import math
import random

import torch


class Conv:
    """Convolution layer for tensors shaped ``(batch, channels, height, width)``."""

    def __init__(
        self,
        InChannel: int,
        OutChannel: int,
        KernelSize: int = 3,
        Padding: int = 1,
        Stride: int = 1,
    ) -> None:
        self.KernelSize = KernelSize
        self.Padding = Padding
        self.Stride = Stride
        self.OutChannel = OutChannel

        scale = math.sqrt(2.0 / (KernelSize * KernelSize * InChannel))
        values = [
            random.gauss(0.0, scale)
            for _ in range(OutChannel * InChannel * KernelSize * KernelSize)
        ]
        self.Weight = torch.tensor(
            values,
            dtype=torch.float32,
        ).reshape(OutChannel, InChannel, KernelSize, KernelSize).detach()
        self.Weight.requires_grad_()
        self.Bias = torch.tensor(
            [0.0] * OutChannel,
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
        if x.ndim != 4:
            raise ValueError("Conv expects (batch, channels, height, width)")

        batch, _, height, width = x.shape
        kernel = self.KernelSize
        padding = self.Padding
        stride = self.Stride

        if padding > 0:
            vertical = torch.zeros(
                (batch, x.shape[1], padding, width),
                dtype=x.dtype,
                device=x.device,
            )
            x = torch.cat((vertical, x, vertical), dim=2)
            horizontal = torch.zeros(
                (batch, x.shape[1], x.shape[2], padding),
                dtype=x.dtype,
                device=x.device,
            )
            x = torch.cat((horizontal, x, horizontal), dim=3)

        height_out = (height + 2 * padding - kernel) // stride + 1
        width_out = (width + 2 * padding - kernel) // stride + 1
        rows = []

        for i in range(height_out):
            columns = []
            for j in range(width_out):
                region = x[
                    :, :, i * stride : i * stride + kernel,
                    j * stride : j * stride + kernel,
                ]
                value = (
                    region.unsqueeze(1) * self.Weight.unsqueeze(0)
                ).sum(dim=(2, 3, 4))
                columns.append(value + self.Bias.unsqueeze(0))
            rows.append(torch.stack(columns, dim=2))

        return torch.stack(rows, dim=2)

    __call__ = forward
