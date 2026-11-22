"""Max-pooling implemented with tensor operations and autograd."""

import torch


class MaxPool:
    def __init__(self, KernelSize: int = 2, Stride: int = 2) -> None:
        self.K = KernelSize
        self.S = Stride

    def parameters(self):
        return []

    def zero_grad(self) -> None:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("MaxPool expects (batch, channels, height, width)")

        batch, channels, height, width = x.shape
        height_out = (height - self.K) // self.S + 1
        width_out = (width - self.K) // self.S + 1
        rows = []

        for i in range(height_out):
            columns = []
            for j in range(width_out):
                region = x[
                    :, :, i * self.S : i * self.S + self.K,
                    j * self.S : j * self.S + self.K,
                ]
                columns.append(region.reshape(batch, channels, -1).max(dim=2).values)
            rows.append(torch.stack(columns, dim=2))

        return torch.stack(rows, dim=2)

    __call__ = forward
