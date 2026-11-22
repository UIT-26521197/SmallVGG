"""Softmax cross-entropy implemented with tensors and autograd."""

import torch


class SoftmaxEntropyLoss:
    def __init__(self) -> None:
        self.x = None
        self.yTrue = None
        self.Out = None
        self._loss = None

    def forward(self, x: torch.Tensor, yTrue: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("SoftmaxEntropyLoss expects (batch, classes)")

        yTrue = yTrue.to(dtype=torch.long).reshape(-1)
        shifted = x - x.max(dim=1, keepdim=True).values
        exp_x = torch.exp(shifted)
        self.Out = exp_x / exp_x.sum(dim=1, keepdim=True)

        batch = x.shape[0]
        indices = torch.tensor(
            list(range(batch)),
            dtype=torch.long,
            device=x.device,
        )
        correct_probability = self.Out[indices, yTrue]
        self.x = x
        self.yTrue = yTrue
        self._loss = -torch.log(correct_probability).mean()
        return self._loss

    def backward(self) -> torch.Tensor:
        """Return d(loss)/d(logits) through PyTorch autograd."""
        if self._loss is None or self.x is None:
            raise RuntimeError("Call forward before backward")
        return torch.autograd.grad(self._loss, self.x, retain_graph=True)[0]

    __call__ = forward
