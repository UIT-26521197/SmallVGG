"""Manual stochastic gradient descent without torch.optim."""

import torch


class SGD:
    def __init__(self, learningRate: float = 0.01) -> None:
        self.learningRate = learningRate

    def update(self, layer) -> None:
        with torch.no_grad():
            for parameter in layer.parameters():
                if parameter.grad is not None:
                    parameter -= self.learningRate * parameter.grad
        layer.zero_grad()

    step = update
