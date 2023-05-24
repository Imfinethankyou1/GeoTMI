from math import pi as PI

import torch
from torch import Tensor
from torch.nn import Module, Parameter

from .envelope import Envelope


class BesselBasisLayer(Module):
    def __init__(
        self, num_radial: int, cutoff: float = 5.0, envelope_exponent: int = 5
    ):
        super().__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        self.freq = Parameter(Tensor(num_radial))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            torch.arange(1, self.freq.numel() + 1, out=self.freq).mul_(PI)
        self.freq.requires_grad_()

    def forward(self, dist: Tensor) -> Tensor:
        dist = dist.unsqueeze(-1) / self.cutoff
        return self.envelope(dist) * (self.freq * dist).sin()
