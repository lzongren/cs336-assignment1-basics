import math

import torch
from torch import nn


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        tensor = torch.Tensor(self.out_features, self.in_features)
        std = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-3 * std, b=3 * std, generator=None)
        self.W = nn.Parameter(tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bio,oy->biy", x, self.W.T)
