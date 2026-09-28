import torch
from torch import nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None) -> None:
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        i = torch.arange(self.d_k // 2, device=device, dtype=torch.float32)
        exponent = -2 * i / self.d_k
        theta_i = torch.pow(self.theta, exponent)

        p = torch.arange(self.max_seq_len, device=device, dtype=torch.float32)
        theta_pi = torch.outer(p, theta_i)

        self.cos_values = torch.cos(theta_pi)
        self.sin_values = torch.sin(theta_pi)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """Process an input tensor of shape (..., seq_len, d_k)

        Args:
            x (torch.Tensor): can be any arbitrary number of batch dimensions.
            token_positions (torch.Tensor): shape of (..., seq_len)

        Returns:
            torch.Tensor: tensor of the same input shape.
        """

        cos = self.cos_values[token_positions]
        sin = self.sin_values[token_positions]

        x_pairs = x.reshape(*x.shape[:-1], self.d_k // 2, 2)
        x_even = x_pairs[..., 0]
        x_odd = x_pairs[..., 1]

        y_even = x_even * cos - x_odd * sin
        y_odd = x_even * sin + x_odd * cos

        y_pairs = torch.stack([y_even, y_odd], dim=-1)
        return y_pairs.reshape_as(x)
