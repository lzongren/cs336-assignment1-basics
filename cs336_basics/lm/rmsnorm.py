import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.eps = eps
        tensor = torch.ones(self.d_model)
        self.weights = nn.Parameter(tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process an input of shape (batch, sequence_length, d_model) and returns a tensor of the same shape

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: tensor after RMSNorm.
        """

        # upcast to prevent overflow when squaring
        in_type = x.dtype
        x = x.to(torch.float32)

        # step 2
        rms = torch.sqrt(torch.mean(torch.square(x), dim=-1, keepdim=True) + self.eps)
        rms_norm = x / rms * self.weights

        return rms_norm.to(in_type)
