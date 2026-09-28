import torch
from torch import nn


class SwiGLU(nn.Module):
    def __init__(
        self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        self.W1 = nn.Parameter(torch.Tensor(self.d_ff, self.d_model))
        self.W2 = nn.Parameter(torch.Tensor(self.d_model, self.d_ff))
        self.W3 = nn.Parameter(torch.Tensor(self.d_ff, self.d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process an input of shape (d_model) and returns a tensor of the same shape

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: tensor after SwiGLU activation.
        """
        input_silu = torch.matmul(x, self.W1.T)
        output_silu = input_silu * torch.sigmoid(input_silu)

        operand = torch.matmul(x, self.W3.T)

        return torch.matmul(output_silu * operand, self.W2.T)
