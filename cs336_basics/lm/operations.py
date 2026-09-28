import torch


def softmax(tensor: torch.Tensor, i: int) -> torch.Tensor:
    """Apply softmax to the i-th dimension of tensor

    Args:
        tensor (torch.Tensor): tensor
        i (int): i-th dimension

    Returns:
        torch.Tensor: original tensor with softmax applied to i-th dimension
    """
    maxmium = tensor.max(dim=i, keepdim=True).values
    e = torch.exp(tensor - maxmium)

    return e / e.sum(dim=i, keepdim=True)
