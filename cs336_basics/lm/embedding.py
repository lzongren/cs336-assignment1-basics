import torch
from jaxtyping import Int
from torch import nn


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Args:
            num_embeddings (int): size of the vocab
            embedding_dim (int): dimension of the embedding vectors, i.e., d_{model}
            device (torch.device | None): Device to store the parameters on
            dtype (torch.dtype | None, optional): Data type of the parameters
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        tensor = torch.Tensor(self.num_embeddings, self.embedding_dim)
        self.embedding = nn.Parameter(tensor)

    def forward(self, token_ids: Int[torch.Tensor, " ..."]) -> torch.Tensor:
        """Lookup the embedding vectors for the given token IDs.

        Args:
            token_ids (Int[torch.Tensor, &quot; ...&quot;]): token IDs

        Returns:
            torch.Tensor:
        """
        return self.embedding[token_ids]
