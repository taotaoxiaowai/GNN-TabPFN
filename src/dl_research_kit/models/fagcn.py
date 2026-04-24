from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import FAConv


class FAGCNEncoder(nn.Module):
    """
    FAGCN-style encoder implemented with PyG FAConv layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        eps: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.dropout = dropout
        self.num_layers = num_layers
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList(
            [FAConv(hidden_dim, eps=eps, dropout=dropout) for _ in range(num_layers)]
        )
        self.output_proj = nn.Linear(hidden_dim, embedding_dim)
        self.output_dim = embedding_dim

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_intermediate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, List[torch.Tensor]]:
        x = F.dropout(x, p=self.dropout, training=self.training)
        h0 = F.relu(self.input_proj(x))
        h0 = F.dropout(h0, p=self.dropout, training=self.training)

        h = h0
        hidden_states: List[torch.Tensor] = []
        for conv in self.convs:
            h = conv(h, h0, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            hidden_states.append(self.output_proj(h))

        out = self.output_proj(h)
        if return_intermediate:
            return out, hidden_states
        return out

    @torch.no_grad()
    def get_intermediate_embeddings(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> List[torch.Tensor]:
        self.eval()
        _, hidden_states = self.forward(x, edge_index, return_intermediate=True)
        return hidden_states

