from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SimpleConv


class H2GCNEncoder(nn.Module):
    """
    H2GCN-style encoder using PyG neighborhood aggregations.

    This implementation follows the key H2 idea by explicitly mixing
    first-order and second-order neighborhood signals in each layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.dropout = dropout
        self.num_layers = num_layers
        self.output_dim = embedding_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.one_hop_conv = SimpleConv(aggr="mean")
        self.two_hop_conv = SimpleConv(aggr="mean")
        self.mix_projs = nn.ModuleList(
            [nn.Linear(2 * hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.output_proj = nn.Linear(hidden_dim, embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_intermediate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, List[torch.Tensor]]:
        x = F.dropout(x, p=self.dropout, training=self.training)
        h = F.relu(self.input_proj(x))
        h = F.dropout(h, p=self.dropout, training=self.training)

        hidden_states: List[torch.Tensor] = []
        for layer_idx in range(self.num_layers):
            h_1 = self.one_hop_conv(h, edge_index)
            h_2 = self.two_hop_conv(h_1, edge_index)
            h = torch.cat([h_1, h_2], dim=-1)
            h = self.mix_projs[layer_idx](h)
            if layer_idx != self.num_layers - 1:
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

