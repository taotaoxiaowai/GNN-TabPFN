from __future__ import annotations

from typing import List

import torch
from torch import nn
from torch_geometric.nn.models import LINKX


class LINKXEncoder(nn.Module):
    """LINKX encoder wrapper with a unified project interface."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        num_layers: int = 1,
        dropout: float = 0.5,
        num_nodes: int = 0,
        num_edge_layers: int = 1,
        num_node_layers: int = 1,
    ) -> None:
        super().__init__()
        if num_nodes <= 0:
            raise ValueError("num_nodes must be > 0 for LINKXEncoder.")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.output_dim = embedding_dim
        self.num_layers = num_layers
        self.model = LINKX(
            num_nodes=num_nodes,
            in_channels=input_dim,
            hidden_channels=hidden_dim,
            out_channels=embedding_dim,
            num_layers=num_layers,
            num_edge_layers=num_edge_layers,
            num_node_layers=num_node_layers,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_intermediate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, List[torch.Tensor]]:
        out = self.model(x, edge_index)
        if return_intermediate:
            return out, [out for _ in range(self.num_layers)]
        return out

    @torch.no_grad()
    def get_intermediate_embeddings(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> List[torch.Tensor]:
        self.eval()
        _, hidden_states = self.forward(x, edge_index, return_intermediate=True)
        return hidden_states

