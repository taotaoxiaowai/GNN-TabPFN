from __future__ import annotations

from typing import List

import torch
from torch import nn
from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):
    """GCN encoder: aggregation only, outputs node embeddings (no classifier head)."""

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
        self.convs = nn.ModuleList()

        # num_layers controls the number of graph aggregation updates.
        # 1 layer: input -> embedding
        # n layers: input -> hidden*(n-1) -> embedding
        if num_layers == 1:
            self.convs.append(GCNConv(input_dim, embedding_dim))
        else:
            self.convs.append(GCNConv(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.convs.append(GCNConv(hidden_dim, embedding_dim))
        self.output_dim = embedding_dim

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_intermediate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, List[torch.Tensor]]:
        # hidden_states[i] is the representation after layer i+1 aggregation.
        hidden_states: List[torch.Tensor] = []

        for layer_idx, conv in enumerate(self.convs):
            # One GCNConv is one message-passing aggregation update.
            x = conv(x, edge_index)
            hidden_states.append(x)

            if layer_idx != len(self.convs) - 1:
                # Keep the last layer linear; use activation/dropout on hidden layers.
                x = torch.relu(x)
                x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        if return_intermediate:
            return x, hidden_states
        return x

    @torch.no_grad()
    def get_intermediate_embeddings(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> List[torch.Tensor]:
        """Return per-layer aggregated embeddings for inspection."""
        self.eval()
        _, hidden_states = self.forward(x, edge_index, return_intermediate=True)
        return hidden_states


class GCNClassifier(GCNEncoder):
    """
    Backward-compatible alias keeping the old class name.
    Current behavior only outputs embeddings; downstream classifier is user-defined.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
