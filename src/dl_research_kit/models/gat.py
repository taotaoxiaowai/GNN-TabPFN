from __future__ import annotations

from typing import List

import torch
from torch import nn
from torch_geometric.nn import GATConv


class GATEncoder(nn.Module):
    """GAT encoder: attention-based aggregation only, outputs node embeddings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        heads: int = 8,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if heads < 1:
            raise ValueError("heads must be >= 1")

        self.dropout = dropout
        self.convs = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(
                GATConv(input_dim, embedding_dim, heads=1, concat=False, dropout=dropout)
            )
        else:
            self.convs.append(
                GATConv(input_dim, hidden_dim, heads=heads, concat=True, dropout=dropout)
            )
            hidden_input_dim = hidden_dim * heads
            for _ in range(num_layers - 2):
                self.convs.append(
                    GATConv(
                        hidden_input_dim,
                        hidden_dim,
                        heads=heads,
                        concat=True,
                        dropout=dropout,
                    )
                )
            self.convs.append(
                GATConv(
                    hidden_input_dim,
                    embedding_dim,
                    heads=1,
                    concat=False,
                    dropout=dropout,
                )
            )

        self.output_dim = embedding_dim

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_intermediate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, List[torch.Tensor]]:
        hidden_states: List[torch.Tensor] = []

        for layer_idx, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            hidden_states.append(x)

            if layer_idx != len(self.convs) - 1:
                x = torch.nn.functional.elu(x)
                x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        if return_intermediate:
            return x, hidden_states
        return x

    @torch.no_grad()
    def get_intermediate_embeddings(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> List[torch.Tensor]:
        self.eval()
        _, hidden_states = self.forward(x, edge_index, return_intermediate=True)
        return hidden_states


class GATClassifier(GATEncoder):
    """Backward-compatible alias mirroring GCNClassifier behavior."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        heads: int = 8,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            heads=heads,
        )
