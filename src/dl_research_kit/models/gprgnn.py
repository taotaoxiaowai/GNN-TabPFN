from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class _GPRProp(MessagePassing):
    """Generalized PageRank propagation with learnable hop coefficients."""

    def __init__(self, k: int, alpha: float = 0.1) -> None:
        super().__init__(aggr="add")
        if k < 1:
            raise ValueError("k must be >= 1 for GPR propagation.")
        self.k = k
        self.alpha = alpha

        gamma = torch.zeros(self.k + 1, dtype=torch.float32)
        for i in range(self.k + 1):
            gamma[i] = alpha * ((1.0 - alpha) ** i)
        gamma[-1] = (1.0 - alpha) ** self.k
        self.gamma = nn.Parameter(gamma)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_intermediate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, List[torch.Tensor]]:
        edge_index, edge_weight = gcn_norm(
            edge_index=edge_index,
            edge_weight=None,
            num_nodes=x.size(0),
            dtype=x.dtype,
            add_self_loops=True,
        )

        hidden_states: List[torch.Tensor] = []
        h = x
        out = self.gamma[0] * h
        hidden_states.append(out)
        for k in range(1, self.k + 1):
            h = self.propagate(edge_index, x=h, edge_weight=edge_weight)
            out = out + self.gamma[k] * h
            hidden_states.append(out)

        if return_intermediate:
            return out, hidden_states
        return out

    def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        return edge_weight.view(-1, 1) * x_j


class GPRGNNEncoder(nn.Module):
    """GPR-GNN encoder built with PyG message passing primitives."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        num_layers: int = 10,
        dropout: float = 0.5,
        alpha: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.dropout = dropout
        self.output_dim = embedding_dim
        self.k = num_layers
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, embedding_dim)
        self.prop = _GPRProp(k=num_layers, alpha=alpha)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_intermediate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, List[torch.Tensor]]:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        if return_intermediate:
            out, hidden_states = self.prop(x, edge_index, return_intermediate=True)
            return out, hidden_states
        return self.prop(x, edge_index)

    @torch.no_grad()
    def get_intermediate_embeddings(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> List[torch.Tensor]:
        self.eval()
        _, hidden_states = self.forward(x, edge_index, return_intermediate=True)
        return hidden_states

