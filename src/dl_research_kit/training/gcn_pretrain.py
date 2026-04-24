from __future__ import annotations

import copy
from typing import Dict, Protocol

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data

class GraphEncoderProtocol(Protocol):
    output_dim: int

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_intermediate: bool = False,
    ) -> torch.Tensor: ...

    def get_intermediate_embeddings(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> list[torch.Tensor]: ...

    def parameters(self): ...

    def train(self, mode: bool = True): ...

    def eval(self): ...


def pretrain_gcn_encoder(
    model: GraphEncoderProtocol,
    data: Data,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor | None,
    num_classes: int,
    epochs: int = 200,
    lr: float = 1e-2,
    weight_decay: float = 5e-4,
    log_every: int = 20,
) -> nn.Linear:
    """
    Supervised pretraining for the GCN encoder with a temporary linear head.
    The returned head can be reused for direct linear-head evaluation.
    """
    device = next(model.parameters()).device
    data = data.to(device)
    head = nn.Linear(model.output_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(head.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    train_mask = train_mask.to(device)
    has_val = val_mask is not None and int(val_mask.sum().item()) > 0
    if val_mask is None:
        val_mask = torch.zeros_like(train_mask, dtype=torch.bool)
    val_mask = val_mask.to(device)
    best_metric = -1.0
    best_epoch = 1
    best_model_state = copy.deepcopy(model.state_dict())
    best_head_state = copy.deepcopy(head.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()
        head.train()
        optimizer.zero_grad()

        z = model(data.x, data.edge_index)
        logits = head(z)
        loss = F.cross_entropy(logits[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        head.eval()
        with torch.no_grad():
            eval_logits = head(model(data.x, data.edge_index))
            pred_train = eval_logits[train_mask].argmax(dim=-1)
            train_acc = (pred_train == data.y[train_mask]).float().mean().item()
            val_acc = None
            if has_val:
                pred_val = eval_logits[val_mask].argmax(dim=-1)
                val_acc = (pred_val == data.y[val_mask]).float().mean().item()
                selection_metric = val_acc
            else:
                selection_metric = train_acc
            if selection_metric > best_metric:
                best_metric = selection_metric
                best_epoch = epoch
                best_model_state = copy.deepcopy(model.state_dict())
                best_head_state = copy.deepcopy(head.state_dict())

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            if val_acc is None:
                print(f"[Pretrain] epoch={epoch:03d} loss={loss.item():.4f} train_acc={train_acc:.4f}")
            else:
                print(
                    f"[Pretrain] epoch={epoch:03d} loss={loss.item():.4f} "
                    f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
                )

    model.load_state_dict(best_model_state)
    head.load_state_dict(best_head_state)
    print(
        f"[Pretrain] selected epoch={best_epoch:03d} "
        f"{'val_acc' if has_val else 'train_acc'}={best_metric:.4f}"
    )

    return head


@torch.no_grad()
def collect_embedding_snapshot(
    model: GraphEncoderProtocol,
    data: Data,
    layer_index: int,
) -> Dict[str, torch.Tensor | tuple[int, ...] | float]:
    hidden_states = model.get_intermediate_embeddings(data.x, data.edge_index)
    if layer_index < 0 or layer_index >= len(hidden_states):
        raise ValueError(
            f"layer_index must be in [0, {len(hidden_states) - 1}], got {layer_index}"
        )
    emb = hidden_states[layer_index].detach().cpu()
    return {
        "shape": tuple(emb.shape),
        "mean": float(emb.mean().item()),
        "std": float(emb.std().item()),
        "max": float(emb.max().item()),
        "min": float(emb.min().item()),
        "embedding": emb,
    }
