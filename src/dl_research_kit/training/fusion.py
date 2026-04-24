from __future__ import annotations

import copy
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data


def fuse_embeddings_sum_standardized(
    embeddings: List[torch.Tensor],
    train_mask: torch.Tensor,
) -> torch.Tensor:
    if len(embeddings) == 0:
        raise ValueError("embeddings must not be empty.")
    fused = torch.stack(embeddings, dim=0).sum(dim=0)
    return _standardize_by_train_stats(fused, train_mask)


def pretrain_weighted_fusion(
    embeddings: List[torch.Tensor],
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    num_classes: int,
    epochs: int = 200,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    log_every: int = 20,
) -> Dict[str, object]:
    if len(embeddings) == 0:
        raise ValueError("embeddings must not be empty.")

    device = embeddings[0].device
    emb_stack = torch.stack(embeddings, dim=0).to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)

    raw_weights = nn.Parameter(torch.zeros(emb_stack.size(0), device=device))
    head = nn.Linear(emb_stack.size(-1), num_classes).to(device)
    optimizer = torch.optim.Adam(
        [raw_weights] + list(head.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    for epoch in range(1, epochs + 1):
        head.train()
        optimizer.zero_grad()

        w = torch.softmax(raw_weights, dim=0).view(-1, 1, 1)
        fused = (w * emb_stack).sum(dim=0)
        logits = head(fused)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            with torch.no_grad():
                pred = logits[train_mask].argmax(dim=-1)
                acc = (pred == labels[train_mask]).float().mean().item()
                w_now = torch.softmax(raw_weights, dim=0).detach().cpu().tolist()
            print(
                f"[FusionPretrain] epoch={epoch:03d} loss={loss.item():.4f} "
                f"train_acc={acc:.4f} weights={[round(v, 4) for v in w_now]}"
            )

    with torch.no_grad():
        final_w = torch.softmax(raw_weights, dim=0).detach()
        final_fused = (final_w.view(-1, 1, 1) * emb_stack).sum(dim=0).detach()

    return {
        "embedding": final_fused,
        "weights": final_w.detach().cpu(),
        "head": head,
    }


def pretrain_weighted_fusion_joint(
    encoders: Dict[str, nn.Module],
    data: Data,
    train_mask: torch.Tensor,
    num_classes: int,
    epochs: int = 200,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    log_every: int = 20,
) -> Dict[str, object]:
    if len(encoders) == 0:
        raise ValueError("encoders must not be empty.")

    model_names = list(encoders.keys())
    first_model = next(iter(encoders.values()))
    device = next(first_model.parameters()).device
    data = data.to(device)
    train_mask = train_mask.to(device)

    for model in encoders.values():
        model.train()

    sample_out_dim = int(first_model.output_dim)
    raw_weights = nn.Parameter(torch.zeros(len(model_names), device=device))
    head = nn.Linear(sample_out_dim, num_classes).to(device)

    param_groups: List[torch.nn.Parameter] = [raw_weights] + list(head.parameters())
    for model in encoders.values():
        param_groups.extend(list(model.parameters()))
    optimizer = torch.optim.Adam(param_groups, lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        for model in encoders.values():
            model.train()
        head.train()
        optimizer.zero_grad()

        emb_list = [encoders[name](data.x, data.edge_index) for name in model_names]
        emb_stack = torch.stack(emb_list, dim=0)
        w = torch.softmax(raw_weights, dim=0).view(-1, 1, 1)
        fused = (w * emb_stack).sum(dim=0)
        logits = head(fused)
        loss = F.cross_entropy(logits[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            with torch.no_grad():
                pred = logits[train_mask].argmax(dim=-1)
                acc = (pred == data.y[train_mask]).float().mean().item()
                w_now = torch.softmax(raw_weights, dim=0).detach().cpu().tolist()
            print(
                f"[FusionJoint] epoch={epoch:03d} loss={loss.item():.4f} "
                f"train_acc={acc:.4f} weights={[round(v, 4) for v in w_now]}"
            )

    with torch.no_grad():
        for model in encoders.values():
            model.eval()
        head.eval()
        emb_list = [encoders[name](data.x, data.edge_index) for name in model_names]
        emb_stack = torch.stack(emb_list, dim=0)
        final_w = torch.softmax(raw_weights, dim=0).detach()
        final_fused = (final_w.view(-1, 1, 1) * emb_stack).sum(dim=0).detach()

    return {
        "embedding": final_fused,
        "weights": final_w.detach().cpu(),
        "head": head,
    }


def pretrain_linear_head_on_embeddings(
    node_embeddings: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor | None,
    num_classes: int,
    epochs: int = 200,
    lr: float = 1e-2,
    weight_decay: float = 5e-4,
    log_every: int = 20,
) -> nn.Linear:
    device = node_embeddings.device
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    has_val = val_mask is not None and int(val_mask.sum().item()) > 0
    if val_mask is None:
        val_mask = torch.zeros_like(train_mask, dtype=torch.bool)
    val_mask = val_mask.to(device)
    head = nn.Linear(node_embeddings.size(-1), num_classes).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    best_metric = -1.0
    best_epoch = 1
    best_head_state = copy.deepcopy(head.state_dict())

    for epoch in range(1, epochs + 1):
        head.train()
        optimizer.zero_grad()
        logits = head(node_embeddings)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        head.eval()
        with torch.no_grad():
            eval_logits = head(node_embeddings)
            pred_train = eval_logits[train_mask].argmax(dim=-1)
            train_acc = (pred_train == labels[train_mask]).float().mean().item()
            val_acc = None
            if has_val:
                pred_val = eval_logits[val_mask].argmax(dim=-1)
                val_acc = (pred_val == labels[val_mask]).float().mean().item()
                selection_metric = val_acc
            else:
                selection_metric = train_acc
            if selection_metric > best_metric:
                best_metric = selection_metric
                best_epoch = epoch
                best_head_state = copy.deepcopy(head.state_dict())

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            if val_acc is None:
                print(f"[LinearHead] epoch={epoch:03d} loss={loss.item():.4f} train_acc={train_acc:.4f}")
            else:
                print(
                    f"[LinearHead] epoch={epoch:03d} loss={loss.item():.4f} "
                    f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
                )

    head.load_state_dict(best_head_state)
    print(
        f"[LinearHead] selected epoch={best_epoch:03d} "
        f"{'val_acc' if has_val else 'train_acc'}={best_metric:.4f}"
    )

    return head


def _standardize_by_train_stats(embedding: torch.Tensor, train_mask: torch.Tensor) -> torch.Tensor:
    train_mask = train_mask.to(embedding.device)
    train_x = embedding[train_mask]
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True, unbiased=False)
    std = torch.where(std < 1e-12, torch.ones_like(std), std)
    return (embedding - mean) / std
