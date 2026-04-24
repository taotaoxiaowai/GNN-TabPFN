from __future__ import annotations

from typing import Dict

import torch
from sklearn.metrics import accuracy_score, classification_report
from torch import nn
from torch_geometric.data import Data


@torch.no_grad()
def run_linear_head_classifier(
    model: nn.Module,
    head: nn.Linear,
    data: Data,
    test_mask: torch.Tensor,
) -> Dict[str, object]:
    device = next(model.parameters()).device
    data = data.to(device)
    test_mask = test_mask.to(device)

    model.eval()
    head.eval()

    z = model(data.x, data.edge_index)
    logits = head(z)
    y_pred = logits[test_mask].argmax(dim=-1).detach().cpu()
    y_true = data.y[test_mask].detach().cpu()

    acc = float(accuracy_score(y_true.numpy(), y_pred.numpy()))
    report = classification_report(y_true.numpy(), y_pred.numpy(), digits=4)
    return {"accuracy": acc, "y_pred": y_pred, "report": report}


@torch.no_grad()
def run_linear_head_on_embeddings(
    head: nn.Linear,
    node_embeddings: torch.Tensor,
    labels: torch.Tensor,
    test_mask: torch.Tensor,
) -> Dict[str, object]:
    device = next(head.parameters()).device
    emb = node_embeddings.to(device)
    labels = labels.to(device)
    test_mask = test_mask.to(device)

    head.eval()
    logits = head(emb)
    y_pred = logits[test_mask].argmax(dim=-1).detach().cpu()
    y_true = labels[test_mask].detach().cpu()

    acc = float(accuracy_score(y_true.numpy(), y_pred.numpy()))
    report = classification_report(y_true.numpy(), y_pred.numpy(), digits=4)
    return {"accuracy": acc, "y_pred": y_pred, "report": report}
