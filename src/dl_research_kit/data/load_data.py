from __future__ import annotations

from typing import Tuple

import torch
import torch_geometric.transforms as T
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import HeterophilousGraphDataset
from torch_geometric.datasets import Actor
from torch_geometric.datasets import WikipediaNetwork

SUPPORTED_PLANETOID_DATASETS = ("Cora", "CiteSeer", "PubMed", "Computers", "Photo","chameleon","squirrel","Actor")
Planetoid_Dataset = ("Cora", "CiteSeer", "PubMed")
Amazon_Dataset = ("Computers", "Photo")

def load_planetoid_data(name: str, root: str = "./data") -> Data:
    if name not in SUPPORTED_PLANETOID_DATASETS:
        supported = ", ".join(SUPPORTED_PLANETOID_DATASETS)
        raise ValueError(f"Unsupported dataset '{name}'. Supported: {supported}.")
    if name in Planetoid_Dataset:
        dataset = Planetoid(root=f"{root}/Planetoid", name=name, transform=T.NormalizeFeatures())
    elif name in Amazon_Dataset:
        dataset = Amazon(root=f"{root}/Amazon", name=name, transform=T.NormalizeFeatures())
    elif name == "Actor":
        dataset = Actor(root=f"{root}/Actor", transform=T.NormalizeFeatures())
    else:
        dataset = WikipediaNetwork(root=f"{root}/Wikipedia", name=name, transform=T.NormalizeFeatures())

    return dataset[0]


def build_split_masks(
    data: Data,
    split_source: str = "mask",
    test_size: float = 0.2,
    train_test_ratio: float = 5.0,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build train/test masks for node-level split.
    - mask: use dataset-provided train/test masks.
    - random: stratified random split over all nodes (supports explicit train:test ratio).
    """
    if split_source == "mask":
        if not hasattr(data, "train_mask") or not hasattr(data, "test_mask"):
            raise ValueError("Dataset does not provide train_mask/test_mask.")
        return data.train_mask.detach().cpu(), data.test_mask.detach().cpu()

    if split_source == "random":
        if train_test_ratio <= 0:
            raise ValueError(f"train_test_ratio must be > 0, got {train_test_ratio}")
        resolved_test_size = 1.0 / (train_test_ratio + 1.0)
        y = data.y.cpu().numpy()
        node_idx = list(range(data.num_nodes))
        train_idx, test_idx = train_test_split(
            node_idx,
            test_size=resolved_test_size if train_test_ratio > 0 else test_size,
            random_state=seed,
            stratify=y,
        )
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        train_mask[train_idx] = True
        test_mask[test_idx] = True
        return train_mask, test_mask

    raise ValueError(f"Unsupported split_source: {split_source}")


def split_embedding_dataset(
    node_embeddings: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_train = node_embeddings[train_mask]
    y_train = labels[train_mask]
    x_test = node_embeddings[test_mask]
    y_test = labels[test_mask]
    return x_train, y_train, x_test, y_test


def validate_embedding_alignment(
    node_embeddings: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
) -> None:
    """Sanity-check index alignment and split validity before downstream training."""
    if node_embeddings.ndim != 2:
        raise ValueError(
            f"node_embeddings must be 2D [num_nodes, dim], got shape={tuple(node_embeddings.shape)}"
        )
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D [num_nodes], got shape={tuple(labels.shape)}")
    if node_embeddings.size(0) != labels.size(0):
        raise ValueError(
            "Mismatch between embedding rows and labels: "
            f"{node_embeddings.size(0)} vs {labels.size(0)}"
        )
    if train_mask.dtype != torch.bool or test_mask.dtype != torch.bool:
        raise ValueError("train_mask/test_mask must be torch.bool tensors.")
    if train_mask.numel() != labels.numel() or test_mask.numel() != labels.numel():
        raise ValueError(
            "Mask length must match number of nodes: "
            f"train={train_mask.numel()}, test={test_mask.numel()}, labels={labels.numel()}"
        )
    overlap = int((train_mask & test_mask).sum().item())
    if overlap > 0:
        raise ValueError(f"train/test masks overlap on {overlap} nodes.")
    if int(train_mask.sum().item()) == 0 or int(test_mask.sum().item()) == 0:
        raise ValueError("train/test split is empty; cannot train/evaluate.")
    if torch.isnan(node_embeddings).any() or torch.isinf(node_embeddings).any():
        raise ValueError("node_embeddings contains NaN/Inf values.")


def standardize_with_train_stats(
    x_train: torch.Tensor,
    x_test: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fit scaler on train embeddings and apply to train/test.
    This prevents test leakage while improving downstream classifier conditioning.
    """
    scaler = StandardScaler()
    x_train_np = scaler.fit_transform(x_train.numpy())
    x_test_np = scaler.transform(x_test.numpy())
    return (
        torch.from_numpy(x_train_np).to(dtype=x_train.dtype),
        torch.from_numpy(x_test_np).to(dtype=x_test.dtype),
    )
