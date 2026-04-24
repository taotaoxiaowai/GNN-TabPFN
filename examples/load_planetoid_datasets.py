from __future__ import annotations

from typing import Dict, Tuple

from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T


def load_planetoid_datasets(root: str = "./data") -> Dict[str, Tuple[Planetoid, Data]]:
    """Load Cora, CiteSeer and PubMed from PyG Planetoid datasets."""
    transform = T.NormalizeFeatures()

    cora = Planetoid(root=f"{root}/Planetoid", name="Cora", transform=transform)
    citeseer = Planetoid(root=f"{root}/Planetoid", name="CiteSeer", transform=transform)
    pubmed = Planetoid(root=f"{root}/Planetoid", name="PubMed", transform=transform)

    # Planetoid datasets each contain a single graph.
    data_cora = cora[0]
    data_citeseer = citeseer[0]
    data_pubmed = pubmed[0]

    return {
        "Cora": (cora, data_cora),
        "CiteSeer": (citeseer, data_citeseer),
        "PubMed": (pubmed, data_pubmed),
    }


def run() -> None:
    datasets = load_planetoid_datasets("./data")

    for name, (dataset, data) in datasets.items():
        print(f"\n{name}")
        print(f"  Num graphs: {len(dataset)}")
        print(f"  Num features: {dataset.num_features}")
        print(f"  Num classes: {dataset.num_classes}")
        print(f"  Num nodes: {data.num_nodes}")
        print(f"  Num edges: {data.num_edges}")
        print(
            "  Train/Val/Test: "
            f"{int(data.train_mask.sum())}/"
            f"{int(data.val_mask.sum())}/"
            f"{int(data.test_mask.sum())}"
        )


if __name__ == "__main__":
    run()
