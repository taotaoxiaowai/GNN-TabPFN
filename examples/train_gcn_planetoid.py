from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

# Ensure local package is importable when running examples directly.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import degree as pyg_degree

from dl_research_kit.data import build_split_masks
from dl_research_kit.data import load_planetoid_data
from dl_research_kit.data import split_embedding_dataset
from dl_research_kit.data import standardize_with_train_stats
from dl_research_kit.data import SUPPORTED_PLANETOID_DATASETS
from dl_research_kit.data import validate_embedding_alignment
from dl_research_kit.inference import run_linear_head_on_embeddings
from dl_research_kit.inference import run_limix_classifier
from dl_research_kit.inference import run_tabpfn_bagging_classifier
from dl_research_kit.inference import run_tabpfn_classifier
from dl_research_kit.inference import run_tabpfn_ensemble_selection_classifier
from dl_research_kit.inference import run_tabpfn_ensemble_average_classifier
from dl_research_kit.models import FAGCNEncoder
from dl_research_kit.models import GATEncoder
from dl_research_kit.models import GCNEncoder
from dl_research_kit.models import GPRGNNEncoder
from dl_research_kit.models import LINKXEncoder
from dl_research_kit.models import SAGEEncoder
from dl_research_kit.training import collect_embedding_snapshot
from dl_research_kit.training import fuse_embeddings_sum_standardized
from dl_research_kit.training import load_tabpfn_frozen_adapter
from dl_research_kit.training import pretrain_encoder_with_frozen_tabpfn
from dl_research_kit.training import pretrain_linear_head_on_embeddings
from dl_research_kit.training import pretrain_gcn_encoder
from dl_research_kit.training import pretrain_weighted_fusion_joint
from dl_research_kit.training import pretrain_weighted_fusion
from dl_research_kit.utils import set_seed


CONCAT_PRE_AGG_EMBEDDING_DATASETS = {"chameleon", "squirrel", "Actor"}
HETEROPHILY_DATASETS = {"actor", "squirrel", "chameleon"}
HOMOPHILY_BACKBONES = ("gcn", "gat", "sage")
HETEROPHILY_BACKBONES = ("linkx", "fagcn", "gprgnn")
HETEROPHILY_BACKBONE_REPLACE_MAP = {
    "gcn": "linkx",
    "gat": "fagcn",
    "sage": "gprgnn",
}
ALL_BACKBONES = HOMOPHILY_BACKBONES + HETEROPHILY_BACKBONES


def _build_pre_aggregation_embedding(
    raw_x: torch.Tensor,
    train_mask: torch.Tensor,
    out_dim: int,
) -> torch.Tensor:
    """
    Build non-aggregated node embeddings with fixed dimensionality via
    train-split-fitted PCA projection.
    """
    x_cpu = raw_x.detach().cpu()
    train_mask_cpu = train_mask.detach().cpu()

    scaler = StandardScaler()
    x_train_np = scaler.fit_transform(x_cpu[train_mask_cpu].numpy())
    x_all_np = scaler.transform(x_cpu.numpy())
    x_all = torch.from_numpy(x_all_np).to(dtype=x_cpu.dtype)

    x_centered = x_all - x_all.mean(dim=0, keepdim=True)
    max_rank = min(x_centered.size(0), x_centered.size(1))
    pca_rank = min(out_dim, max_rank)
    if pca_rank <= 0:
        raise ValueError(f"Invalid PCA rank computed: {pca_rank}")

    _, _, v = torch.pca_lowrank(x_centered, q=pca_rank, center=False)
    pre_agg = x_centered @ v[:, :pca_rank]
    if pca_rank < out_dim:
        pad = torch.zeros(pre_agg.size(0), out_dim - pca_rank, dtype=pre_agg.dtype)
        pre_agg = torch.cat([pre_agg, pad], dim=-1)
    return pre_agg


def _build_graph_engineered_features(
    x: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """
    Build 3 node-level engineered features:
    1) Random-walk Laplacian response norm: ||(I - D^{-1}A)X||_2
    2) Unique neighbor count (excluding self-loop)
    3) Node degree (edge multiplicity-aware on row index)
    """
    x_cpu = x.detach().cpu().to(torch.float32)
    edge_index_cpu = edge_index.detach().cpu().to(torch.long)
    num_nodes = x_cpu.size(0)
    row, col = edge_index_cpu[0], edge_index_cpu[1]

    deg = pyg_degree(row, num_nodes=num_nodes, dtype=torch.float32)
    inv_deg = 1.0 / deg.clamp(min=1.0)

    # P = D^{-1}A (row-normalized transition); aggregate neighbor features.
    rw_agg = torch.zeros_like(x_cpu)
    rw_agg.index_add_(0, row, x_cpu[col] * inv_deg[row].unsqueeze(-1))
    lap_rw_resp_norm = (x_cpu - rw_agg).norm(p=2, dim=1, keepdim=True)

    non_self = row != col
    row_ns = row[non_self]
    col_ns = col[non_self]
    if row_ns.numel() == 0:
        neighbor_count = torch.zeros((num_nodes, 1), dtype=torch.float32)
    else:
        unique_adj = torch.sparse_coo_tensor(
            indices=torch.stack([row_ns, col_ns], dim=0),
            values=torch.ones(row_ns.numel(), dtype=torch.float32),
            size=(num_nodes, num_nodes),
        ).coalesce()
        neighbor_count = torch.zeros(num_nodes, dtype=torch.float32)
        neighbor_count.index_add_(
            0,
            unique_adj.indices()[0],
            torch.ones(unique_adj.indices().size(1), dtype=torch.float32),
        )
        neighbor_count = neighbor_count.unsqueeze(-1)

    degree_col = deg.unsqueeze(-1)
    return torch.cat([lap_rw_resp_norm, neighbor_count, degree_col], dim=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain graph encoder, then run downstream prediction on node embeddings"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Cora",
        choices=list(SUPPORTED_PLANETOID_DATASETS),
    )
    parser.add_argument("--root", type=str, default="./data")
    parser.add_argument(
        "--base-model",
        type=str,
        default="gcn",
        choices=list(ALL_BACKBONES),
        help="Graph aggregation backbone to generate node embeddings.",
    )
    parser.add_argument(
        "--encoder-fusion",
        type=str,
        default="none",
        choices=["none", "sum", "weighted"],
        help="Encoder-level fusion: none=single backbone; sum/weighted use dataset-specific backbone set.",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--gat-heads", type=int, default=8)
    parser.add_argument("--linkx-edge-layers", type=int, default=1)
    parser.add_argument("--linkx-node-layers", type=int, default=1)
    parser.add_argument("--fagcn-eps", type=float, default=0.1)
    parser.add_argument("--gpr-alpha", type=float, default=0.1)
    parser.add_argument("--inspect-layer", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-source", type=str, default="random", choices=["mask", "random"])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Validation ratio split from the original training set for linear pretraining.",
    )
    parser.add_argument(
        "--train-test-ratio",
        type=float,
        default=5.0,
        help="Only used when --split-source random. Example: 5 means train:test ~= 5:1.",
    )
    parser.add_argument("--pretrain-epochs", type=int, default=200)
    parser.add_argument("--pretrain-lr", type=float, default=1e-2)
    parser.add_argument("--pretrain-weight-decay", type=float, default=5e-4)
    parser.add_argument("--pretrain-log-every", type=int, default=20)
    parser.add_argument(
        "--encoder-pretrain-head",
        type=str,
        default="linear",
        choices=["linear", "tabpfn-frozen-forward"],
        help="Encoder pretraining objective head. tabpfn-frozen-forward requires a differentiable TabPFN adapter.",
    )
    parser.add_argument(
        "--tabpfn-frozen-adapter",
        type=str,
        default="",
        help="Required when --encoder-pretrain-head tabpfn-frozen-forward. Format: module.path:factory",
    )
    parser.add_argument(
        "--tabpfn-frozen-model-path",
        type=str,
        default="",
        help="Optional model/checkpoint path passed to frozen TabPFN adapter factory.",
    )
    parser.add_argument(
        "--tabpfn-frozen-config-path",
        type=str,
        default="",
        help="Optional config path passed to frozen TabPFN adapter factory.",
    )
    parser.add_argument("--fusion-epochs", type=int, default=200)
    parser.add_argument("--fusion-lr", type=float, default=1e-2)
    parser.add_argument("--fusion-weight-decay", type=float, default=0.0)
    parser.add_argument("--fusion-log-every", type=int, default=20)
    parser.add_argument(
        "--weighted-fusion-train-strategy",
        type=str,
        default="two-stage",
        choices=["two-stage", "joint"],
        help="Only used when --encoder-fusion weighted. two-stage=pretrain encoders then train w; joint=train encoders and w together.",
    )
    parser.add_argument(
        "--prediction-head",
        type=str,
        default="tabpfn",
        choices=["tabpfn", "linear", "limix", "tabpfn-ensemble-selection", "tabpfn-ensemble-average"],
        help="Downstream predictor. 'linear' reuses the linear head trained in GNN pretraining.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Global device for data and GNN models. Use cpu to force CPU-only mode.",
    )
    parser.add_argument("--tabpfn-device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--tabpfn-bagging",
        action="store_true",
        help="Enable bagging over TabPFN contexts with bootstrap resampling.",
    )
    parser.add_argument(
        "--tabpfn-bagging-m",
        type=int,
        default=8,
        help="Number of bootstrap context bags for TabPFN.",
    )
    parser.add_argument(
        "--tabpfn-bagging-context-size",
        type=int,
        default=0,
        help="Bootstrap sample size per bag. 0 means full training context size.",
    )
    parser.add_argument(
        "--tabpfn-frozen-subset-size",
        type=int,
        default=0,
        help="Subsample size for context/query during frozen TabPFN encoder pretraining. 0 means full training set.",
    )
    parser.add_argument(
        "--tabpfn-frozen-context-ratio",
        type=float,
        default=0.75,
        help="Context/query ratio for frozen TabPFN encoder pretraining. 0.75 means 3:1 context:query split.",
    )
    parser.add_argument(
        "--tabpfn-bagging-aggregation",
        type=str,
        default="average",
        choices=["average", "vote"],
        help="Bagging result aggregation: average probabilities or majority vote.",
    )
    parser.add_argument(
        "--tabpfn-bagging-n-jobs",
        type=int,
        default=1,
        help="Parallel workers for TabPFN bagging.",
    )
    parser.add_argument(
        "--tabpfn-bagging-feature-drop-rate",
        type=float,
        default=0.0,
        help="Optional random feature-column drop rate per bag. 0.0 disables feature masking.",
    )
    parser.add_argument(
        "--tabpfn-ens-sources",
        type=str,
        default="current",
        choices=["current", "all-backbones"],
        help="Source tables for ensemble branch: current prediction table or all available per-backbone/fused tables.",
    )
    parser.add_argument(
        "--tabpfn-ens-val-size",
        type=float,
        default=0.2,
        help="Validation ratio split from train context for ensemble selection.",
    )
    parser.add_argument(
        "--tabpfn-ens-candidates-per-table",
        type=int,
        default=8,
        help="How many row+column subsampled TabPFN candidates to build per table source.",
    )
    parser.add_argument(
        "--tabpfn-ens-context-size",
        type=int,
        default=0,
        help="Bootstrap context size for each ensemble-selection candidate. 0 means fit-split full size.",
    )
    parser.add_argument(
        "--tabpfn-ens-colsample-min-rate",
        type=float,
        default=0.4,
        help="Minimum feature-column sampling rate per candidate.",
    )
    parser.add_argument(
        "--tabpfn-ens-colsample-max-rate",
        type=float,
        default=1.0,
        help="Maximum feature-column sampling rate per candidate.",
    )
    parser.add_argument(
        "--tabpfn-ens-max-selected",
        type=int,
        default=32,
        help="Maximum number of greedy-selected candidates (with replacement).",
    )
    parser.add_argument(
        "--tabpfn-ens-n-jobs",
        type=int,
        default=1,
        help="Parallel workers for ensemble-selection candidate inference.",
    )
    parser.add_argument(
        "--limix-model-path",
        type=str,
        default="",
        help="Optional LimiX model/checkpoint path.",
    )
    parser.add_argument(
        "--limix-config-path",
        type=str,
        default="",
        help="Optional LimiX inference config path.",
    )
    parser.add_argument(
        "--feature-normalization",
        type=str,
        default="standardize",
        choices=["none", "standardize"],
        help="Normalization for node embeddings before downstream classifier.",
    )
    parser.add_argument(
        "--embedding-branch",
        type=str,
        default="branch2",
        choices=["branch1", "branch2"],
        help="branch1: graph embedding (+ optional PCA pre-agg); branch2: branch1 + 3 engineered graph features.",
    )
    parser.add_argument(
        "--pre-agg-pca-dim",
        type=int,
        default=-1,
        help="Extra PCA pre-aggregation feature dims to concatenate before prediction. -1=auto(64 on Actor/squirrel/chameleon else 0), 0=off.",
    )
    parser.add_argument("--save-prefix", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    dataset_name_lower = args.dataset.lower()
    is_heterophily_dataset = dataset_name_lower in HETEROPHILY_DATASETS

    if is_heterophily_dataset and args.base_model in HETEROPHILY_BACKBONE_REPLACE_MAP:
        replaced = HETEROPHILY_BACKBONE_REPLACE_MAP[args.base_model]
        print(
            f"[INFO] Dataset '{args.dataset}' is heterophilous. "
            f"Replacing base backbone '{args.base_model}' -> '{replaced}'."
        )
        args.base_model = replaced

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    else:
        device = args.device
    data = load_planetoid_data(args.dataset, root=args.root).to(device)
    train_mask, test_mask = build_split_masks(
        data=data,
        split_source=args.split_source,
        test_size=args.test_size,
        train_test_ratio=args.train_test_ratio,
        seed=args.seed,
    )
    linear_train_mask = train_mask.clone()
    linear_val_mask = torch.zeros_like(train_mask, dtype=torch.bool)
    if args.encoder_pretrain_head == "linear":
        if not (0.0 < args.val_size < 1.0):
            raise ValueError(f"--val-size must be in (0, 1), got {args.val_size}")
        train_idx = torch.where(train_mask)[0].cpu().numpy()
        if train_idx.size > 1:
            y_train = data.y[train_mask].detach().cpu().numpy()
            try:
                train_sub_idx, val_idx = train_test_split(
                    train_idx,
                    test_size=args.val_size,
                    random_state=args.seed,
                    stratify=y_train,
                )
            except ValueError:
                train_sub_idx, val_idx = train_test_split(
                    train_idx,
                    test_size=args.val_size,
                    random_state=args.seed,
                    stratify=None,
                )
            linear_train_mask = torch.zeros_like(train_mask, dtype=torch.bool)
            linear_val_mask = torch.zeros_like(train_mask, dtype=torch.bool)
            linear_train_mask[torch.from_numpy(train_sub_idx)] = True
            linear_val_mask[torch.from_numpy(val_idx)] = True
        print(
            "Linear pretrain split: "
            f"train={int(linear_train_mask.sum().item())}, "
            f"val={int(linear_val_mask.sum().item())}, "
            f"test={int(test_mask.sum().item())}"
        )
    print(
        "Split setting: "
        f"source={args.split_source}, requested_train_test_ratio={args.train_test_ratio:.4f}"
    )

    num_classes = int(data.y.max().item()) + 1

    linear_head = None
    fusion_weights = None
    source_embeddings: Dict[str, torch.Tensor] = {}
    labels = data.y.detach().cpu()
    frozen_tabpfn_adapter = None
    if args.encoder_pretrain_head == "tabpfn-frozen-forward":
        frozen_tabpfn_adapter = load_tabpfn_frozen_adapter(
            adapter_spec=args.tabpfn_frozen_adapter,
            model_path=args.tabpfn_frozen_model_path or None,
            config_path=args.tabpfn_frozen_config_path or None,
        )
        print(f"Encoder pretrain head: {args.encoder_pretrain_head}")

    def _pretrain_encoder(model: torch.nn.Module):
        if args.encoder_pretrain_head == "linear":
            return pretrain_gcn_encoder(
                model=model,
                data=data,
                train_mask=linear_train_mask,
                val_mask=linear_val_mask,
                num_classes=num_classes,
                epochs=args.pretrain_epochs,
                lr=args.pretrain_lr,
                weight_decay=args.pretrain_weight_decay,
                log_every=args.pretrain_log_every,
            )
        print(
            "Starting frozen TabPFN encoder pretraining: "
            f"epochs={args.pretrain_epochs}, subset_size={args.tabpfn_frozen_subset_size}, "
            f"context_ratio={args.tabpfn_frozen_context_ratio:.2f}, device={device}"
        )
        pretrain_encoder_with_frozen_tabpfn(
            model=model,
            data=data,
            train_mask=train_mask,
            tabpfn_module=frozen_tabpfn_adapter,
            epochs=args.pretrain_epochs,
            lr=args.pretrain_lr,
            weight_decay=args.pretrain_weight_decay,
            log_every=args.pretrain_log_every,
            subset_size=args.tabpfn_frozen_subset_size,
            context_ratio=args.tabpfn_frozen_context_ratio,
        )
        return None

    def _build_encoder(name: str):
        if name == "gcn":
            return GCNEncoder(
                input_dim=data.num_features,
                hidden_dim=args.hidden_dim,
                embedding_dim=args.embedding_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
            ).to(device)
        if name == "gat":
            return GATEncoder(
                input_dim=data.num_features,
                hidden_dim=args.hidden_dim,
                embedding_dim=args.embedding_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                heads=args.gat_heads,
            ).to(device)
        if name == "sage":
            return SAGEEncoder(
                input_dim=data.num_features,
                hidden_dim=args.hidden_dim,
                embedding_dim=args.embedding_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
            ).to(device)
        if name == "linkx":
            return LINKXEncoder(
                input_dim=data.num_features,
                hidden_dim=args.hidden_dim,
                embedding_dim=args.embedding_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                num_nodes=data.num_nodes,
                num_edge_layers=args.linkx_edge_layers,
                num_node_layers=args.linkx_node_layers,
            ).to(device)
        if name == "fagcn":
            return FAGCNEncoder(
                input_dim=data.num_features,
                hidden_dim=args.hidden_dim,
                embedding_dim=args.embedding_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                eps=args.fagcn_eps,
            ).to(device)
        if name == "gprgnn":
            return GPRGNNEncoder(
                input_dim=data.num_features,
                hidden_dim=args.hidden_dim,
                embedding_dim=args.embedding_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                alpha=args.gpr_alpha,
            ).to(device)
        raise ValueError(f"Unknown encoder backbone: {name}")

    if args.encoder_fusion == "none":
        model = _build_encoder(args.base_model)
        print(
            "Backbone setting: "
            f"base_model={args.base_model}, hidden_dim={args.hidden_dim}, "
            f"embedding_dim={args.embedding_dim}, num_layers={args.num_layers}, "
            f"dropout={args.dropout}, gat_heads={args.gat_heads}, "
            f"linkx_edge_layers={args.linkx_edge_layers}, linkx_node_layers={args.linkx_node_layers}, "
            f"fagcn_eps={args.fagcn_eps}, gpr_alpha={args.gpr_alpha}"
        )

        linear_head = _pretrain_encoder(model)

        model.eval()
        with torch.no_grad():
            final_embedding = model(data.x, data.edge_index).detach().cpu()
        source_embeddings[args.base_model] = final_embedding

        snapshot = collect_embedding_snapshot(model, data, layer_index=args.inspect_layer)
        print(
            f"[{args.dataset}] embedding built after pretraining "
            f"(shape={tuple(final_embedding.shape)})"
        )
        print(
            "  selected-layer embedding "
            f"(layer={args.inspect_layer + 1}/{args.num_layers}) "
            f"shape={snapshot['shape']} mean={snapshot['mean']:.4f} "
            f"std={snapshot['std']:.4f} min={snapshot['min']:.4f} "
            f"max={snapshot['max']:.4f}"
        )
    else:
        fusion_model_names = (
            list(HETEROPHILY_BACKBONES)
            if is_heterophily_dataset
            else list(HOMOPHILY_BACKBONES)
        )
        print(
            "Fusion setting: "
            f"encoder_fusion={args.encoder_fusion}, models={fusion_model_names}, "
            f"hidden_dim={args.hidden_dim}, embedding_dim={args.embedding_dim}, "
            f"num_layers={args.num_layers}, dropout={args.dropout}, gat_heads={args.gat_heads}, "
            f"weighted_fusion_train_strategy={args.weighted_fusion_train_strategy}"
        )
        encoders: Dict[str, torch.nn.Module] = {
            name: _build_encoder(name) for name in fusion_model_names
        }
        per_model_embeddings: List[torch.Tensor] = []
        per_model_embedding_map: Dict[str, torch.Tensor] = {}
        need_individual_encoder_pretrain = (
            args.encoder_fusion == "sum"
            or (args.encoder_fusion == "weighted" and args.weighted_fusion_train_strategy == "two-stage")
        )

        if need_individual_encoder_pretrain:
            for name, model in encoders.items():
                print(f"Pretraining encoder: {name}")
                _ = _pretrain_encoder(model)
                model.eval()
                with torch.no_grad():
                    emb = model(data.x, data.edge_index).detach()
                per_model_embeddings.append(emb)
                per_model_embedding_map[name] = emb.detach().cpu()
                print(f"  {name} embedding shape: {tuple(emb.shape)}")

        if args.encoder_fusion == "sum":
            fused = fuse_embeddings_sum_standardized(per_model_embeddings, train_mask.to(device))
            final_embedding = fused.detach().cpu()
            source_embeddings.update(per_model_embedding_map)
            source_embeddings["fused_sum"] = final_embedding
            fusion_weights = torch.tensor([1.0 / 3, 1.0 / 3, 1.0 / 3], dtype=torch.float32)
            if args.prediction_head == "linear":
                linear_head = pretrain_linear_head_on_embeddings(
                    node_embeddings=fused,
                    labels=labels.to(device),
                    train_mask=linear_train_mask.to(device),
                    val_mask=linear_val_mask.to(device),
                    num_classes=num_classes,
                    epochs=args.fusion_epochs,
                    lr=args.fusion_lr,
                    weight_decay=args.fusion_weight_decay,
                    log_every=args.fusion_log_every,
                )
        else:
            if args.weighted_fusion_train_strategy == "joint":
                fusion_result = pretrain_weighted_fusion_joint(
                    encoders=encoders,
                    data=data,
                    train_mask=train_mask.to(device),
                    num_classes=num_classes,
                    epochs=args.fusion_epochs,
                    lr=args.fusion_lr,
                    weight_decay=args.fusion_weight_decay,
                    log_every=args.fusion_log_every,
                )
                for name, model in encoders.items():
                    model.eval()
                    with torch.no_grad():
                        source_embeddings[name] = model(data.x, data.edge_index).detach().cpu()
            else:
                fusion_result = pretrain_weighted_fusion(
                    embeddings=per_model_embeddings,
                    labels=labels.to(device),
                    train_mask=train_mask.to(device),
                    num_classes=num_classes,
                    epochs=args.fusion_epochs,
                    lr=args.fusion_lr,
                    weight_decay=args.fusion_weight_decay,
                    log_every=args.fusion_log_every,
                )
                source_embeddings.update(per_model_embedding_map)
            final_embedding = fusion_result["embedding"].detach().cpu()
            source_embeddings["fused_weighted"] = final_embedding
            fusion_weights = fusion_result["weights"]
            if args.prediction_head == "linear":
                linear_head = fusion_result["head"]

        snapshot = {
            "shape": tuple(final_embedding.shape),
            "mean": float(final_embedding.mean().item()),
            "std": float(final_embedding.std().item()),
            "max": float(final_embedding.max().item()),
            "min": float(final_embedding.min().item()),
            "embedding": final_embedding,
        }
        print(
            f"[{args.dataset}] fused embedding built "
            f"(shape={tuple(final_embedding.shape)})"
        )
        print(
            "  fused embedding stats "
            f"shape={snapshot['shape']} mean={snapshot['mean']:.4f} "
            f"std={snapshot['std']:.4f} min={snapshot['min']:.4f} "
            f"max={snapshot['max']:.4f}"
        )
        if fusion_weights is not None:
            print(
                f"  fusion weights {fusion_model_names} = "
                f"{[round(float(v), 4) for v in fusion_weights.tolist()]}"
            )

    if args.pre_agg_pca_dim < -1:
        raise ValueError(f"--pre-agg-pca-dim must be >= -1, got {args.pre_agg_pca_dim}")
    pre_agg_pca_dim = (
        64
        if args.pre_agg_pca_dim == -1 and args.dataset in CONCAT_PRE_AGG_EMBEDDING_DATASETS
        else max(args.pre_agg_pca_dim, 0)
    )
    add_graph_engineered_features = args.embedding_branch == "branch2"
    graph_engineered_features = None
    if add_graph_engineered_features:
        graph_engineered_features = _build_graph_engineered_features(
            x=data.x,
            edge_index=data.edge_index,
        )
    pre_agg_cache: Dict[int, torch.Tensor] = {}

    def _augment_prediction_embedding(base_embedding: torch.Tensor) -> torch.Tensor:
        local_prediction_embedding = base_embedding
        if pre_agg_pca_dim > 0:
            if pre_agg_pca_dim not in pre_agg_cache:
                pre_agg_cache[pre_agg_pca_dim] = _build_pre_aggregation_embedding(
                    raw_x=data.x,
                    train_mask=train_mask,
                    out_dim=pre_agg_pca_dim,
                )
            local_prediction_embedding = torch.cat(
                [local_prediction_embedding, pre_agg_cache[pre_agg_pca_dim]],
                dim=-1,
            )
        if add_graph_engineered_features:
            local_prediction_embedding = torch.cat(
                [local_prediction_embedding, graph_engineered_features],
                dim=-1,
            )
        return local_prediction_embedding

    prediction_embedding = _augment_prediction_embedding(final_embedding)
    print(
        "Prediction embedding branch setting: "
        f"branch={args.embedding_branch}, pre_agg_pca_dim={pre_agg_pca_dim}, "
        f"add_graph_engineered_features={add_graph_engineered_features}"
    )
    if pre_agg_pca_dim > 0:
        print(
            f"[{args.dataset}] prediction uses concatenated embedding: "
            f"gnn_dim={final_embedding.size(1)} + pre_agg_dim={pre_agg_cache[pre_agg_pca_dim].size(1)} "
            f"-> total_dim={prediction_embedding.size(1)}"
        )
    if add_graph_engineered_features:
        print(
            "Appended graph engineered features to prediction embedding: "
            "[lap_rw_resp_norm, neighbor_count, degree], "
            f"new_total_dim={prediction_embedding.size(1)}"
        )
    else:
        print(f"Graph engineered features disabled. total_dim={prediction_embedding.size(1)}")

    if args.prediction_head in ("tabpfn", "limix", "tabpfn-ensemble-selection", "tabpfn-ensemble-average"):
        validate_embedding_alignment(
            node_embeddings=prediction_embedding,
            labels=labels,
            train_mask=train_mask,
            test_mask=test_mask,
        )
        x_train, y_train, x_test, y_test = split_embedding_dataset(
            node_embeddings=prediction_embedding,
            labels=labels,
            train_mask=train_mask,
            test_mask=test_mask,
        )
        if args.feature_normalization == "standardize":
            x_train, x_test = standardize_with_train_stats(x_train=x_train, x_test=x_test)
            print("Applied feature normalization: standardize (fit on train only).")
        else:
            print("Applied feature normalization: none.")

        print(
            "Split complete: "
            f"x_train={tuple(x_train.shape)}, x_test={tuple(x_test.shape)}, "
            f"num_classes={int(torch.unique(y_train).numel())}, "
            f"actual_train_test_ratio={x_train.shape[0] / max(1, x_test.shape[0]):.4f}"
        )

        if args.prediction_head == "tabpfn":
            try:
                if args.tabpfn_bagging:
                    context_size = (
                        None if args.tabpfn_bagging_context_size <= 0 else args.tabpfn_bagging_context_size
                    )
                    print(
                        "TabPFN bagging enabled: "
                        f"m={args.tabpfn_bagging_m}, context_size={context_size or 'full'}, "
                        f"aggregation={args.tabpfn_bagging_aggregation}, n_jobs={args.tabpfn_bagging_n_jobs}, "
                        f"feature_drop_rate={args.tabpfn_bagging_feature_drop_rate}"
                    )
                    result = run_tabpfn_bagging_classifier(
                        x_train=x_train,
                        y_train=y_train,
                        x_test=x_test,
                        y_test=y_test,
                        device=args.tabpfn_device,
                        num_bags=args.tabpfn_bagging_m,
                        context_size=context_size,
                        aggregation=args.tabpfn_bagging_aggregation,
                        random_seed=args.seed,
                        n_jobs=args.tabpfn_bagging_n_jobs,
                        feature_drop_rate=args.tabpfn_bagging_feature_drop_rate,
                    )
                else:
                    result = run_tabpfn_classifier(
                        x_train=x_train,
                        y_train=y_train,
                        x_test=x_test,
                        y_test=y_test,
                        device=args.tabpfn_device,
                    )
            except ImportError as e:
                print("TabPFN is not installed. Install with: pip install tabpfn")
                raise e

            print(f"TabPFN test accuracy: {result['accuracy']:.4f}")
            print("TabPFN classification report:")
            print(result["report"])
        elif args.prediction_head == "tabpfn-ensemble-selection":
            try:
                if args.tabpfn_ens_sources == "all-backbones":
                    candidate_source_keys = sorted(source_embeddings.keys())
                else:
                    candidate_source_keys = ["current"]

                table_train_list: List[Dict[str, object]] = []
                table_test_list: List[Dict[str, object]] = []
                for source_key in candidate_source_keys:
                    if source_key == "current":
                        src_emb = prediction_embedding
                    else:
                        src_emb = _augment_prediction_embedding(source_embeddings[source_key])
                    validate_embedding_alignment(
                        node_embeddings=src_emb,
                        labels=labels,
                        train_mask=train_mask,
                        test_mask=test_mask,
                    )
                    s_x_train, _, s_x_test, _ = split_embedding_dataset(
                        node_embeddings=src_emb,
                        labels=labels,
                        train_mask=train_mask,
                        test_mask=test_mask,
                    )
                    if args.feature_normalization == "standardize":
                        s_x_train, s_x_test = standardize_with_train_stats(
                            x_train=s_x_train,
                            x_test=s_x_test,
                        )
                    table_train_list.append(
                        {
                            "name": source_key,
                            "x_train": s_x_train.numpy(),
                        }
                    )
                    table_test_list.append(
                        {
                            "name": source_key,
                            "x_test": s_x_test.numpy(),
                        }
                    )

                ens_context_size = None if args.tabpfn_ens_context_size <= 0 else args.tabpfn_ens_context_size
                print(
                    "TabPFN ensemble-selection enabled: "
                    f"sources={candidate_source_keys}, "
                    f"val_size={args.tabpfn_ens_val_size}, "
                    f"candidates_per_table={args.tabpfn_ens_candidates_per_table}, "
                    f"context_size={ens_context_size or 'fit-full'}, "
                    f"colsample=[{args.tabpfn_ens_colsample_min_rate}, {args.tabpfn_ens_colsample_max_rate}], "
                    f"max_selected={args.tabpfn_ens_max_selected}, "
                    f"n_jobs={args.tabpfn_ens_n_jobs}"
                )
                result = run_tabpfn_ensemble_selection_classifier(
                    tables=table_train_list,
                    y_train=y_train,
                    x_test_tables=table_test_list,
                    y_test=y_test,
                    device=args.tabpfn_device,
                    val_size=args.tabpfn_ens_val_size,
                    candidates_per_table=args.tabpfn_ens_candidates_per_table,
                    context_size=ens_context_size,
                    colsample_min_rate=args.tabpfn_ens_colsample_min_rate,
                    colsample_max_rate=args.tabpfn_ens_colsample_max_rate,
                    max_selected=args.tabpfn_ens_max_selected,
                    n_jobs=args.tabpfn_ens_n_jobs,
                    random_seed=args.seed,
                )
            except ImportError as e:
                print("TabPFN is not installed. Install with: pip install tabpfn")
                raise e

            print(f"TabPFN ensemble-selection test accuracy: {result['accuracy']:.4f}")
            print(
                "TabPFN ensemble-selection summary: "
                f"fit_size={result['ensemble_fit_size']}, "
                f"val_size={result['ensemble_val_size']}, "
                f"num_candidates={result['ensemble_num_candidates']}, "
                f"num_selected={result['ensemble_num_selected']}, "
                f"table_weights={result['ensemble_table_weights']}"
            )
            print("TabPFN ensemble-selection classification report:")
            print(result["report"])
        elif args.prediction_head == "tabpfn-ensemble-average":
            try:
                if args.tabpfn_ens_sources == "all-backbones":
                    candidate_source_keys = sorted(source_embeddings.keys())
                else:
                    candidate_source_keys = ["current"]

                table_train_list: List[Dict[str, object]] = []
                table_test_list: List[Dict[str, object]] = []
                for source_key in candidate_source_keys:
                    if source_key == "current":
                        src_emb = prediction_embedding
                    else:
                        src_emb = _augment_prediction_embedding(source_embeddings[source_key])
                    validate_embedding_alignment(
                        node_embeddings=src_emb,
                        labels=labels,
                        train_mask=train_mask,
                        test_mask=test_mask,
                    )
                    s_x_train, _, s_x_test, _ = split_embedding_dataset(
                        node_embeddings=src_emb,
                        labels=labels,
                        train_mask=train_mask,
                        test_mask=test_mask,
                    )
                    if args.feature_normalization == "standardize":
                        s_x_train, s_x_test = standardize_with_train_stats(
                            x_train=s_x_train,
                            x_test=s_x_test,
                        )
                    table_train_list.append(
                        {
                            "name": source_key,
                            "x_train": s_x_train.numpy(),
                        }
                    )
                    table_test_list.append(
                        {
                            "name": source_key,
                            "x_test": s_x_test.numpy(),
                        }
                    )

                ens_context_size = None if args.tabpfn_ens_context_size <= 0 else args.tabpfn_ens_context_size
                print(
                    "TabPFN ensemble-average enabled: "
                    f"sources={candidate_source_keys}, "
                    f"candidates_per_table={args.tabpfn_ens_candidates_per_table}, "
                    f"context_size={ens_context_size or 'fit-full'}, "
                    f"colsample=[{args.tabpfn_ens_colsample_min_rate}, {args.tabpfn_ens_colsample_max_rate}], "
                    f"n_jobs={args.tabpfn_ens_n_jobs}"
                )
                result = run_tabpfn_ensemble_average_classifier(
                    tables=table_train_list,
                    y_train=y_train,
                    x_test_tables=table_test_list,
                    y_test=y_test,
                    device=args.tabpfn_device,
                    candidates_per_table=args.tabpfn_ens_candidates_per_table,
                    context_size=ens_context_size,
                    colsample_min_rate=args.tabpfn_ens_colsample_min_rate,
                    colsample_max_rate=args.tabpfn_ens_colsample_max_rate,
                    n_jobs=args.tabpfn_ens_n_jobs,
                    random_seed=args.seed,
                )
            except ImportError as e:
                print("TabPFN is not installed. Install with: pip install tabpfn")
                raise e

            print(f"TabPFN ensemble-average test accuracy: {result['accuracy']:.4f}")
            print(
                "TabPFN ensemble-average summary: "
                f"num_candidates={result['ensemble_num_candidates']}, "
                f"table_weights={result['ensemble_table_weights']}"
            )
            print("TabPFN ensemble-average classification report:")
            print(result["report"])
        else:
            try:
                result = run_limix_classifier(
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    y_test=y_test,
                    model_path=args.limix_model_path or None,
                    config_path=args.limix_config_path or None,
                )
            except ImportError as e:
                print(
                    "LimiX classifier is not installed or not importable. "
                    "Please install LimiX and verify its import path."
                )
                raise e

            print(f"LimiX test accuracy: {result['accuracy']:.4f}")
            print("LimiX classification report:")
            print(result["report"])
    else:
        linear_head = pretrain_linear_head_on_embeddings(
            node_embeddings=prediction_embedding.to(device),
            labels=labels.to(device),
            train_mask=linear_train_mask.to(device),
            val_mask=linear_val_mask.to(device),
            num_classes=num_classes,
            epochs=args.fusion_epochs,
            lr=args.fusion_lr,
            weight_decay=args.fusion_weight_decay,
            log_every=args.fusion_log_every,
        )
        result = run_linear_head_on_embeddings(
            head=linear_head,
            node_embeddings=prediction_embedding,
            labels=labels,
            test_mask=test_mask,
        )
        print(f"Linear-head test accuracy: {result['accuracy']:.4f}")
        print("Linear-head classification report:")
        print(result["report"])

    if args.save_prefix:
        torch.save(final_embedding, f"{args.save_prefix}_final_embedding.pt")
        torch.save(snapshot["embedding"], f"{args.save_prefix}_inspect_layer_embedding.pt")
        torch.save(torch.as_tensor(result["y_pred"]), f"{args.save_prefix}_{args.prediction_head}_pred.pt")
        if fusion_weights is not None:
            torch.save(fusion_weights, f"{args.save_prefix}_fusion_weights.pt")
        print(f"Saved outputs with prefix: {args.save_prefix}")


if __name__ == "__main__":
    main()
