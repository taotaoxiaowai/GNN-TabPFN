from __future__ import annotations

import importlib
import math
from contextlib import nullcontext
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data


def pretrain_encoder_with_frozen_tabpfn(
    model: nn.Module,
    data: Data,
    train_mask: torch.Tensor,
    tabpfn_module: nn.Module,
    epochs: int = 200,
    lr: float = 1e-2,
    weight_decay: float = 5e-4,
    log_every: int = 20,
    subset_size: int = 0,
    context_ratio: float = 0.75,
) -> None:
    """
    Train graph encoder while freezing TabPFN module parameters.
    tabpfn_module must implement:
      logits = tabpfn_module(x_context, y_context, x_query)
    where logits has shape [num_query, num_classes].

    context_ratio controls the relative size of the frozen TabPFN context set.
    For example, 0.75 means a 3:1 context:query split.
    """
    device = next(model.parameters()).device
    data = data.to(device)
    train_mask = train_mask.to(device)
    tabpfn_module = tabpfn_module.to(device)
    tabpfn_module.eval()

    for p in tabpfn_module.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        z = model(data.x, data.edge_index)
        x_train = z[train_mask]
        y_train = data.y[train_mask]
        n_train = x_train.shape[0]

        if subset_size and subset_size > 0:
            sample_size = min(int(subset_size), n_train)
        else:
            sample_size = n_train

        if sample_size < 2:
            x_context = x_train
            y_context = y_train
            x_query = x_train
            y_query = y_train
        else:
            perm = torch.randperm(n_train, device=device)[:sample_size]
            query_size = max(1, min(sample_size - 1, math.ceil(sample_size * (1.0 - context_ratio))))
            context_size = sample_size - query_size
            if context_size < 1:
                context_size = sample_size - 1
                query_size = 1

            context_idx = perm[:context_size]
            query_idx = perm[context_size:context_size + query_size]

            x_context = x_train[context_idx]
            y_context = y_train[context_idx]
            x_query = x_train[query_idx]
            y_query = y_train[query_idx]

        amp_ctx = (
            torch.amp.autocast("cuda", dtype=torch.float16)
            if amp_enabled
            else nullcontext()
        )
        try:
            with amp_ctx:
                logits = tabpfn_module(x_context, y_context, x_query)
                loss = F.cross_entropy(logits, y_query)
        except RuntimeError as exc:
            if "CUDA_R_16F" in str(exc) or "cublasGemmEx" in str(exc):
                # Fall back to float32 for TabPFN internals when mixed precision is unsupported.
                with nullcontext():
                    logits = tabpfn_module(x_context.float(), y_context, x_query.float())
                    loss = F.cross_entropy(logits, y_query)
                loss.backward()
                optimizer.step()
                scaler.update()
                scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
                continue
            raise

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                acc = (pred == y_query).float().mean().item()
            print(f"[FrozenTabPFN] epoch={epoch:03d} loss={loss.item():.4f} train_acc={acc:.4f}")


def load_tabpfn_frozen_adapter(
    adapter_spec: str,
    model_path: str | None = None,
    config_path: str | None = None,
) -> nn.Module:
    """
    Load a user-provided differentiable TabPFN adapter from:
      <python.module.path>:<factory_function_name>
    Factory signature recommendation:
      factory(model_path: str | None = None, config_path: str | None = None) -> nn.Module
    """
    if not adapter_spec or ":" not in adapter_spec:
        raise ValueError(
            "tabpfn frozen adapter is required for differentiable forward. "
            "Use --tabpfn-frozen-adapter <module.path>:<factory>."
        )

    module_name, factory_name = adapter_spec.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name)
    if not callable(factory):
        raise TypeError(f"Adapter factory is not callable: {adapter_spec}")

    built = _call_factory(factory, model_path=model_path, config_path=config_path)
    if not isinstance(built, nn.Module):
        raise TypeError(
            f"Adapter factory must return torch.nn.Module, got {type(built).__name__}."
        )
    return built


def _call_factory(
    factory: Callable[..., object],
    model_path: str | None,
    config_path: str | None,
) -> object:
    try:
        return factory(model_path=model_path, config_path=config_path)
    except TypeError:
        try:
            return factory(model_path=model_path)
        except TypeError:
            try:
                return factory(config_path=config_path)
            except TypeError:
                return factory()
