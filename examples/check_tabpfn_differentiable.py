from __future__ import annotations

import argparse
import importlib
import sys

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether a TabPFN adapter supports differentiable forward/backward."
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="examples.tabpfn_adapter:build_frozen_tabpfn",
        help="Adapter factory in format module.path:factory_name",
    )
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--config-path", type=str, default="")
    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--ctx-size", type=int, default=32)
    parser.add_argument("--query-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    return parser.parse_args()


def load_factory(spec: str):
    if ":" not in spec:
        raise ValueError(f"Invalid adapter spec: {spec}. Expected module.path:factory")
    module_name, factory_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name)
    if not callable(factory):
        raise TypeError(f"Factory is not callable: {spec}")
    return factory


def main() -> None:
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    # 1) Adapter load
    try:
        factory = load_factory(args.adapter)
    except Exception as e:
        print(f"[FAIL] Adapter load failed: {e}")
        sys.exit(1)

    try:
        adapter = factory(
            model_path=args.model_path or None,
            config_path=args.config_path or None,
        )
    except TypeError:
        adapter = factory()
    except Exception as e:
        print(f"[FAIL] Adapter factory execution failed: {e}")
        sys.exit(1)

    if not isinstance(adapter, torch.nn.Module):
        print(f"[FAIL] Adapter factory must return torch.nn.Module, got {type(adapter).__name__}")
        sys.exit(1)
    adapter = adapter.to(device)

    # 2) Freeze adapter params
    trainable_before = sum(int(p.requires_grad) for p in adapter.parameters())
    for p in adapter.parameters():
        p.requires_grad = False
    trainable_after = sum(int(p.requires_grad) for p in adapter.parameters())

    # 3) Build fake encoder output as differentiable input
    x_context = torch.randn(args.ctx_size, args.input_dim, device=device, requires_grad=True)
    y_context = torch.randint(0, args.num_classes, (args.ctx_size,), device=device)
    x_query = torch.randn(args.query_size, args.input_dim, device=device, requires_grad=True)
    y_query = torch.randint(0, args.num_classes, (args.query_size,), device=device)

    # 4) Differentiability check: forward + CE + backward
    try:
        logits = adapter(x_context, y_context, x_query)
    except Exception as e:
        print(f"[FAIL] Adapter forward failed: {e}")
        sys.exit(1)

    if not isinstance(logits, torch.Tensor):
        print(f"[FAIL] Forward output must be torch.Tensor, got {type(logits).__name__}")
        sys.exit(1)
    if logits.ndim != 2:
        print(f"[FAIL] Logits must be 2D [N_query, C], got shape={tuple(logits.shape)}")
        sys.exit(1)
    if logits.shape[0] != args.query_size:
        print(
            f"[FAIL] Logits first dim mismatch. expected {args.query_size}, got {logits.shape[0]}"
        )
        sys.exit(1)

    try:
        loss = torch.nn.functional.cross_entropy(logits, y_query)
        loss.backward()
    except Exception as e:
        print(f"[FAIL] Backward failed: {e}")
        sys.exit(1)

    xq_grad_ok = x_query.grad is not None and torch.isfinite(x_query.grad).all().item()
    xc_grad_ok = x_context.grad is not None and torch.isfinite(x_context.grad).all().item()

    print("[PASS] Differentiability check completed.")
    print(f"  trainable_params_before_freeze={trainable_before}")
    print(f"  trainable_params_after_freeze={trainable_after}")
    print(f"  logits_shape={tuple(logits.shape)}")
    print(f"  loss={float(loss.item()):.6f}")
    print(f"  grad_to_x_query={bool(xq_grad_ok)}")
    print(f"  grad_to_x_context={bool(xc_grad_ok)}")

    if not xq_grad_ok:
        print("[FAIL] Gradient did not flow to x_query.")
        sys.exit(2)

    print("[OK] Adapter is usable for frozen-forward encoder training.")


if __name__ == "__main__":
    main()
