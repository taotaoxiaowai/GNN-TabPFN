from __future__ import annotations

import inspect
import warnings
from typing import Optional

import torch
from torch import nn


class FrozenTabPFNAdapter(nn.Module):
    """
    Differentiable adapter for TabPFN frozen-forward encoder pretraining.

    This adapter uses TabPFN's dedicated differentiable API when available:
      - TabPFNClassifier(..., differentiable_input=True)
      - fit_with_differentiable_input(x_context, y_context, x_query)

    If those APIs are unavailable, it falls back to the existing TabPFN
    fit/predict interface and keeps the training script compatible.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.config_path = config_path
        self.device = device
        self.core = self._build_or_load_tabpfn_core(model_path, config_path)

    def _build_or_load_tabpfn_core(
        self, model_path: Optional[str], config_path: Optional[str]
    ) -> object:
        try:
            from tabpfn import TabPFNClassifier
        except ImportError as exc:
            raise ImportError(
                "TabPFN is required for FrozenTabPFNAdapter. "
                "Install it into your environment before using this adapter."
            ) from exc

        classifier_kwargs = {
            "device": self.device,
            "ignore_pretraining_limits": True,
            # Use single ensemble member during frozen-forward training to reduce
            # GPU memory use. Final inference in the prediction head uses a
            # fresh TabPFN classifier with the default ensemble setting.
            "ensemble_members": 1,
        }

        signature = inspect.signature(TabPFNClassifier)
        supports_differentiable_input = "differentiable_input" in signature.parameters
        if supports_differentiable_input:
            classifier_kwargs["differentiable_input"] = True
            # Differentiable mode requires preprocessing-fit path; setting this
            # explicitly avoids repeated internal mode switching on CPU.
            if "fit_mode" in signature.parameters:
                classifier_kwargs["fit_mode"] = "fit_preprocessors"
        else:
            if "fit_mode" in signature.parameters:
                classifier_kwargs["fit_mode"] = "low_memory"
            if "memory_saving_mode" in signature.parameters:
                classifier_kwargs["memory_saving_mode"] = "auto"
        if "ensemble_members" not in signature.parameters:
            classifier_kwargs.pop("ensemble_members", None)

        try:
            return TabPFNClassifier(**classifier_kwargs)
        except TypeError:
            classifier_kwargs.pop("differentiable_input", None)
            return TabPFNClassifier(**classifier_kwargs)

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        if hasattr(self, "core") and hasattr(self.core, "to"):
            try:
                self.core.to(*args, **kwargs)
            except Exception:
                pass
        return self

    def forward(
        self,
        x_context: torch.Tensor,
        y_context: torch.Tensor,
        x_query: torch.Tensor,
    ) -> torch.Tensor:
        if hasattr(self.core, "fit_with_differentiable_input"):
            fit_fn = self.core.fit_with_differentiable_input
            fit_sig = inspect.signature(fit_fn)

            if len(fit_sig.parameters) == 2:
                maybe_self = fit_fn(x_context, y_context)
            else:
                maybe_self = fit_fn(x_context, y_context, x_query)

            if isinstance(maybe_self, torch.Tensor):
                logits = maybe_self
            else:
                # Directly invoke the executor-backed inference path to preserve
                # autograd, bypassing wrapper methods that detach tensors.
                if hasattr(self.core, "_raw_predict"):
                    logits = self.core._raw_predict(x_query, return_logits=True)
                elif hasattr(self.core, "forward"):
                    logits = self.core.forward(
                        x_query,
                        use_inference_mode=False,
                        return_logits=True,
                    )
                else:
                    raise RuntimeError(
                        "TabPFN differentiable API returned model object but no low-level inference path was found."
                    )

            if isinstance(logits, tuple):
                logits = logits[0]
            if not isinstance(logits, torch.Tensor):
                logits = torch.as_tensor(logits, device=x_query.device)
            
            if not logits.requires_grad:
                raise RuntimeError(
                    "TabPFN low-level inference produced detached logits. "
                    "Gradient flow was not preserved."
                )
            return logits

        warnings.warn(
            "TabPFN differentiable API is unavailable; falling back to non-differentiable "
            "fit/predict execution. Gradient flow will not be preserved.",
            UserWarning,
        )

        x_context_np = x_context.detach().cpu().numpy()
        y_context_np = y_context.detach().cpu().numpy()
        x_query_np = x_query.detach().cpu().numpy()

        self.core.fit(x_context_np, y_context_np)

        if hasattr(self.core, "predict_proba"):
            probs = self.core.predict_proba(x_query_np)
            return torch.from_numpy(probs).to(x_query.device)

        y_pred = self.core.predict(x_query_np)
        num_classes = int(y_context.max().item()) + 1
        logits = torch.full(
            (x_query.shape[0], num_classes), float("-inf"), device=x_query.device
        )
        logits[torch.arange(x_query.shape[0], device=x_query.device),
               torch.from_numpy(y_pred).to(x_query.device)] = 0.0
        return logits


def build_frozen_tabpfn(
    model_path: Optional[str] = None,
    config_path: Optional[str] = None,
) -> nn.Module:
    """
    Factory function consumed by:
      --tabpfn-frozen-adapter examples.tabpfn_adapter:build_frozen_tabpfn
    """
    return FrozenTabPFNAdapter(model_path=model_path, config_path=config_path)
