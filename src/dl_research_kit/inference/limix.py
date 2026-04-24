from __future__ import annotations

import inspect
from typing import Dict, Tuple

import torch
from sklearn.metrics import accuracy_score, classification_report


def _prepare_limix_input(
    x_train: torch.Tensor, y_train: torch.Tensor, x_test: torch.Tensor
) -> Tuple[object, object, object]:
    return x_train.numpy(), y_train.numpy(), x_test.numpy()


def _maybe_pass_known_kwargs(cls_or_fn, kwargs: dict[str, object]) -> dict[str, object]:
    try:
        sig = inspect.signature(cls_or_fn)
    except (TypeError, ValueError):
        return {}
    valid = {}
    for name, value in kwargs.items():
        if value is None:
            continue
        if name in sig.parameters:
            valid[name] = value
    return valid


def _create_limix_classifier(
    model_path: str | None = None,
    config_path: str | None = None,
):
    ctor_kwargs = {
        "model_path": model_path,
        "model_file": model_path,
        "checkpoint_path": model_path,
        "inference_config_path": config_path,
        "config_path": config_path,
    }

    import_errors: list[str] = []

    try:
        from inference.predictor import LimiXClassifier

        return LimiXClassifier(**_maybe_pass_known_kwargs(LimiXClassifier, ctor_kwargs))
    except Exception as e:  # pragma: no cover - depends on external package layout.
        import_errors.append(f"inference.predictor.LimiXClassifier: {e}")

    try:
        from limix import LimiXClassifier

        return LimiXClassifier(**_maybe_pass_known_kwargs(LimiXClassifier, ctor_kwargs))
    except Exception as e:  # pragma: no cover - depends on external package layout.
        import_errors.append(f"limix.LimiXClassifier: {e}")

    raise ImportError(
        "Cannot import a LimiX classifier. "
        "Tried inference.predictor.LimiXClassifier and limix.LimiXClassifier. "
        f"Details: {' | '.join(import_errors)}"
    )


def run_limix_classifier(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    model_path: str | None = None,
    config_path: str | None = None,
) -> Dict[str, object]:
    x_train_np, y_train_np, x_test_np = _prepare_limix_input(x_train, y_train, x_test)
    clf = _create_limix_classifier(model_path=model_path, config_path=config_path)

    # LimiX style-1: predict(x_train, y_train, x_test)
    # LimiX style-2: sklearn style fit/predict.
    try:
        y_pred = clf.predict(x_train_np, y_train_np, x_test_np)
    except TypeError:
        if not hasattr(clf, "fit") or not hasattr(clf, "predict"):
            raise RuntimeError(
                "Unsupported LimiX API. Expected either predict(X_train, y_train, X_test) "
                "or sklearn-style fit/predict."
            )
        clf.fit(x_train_np, y_train_np)
        y_pred = clf.predict(x_test_np)

    acc = float(accuracy_score(y_test.numpy(), y_pred))
    report = classification_report(y_test.numpy(), y_pred, digits=4)
    return {"accuracy": acc, "y_pred": y_pred, "report": report}
