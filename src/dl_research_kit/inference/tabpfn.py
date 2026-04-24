from __future__ import annotations

import concurrent.futures
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy import special as scipy_special
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def _prepare_tabpfn_input(
    x_train: torch.Tensor, y_train: torch.Tensor, x_test: torch.Tensor
) -> Tuple[object, object, object]:
    return x_train.numpy(), y_train.numpy(), x_test.numpy()


def _create_tabpfn_classifier(device: str):
    import inspect
    from tabpfn import TabPFNClassifier

    classifier_kwargs = {
        "device": device,
        "ignore_pretraining_limits": True,
        "fit_mode": "low_memory",
        "memory_saving_mode": "auto",
    }
    if "inference_precision" in inspect.signature(TabPFNClassifier).parameters:
        classifier_kwargs["inference_precision"] = torch.float32

    try:
        return TabPFNClassifier(**classifier_kwargs)
    except TypeError:
        # Backward compatibility for older TabPFN signatures.
        classifier_kwargs.pop("inference_precision", None)
        try:
            return TabPFNClassifier(**classifier_kwargs)
        except TypeError:
            classifier_kwargs.pop("fit_mode", None)
            classifier_kwargs.pop("memory_saving_mode", None)
            return TabPFNClassifier(device=device, ignore_pretraining_limits=True)


def run_tabpfn_classifier(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    device: str = "cpu",
) -> Dict[str, object]:
    x_train_np, y_train_np, x_test_np = _prepare_tabpfn_input(x_train, y_train, x_test)
    if device == "cpu" and x_train_np.shape[0] > 1000:
        # TabPFN blocks large CPU datasets by default; enable explicit override.
        os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
    if device == "cuda":
        torch.cuda.empty_cache()
    clf = _create_tabpfn_classifier(device=device)
    clf.fit(x_train_np, y_train_np)
    y_pred = clf.predict(x_test_np)
    if device == "cuda":
        torch.cuda.empty_cache()

    acc = float(accuracy_score(y_test.numpy(), y_pred))
    report = classification_report(y_test.numpy(), y_pred, digits=4)
    return {"accuracy": acc, "y_pred": y_pred, "report": report}


def run_tabpfn_bagging_classifier(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    device: str = "cpu",
    num_bags: int = 8,
    context_size: int | None = None,
    aggregation: str = "average",
    random_seed: int = 42,
    n_jobs: int = 1,
    feature_drop_rate: float = 0.0,
) -> Dict[str, object]:
    if num_bags < 1:
        raise ValueError(f"num_bags must be >= 1, got {num_bags}")
    if aggregation not in ("average", "vote"):
        raise ValueError(f"aggregation must be one of ['average', 'vote'], got {aggregation}")
    if n_jobs < 1:
        raise ValueError(f"n_jobs must be >= 1, got {n_jobs}")
    if not (0.0 <= feature_drop_rate < 1.0):
        raise ValueError(
            f"feature_drop_rate must be in [0, 1), got {feature_drop_rate}"
        )

    x_train_np, y_train_np, x_test_np = _prepare_tabpfn_input(x_train, y_train, x_test)
    y_test_np = y_test.numpy()
    n_train = x_train_np.shape[0]
    n_features = x_train_np.shape[1]
    if n_train < 1:
        raise ValueError("x_train is empty.")
    if n_features < 1:
        raise ValueError("x_train has no feature columns.")
    sample_size = n_train if context_size is None else int(context_size)
    if sample_size < 1:
        raise ValueError(f"context_size must be >= 1, got {context_size}")

    if device == "cpu" and sample_size > 1000:
        os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

    classes = np.unique(y_train_np)
    class_to_idx = {int(c): i for i, c in enumerate(classes.tolist())}
    rng = np.random.default_rng(random_seed)
    bag_seeds = rng.integers(0, 2**31 - 1, size=num_bags, dtype=np.int64)

    def _single_bag(seed: int) -> Tuple[np.ndarray, np.ndarray]:
        if device == "cuda":
            torch.cuda.empty_cache()
        local_rng = np.random.default_rng(int(seed))
        boot_idx = local_rng.integers(0, n_train, size=sample_size)
        x_boot = x_train_np[boot_idx]
        y_boot = y_train_np[boot_idx]
        x_test_local = x_test_np

        if feature_drop_rate > 0.0 and n_features > 1:
            keep_count = max(1, int(round(n_features * (1.0 - feature_drop_rate))))
            keep_count = min(keep_count, n_features)
            feat_idx = local_rng.choice(n_features, size=keep_count, replace=False)
            x_boot = x_boot[:, feat_idx]
            x_test_local = x_test_np[:, feat_idx]

        clf = _create_tabpfn_classifier(device=device)
        clf.fit(x_boot, y_boot)
        y_pred_local = clf.predict(x_test_local)

        if device == "cuda":
            torch.cuda.empty_cache()

        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(x_test_local)
            if probs.shape[1] != len(classes):
                aligned = np.zeros((probs.shape[0], len(classes)), dtype=np.float64)
                local_classes = getattr(clf, "classes_", classes)
                for j, c in enumerate(local_classes):
                    if int(c) in class_to_idx:
                        aligned[:, class_to_idx[int(c)]] = probs[:, j]
                probs = aligned
        else:
            probs = np.zeros((x_test_np.shape[0], len(classes)), dtype=np.float64)
            for i, c in enumerate(y_pred_local):
                probs[i, class_to_idx[int(c)]] = 1.0

        del clf
        import gc

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        return y_pred_local, probs

    if n_jobs == 1:
        bag_outputs = [_single_bag(int(s)) for s in bag_seeds]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as pool:
            bag_outputs = list(pool.map(lambda s: _single_bag(int(s)), bag_seeds.tolist()))

    if device == "cuda":
        torch.cuda.empty_cache()

    pred_list = [pred for pred, _ in bag_outputs]
    prob_list = [prob for _, prob in bag_outputs]
    probs_stack = np.stack(prob_list, axis=0)

    if aggregation == "average":
        probs_mean = probs_stack.mean(axis=0)
        y_pred = classes[np.argmax(probs_mean, axis=1)]
    else:
        preds_stack = np.stack(pred_list, axis=0)
        votes = np.zeros((preds_stack.shape[1], len(classes)), dtype=np.int32)
        for b in range(preds_stack.shape[0]):
            for i, c in enumerate(preds_stack[b]):
                votes[i, class_to_idx[int(c)]] += 1
        y_pred = classes[np.argmax(votes, axis=1)]
        probs_mean = probs_stack.mean(axis=0)

    acc = float(accuracy_score(y_test_np, y_pred))
    report = classification_report(y_test_np, y_pred, digits=4)
    return {
        "accuracy": acc,
        "y_pred": y_pred,
        "report": report,
        "bagging_num_bags": num_bags,
        "bagging_aggregation": aggregation,
        "bagging_feature_drop_rate": feature_drop_rate,
        "bagging_mean_proba": probs_mean,
    }


def run_tabpfn_ensemble_selection_classifier(
    tables: List[Dict[str, object]],
    y_train: torch.Tensor,
    x_test_tables: List[Dict[str, object]],
    y_test: torch.Tensor,
    device: str = "cpu",
    val_size: float = 0.2,
    candidates_per_table: int = 8,
    context_size: int | None = None,
    colsample_min_rate: float = 0.4,
    colsample_max_rate: float = 1.0,
    max_selected: int = 32,
    n_jobs: int = 1,
    random_seed: int = 42,
) -> Dict[str, object]:
    if len(tables) == 0:
        raise ValueError("tables must not be empty.")
    if len(x_test_tables) != len(tables):
        raise ValueError("x_test_tables length must match tables length.")
    if not (0.0 < val_size < 1.0):
        raise ValueError(f"val_size must be in (0, 1), got {val_size}")
    if candidates_per_table < 1:
        raise ValueError(f"candidates_per_table must be >= 1, got {candidates_per_table}")
    if max_selected < 1:
        raise ValueError(f"max_selected must be >= 1, got {max_selected}")
    if n_jobs < 1:
        raise ValueError(f"n_jobs must be >= 1, got {n_jobs}")
    if not (0.0 < colsample_min_rate <= colsample_max_rate <= 1.0):
        raise ValueError(
            "Require 0 < colsample_min_rate <= colsample_max_rate <= 1.0, "
            f"got ({colsample_min_rate}, {colsample_max_rate})"
        )

    y_train_np = y_train.numpy()
    y_test_np = y_test.numpy()
    num_rows = y_train_np.shape[0]
    if num_rows < 3:
        raise ValueError("Need at least 3 train rows for train/validation split.")

    row_idx = np.arange(num_rows)
    try:
        fit_idx, val_idx = train_test_split(
            row_idx,
            test_size=val_size,
            random_state=random_seed,
            stratify=y_train_np,
        )
    except ValueError:
        fit_idx, val_idx = train_test_split(
            row_idx,
            test_size=val_size,
            random_state=random_seed,
            stratify=None,
        )
    y_fit = y_train_np[fit_idx]
    y_val = y_train_np[val_idx]

    classes = np.unique(y_fit)
    if classes.shape[0] < 2:
        raise ValueError("Need at least 2 classes in fit split for classification.")
    class_to_idx = {int(c): i for i, c in enumerate(classes.tolist())}

    if device == "cpu" and (context_size or len(fit_idx)) > 1000:
        os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

    meta_candidates: List[Dict[str, object]] = []
    for t_idx, table in enumerate(tables):
        table_name = str(table["name"])
        x_train_np = np.asarray(table["x_train"])
        x_test_np = np.asarray(x_test_tables[t_idx]["x_test"])
        if x_train_np.shape[0] != num_rows:
            raise ValueError(
                f"Table '{table_name}' row mismatch: x_train rows={x_train_np.shape[0]} vs y_train rows={num_rows}"
            )
        if x_train_np.shape[1] < 1:
            raise ValueError(f"Table '{table_name}' has no feature columns.")
        for c_idx in range(candidates_per_table):
            meta_candidates.append(
                {
                    "table_name": table_name,
                    "table_idx": t_idx,
                    "candidate_idx": c_idx,
                    "x_train": x_train_np,
                    "x_test": x_test_np,
                }
            )

    global_rng = np.random.default_rng(random_seed)
    candidate_seeds = global_rng.integers(0, 2**31 - 1, size=len(meta_candidates), dtype=np.int64)

    def _single_candidate(idx_and_seed: Tuple[int, int]) -> Dict[str, object]:
        idx, seed = idx_and_seed
        spec = meta_candidates[idx]
        x_train_full = spec["x_train"]
        x_test_full = spec["x_test"]
        local_rng = np.random.default_rng(int(seed))

        x_fit_pool = x_train_full[fit_idx]
        x_val = x_train_full[val_idx]
        n_fit = x_fit_pool.shape[0]
        n_cols = x_fit_pool.shape[1]

        sample_size = n_fit if context_size is None else min(int(context_size), n_fit)
        if sample_size < 1:
            raise ValueError("context_size produced empty context.")
        boot_idx = local_rng.integers(0, n_fit, size=sample_size)
        x_context = x_fit_pool[boot_idx]
        y_context = y_fit[boot_idx]

        if n_cols == 1:
            feat_idx = np.array([0], dtype=np.int64)
        else:
            col_rate = float(local_rng.uniform(colsample_min_rate, colsample_max_rate))
            keep_cols = int(round(n_cols * col_rate))
            keep_cols = max(1, min(n_cols, keep_cols))
            feat_idx = local_rng.choice(n_cols, size=keep_cols, replace=False)

        x_context = x_context[:, feat_idx]
        x_val = x_val[:, feat_idx]
        x_test = x_test_full[:, feat_idx]

        if device == "cuda":
            torch.cuda.empty_cache()
        clf = _create_tabpfn_classifier(device=device)
        clf.fit(x_context, y_context)
        val_probs = clf.predict_proba(x_val)
        test_probs = clf.predict_proba(x_test)
        local_classes = getattr(clf, "classes_", classes)

        aligned_val = np.zeros((val_probs.shape[0], len(classes)), dtype=np.float64)
        aligned_test = np.zeros((test_probs.shape[0], len(classes)), dtype=np.float64)
        for j, c in enumerate(local_classes):
            c_int = int(c)
            if c_int in class_to_idx:
                aligned_val[:, class_to_idx[c_int]] = val_probs[:, j]
                aligned_test[:, class_to_idx[c_int]] = test_probs[:, j]

        del clf
        if device == "cuda":
            torch.cuda.empty_cache()
        return {
            "table_name": spec["table_name"],
            "table_idx": spec["table_idx"],
            "candidate_idx": spec["candidate_idx"],
            "seed": int(seed),
            "num_cols": int(feat_idx.shape[0]),
            "val_probs": aligned_val,
            "test_probs": aligned_test,
        }

    packed = list(zip(range(len(meta_candidates)), candidate_seeds.tolist()))
    if n_jobs == 1:
        candidates = [_single_candidate(x) for x in packed]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as pool:
            candidates = list(pool.map(_single_candidate, packed))

    current_sum_val = None
    selected_indices: List[int] = []
    selection_trace: List[Dict[str, object]] = []
    current_best_acc = -1.0

    for step in range(max_selected):
        best_idx = -1
        best_acc = current_best_acc
        for idx, cand in enumerate(candidates):
            if current_sum_val is None:
                trial = cand["val_probs"]
                denom = 1.0
            else:
                trial = current_sum_val + cand["val_probs"]
                denom = float(len(selected_indices) + 1)
            trial_pred = classes[np.argmax(trial / denom, axis=1)]
            trial_acc = float(accuracy_score(y_val, trial_pred))
            if trial_acc > best_acc + 1e-12:
                best_acc = trial_acc
                best_idx = idx

        if best_idx < 0:
            break

        selected_indices.append(best_idx)
        current_best_acc = best_acc
        if current_sum_val is None:
            current_sum_val = candidates[best_idx]["val_probs"].copy()
        else:
            current_sum_val = current_sum_val + candidates[best_idx]["val_probs"]
        selection_trace.append(
            {
                "step": step + 1,
                "selected_table": candidates[best_idx]["table_name"],
                "candidate_idx": int(candidates[best_idx]["candidate_idx"]),
                "val_acc": current_best_acc,
            }
        )

    if len(selected_indices) == 0:
        raise RuntimeError("No candidate improved validation accuracy; ensemble is empty.")

    # Re-infer selected candidates using full training data (fit + val combined)
    def _reinfer_candidate_with_full_train(selected_idx: int) -> np.ndarray:
        """Re-infer a candidate using full training data as context."""
        cand = candidates[selected_idx]
        seed = int(cand["seed"])
        spec = meta_candidates[selected_idx]  # Directly map back to meta_candidates
        x_train_full = spec["x_train"]
        x_test_full = spec["x_test"]
        
        # Reconstruct full training data from fit and val
        x_train_full_np = np.vstack([x_train_full[fit_idx], x_train_full[val_idx]])
        y_train_full_np = np.concatenate([y_fit, y_val])
        
        local_rng = np.random.default_rng(seed)
        n_cols = x_test_full.shape[1]
        
        # Regenerate feature selection with same seed
        if n_cols == 1:
            feat_idx = np.array([0], dtype=np.int64)
        else:
            col_rate = float(local_rng.uniform(colsample_min_rate, colsample_max_rate))
            keep_cols = int(round(n_cols * col_rate))
            keep_cols = max(1, min(n_cols, keep_cols))
            feat_idx = local_rng.choice(n_cols, size=keep_cols, replace=False)
        
        x_train_selected = x_train_full_np[:, feat_idx]
        x_test_selected = x_test_full[:, feat_idx]
        
        if device == "cuda":
            torch.cuda.empty_cache()
        clf = _create_tabpfn_classifier(device=device)
        clf.fit(x_train_selected, y_train_full_np)
        test_probs = clf.predict_proba(x_test_selected)
        local_classes = getattr(clf, "classes_", classes)
        
        aligned_test = np.zeros((test_probs.shape[0], len(classes)), dtype=np.float64)
        for j, c in enumerate(local_classes):
            c_int = int(c)
            if c_int in class_to_idx:
                aligned_test[:, class_to_idx[c_int]] = test_probs[:, j]
        
        del clf
        if device == "cuda":
            torch.cuda.empty_cache()
        return aligned_test
    
    # Re-infer all selected candidates
    reinferred_test_probs = []
    for selected_idx in selected_indices:
        reinfered_probs = _reinfer_candidate_with_full_train(selected_idx)
        reinferred_test_probs.append(reinfered_probs)
    
    # Aggregate re-inferred probabilities
    ensemble_test_probs = np.mean(np.stack(reinferred_test_probs, axis=0), axis=0)
    y_pred = classes[np.argmax(ensemble_test_probs, axis=1)]

    table_count: Dict[str, int] = {}
    for idx in selected_indices:
        table_name = str(candidates[idx]["table_name"])
        table_count[table_name] = table_count.get(table_name, 0) + 1
    table_weights = {
        name: float(count) / float(len(selected_indices))
        for name, count in sorted(table_count.items(), key=lambda kv: kv[0])
    }

    acc = float(accuracy_score(y_test_np, y_pred))
    report = classification_report(y_test_np, y_pred, digits=4)
    return {
        "accuracy": acc,
        "y_pred": y_pred,
        "report": report,
        "ensemble_val_size": int(len(val_idx)),
        "ensemble_fit_size": int(len(fit_idx)),
        "ensemble_num_candidates": int(len(candidates)),
        "ensemble_num_selected": int(len(selected_indices)),
        "ensemble_table_weights": table_weights,
        "ensemble_selection_trace": selection_trace,
    }


def run_tabpfn_ensemble_average_classifier(
    tables: List[Dict[str, object]],
    y_train: torch.Tensor,
    x_test_tables: List[Dict[str, object]],
    y_test: torch.Tensor,
    device: str = "cpu",
    candidates_per_table: int = 8,
    context_size: int | None = None,
    colsample_min_rate: float = 0.4,
    colsample_max_rate: float = 1.0,
    n_jobs: int = 1,
    random_seed: int = 42,
) -> Dict[str, object]:
    """
    Run TabPFN ensemble averaging classifier.
    
    This function generates all candidates from the provided tables and directly
    averages their test probabilities without greedy selection.
    """
    if len(tables) == 0:
        raise ValueError("tables must not be empty.")
    if len(x_test_tables) != len(tables):
        raise ValueError("x_test_tables length must match tables length.")
    if candidates_per_table < 1:
        raise ValueError(f"candidates_per_table must be >= 1, got {candidates_per_table}")
    if n_jobs < 1:
        raise ValueError(f"n_jobs must be >= 1, got {n_jobs}")
    if not (0.0 < colsample_min_rate <= colsample_max_rate <= 1.0):
        raise ValueError(
            "Require 0 < colsample_min_rate <= colsample_max_rate <= 1.0, "
            f"got ({colsample_min_rate}, {colsample_max_rate})"
        )

    y_train_np = y_train.numpy()
    y_test_np = y_test.numpy()
    num_rows = y_train_np.shape[0]
    if num_rows < 1:
        raise ValueError("y_train is empty.")

    classes = np.unique(y_train_np)
    class_to_idx = {int(c): i for i, c in enumerate(classes.tolist())}

    meta_candidates: List[Dict[str, object]] = []
    for t_idx, table in enumerate(tables):
        table_name = str(table["name"])
        x_train_np = np.asarray(table["x_train"])
        x_test_np = np.asarray(x_test_tables[t_idx]["x_test"])
        if x_train_np.shape[0] != num_rows:
            raise ValueError(
                f"Table '{table_name}' row mismatch: x_train rows={x_train_np.shape[0]} vs y_train rows={num_rows}"
            )
        if x_train_np.shape[1] < 1:
            raise ValueError(f"Table '{table_name}' has no feature columns.")
        for c_idx in range(candidates_per_table):
            meta_candidates.append(
                {
                    "table_name": table_name,
                    "table_idx": t_idx,
                    "candidate_idx": c_idx,
                    "x_train": x_train_np,
                    "x_test": x_test_np,
                }
            )

    global_rng = np.random.default_rng(random_seed)
    candidate_seeds = global_rng.integers(0, 2**31 - 1, size=len(meta_candidates), dtype=np.int64)

    def _single_candidate(idx_and_seed: Tuple[int, int]) -> Dict[str, object]:
        idx, seed = idx_and_seed
        spec = meta_candidates[idx]
        x_train_full = spec["x_train"]
        x_test_full = spec["x_test"]
        local_rng = np.random.default_rng(int(seed))

        n_fit = x_train_full.shape[0]
        n_cols = x_train_full.shape[1]

        sample_size = n_fit if context_size is None else min(int(context_size), n_fit)
        if sample_size < 1:
            raise ValueError("context_size produced empty context.")
        boot_idx = local_rng.integers(0, n_fit, size=sample_size)
        x_context = x_train_full[boot_idx]
        y_context = y_train_np[boot_idx]

        if n_cols == 1:
            feat_idx = np.array([0], dtype=np.int64)
        else:
            col_rate = float(local_rng.uniform(colsample_min_rate, colsample_max_rate))
            keep_cols = int(round(n_cols * col_rate))
            keep_cols = max(1, min(n_cols, keep_cols))
            feat_idx = local_rng.choice(n_cols, size=keep_cols, replace=False)

        x_context = x_context[:, feat_idx]
        x_test = x_test_full[:, feat_idx]

        if device == "cuda":
            torch.cuda.empty_cache()
        clf = _create_tabpfn_classifier(device=device)
        clf.fit(x_context, y_context)
        test_probs = clf.predict_proba(x_test)
        local_classes = getattr(clf, "classes_", classes)

        aligned_test = np.zeros((test_probs.shape[0], len(classes)), dtype=np.float64)
        for j, c in enumerate(local_classes):
            c_int = int(c)
            if c_int in class_to_idx:
                aligned_test[:, class_to_idx[c_int]] = test_probs[:, j]

        del clf
        if device == "cuda":
            torch.cuda.empty_cache()
        # Convert probabilities to logits by taking log
        aligned_logits = np.log(np.clip(aligned_test, 1e-15, 1.0))
        return {
            "table_name": spec["table_name"],
            "test_logits": aligned_logits,
        }

    packed = list(zip(range(len(meta_candidates)), candidate_seeds.tolist()))
    if n_jobs == 1:
        candidates = [_single_candidate(x) for x in packed]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as pool:
            candidates = list(pool.map(_single_candidate, packed))

    if device == "cuda":
        torch.cuda.empty_cache()

    # Average all candidate logits, then apply softmax
    all_test_logits = np.stack([cand["test_logits"] for cand in candidates], axis=0)
    ensemble_test_logits = all_test_logits.mean(axis=0)
    ensemble_test_probs = scipy_special.softmax(ensemble_test_logits, axis=1)
    y_pred = classes[np.argmax(ensemble_test_probs, axis=1)]

    table_count: Dict[str, int] = {}
    for cand in candidates:
        table_name = str(cand["table_name"])
        table_count[table_name] = table_count.get(table_name, 0) + 1
    table_weights = {
        name: float(count) / float(len(candidates))
        for name, count in sorted(table_count.items(), key=lambda kv: kv[0])
    }

    acc = float(accuracy_score(y_test_np, y_pred))
    report = classification_report(y_test_np, y_pred, digits=4)
    return {
        "accuracy": acc,
        "y_pred": y_pred,
        "report": report,
        "ensemble_num_candidates": int(len(candidates)),
        "ensemble_table_weights": table_weights,
    }
