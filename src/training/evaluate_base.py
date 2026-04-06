"""
evaluates a single base model on our datasets using k fold cv.
keeps things minimal and handles both tabular and sequence models.

python -m src.training.evaluate_base 
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, GroupKFold

from src.data.data import (
    COMPOUND_CLASSES,
    infer_feature_types,
    build_feature_sequences,
    make_race_group_folds,
    load_stage1_dataset,
    get_stage1_xy,
    load_stage2_dataset,
    get_stage2_xy,
    encode_y_compound,
)
from src.data.preprocessing import make_preprocessor_for_model
from src.models.models import (
    build_model_pipeline,
    get_stage1_tabular_models,
    get_stage2_tabular_models,
    get_stage1_sequential_models,
    get_stage2_sequential_models,
)


def compute_binary_metrics(y_true: np.ndarray, proba_pos: np.ndarray) -> dict[str, float]:
    """compute common binary classification metrics"""
    y_pred = (proba_pos >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba_pos)),
        "pr_auc": float(average_precision_score(y_true, proba_pos)),
        "logloss": float(log_loss(y_true, proba_pos, labels=[0, 1])),
    }


def compute_multiclass_metrics(y_true: np.ndarray, proba_full: np.ndarray) -> dict[str, float]:
    """compute common multiclass metrics"""
    y_pred = np.argmax(proba_full, axis=1)
    labels = np.arange(proba_full.shape[1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "logloss": float(log_loss(y_true, proba_full, labels=labels)),
    }


def summarize_fold_metrics(folds: list[dict[str, Any]], keys: list[str]) -> dict[str, float]:
    """average the metrics across all folds"""
    return {f"{k}_mean": float(np.mean([fm[k] for fm in folds])) for k in keys}


def run_tabular_cv(x, y, task: str, model_name: str, outdir: Path):
    """run k fold cross validation for our tabular models"""
    seed = 42
    n_splits = 5
    
    num_cols, cat_cols = infer_feature_types(x)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(skf.split(x, y))
    
    if task == "binary":
        metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "logloss"]
        k_classes = 2
    else:
        k_classes = len(COMPOUND_CLASSES)
        metric_keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "logloss"]
        
    fold_metrics = []
    
    for fold, (tr_idx, va_idx) in enumerate(folds):
        x_tr, y_tr = x.iloc[tr_idx], y.iloc[tr_idx]
        x_va, y_va = x.iloc[va_idx], y.iloc[va_idx]
        
        # fresh instance per fold to avoid any leakage
        if task == "binary":
            est = get_stage1_tabular_models(cfg)[model_name]
        else:
            est = get_stage2_tabular_models(cfg, n_classes=k_classes)[model_name]
            
        pre = make_preprocessor_for_model(model_name, num_cols=num_cols, cat_cols=cat_cols)
        pipe = build_model_pipeline(pre, est)
        
        print(f"training fold {fold} on {len(tr_idx)} samples")
        pipe.fit(x_tr, y_tr)
        
        if task == "binary":
            proba = pipe.predict_proba(x_va)[:, 1]
            m = compute_binary_metrics(y_va.to_numpy(), proba)
        else:
            proba_fold = pipe.predict_proba(x_va)
            
            # map back to full classes in case a fold misses some
            classes_seen = pipe.named_steps["model"].classes_ if hasattr(pipe.named_steps["model"], "classes_") else getattr(pipe, "classes_")
            proba_full = np.zeros((len(va_idx), k_classes), dtype=float)
            for j, cls in enumerate(classes_seen):
                proba_full[:, int(cls)] = proba_fold[:, j]
                
            m = compute_multiclass_metrics(y_va.to_numpy(), proba_full)
            
        row = {"fold": fold, "n_valid": len(va_idx), **m}
        fold_metrics.append(row)
        
        print(f"fold {fold} done: " + " ".join(f"{k} {row[k]:.4f}" for k in metric_keys))
        
    return {
        "task": task,
        "model": model_name,
        "n_splits": n_splits,
        **summarize_fold_metrics(fold_metrics, metric_keys)
    }, fold_metrics


def run_seq_cv(df, task: str, model_name: str, outdir: Path):
    """run grouped cross validation for sequential models"""
    seed = 42
    n_splits = 5
    seq_len = 8 if task == "binary" else 12
    
    
    df = df.reset_index(drop=True)
    keys = df[["race_id", "driver_id", "lapno"]].copy()
    
    if task == "binary":
        y_full = df["y_pit"].astype(int)
        x_tab = df.drop(columns=["y_pit", "y_compound", "race_id", "driver_id"], errors="ignore").copy()
    else:
        y_full = df["y_compound_encoded"].fillna(-1).astype(int)
        x_tab = df.drop(columns=["y_pit", "y_compound", "y_compound_encoded", "race_id", "driver_id"], errors="ignore").copy()

    num_cols, cat_cols = infer_feature_types(x_tab)
    
    # keep races completely isolated from each other
    if task == "binary":
        groups = (df["race_id"].astype(str) + "_" + df["driver_id"].astype(str)).to_numpy()
        folds = list(GroupKFold(n_splits=n_splits).split(np.arange(len(groups)), groups=groups))
    else:
        fb = make_race_group_folds(df, group_col="race_id", target_col="y_compound_encoded", n_splits=n_splits, seed=seed)
        folds = fb.folds

    if task == "binary":
        k_classes = 2
        metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "logloss"]
    else:
        k_classes = len(COMPOUND_CLASSES)
        metric_keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "logloss"]

    fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(folds):
        x_tr = x_tab.iloc[tr_idx]
        
        pre = make_preprocessor_for_model(model_name, num_cols=num_cols, cat_cols=cat_cols)
        if task == "binary":
            pre.fit(x_tr, y_full.iloc[tr_idx])
        else:
            pre.fit(x_tr)
            
        xt_all = pre.transform(x_tab)
        if hasattr(xt_all, "toarray"):
            xt_all = xt_all.toarray()
        xt_all = xt_all.astype(np.float32)

        x_seq, y_seq, idx_last, seq_idx, _ = build_feature_sequences(
            keys, xt_all, y_full, seq_len=seq_len, pad_left=True, add_timestep_mask=True
        )

        def all_in(s_idx, allowed):
            return np.all((s_idx == -1) | np.isin(s_idx, allowed), axis=1)

        tr_mask = all_in(seq_idx, tr_idx)
        va_mask = all_in(seq_idx, va_idx)

        if task == "multiclass":
            tr_mask &= (y_seq >= 0)
            va_mask &= (y_seq >= 0)

        x_seq_tr, y_seq_tr = x_seq[tr_mask], y_seq[tr_mask]
        x_seq_va, y_seq_va = x_seq[va_mask], y_seq[va_mask]

        if len(y_seq_tr) == 0 or len(y_seq_va) == 0:
            print(f"skipping fold {fold} as there are no valid sequences")
            continue

        if task == "binary":
            model = get_stage1_sequential_models(cfg)[model_name]
            n_pos = int(np.sum(y_seq_tr == 1))
            n_neg = int(np.sum(y_seq_tr == 0))
            pos_w = min(20.0, float(n_neg / n_pos)) if n_pos > 0 else 1.0
            model.fit(x_seq_tr, y_seq_tr, class_weight={0: 1.0, 1: pos_w})
            
            proba = model.predict_proba(x_seq_va)[:, 1]
            m = compute_binary_metrics(y_seq_va, proba)
        else:
            model = get_stage2_sequential_models(cfg, n_classes=k_classes)[model_name]
            present = np.unique(y_seq_tr)
            from sklearn.utils.class_weight import compute_class_weight
            w = compute_class_weight(class_weight="balanced", classes=present, y=y_seq_tr)
            class_w = {int(c): float(wi) for c, wi in zip(present, w)}
            for c in range(k_classes):
                class_w.setdefault(c, 1.0)
                
            model.fit(x_seq_tr, y_seq_tr, class_weight=class_w)
            
            proba = model.predict_proba(x_seq_va)
            m = compute_multiclass_metrics(y_seq_va, proba)

        row = {"fold": fold, "n_valid": len(y_seq_va), **m}
        fold_metrics.append(row)
        
        print(f"fold {fold} done: " + " ".join(f"{k} {row[k]:.4f}" for k in metric_keys))

    return {
        "task": task,
        "model": model_name,
        "seq_len": seq_len,
        "n_splits": n_splits,
        **summarize_fold_metrics(fold_metrics, metric_keys)
    }, fold_metrics


def main() -> None:
    data_stage1 = "data/processed/dataset1.csv"
    data_stage2 = "data/processed/dataset2.csv"
    task = "binary" # "binary"/"multiclass"
    model = "xgb" # e.g. "xgb", "rf", "svm", "lstm", "tcn_gru", "hybrid_vse"
    outdir_base = "runs/base"

    outdir = Path(outdir_base) / task / model
    outdir.mkdir(parents=True, exist_ok=True)

    seq_models = {"tcn", "gru", "lstm", "tcn_gru", "hybrid_vse"}

    if task == "binary":
        df = load_stage1_dataset(data_stage1)
        if model.lower() in seq_models:
            summary, fold_metrics = run_seq_cv(df, task, model.lower(), outdir)
        else:
            x, y = get_stage1_xy(df)
            summary, fold_metrics = run_tabular_cv(x, y, task, model.lower(), outdir)

    elif task == "multiclass":
        df = load_stage2_dataset(data_stage2)
        if model.lower() in seq_models:
            df = encode_y_compound(df, col="y_compound", out_col="y_compound_encoded")
            summary, fold_metrics = run_seq_cv(df, task, model.lower(), outdir)
        else:
            x, y = get_stage2_xy(df)
            summary, fold_metrics = run_tabular_cv(x, y, task, model.lower(), outdir)

    else:
        raise ValueError("must be 'binary' or 'multiclass'")

    (outdir / "fold_metrics.json").write_text(json.dumps(fold_metrics, indent=2))
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()