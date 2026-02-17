# src/evaluate_base.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from sklearn.model_selection import GroupKFold
from sklearn.base import clone

import numpy as np
from joblib import dump
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src.data.data import (
    COMPOUND_CLASSES,
    infer_feature_types,
    build_prob_sequences_4lap,
    build_feature_sequences,
)
from src.data.preprocessing import make_preprocessor_for_model   # <- instead of build_preprocessor
from src.models.models import (
    ModelConfig,
    build_model_pipeline,
    get_binary_base_learners,
    get_multiclass_base_learners,
    make_ann_binary_vse_ffnn, 
    make_lstm_binary_head_vse,
    make_tcn_binary,
)
from src.models.models import (
    make_tcn_binary,
    make_gru_binary,
    make_lstm_binary,
    make_tcn_gru_binary,
)


# -----------------------------
# Metrics
# -----------------------------

def compute_binary_metrics(
    y_true: np.ndarray,
    proba_pos: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_pred = (proba_pos >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba_pos)),
        "pr_auc": float(average_precision_score(y_true, proba_pos)),
        "logloss": float(log_loss(y_true, proba_pos, labels=[0, 1])),
    }


def compute_multiclass_metrics(y_true: np.ndarray, proba_full: np.ndarray) -> Dict[str, float]:
    """
    proba_full: (n_samples, K) where columns correspond to classes 0..K-1.
    """
    y_pred = np.argmax(proba_full, axis=1)
    labels = np.arange(proba_full.shape[1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "logloss": float(log_loss(y_true, proba_full, labels=labels)),
    }


def summarize_fold_metrics(folds: List[Dict[str, Any]], keys: Sequence[str]) -> Dict[str, float]:
    return {f"{k}_mean": float(np.mean([fm[k] for fm in folds])) for k in keys}


# -----------------------------
# Fold helpers
# -----------------------------

def _make_group_folds(
    groups: np.ndarray,
    *,
    n_splits: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    gkf = GroupKFold(n_splits=n_splits)
    idx = np.arange(len(groups))
    return [(tr, va) for tr, va in gkf.split(idx, y=None, groups=groups)]


def _make_folds_from_y(
    X,
    y,
    *,
    n_splits: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, va) for tr, va in skf.split(X, y)]


def _subset_with_fold_indices(
    X_full,
    y_full,
    folds_full: List[Tuple[np.ndarray, np.ndarray]],
):
    """
    Given folds expressed in FULL dataframe indices, subset X/y down to only the rows
    that appear in the folds, and convert folds into SUBSET coordinates.

    Returns: (X_sub, y_sub, folds_sub, full_idx_kept)
    """
    used = np.unique(np.concatenate([np.concatenate([tr, va]) for tr, va in folds_full]))
    used = used.astype(int)
    used_sorted = np.sort(used)

    pos = {int(full_idx): i for i, full_idx in enumerate(used_sorted)}
    folds_sub = []
    for tr_full, va_full in folds_full:
        tr_sub = np.array([pos[int(i)] for i in tr_full], dtype=int)
        va_sub = np.array([pos[int(i)] for i in va_full], dtype=int)
        folds_sub.append((tr_sub, va_sub))

    X_sub = X_full.iloc[used_sorted].reset_index(drop=True)
    y_sub = y_full.iloc[used_sorted].reset_index(drop=True)

    return X_sub, y_sub, folds_sub, used_sorted


# -----------------------------
# CV routine
# -----------------------------

def run_cv(
    X,
    y,
    *,
    task: str,
    model_name: str,
    preprocessor,
    cfg: ModelConfig,
    folds: Optional[List[Tuple[np.ndarray, np.ndarray]]],
    n_splits: int,
    seed: int,
    threshold: float,
    n_classes: Optional[int],
    save_models: bool,
    outdir: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    # Folds
    if folds is None:
        folds = _make_folds_from_y(X, y, n_splits=n_splits, seed=seed)

    # Learners + metric keys
    if task == "binary":
        learners = get_binary_base_learners(cfg)
        if model_name not in learners:
            raise ValueError(f"Unknown binary model '{model_name}'. Available: {list(learners.keys())}")
        metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "logloss"]
        oof_pred = np.zeros(len(y), dtype=float)
        K = 2
    elif task == "multiclass":
        if n_classes is None:
            n_classes = int(np.unique(np.asarray(y)).shape[0])
        learners = get_multiclass_base_learners(cfg, n_classes=n_classes)
        if model_name not in learners:
            raise ValueError(f"Unknown multiclass model '{model_name}'. Available: {list(learners.keys())}")
        metric_keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "logloss"]
        oof_pred = np.zeros((len(y), n_classes), dtype=float)
        K = n_classes
    else:
        raise ValueError("--task must be 'binary' or 'multiclass'")

    fold_metrics: List[Dict[str, Any]] = []

    for fold, (tr_idx, va_idx) in enumerate(folds):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

        # IMPORTANT: clone both preprocessor + estimator per fold (no leakage/state carryover)
        est = clone(learners[model_name])
        pre = clone(preprocessor)

        pipe = build_model_pipeline(pre, est)
        print(f"[fold {fold}] fitting... X_tr={X_tr.shape}, y_tr={y_tr.shape}", flush=True)
        pipe.fit(X_tr, y_tr)
        print(f"[fold {fold}] done fit", flush=True)

        if task == "binary":
            proba_pos = pipe.predict_proba(X_va)[:, 1]
            oof_pred[va_idx] = proba_pos
            m = compute_binary_metrics(y_va.to_numpy(), proba_pos, threshold=threshold)

        else:
            # IMPORTANT: in CV, some folds may not include all classes -> proba may have <K columns
            proba_fold = pipe.predict_proba(X_va)  # (n_valid, k_fold)

            # Find the classes seen in training
            classes_seen = None
            if hasattr(pipe, "classes_"):
                classes_seen = getattr(pipe, "classes_")
            elif hasattr(pipe, "named_steps") and "model" in pipe.named_steps and hasattr(pipe.named_steps["model"], "classes_"):
                classes_seen = pipe.named_steps["model"].classes_
            else:
                raise RuntimeError("Could not determine classes_ from pipeline for multiclass alignment.")

            proba_full = np.zeros((len(va_idx), K), dtype=float)
            for j, cls in enumerate(classes_seen):
                proba_full[:, int(cls)] = proba_fold[:, j]

            oof_pred[va_idx, :] = proba_full
            m = compute_multiclass_metrics(y_va.to_numpy(), proba_full)

        row = {"fold": int(fold), "n_valid": int(len(va_idx)), **m}
        fold_metrics.append(row)

        if save_models:
            dump(pipe, outdir / f"pipeline_fold_{fold}.joblib")

        shown = " ".join(f"{k}={row[k]:.4f}" for k in metric_keys)
        print(f"[fold {fold}] {shown}")

    # Save OOF predictions
    if task == "binary":
        np.save(outdir / "oof_pred_proba_pos.npy", oof_pred)
    else:
        np.save(outdir / "oof_pred_proba.npy", oof_pred)

    summary = {
        "task": task,
        "model": model_name,
        "n_splits": len(folds),
        "seed": seed,
        "threshold": threshold if task == "binary" else None,
        "n_classes": K,
        **summarize_fold_metrics(fold_metrics, metric_keys),
    }
    return fold_metrics, summary

def run_cv_hybrid_vse(
    df,
    *,
    model_name: str,
    cfg: ModelConfig,
    n_splits: int,
    seed: int,
    threshold: float,
    save_models: bool,
    outdir: Path,
    seq_len: int = 4,
):
    df = df.reset_index(drop=True)
    y_full = df["y_pit"].astype(int)
    keys = df[["race_id", "driver_id", "lapno"]].copy()

    X_tab = df.drop(columns=["y_pit", "y_compound", "race_id", "driver_id"], errors="ignore").copy()

    num_cols, cat_cols = infer_feature_types(X_tab)
    base_pre = make_preprocessor_for_model(model_name, num_cols=num_cols, cat_cols=cat_cols)

    # Use grouped folds so 4-lap sequences don't get split across train/valid
    groups = (df["race_id"].astype(str) + "_" + df["driver_id"].astype(str)).to_numpy()
    folds = _make_group_folds(groups, n_splits=n_splits)


    fold_metrics = []
    oof_pred = np.full(len(df), np.nan, dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(folds):
        X_tr, y_tr = X_tab.iloc[tr_idx], y_full.iloc[tr_idx]

        X_va, y_va = X_tab.iloc[va_idx], y_full.iloc[va_idx]

        # fresh preprocessor each fold
        pre = clone(base_pre)

        # ---- Stage A: FFNN ----
        ffnn = make_ann_binary_vse_ffnn(cfg)
        pipe = build_model_pipeline(pre, ffnn)

        pipe.fit(X_tr, y_tr, model__class_weight={0: 1.0, 1: 5.0})

        # bypass sklearn Pipeline.predict_proba tag issues
        Xt_all = pipe.named_steps["preprocess"].transform(X_tab)
        proba_all = pipe.named_steps["model"].predict_proba(Xt_all)[:, 1]

        print(
            "DEBUG index check:",
            "len(df)=", len(df),
            "df.index.min/max=", int(df.index.min()), int(df.index.max()),
            "len(proba_all)=", len(proba_all),
            flush=True
        )

        # ---- Build sequences ----
        X_seq, y_seq, idx_last, seq_idx = build_prob_sequences_4lap(
            keys, proba_all, y_full, seq_len=seq_len
        )

        # NO LEAKAGE: all 4 indices must belong to the fold split
        tr_mask = np.all(np.isin(seq_idx, tr_idx), axis=1)
        va_mask = np.all(np.isin(seq_idx, va_idx), axis=1)

        X_seq_tr, y_seq_tr = X_seq[tr_mask], y_seq[tr_mask]
        X_seq_va, y_seq_va = X_seq[va_mask], y_seq[va_mask]
        idx_last_va = idx_last[va_mask]

        if len(y_seq_tr) == 0 or len(y_seq_va) == 0:
            print(f"[fold {fold}] skipping (no full sequences in train/valid after leakage filter)")
            continue

        # ---- Stage B: LSTM head ----
        lstm = make_lstm_binary_head_vse(cfg, seq_len=seq_len)

        lstm.fit(X_seq_tr, y_seq_tr, class_weight={0: 1.0, 1: 5.0})

        proba_pos = lstm.predict_proba(X_seq_va)[:, 1]
        oof_pred[idx_last_va] = proba_pos

        m = compute_binary_metrics(y_seq_va, proba_pos, threshold=threshold)
        row = {"fold": int(fold), "n_valid_seq": int(len(y_seq_va)), **m}
        fold_metrics.append(row)

        if save_models:
            dump(pipe, outdir / f"ffnn_pipeline_fold_{fold}.joblib")
            dump(lstm, outdir / f"lstm_head_fold_{fold}.joblib")

        shown = " ".join(f"{k}={row[k]:.4f}" for k in ["accuracy","precision","recall","f1","roc_auc","pr_auc","logloss"])
        print(f"[fold {fold}] {shown} (n_valid_seq={row['n_valid_seq']})")

    np.save(outdir / "oof_pred_proba_pos.npy", oof_pred)

    summary = {
        "task": "binary",
        "model": model_name,
        "n_splits": len(folds),
        "seed": seed,
        "threshold": threshold,
        "n_classes": 2,
        **summarize_fold_metrics(fold_metrics, ["accuracy","precision","recall","f1","roc_auc","pr_auc","logloss"]),
    }
    return fold_metrics, summary

def run_cv_tcn(
    df,
    *,
    cfg: ModelConfig,
    n_splits: int,
    seed: int,
    threshold: float,
    save_models: bool,
    outdir: Path,
    seq_len: int = 8,
):
    df = df.reset_index(drop=True)
    y_full = df["y_pit"].astype(int)
    keys = df[["race_id", "driver_id", "lapno"]].copy()

    # Tabular features only (drop ids + targets)
    X_tab = df.drop(columns=["y_pit", "y_compound", "race_id", "driver_id"], errors="ignore").copy()

    num_cols, cat_cols = infer_feature_types(X_tab)
    base_pre = make_preprocessor_for_model("tcn", num_cols=num_cols, cat_cols=cat_cols)

    # group = one time-series per driver per race
    groups = (df["race_id"].astype(str) + "_" + df["driver_id"].astype(str)).to_numpy()
    folds = _make_group_folds(groups, n_splits=n_splits)

    fold_metrics = []
    oof_pred = np.full(len(df), np.nan, dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(folds):
        X_tr, y_tr = X_tab.iloc[tr_idx], y_full.iloc[tr_idx]
        X_va, y_va = X_tab.iloc[va_idx], y_full.iloc[va_idx]

        # Fit preprocessor ONLY on train fold
        pre = clone(base_pre)
        pre.fit(X_tr, y_tr)

        # Transform all rows using train-fitted preprocessor (safe)
        Xt_all = pre.transform(X_tab)
        if hasattr(Xt_all, "toarray"):
            Xt_all = Xt_all.toarray()
        Xt_all = Xt_all.astype(np.float32)

        # Build sequences over the whole race timeline
        X_seq, y_seq, idx_last, seq_idx, eff_len = build_feature_sequences(
            keys, Xt_all, y_full,
            seq_len=seq_len,
            pad_left=True,
            add_timestep_mask=True,
        )

        # NO LEAKAGE: only keep sequences whose ALL timesteps are inside the fold split
        def _all_in_or_pad(seq_idx: np.ndarray, allowed_idx: np.ndarray) -> np.ndarray:
            return np.all((seq_idx == -1) | np.isin(seq_idx, allowed_idx), axis=1)

        tr_mask = _all_in_or_pad(seq_idx, tr_idx)
        va_mask = _all_in_or_pad(seq_idx, va_idx)

        X_seq_tr, y_seq_tr = X_seq[tr_mask], y_seq[tr_mask]
        X_seq_va, y_seq_va = X_seq[va_mask], y_seq[va_mask]
        idx_last_va = idx_last[va_mask]

        if len(y_seq_tr) == 0 or len(y_seq_va) == 0:
            print(f"[fold {fold}] skipping (no full sequences in train/valid after leakage filter)")
            continue

        # Imbalance handling:
        # compute pos_weight ~= n_neg / n_pos and cap to avoid instability
        n_pos = int(np.sum(y_seq_tr == 1))
        n_neg = int(np.sum(y_seq_tr == 0))
        if n_pos == 0:
            print(f"[fold {fold}] skipping (no positives in training sequences)")
            continue
        pos_w = min(20.0, float(n_neg / n_pos))
        class_w = {0: 1.0, 1: pos_w}

        # Train TCN
        tcn = make_tcn_binary(cfg, seq_len=seq_len)
        tcn.fit(X_seq_tr, y_seq_tr, class_weight=class_w)

        # Predict + write OOF probs back to last-lap indices
        proba_pos = tcn.predict_proba(X_seq_va)[:, 1]
        oof_pred[idx_last_va] = proba_pos

        m = compute_binary_metrics(y_seq_va, proba_pos, threshold=threshold)
        row = {"fold": int(fold), "n_valid_seq": int(len(y_seq_va)), "pos_weight": float(pos_w), **m}
        fold_metrics.append(row)

        if save_models:
            dump(pre, outdir / f"tcn_pre_fold_{fold}.joblib")

            # Save the underlying tf.keras.Model in native Keras format
            model_path = outdir / f"tcn_model_fold_{fold}.keras"
            tcn.model_.save(model_path)



        shown = " ".join(f"{k}={row[k]:.4f}" for k in ["accuracy","precision","recall","f1","roc_auc","pr_auc","logloss"])
        print(f"[fold {fold}] {shown} (n_valid_seq={row['n_valid_seq']} pos_w={pos_w:.2f})")

    np.save(outdir / "oof_pred_proba_pos.npy", oof_pred)

    summary = {
        "task": "binary",
        "model": "tcn",
        "seq_len": int(seq_len),
        "n_splits": len(folds),
        "seed": seed,
        "threshold": threshold,
        **summarize_fold_metrics(fold_metrics, ["accuracy","precision","recall","f1","roc_auc","pr_auc","logloss"]),
    }
    return fold_metrics, summary

SEQ_MODEL_FACTORIES = {
    "tcn": lambda cfg, seq_len: make_tcn_binary(cfg, seq_len=seq_len),
    "gru": lambda cfg, seq_len: make_gru_binary(cfg),
    "lstm": lambda cfg, seq_len: make_lstm_binary(cfg),
    "tcn_gru": lambda cfg, seq_len: make_tcn_gru_binary(cfg),
}

def run_cv_seq(
    df,
    *,
    seq_model: str,
    cfg: ModelConfig,
    n_splits: int,
    seed: int,
    threshold: float,
    save_models: bool,
    outdir: Path,
    seq_len: int = 8,
):
    df = df.reset_index(drop=True)
    y_full = df["y_pit"].astype(int)
    keys = df[["race_id", "driver_id", "lapno"]].copy()

    X_tab = df.drop(columns=["y_pit", "y_compound", "race_id", "driver_id"], errors="ignore").copy()
    num_cols, cat_cols = infer_feature_types(X_tab)

    # IMPORTANT: use the SAME preprocessor as TCN for all seq models
    base_pre = make_preprocessor_for_model("tcn", num_cols=num_cols, cat_cols=cat_cols)

    groups = (df["race_id"].astype(str) + "_" + df["driver_id"].astype(str)).to_numpy()
    folds = _make_group_folds(groups, n_splits=n_splits)

    fold_metrics = []
    oof_pred = np.full(len(df), np.nan, dtype=float)

    if seq_model not in SEQ_MODEL_FACTORIES:
        raise ValueError(f"Unknown seq_model={seq_model}. Choose from {list(SEQ_MODEL_FACTORIES)}")

    for fold, (tr_idx, va_idx) in enumerate(folds):
        X_tr, y_tr = X_tab.iloc[tr_idx], y_full.iloc[tr_idx]

        pre = clone(base_pre)
        pre.fit(X_tr, y_tr)

        Xt_all = pre.transform(X_tab)
        if hasattr(Xt_all, "toarray"):
            Xt_all = Xt_all.toarray()
        Xt_all = Xt_all.astype(np.float32)

        X_seq, y_seq, idx_last, seq_idx, eff_len = build_feature_sequences(
            keys, Xt_all, y_full,
            seq_len=seq_len,
            pad_left=True,
            add_timestep_mask=True,
        )

        def _all_in_or_pad(seq_idx, allowed_idx):
            return np.all((seq_idx == -1) | np.isin(seq_idx, allowed_idx), axis=1)

        tr_mask = _all_in_or_pad(seq_idx, tr_idx)
        va_mask = _all_in_or_pad(seq_idx, va_idx)

        X_seq_tr, y_seq_tr = X_seq[tr_mask], y_seq[tr_mask]
        X_seq_va, y_seq_va = X_seq[va_mask], y_seq[va_mask]
        idx_last_va = idx_last[va_mask]

        if len(y_seq_tr) == 0 or len(y_seq_va) == 0:
            print(f"[fold {fold}] skipping (no full sequences in train/valid after leakage filter)")
            continue

        n_pos = int(np.sum(y_seq_tr == 1))
        n_neg = int(np.sum(y_seq_tr == 0))
        if n_pos == 0:
            print(f"[fold {fold}] skipping (no positives in training sequences)")
            continue

        pos_w = min(20.0, float(n_neg / n_pos))
        class_w = {0: 1.0, 1: pos_w}

        model = SEQ_MODEL_FACTORIES[seq_model](cfg, seq_len)
        model.fit(X_seq_tr, y_seq_tr, class_weight=class_w)

        proba_pos = model.predict_proba(X_seq_va)[:, 1]
        oof_pred[idx_last_va] = proba_pos

        m = compute_binary_metrics(y_seq_va, proba_pos, threshold=threshold)
        row = {"fold": int(fold), "n_valid_seq": int(len(y_seq_va)), "pos_weight": float(pos_w), **m}
        fold_metrics.append(row)

        if save_models:
            dump(pre, outdir / f"{seq_model}_pre_fold_{fold}.joblib")
            model_path = outdir / f"{seq_model}_model_fold_{fold}.keras"
            model.model_.save(model_path)

        shown = " ".join(f"{k}={row[k]:.4f}" for k in ["accuracy","precision","recall","f1","roc_auc","pr_auc","logloss"])
        print(f"[{seq_model} fold {fold}] {shown} (n_valid_seq={row['n_valid_seq']} pos_w={pos_w:.2f})")

    np.save(outdir / "oof_pred_proba_pos.npy", oof_pred)

    summary = {
        "task": "binary",
        "model": seq_model,
        "seq_len": int(seq_len),
        "n_splits": len(folds),
        "seed": seed,
        "threshold": threshold,
        **summarize_fold_metrics(fold_metrics, ["accuracy","precision","recall","f1","roc_auc","pr_auc","logloss"]),
    }
    return fold_metrics, summary


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_stage1", required=False, help="Path to stage1 dataset (pit/no-pit)")
    ap.add_argument("--data_stage2", required=False, help="Path to stage2 dataset (pit-stop-only compound)")

    ap.add_argument("--task", choices=["binary", "multiclass"], required=True)
    ap.add_argument("--model", required=True, help="Model name (e.g., svm, rf, xgb, ann, hybrid_vse, tcn)")
    ap.add_argument("--stage2", action="store_true", help="Run stage-2 compound task on stage2 dataset")
    
    ap.add_argument("--outdir", default="runs/base", help="Base output directory")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--n_classes", type=int, default=None)
    ap.add_argument("--save_models", action="store_true")
    args = ap.parse_args()

    stage_tag = "stage2" if args.stage2 else "stage1"
    outdir = Path(args.outdir) / stage_tag / args.task / args.model
    outdir.mkdir(parents=True, exist_ok=True)

    # -----------------------
    # Load the correct dataset
    # -----------------------
    if args.stage2:
        if args.task != "multiclass":
            raise ValueError("--stage2 requires --task multiclass")
        if args.data_stage2 is None:
            raise ValueError("For --stage2 you must pass --data_stage2")

        from src.data.data import load_stage2_dataset, get_stage2_xy
        df = load_stage2_dataset(args.data_stage2, strict=True)
        X, y = get_stage2_xy(df)

        args.n_classes = len(COMPOUND_CLASSES)
        (outdir / "classes.json").write_text(json.dumps({"classes": list(COMPOUND_CLASSES)}, indent=2))

    else:
        if args.data_stage1 is None:
            raise ValueError("For stage1 you must pass --data_stage1")

        from src.data.data import load_stage1_dataset, get_stage1_xy
        df = load_stage1_dataset(args.data_stage1)
        X, y = get_stage1_xy(df)

    # -----------------------
    # Special sequence models (stage1 only)
    # -----------------------
    if (not args.stage2) and args.task == "binary" and args.model.lower() == "hybrid_vse":
        fold_metrics, summary = run_cv_hybrid_vse(
            df,
            model_name="hybrid_vse",
            cfg=ModelConfig(random_state=args.seed),
            n_splits=args.n_splits,
            seed=args.seed,
            threshold=args.threshold,
            save_models=args.save_models,
            outdir=outdir,
            seq_len=4,
        )
        (outdir / "fold_metrics.json").write_text(json.dumps(fold_metrics, indent=2))
        (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return

    if (not args.stage2) and args.task == "binary" and args.model.lower() in {"tcn","gru","lstm","tcn_gru"}:
        fold_metrics, summary = run_cv_seq(
            df,
            seq_model=args.model.lower(),
            cfg=ModelConfig(random_state=args.seed),
            n_splits=args.n_splits,
            seed=args.seed,
            threshold=args.threshold,
            save_models=args.save_models,
            outdir=outdir,
            seq_len=8,
        )
        (outdir / "fold_metrics.json").write_text(json.dumps(fold_metrics, indent=2))
        (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return


    # -----------------------
    # Generic base evaluation
    # -----------------------
    folds = None
    num_cols, cat_cols = infer_feature_types(X)
    pre = make_preprocessor_for_model(args.model, num_cols=num_cols, cat_cols=cat_cols)

    cfg = ModelConfig(random_state=args.seed)

    fold_metrics, summary = run_cv(
        X, y,
        task=args.task,
        model_name=args.model,
        preprocessor=pre,
        cfg=cfg,
        folds=folds,
        n_splits=args.n_splits,
        seed=args.seed,
        threshold=args.threshold,
        n_classes=args.n_classes,
        save_models=args.save_models,
        outdir=outdir,
    )

    (outdir / "fold_metrics.json").write_text(json.dumps(fold_metrics, indent=2))
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
