#!/usr/bin/env python3
# src/evaluate_stack.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from joblib import dump
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
    average_precision_score,
)

from src.data.data import (
    load_stage1_dataset, load_stage2_dataset,
    get_stage1_xy, get_stage2_xy,
    make_race_group_folds,
    infer_feature_types,
    COMPOUND_CLASSES,
    build_feature_sequences
)
from src.data.preprocessing import build_preprocessor, PreprocessConfig
from src.models.models import (
    ModelConfig,
    build_model_pipeline,
    get_binary_base_learners,
    get_multiclass_base_learners,
    make_meta_binary_lr,
    make_meta_multinomial_lr,
    make_meta_binary_xgb,
    make_meta_binary_mlp,
    make_meta_multiclass_xgb,
    make_meta_multiclass_mlp,
    make_tcn_binary,
)

from src.data.preprocessing import make_preprocessor_for_model  # (you used this in evaluate_base.py)


import pandas as pd
COMPOUND_TO_INT: Dict[str, int] = {c: i for i, c in enumerate(COMPOUND_CLASSES)}

# -----------------------------
# helpers
# -----------------------------

def vprint(verbose: bool, *args, **kwargs):
    """Print only when verbose is enabled."""
    if verbose:
        print(*args, **kwargs, flush=True)

def encode_compound_labels(y: "pd.Series") -> "pd.Series":
    """
    Accepts y as strings (e.g., 'SOFT') or ints (0..K-1) and returns int labels.
    Drops/keeps NaN as-is (you should dropna before calling if needed).
    """
    # already numeric?
    if pd.api.types.is_numeric_dtype(y):
        return y.astype(int)

    # normalise strings
    y_str = y.astype(str).str.strip().str.upper()

    # map to ints; unmapped -> NaN
    y_enc = y_str.map(COMPOUND_TO_INT)

    # fail early if unexpected labels exist
    bad = y_str[y_enc.isna() & y.notna()].unique()
    if len(bad) > 0:
        raise ValueError(f"Unknown y_compound labels encountered: {list(bad)}. Update COMPOUND_TO_INT.")

    return y_enc.astype(int)


# -----------------------------
# Metrics
# -----------------------------

def binary_metrics(y_true: np.ndarray, proba_pos: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (proba_pos >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, proba_pos)),
        "pr_auc": float(average_precision_score(y_true, proba_pos)),
        "logloss": float(log_loss(y_true, proba_pos, labels=[0, 1])),
    }


def multiclass_metrics(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    y_pred = np.argmax(proba, axis=1)
    labels = np.arange(proba.shape[1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "logloss": float(log_loss(y_true, proba, labels=labels)),
    }


def summarize(metrics_list: List[Dict[str, Any]], keys: List[str]) -> Dict[str, float]:
    if not metrics_list:
        return {f"{k}_mean": float("nan") for k in keys}
    return {f"{k}_mean": float(np.mean([m[k] for m in metrics_list])) for k in keys}


def get_meta_binary(cfg, name: str):
    if name != "xgb":
        raise ValueError(f"Only 'xgb' meta learner is supported, got: {name}")
    return make_meta_binary_xgb(cfg)

def get_meta_multiclass(cfg, name: str, n_classes: int):
    if name != "xgb":
        raise ValueError(f"Only 'xgb' meta learner is supported, got: {name}")
    return make_meta_multiclass_xgb(cfg, n_classes=n_classes)

# -----------------------------
# Core stacking utilities
# -----------------------------

from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.base import clone

def _collect_oof_tcn_preds_binary(
    df1: "pd.DataFrame",
    folds: List[Tuple[np.ndarray, np.ndarray]],
    *,
    cfg: ModelConfig,
    seq_len: int = 8,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute OOF p(y=1) for the TCN using the SAME (race-grouped) folds as stacking.

    Returns:
        oof_pred: (N,) float, OOF p(y=1) written to each row index (should be fully filled with pad_left=True,
                  unless a fold is skipped and you don't handle it).
        oof_eff_len: (N,) float/int, effective real-timestep count (1..seq_len) for that prediction.
    """
    y_full = df1["y_pit"].astype(int)
    keys = df1[["race_id", "driver_id", "lapno"]].copy()

    oof_pred = np.full(len(df1), np.nan, dtype=float)
    oof_eff_len = np.zeros(len(df1), dtype=float)  # 0 means “not written yet”

    # tabular features used for TCN preprocessing
    X_tab = df1.drop(columns=["y_pit", "y_compound", "race_id", "driver_id"], errors="ignore").copy()
    num_cols, cat_cols = infer_feature_types(X_tab)
    base_pre = make_preprocessor_for_model("tcn", num_cols=num_cols, cat_cols=cat_cols)

    for fold, (tr_idx, va_idx) in enumerate(folds):
        if verbose:
            print(f"[tcn-oof][fold {fold}] train={len(tr_idx)} valid={len(va_idx)}", flush=True)

        X_tr, y_tr = X_tab.iloc[tr_idx], y_full.iloc[tr_idx]

        # Fit preprocessor ONLY on train rows
        pre = clone(base_pre)
        pre.fit(X_tr, y_tr)

        # Transform all rows using train-fitted preprocessor
        Xt_all = pre.transform(X_tab)
        if hasattr(Xt_all, "toarray"):
            Xt_all = Xt_all.toarray()
        Xt_all = Xt_all.astype(np.float32)

        # Build padded sequences (now includes early laps) + timestep mask channel
        X_seq, y_seq, idx_last, seq_idx, eff_len = build_feature_sequences(
            keys, Xt_all, y_full, seq_len=seq_len, pad_left=True, add_timestep_mask=True
        )

        # allow padded timesteps (-1) in fold membership checks
        tr_mask = np.all((seq_idx == -1) | np.isin(seq_idx, tr_idx), axis=1)
        va_mask = np.all((seq_idx == -1) | np.isin(seq_idx, va_idx), axis=1)

        X_seq_tr, y_seq_tr = X_seq[tr_mask], y_seq[tr_mask]
        X_seq_va = X_seq[va_mask]
        idx_last_va = idx_last[va_mask]
        eff_len_va = eff_len[va_mask]

        if len(y_seq_tr) == 0 or len(X_seq_va) == 0:
            if verbose:
                print(f"[tcn-oof][fold {fold}] skipping (no sequences after filtering)", flush=True)
            continue

        # class weights
        n_pos = int(np.sum(y_seq_tr == 1))
        n_neg = int(np.sum(y_seq_tr == 0))

        # If degenerate training labels, fall back to constant predictions (avoids NaNs downstream)
        if n_pos == 0 or n_neg == 0:
            const_p = 1.0 if n_neg == 0 else 0.0  # if only positives, predict 1; if only negatives, predict 0
            oof_pred[idx_last_va] = const_p
            oof_eff_len[idx_last_va] = eff_len_va
            if verbose:
                print(f"[tcn-oof][fold {fold}] degenerate y (n_pos={n_pos}, n_neg={n_neg}) -> const_p={const_p}", flush=True)
            continue

        pos_w = min(20.0, float(n_neg / n_pos))
        class_w = {0: 1.0, 1: pos_w}

        tcn = make_tcn_binary(cfg, seq_len=seq_len)
        tcn.fit(X_seq_tr, y_seq_tr, class_weight=class_w)

        proba_pos = tcn.predict_proba(X_seq_va)[:, 1]
        oof_pred[idx_last_va] = proba_pos
        oof_eff_len[idx_last_va] = eff_len_va

        if verbose:
            print(f"[tcn-oof][fold {fold}] wrote {len(idx_last_va)} preds (pos_w={pos_w:.2f})", flush=True)

    return oof_pred, oof_eff_len



def _collect_oof_base_preds_binary(
    X,
    y,
    folds,
    base_learners,
    preprocessor,
    *,
    verbose: bool = False,
):
    """
    Returns meta_X: (n_samples, n_models) with p(y=1) per base model.
    Uses provided folds (race-grouped).
    """
    n = len(y)
    model_names = list(base_learners.keys())
    meta_X = np.zeros((n, len(model_names)), dtype=float)

    vprint(verbose, f"[base-oof/binary] n={n} models={model_names}")

    for fold, (tr_idx, va_idx) in enumerate(folds):
        vprint(verbose, f"[base-oof/binary][fold {fold}] train={len(tr_idx)} valid={len(va_idx)}")
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va = X.iloc[va_idx]

        for j, name in enumerate(model_names):
            vprint(verbose, f"  -> fitting base '{name}'")
            est = clone(base_learners[name])
            # pick preprocessor based on model type
            num_cols, cat_cols = infer_feature_types(X)

            if name in {"ann"}:  # any keras tabular model
                pre = make_preprocessor_for_model("ann", num_cols=num_cols, cat_cols=cat_cols)
            elif name in {"svm"}:
                pre = make_preprocessor_for_model("svm", num_cols=num_cols, cat_cols=cat_cols)
            elif name in {"rf"}:
                pre = make_preprocessor_for_model("rf", num_cols=num_cols, cat_cols=cat_cols)
            elif name in {"xgb"}:
                pre = make_preprocessor_for_model("xgb", num_cols=num_cols, cat_cols=cat_cols)
            else:
                pre = clone(preprocessor)

            pipe = build_model_pipeline(pre, est)
            pipe.fit(X_tr, y_tr)

            proba_pos = pipe.predict_proba(X_va)[:, 1]
            meta_X[va_idx, j] = proba_pos

            vprint(verbose, f"     done '{name}': mean(p)= {float(np.mean(proba_pos)):.4f}")

    return meta_X

def make_meta_feature_df_binary(meta_X: np.ndarray, base_model_names: list[str], X_context: pd.DataFrame) -> pd.DataFrame:
    meta_prob_cols = [f"p_{name}" for name in base_model_names]
    df_probs = pd.DataFrame(meta_X, columns=meta_prob_cols)
    # join with context features (same row ordering)
    return pd.concat([df_probs.reset_index(drop=True), X_context.reset_index(drop=True)], axis=1)

def make_meta_feature_df_multiclass(meta_X: np.ndarray, base_model_names: list[str], n_classes: int, X_context: pd.DataFrame) -> pd.DataFrame:
    cols = []
    for name in base_model_names:
        for k in range(n_classes):
            cols.append(f"p_{name}_c{k}")
    df_probs = pd.DataFrame(meta_X, columns=cols)
    return pd.concat([df_probs.reset_index(drop=True), X_context.reset_index(drop=True)], axis=1)

def _collect_oof_base_preds_multiclass(
    X,
    y,
    folds,
    base_learners,
    preprocessor,
    n_classes: int,
    *,
    verbose: bool = False,
):
    """
    Returns meta_X: (n_samples, n_models * n_classes) by concatenating prob vectors.
    Uses fold-based training and aligns predict_proba columns to [0..K-1].
    """
    n = len(y)
    model_names = list(base_learners.keys())
    meta_X = np.zeros((n, len(model_names) * n_classes), dtype=float)

    vprint(verbose, f"[base-oof/multiclass] n={n} models={model_names} K={n_classes}")

    for fold, (tr_idx, va_idx) in enumerate(folds):
        vprint(verbose, f"[base-oof/multiclass][fold {fold}] train={len(tr_idx)} valid={len(va_idx)}")
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va = X.iloc[va_idx]

        for j, name in enumerate(model_names):
            vprint(verbose, f"  -> fitting base '{name}'")
            est = clone(base_learners[name])

            # pick preprocessor based on model type (key: keras models must be dense)
            num_cols, cat_cols = infer_feature_types(X)

            if name in {"ann", "vse_compound_ann"}:
                pre = make_preprocessor_for_model("ann", num_cols=num_cols, cat_cols=cat_cols)  # dense
            elif name == "svm":
                pre = make_preprocessor_for_model("svm", num_cols=num_cols, cat_cols=cat_cols)
            elif name == "rf":
                pre = make_preprocessor_for_model("rf", num_cols=num_cols, cat_cols=cat_cols)
            elif name == "xgb":
                pre = make_preprocessor_for_model("xgb", num_cols=num_cols, cat_cols=cat_cols)
            else:
                pre = clone(preprocessor)

            pipe = build_model_pipeline(pre, est)

            pipe.fit(X_tr, y_tr)
            proba_fold = pipe.predict_proba(X_va)

            # classes seen in this fold (may be subset)
            if hasattr(pipe, "named_steps") and "model" in pipe.named_steps and hasattr(pipe.named_steps["model"], "classes_"):
                classes_seen = pipe.named_steps["model"].classes_
            elif hasattr(pipe, "classes_"):
                classes_seen = pipe.classes_
            else:
                raise RuntimeError("Could not determine classes_ for multiclass alignment.")

            # align into full K columns
            proba_full = np.zeros((len(va_idx), n_classes), dtype=float)
            for j2, cls in enumerate(classes_seen):
                proba_full[:, int(cls)] = proba_fold[:, j2]

            start = j * n_classes
            meta_X[va_idx, start:start + n_classes] = proba_full

            vprint(verbose, f"     done '{name}': mean(maxp)= {float(np.mean(np.max(proba_full, axis=1))):.4f}")

    return meta_X



# -----------------------------
# Stage 1: Binary stacking CV
# -----------------------------

def run_binary_stacking_cv(
    df1, X, y,
    *,
    cfg: ModelConfig,
    preprocessor,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    meta_model_names: List[str],  # <-- CHANGED
    threshold: float,
    outdir: Path,
    save_models: bool,
    verbose: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:  # <-- CHANGED return type
    outdir.mkdir(parents=True, exist_ok=True)

    base = get_binary_base_learners(cfg)
    model_names = list(base.keys())

    vprint(verbose, f"\n[stage1] Binary stacking CV start: n={len(y)} folds={len(folds)}")
    vprint(verbose, f"[stage1] base models: {list(base.keys())}")

    # OOF base predictions -> meta features
    vprint(verbose, "[stage1] collecting OOF base predictions...")
    meta_X = _collect_oof_base_preds_binary(X, y, folds, base, preprocessor, verbose=verbose)
    vprint(verbose, f"[stage1] meta_X shape: {meta_X.shape}")


    # ---- Add TCN OOF as an extra base learner feature ----
    seq_len_tcn = 8  # keep this explicit so we can scale eff_len consistently

    tcn_oof, tcn_eff_len = _collect_oof_tcn_preds_binary(
        df1, folds, cfg=cfg, seq_len=seq_len_tcn, verbose=verbose
    )

    # With pad_left=True, tcn_oof should be fully filled; but keep it robust anyway
    tcn_oof_filled = np.nan_to_num(tcn_oof, nan=0.0)

    # Scale effective length to [0,1] so the meta learner can use it smoothly
    tcn_eff_len_scaled = tcn_eff_len / float(seq_len_tcn)

    # ---- Save base OOF probs INCLUDING TCN ----
    meta_X_with_tcn = np.column_stack([meta_X, tcn_oof_filled, tcn_eff_len_scaled])

    base_prob_columns = [f"p_{name}" for name in model_names] + ["tcn_proba", "tcn_effective_len"]
    (outdir / "base_oof_prob_columns.json").write_text(
        json.dumps({"columns": base_prob_columns}, indent=2)
    )

    np.save(outdir / "oof_pred_proba_pos.npy", meta_X_with_tcn)

    # Start from base probs + context
    X_meta_df = make_meta_feature_df_binary(meta_X, model_names, X)

    # Add TCN features to the meta dataframe
    X_meta_df["tcn_proba"] = tcn_oof_filled
    X_meta_df["tcn_effective_len"] = tcn_eff_len_scaled

    #save meta feature names
    (outdir / "meta_features_names.json").write_text(json.dumps({
        "columns": list(X_meta_df.columns)
    }, indent=2))
    X_meta_df.to_parquet(outdir / "meta_features_oof.parquet", index=False)

    # ----------------------------
    # Meta learner evaluation (multiple meta models, reuse same X_meta_df)
    # ----------------------------
    meta_rows_by_model: Dict[str, Any] = {}
    meta_summaries_by_model: Dict[str, Any] = {}

    for meta_model_name in meta_model_names:
        fold_rows: List[Dict[str, Any]] = []
        oof_meta_proba = np.zeros(len(y), dtype=float)

        vprint(verbose, f"[stage1] training/evaluating meta learner='{meta_model_name}' fold-by-fold...")
        for fold, (tr_idx, va_idx) in enumerate(folds):
            X_meta_tr = X_meta_df.iloc[tr_idx]
            X_meta_va = X_meta_df.iloc[va_idx]
            y_tr = y.iloc[tr_idx].to_numpy()
            y_va = y.iloc[va_idx].to_numpy()

            meta = clone(get_meta_binary(cfg, meta_model_name))

            # meta needs preprocessing (X_meta_df includes categoricals)
            num_m, cat_m = infer_feature_types(X_meta_df)
            meta_pre = build_preprocessor(
                num_cols=num_m,
                cat_cols=cat_m,
                cfg=PreprocessConfig(scale_numeric=False, sparse_onehot=True)
            )

            meta_pipe = build_model_pipeline(meta_pre, meta)
            meta_pipe.fit(X_meta_tr, y_tr)

            proba_pos = meta_pipe.predict_proba(X_meta_va)[:, 1]
            oof_meta_proba[va_idx] = proba_pos

            m = binary_metrics(y_va, proba_pos, threshold=threshold)
            fold_rows.append({"fold": int(fold), "n_valid": int(len(va_idx)), **m})

            print(
                f"[stage1][{meta_model_name}][meta fold {fold}] "
                + " ".join(f"{k}={m[k]:.4f}" for k in ["accuracy", "f1", "roc_auc", "pr_auc", "logloss"]),
                flush=True,
            )

            if save_models:
                dump(meta_pipe, outdir / f"meta_{meta_model_name}_pipe_fold_{fold}.joblib")

        # save OOF meta probs per meta model
        np.save(outdir / f"oof_meta_{meta_model_name}_proba_pos.npy", oof_meta_proba)

        summary = {
            "stage": "stage1_binary",
            "meta_model": meta_model_name,
            "n_folds": len(folds),
            "threshold": threshold,
            **summarize(fold_rows, ["accuracy", "f1", "roc_auc", "pr_auc", "logloss"]),
        }

        meta_rows_by_model[meta_model_name] = fold_rows
        meta_summaries_by_model[meta_model_name] = summary

    # also write combined summaries
    (outdir / "summary_by_meta.json").write_text(json.dumps(meta_summaries_by_model, indent=2))

    # Save meta features (already includes TCN in the parquet)
    X_meta_df.to_parquet(outdir / "meta_features_oof.parquet", index=False)

    return meta_rows_by_model, meta_summaries_by_model



# -----------------------------
# Stage 2: Multiclass stacking CV (true gate == 1 subset)
# -----------------------------

def run_multiclass_stacking_cv(
    X, y,
    *,
    cfg: ModelConfig,
    preprocessor,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    meta_model_name: str = "lr",
    n_classes: int,
    outdir: Path,
    save_models: bool,
    verbose: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    outdir.mkdir(parents=True, exist_ok=True)

    base = get_multiclass_base_learners(cfg, n_classes=n_classes)
    model_names = list(base.keys())

    # Ensure y is encoded to ints 0..K-1
    y_enc = encode_compound_labels(y)

    vprint(verbose, f"\n[stage2] Multiclass stacking CV start: n={len(y_enc)} folds={len(folds)} K={n_classes}")
    vprint(verbose, f"[stage2] base models: {list(base.keys())}")

    meta_X = _collect_oof_base_preds_multiclass(
        X, y_enc, folds, base, preprocessor, n_classes=n_classes, verbose=verbose
    )
    vprint(verbose, f"[stage2] meta_X shape: {meta_X.shape}")
    X_meta_df = make_meta_feature_df_multiclass(meta_X, model_names, n_classes, X)

    # --- Debug: confirm meta inputs include race context ---
    prob_cols = []
    for name in model_names:
        for k in range(n_classes):
            prob_cols.append(f"p_{name}_c{k}")

    all_cols = list(X_meta_df.columns)
    ctx_cols = [c for c in all_cols if c not in set(prob_cols)]

    vprint(verbose, f"[stage2] meta_X shape (probs only): {meta_X.shape}")
    vprint(verbose, f"[stage2] X_meta_df shape (probs + context): {X_meta_df.shape}")
    vprint(verbose, f"[stage2] n_prob_cols={len(prob_cols)} n_context_cols={len(ctx_cols)}")

    # print a clean list of context feature names
    vprint(verbose, "[stage2] Context feature columns passed to meta learner:")
    for c in ctx_cols:
        vprint(verbose, f"  - {c}")

    # optional: sanity check that prob columns are exactly what we expect
    missing_prob = [c for c in prob_cols if c not in X_meta_df.columns]
    extra_prob = [c for c in X_meta_df.columns if c.startswith("p_") and c not in set(prob_cols)]
    if missing_prob:
        vprint(verbose, f"[stage2][WARN] Missing expected prob cols: {missing_prob[:10]}{'...' if len(missing_prob)>10 else ''}")
    if extra_prob:
        vprint(verbose, f"[stage2][WARN] Unexpected prob cols: {extra_prob[:10]}{'...' if len(extra_prob)>10 else ''}")

    fold_rows: List[Dict[str, Any]] = []
    oof_meta_proba = np.zeros((len(y_enc), n_classes), dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(folds):
        X_meta_tr = X_meta_df.iloc[tr_idx]
        X_meta_va = X_meta_df.iloc[va_idx]
        y_tr = y_enc.iloc[tr_idx].to_numpy()
        y_va = y_enc.iloc[va_idx].to_numpy()

        meta = clone(get_meta_multiclass(cfg, meta_model_name, n_classes=n_classes))

        num_m, cat_m = infer_feature_types(X_meta_df)
        meta_pre = build_preprocessor(
            num_cols=num_m,
            cat_cols=cat_m,
            cfg=PreprocessConfig(scale_numeric=False, sparse_onehot=True)
        )

        meta_pipe = build_model_pipeline(meta_pre, meta)
        meta_pipe.fit(X_meta_tr, y_tr)

        proba = meta_pipe.predict_proba(X_meta_va)

        # align meta outputs too
        proba_full = np.zeros((len(va_idx), n_classes), dtype=float)
        fitted_meta = meta_pipe.named_steps["model"]
        for j, cls in enumerate(fitted_meta.classes_):
            proba_full[:, int(cls)] = proba[:, j]

        oof_meta_proba[va_idx] = proba_full

        m = multiclass_metrics(y_enc.iloc[va_idx].to_numpy(), proba_full)
        fold_rows.append({"fold": int(fold), "n_valid": int(len(va_idx)), **m})

        print(
            f"[stage2][meta fold {fold}] "
            + " ".join(f"{k}={m[k]:.4f}" for k in ["accuracy", "f1_macro", "logloss"]),
            flush=True,
        )

        if save_models:
            dump(meta_pipe, outdir / f"meta_multiclass_{meta_model_name}_pipe_fold_{fold}.joblib")

    np.save(outdir / "oof_meta_proba.npy", oof_meta_proba)
    np.save(outdir / "meta_features_oof.npy", meta_X)

    summary = {
        "stage": "stage2_multiclass",
        "n_folds": len(folds),
        "n_classes": int(n_classes),
        **summarize(fold_rows, ["accuracy", "f1_macro", "logloss"]),
    }
    return fold_rows, summary


# -----------------------------
# CLI entrypoint
# -----------------------------



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_stage1", required=True, help="Path to stage1 dataset (pit/no-pit)")
    ap.add_argument("--data_stage2", required=True, help="Path to stage2 dataset (pit-stop-only compound)")
    ap.add_argument("--outdir", default="runs/stack", help="Output directory root")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--save_models", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--meta_model_stage1",
        choices=["xgb"],
        default="xgb",
        help="Stage1 meta learner (fixed to xgb).",
    )
    ap.add_argument(
        "--meta_model_stage2",
        choices=["xgb"],
        default="xgb",
        help="Stage2 meta learner (fixed to xgb).",
    )
    ap.add_argument(
        "--only_stage",
        choices=["all", "stage1", "stage2"],
        default="all",
        help="Run stage1, stage2, or both",
    )
    args = ap.parse_args()

    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)

    # ---- Load datasets ----
    df1 = load_stage1_dataset(args.data_stage1)
    df2 = load_stage2_dataset(args.data_stage2, strict=True)

    X1, y1 = get_stage1_xy(df1)
    X2, y2 = get_stage2_xy(df2)

    # ---- Race-grouped folds (no leakage across races) ----
    fb1 = make_race_group_folds(df1, target_col="y_pit", n_splits=args.n_splits, seed=args.seed)
    folds1 = fb1.folds

    # Stage2: make sure target exists in df2 for balancing (encode first)
    df2_tmp = df2.copy()
    df2_tmp = df2_tmp.assign(y_compound_encoded=df2_tmp["y_compound"].map(COMPOUND_TO_INT))
    fb2 = make_race_group_folds(df2_tmp, target_col="y_compound_encoded", n_splits=args.n_splits, seed=args.seed)
    folds2 = fb2.folds

    vprint(args.verbose, f"[main] stage1: X1={X1.shape} y1={y1.shape} folds={len(folds1)}")
    vprint(args.verbose, f"[main] stage2: X2={X2.shape} y2={y2.shape} folds={len(folds2)}")

    # ---- Preprocessors per dataset ----
    num1, cat1 = infer_feature_types(X1)
    pre1 = build_preprocessor(num_cols=num1, cat_cols=cat1)

    num2, cat2 = infer_feature_types(X2)
    pre2 = build_preprocessor(num_cols=num2, cat_cols=cat2)

    cfg = ModelConfig(random_state=args.seed)

    meta_models_stage1 = [args.meta_model_stage1]  # always ["xgb"]

    # ---- Stage 1 ----
    if args.only_stage in {"all", "stage1"}:
        out1 = root / "stage1_binary"
        meta_rows, meta_summaries = run_binary_stacking_cv(
            df1, X1, y1,
            cfg=cfg,
            preprocessor=pre1,
            folds=folds1,
            threshold=args.threshold,
            outdir=out1,
            save_models=args.save_models,
            verbose=args.verbose,
            meta_model_names=meta_models_stage1,
        )
        (out1 / "fold_metrics_by_meta.json").write_text(json.dumps(meta_rows, indent=2))
        (out1 / "summary_by_meta.json").write_text(json.dumps(meta_summaries, indent=2))
        print("\nStage 1 summaries:\n", json.dumps(meta_summaries, indent=2))


    # ---- Stage 2 ----
    if args.only_stage in {"all", "stage2"}:
        out2 = root / "stage2_multiclass"
        n_classes = len(COMPOUND_CLASSES)

        fold2, sum2 = run_multiclass_stacking_cv(
            X2, y2,
            cfg=cfg,
            preprocessor=pre2,
            folds=folds2,
            meta_model_name=args.meta_model_stage2,
            n_classes=n_classes,
            outdir=out2,
            save_models=args.save_models,
            verbose=args.verbose,
        )
        (out2 / "fold_metrics.json").write_text(json.dumps(fold2, indent=2))
        (out2 / "summary.json").write_text(json.dumps(sum2, indent=2))
        (out2 / "classes.json").write_text(json.dumps({"classes": list(COMPOUND_CLASSES)}, indent=2))
        print("\nStage 2 summary:\n", json.dumps(sum2, indent=2))

if __name__ == "__main__":
    main()