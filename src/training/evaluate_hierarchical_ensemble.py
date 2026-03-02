#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score

from src.data.data import (
    load_stage2_dataset,
    get_stage2_xy,
    make_race_group_folds,
    infer_feature_types,
    COMPOUND_CLASSES,
)
from src.data.preprocessing import build_preprocessor, PreprocessConfig, make_preprocessor_for_model
from src.models.models import (
    ModelConfig,
    build_model_pipeline,
    make_meta_binary_lr,            # logistic gate
    make_meta_multiclass_xgb,       # final meta
    make_rf_binary, make_svm_binary, make_xgb_binary,
    make_rf_multiclass, make_svm_multiclass, make_xgb_multiclass,
    make_vse_compound_ann,          # use for both dry (3-way) and wet (2-way) experts
)

# -----------------------------
# Class sets / encodings
# -----------------------------
DRY_CLASSES = ["SOFT", "MEDIUM", "HARD"]
WET_CLASSES = ["INTERMEDIATE", "WET"]

CLS_TO_INT: Dict[str, int] = {c: i for i, c in enumerate(COMPOUND_CLASSES)}
INT_TO_CLS: Dict[int, str] = {i: c for c, i in CLS_TO_INT.items()}

DRY_TO_LOCAL = {c: i for i, c in enumerate(DRY_CLASSES)}   # 0..2
WET_TO_LOCAL = {c: i for i, c in enumerate(WET_CLASSES)}   # 0..1


def vprint(verbose: bool, *args, **kwargs):
    if verbose:
        print(*args, **kwargs, flush=True)


def encode_y_global(y: pd.Series) -> np.ndarray:
    y_str = y.astype(str).str.strip().str.upper()
    y_enc = y_str.map(CLS_TO_INT)
    bad = y_str[y_enc.isna() & y.notna()].unique()
    if len(bad) > 0:
        raise ValueError(f"Unknown y_compound labels: {list(bad)}")
    return y_enc.astype(int).to_numpy()


def make_gate_y(y: pd.Series) -> np.ndarray:
    y_str = y.astype(str).str.strip().str.upper()
    return np.isin(y_str, WET_CLASSES).astype(int)


def make_local_y(y: pd.Series, classes: List[str], mapping: Dict[str, int]) -> np.ndarray:
    y_str = y.astype(str).str.strip().str.upper()
    out = np.full(len(y_str), -1, dtype=int)
    m = np.isin(y_str, classes)
    out[m] = y_str[m].map(mapping).astype(int).to_numpy()
    return out


# -----------------------------
# Metrics
# -----------------------------
def multiclass_metrics(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    y_pred = np.argmax(proba, axis=1)
    labels = np.arange(proba.shape[1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "logloss": float(log_loss(y_true, proba, labels=labels)),
    }


def summarize(rows: List[Dict[str, Any]], keys: List[str]) -> Dict[str, float]:
    return {f"{k}_mean": float(np.mean([r[k] for r in rows])) for k in keys}


# -----------------------------
# OOF feature builders
# -----------------------------
def _fit_predict_proba_pipe(
    X_tr: pd.DataFrame, y_tr: np.ndarray,
    X_va: pd.DataFrame,
    estimator,
    preprocessor,
    *,
    sample_weight: Optional[np.ndarray] = None
) -> np.ndarray:
    pipe = build_model_pipeline(clone(preprocessor), clone(estimator))
    if sample_weight is None:
        pipe.fit(X_tr, y_tr)
    else:
        pipe.fit(X_tr, y_tr, model__sample_weight=sample_weight)
    return pipe.predict_proba(X_va)


def collect_oof_gate_lr(
    X: pd.DataFrame,
    y_gate: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    *,
    cfg_gate: ModelConfig,
    verbose: bool = False,
) -> np.ndarray:
    """
    OOF vector p(wet-group=1 | x) from Logistic Regression gate.
    """
    n = len(y_gate)
    oof_pwet = np.zeros(n, dtype=float)

    num_cols, cat_cols = infer_feature_types(X)
    gate_pre = build_preprocessor(
        num_cols=num_cols,
        cat_cols=cat_cols,
        cfg=PreprocessConfig(scale_numeric=True, sparse_onehot=True),
    )
    gate_est = make_meta_binary_lr(cfg_gate)  # logistic regression

    for fold, (tr_idx, va_idx) in enumerate(folds):
        X_tr, y_tr = X.iloc[tr_idx], y_gate[tr_idx]
        X_va = X.iloc[va_idx]

        # degenerate fold fallback
        if len(np.unique(y_tr)) < 2:
            const_p = float(np.mean(y_tr))
            oof_pwet[va_idx] = const_p
            vprint(verbose, f"[gate][fold {fold}] degenerate -> const_p={const_p:.3f}")
            continue

        # optional: explicit weighting (in addition to class_weight="balanced" if you set it)
        neg = int(np.sum(y_tr == 0))
        pos = int(np.sum(y_tr == 1))
        w_pos = (neg / max(pos, 1))
        sw = np.where(y_tr == 1, w_pos, 1.0).astype(float)

        proba = _fit_predict_proba_pipe(X_tr, y_tr, X_va, gate_est, gate_pre, sample_weight=sw)
        oof_pwet[va_idx] = proba[:, 1]
        vprint(verbose, f"[gate][fold {fold}] mean(pwet)={float(np.mean(proba[:,1])):.4f}")

    return oof_pwet


def _pick_preprocessor_for_base(name: str, X: pd.DataFrame):
    num_cols, cat_cols = infer_feature_types(X)
    # mirror your evaluate_stack logic
    if name in {"ann", "vse_compound_ann"}:
        return make_preprocessor_for_model("ann", num_cols=num_cols, cat_cols=cat_cols)  # dense
    if name == "svm":
        return make_preprocessor_for_model("svm", num_cols=num_cols, cat_cols=cat_cols)
    if name == "rf":
        return make_preprocessor_for_model("rf", num_cols=num_cols, cat_cols=cat_cols)
    if name == "xgb":
        return make_preprocessor_for_model("xgb", num_cols=num_cols, cat_cols=cat_cols)
    return make_preprocessor_for_model("xgb", num_cols=num_cols, cat_cols=cat_cols)


def collect_oof_expert_multiclass(
    X: pd.DataFrame,
    y_local: np.ndarray,                 # -1 outside expert domain
    folds: List[Tuple[np.ndarray, np.ndarray]],
    base_models: Dict[str, Any],
    *,
    n_classes: int,
    verbose: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """
    Returns meta_X: (N, M*n_classes) where each block is a prob vector from a base.
    Bases are trained only on in-domain rows in the TRAIN split, but predicted on all VAL rows.
    """
    n = len(y_local)
    names = list(base_models.keys())
    meta_X = np.zeros((n, len(names) * n_classes), dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(folds):
        tr_mask = (y_local[tr_idx] >= 0)
        X_va = X.iloc[va_idx]

        if int(np.sum(tr_mask)) == 0:
            # no in-domain training data
            uniform = np.full((len(va_idx), n_classes), 1.0 / n_classes, dtype=float)
            for j in range(len(names)):
                meta_X[va_idx, j*n_classes:(j+1)*n_classes] = uniform
            vprint(verbose, f"[expert {n_classes}c][fold {fold}] no train data -> uniform")
            continue

        X_tr = X.iloc[tr_idx[tr_mask]]
        y_tr = y_local[tr_idx[tr_mask]]

        # degenerate in-domain labels
        if len(np.unique(y_tr)) < 2:
            only = int(np.unique(y_tr)[0])
            const = np.zeros((len(va_idx), n_classes), dtype=float)
            const[:, only] = 1.0
            for j in range(len(names)):
                meta_X[va_idx, j*n_classes:(j+1)*n_classes] = const
            vprint(verbose, f"[expert {n_classes}c][fold {fold}] degenerate -> class={only}")
            continue

        for j, name in enumerate(names):
            est = clone(base_models[name])
            pre = _pick_preprocessor_for_base(name, X)
            proba = _fit_predict_proba_pipe(X_tr, y_tr, X_va, est, pre)

            # align to full local 0..K-1
            proba_full = np.zeros((len(va_idx), n_classes), dtype=float)
            fitted = build_model_pipeline(pre, est)  # dummy only to get attr? don't do this
            # Instead, rely on estimator's classes_ after fit via pipe:
            pipe = build_model_pipeline(pre, clone(base_models[name]))
            pipe.fit(X_tr, y_tr)
            proba = pipe.predict_proba(X_va)

            fitted_model = pipe.named_steps["model"]
            classes_seen = getattr(fitted_model, "classes_", None)
            if classes_seen is None:
                raise RuntimeError(f"Could not determine classes_ for base '{name}'")

            for jj, cls in enumerate(classes_seen):
                proba_full[:, int(cls)] = proba[:, jj]

            meta_X[va_idx, j*n_classes:(j+1)*n_classes] = proba_full

        vprint(verbose, f"[expert {n_classes}c][fold {fold}] done")

    return meta_X, names


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_stage2", required=True)
    ap.add_argument("--outdir", default="runs/hierarchical_gate_lr")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_models", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no_ann", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = ModelConfig(random_state=args.seed)

    # Gate config: turn on balanced weighting (recommended for ~8% wet-group)
    cfg_gate = ModelConfig(random_state=args.seed, n_jobs=cfg.n_jobs, use_class_weight=True)

    # Load stage2
    df2 = load_stage2_dataset(args.data_stage2, strict=True)
    X2, y2 = get_stage2_xy(df2)
    y_global = encode_y_global(y2)
    y_gate = make_gate_y(y2)
    y_dry_local = make_local_y(y2, DRY_CLASSES, DRY_TO_LOCAL)
    y_wet_local = make_local_y(y2, WET_CLASSES, WET_TO_LOCAL)

    # folds (stratify on global)
    df2_tmp = df2.copy()
    df2_tmp["y_compound_encoded"] = y_global
    fb = make_race_group_folds(df2_tmp, target_col="y_compound_encoded", n_splits=args.n_splits, seed=args.seed)
    folds = fb.folds

    vprint(args.verbose, f"[data] N={len(X2)} dry={(y_dry_local>=0).sum()} wet={(y_wet_local>=0).sum()}")

    # -------------------------
    # Build base learners (2 experts × 4)
    # -------------------------
    # DRY expert (3-way)
    dry_bases: Dict[str, Any] = {
        "rf": make_rf_multiclass(cfg),
        "svm": make_svm_multiclass(cfg),
        "xgb": make_xgb_multiclass(cfg, n_classes=3),
    }
    if not args.no_ann:
        dry_bases["vse_compound_ann"] = make_vse_compound_ann(cfg, n_classes=3)

    # WET expert (2-way) -> use binary models + VSE softmax with 2 classes
    wet_bases: Dict[str, Any] = {
        "rf": make_rf_binary(cfg),
        "svm": make_svm_binary(cfg),
        "xgb": make_xgb_binary(cfg),
    }
    if not args.no_ann:
        wet_bases["vse_compound_ann"] = make_vse_compound_ann(cfg, n_classes=2)

    vprint(args.verbose, f"[models] dry bases={list(dry_bases.keys())}")
    vprint(args.verbose, f"[models] wet bases={list(wet_bases.keys())}")

    # -------------------------
    # 1) OOF features
    # -------------------------
    vprint(args.verbose, "[step] OOF gate pwet via LogisticRegression...")
    oof_gate_pwet = collect_oof_gate_lr(X2, y_gate, folds, cfg_gate=cfg_gate, verbose=args.verbose)

    vprint(args.verbose, "[step] OOF dry expert base prob vectors...")
    oof_dry_X, dry_names = collect_oof_expert_multiclass(
        X2, y_dry_local, folds, dry_bases, n_classes=3, verbose=args.verbose
    )

    vprint(args.verbose, "[step] OOF wet expert base prob vectors...")
    oof_wet_X, wet_names = collect_oof_expert_multiclass(
        X2, y_wet_local, folds, wet_bases, n_classes=2, verbose=args.verbose
    )

    # Build final meta dataframe: [gate] + [dry probs] + [wet probs] + [context]
    def df_from_blocks():
        cols = ["gate_pwet"]
        blocks = [oof_gate_pwet.reshape(-1, 1)]

        # dry cols
        for name in dry_names:
            for k in range(3):
                cols.append(f"dry_p_{name}_c{k}")
        blocks.append(oof_dry_X)

        # wet cols
        for name in wet_names:
            for k in range(2):
                cols.append(f"wet_p_{name}_c{k}")
        blocks.append(oof_wet_X)

        X_probs = np.hstack(blocks)
        df_probs = pd.DataFrame(X_probs, columns=cols)
        return pd.concat([df_probs.reset_index(drop=True), X2.reset_index(drop=True)], axis=1)

    X_meta = df_from_blocks()

    (outdir / "meta_columns.json").write_text(json.dumps({"columns": list(X_meta.columns)}, indent=2))
    np.save(outdir / "oof_gate_pwet.npy", oof_gate_pwet)
    np.save(outdir / "oof_dry_probs.npy", oof_dry_X)
    np.save(outdir / "oof_wet_probs.npy", oof_wet_X)
    X_meta.to_parquet(outdir / "meta_features_oof.parquet", index=False)

    # -------------------------
    # 2) Final meta learner CV (XGB 5-way)
    # -------------------------
    final_meta = make_meta_multiclass_xgb(cfg, n_classes=len(COMPOUND_CLASSES))

    fold_rows: List[Dict[str, Any]] = []
    oof_final_proba = np.zeros((len(X2), len(COMPOUND_CLASSES)), dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(folds):
        X_tr, y_tr = X_meta.iloc[tr_idx], y_global[tr_idx]
        X_va, y_va = X_meta.iloc[va_idx], y_global[va_idx]

        num_m, cat_m = infer_feature_types(X_meta)
        meta_pre = build_preprocessor(
            num_cols=num_m,
            cat_cols=cat_m,
            cfg=PreprocessConfig(scale_numeric=False, sparse_onehot=True),
        )

        meta_pipe = build_model_pipeline(meta_pre, clone(final_meta))
        meta_pipe.fit(X_tr, y_tr)

        proba = meta_pipe.predict_proba(X_va)

        # align to all 5 classes
        proba_full = np.zeros((len(va_idx), len(COMPOUND_CLASSES)), dtype=float)
        fitted = meta_pipe.named_steps["model"]
        for j, cls in enumerate(fitted.classes_):
            proba_full[:, int(cls)] = proba[:, j]

        oof_final_proba[va_idx] = proba_full

        m = multiclass_metrics(y_va, proba_full)
        fold_rows.append({"fold": int(fold), "n_valid": int(len(va_idx)), **m})

        print(
            f"[final][fold {fold}] "
            + " ".join(f"{k}={m[k]:.4f}" for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "logloss"]),
            flush=True,
        )

        if args.save_models:
            dump(meta_pipe, outdir / f"final_meta_pipe_fold_{fold}.joblib")

    summary = {
        "model": "gate_lr + (dry4,wET4) -> final_xgb",
        "n_splits": int(len(folds)),
        "classes": list(COMPOUND_CLASSES),
        **summarize(fold_rows, ["accuracy", "precision_macro", "recall_macro", "f1_macro", "logloss"]),
    }

    (outdir / "fold_metrics.json").write_text(json.dumps(fold_rows, indent=2))
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    np.save(outdir / "oof_final_proba.npy", oof_final_proba)

    print("\nSummary:\n", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()