#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.base import clone
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.data import load_stage1_dataset, make_race_group_folds
from data.preprocessing import build_preprocessor, PreprocessConfig
from src.models.models import ModelConfig, make_meta_binary_mlp


def binary_metrics(y_true: np.ndarray, proba_pos: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (proba_pos >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, proba_pos)),
        "pr_auc": float(average_precision_score(y_true, proba_pos)),
        "logloss": float(log_loss(y_true, proba_pos, labels=[0, 1])),
    }


def summarize(rows: List[Dict[str, Any]], keys: List[str]) -> Dict[str, float]:
    return {f"{k}_mean": float(np.mean([r[k] for r in rows])) for k in keys}


def make_row_id(df: pd.DataFrame) -> pd.Series:
    # Matches your parquet format: "raceid_driverid_lapno"
    return df["race_id"].astype(str) + "_" + df["driver_id"].astype(str) + "_" + df["lapno"].astype(str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1_csv", required=True, help="Path to stage1 dataset (must include race_id,driver_id,lapno,y_pit)")
    ap.add_argument("--meta_parquet", required=True, help="Path to meta_features_oof.parquet (must include row_id)")
    ap.add_argument("--outdir", default="runs/meta_mlp_only", help="Where to save results/models")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--svd_components", type=int, default=256)
    ap.add_argument("--drop_cols", default="race_track", help="Comma-separated meta feature columns to drop (default: race_track)")
    ap.add_argument("--save_models", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Load meta features (probs + context) ----
    X_meta = pd.read_parquet(args.meta_parquet)
    if "row_id" not in X_meta.columns:
        raise ValueError("meta_parquet must include 'row_id' (e.g., 'race_driver_lap').")

    # Ensure row_id is string (parquet sometimes stores as categorical)
    X_meta["row_id"] = X_meta["row_id"].astype(str)

    # ---- Load stage1 df (labels + folds) ----
    df1 = load_stage1_dataset(args.stage1_csv)
    required = ["race_id", "driver_id", "lapno", "y_pit"]
    missing = [c for c in required if c not in df1.columns]
    if missing:
        raise ValueError(f"stage1 dataset missing required columns: {missing}")

    df1 = df1.copy()
    df1["row_id"] = make_row_id(df1)
    df1["row_id"] = df1["row_id"].astype(str)

    # ---- Join labels onto meta features using row_id ----
    y_df = df1[["row_id", "y_pit"]].copy()
    merged = X_meta.merge(y_df, on="row_id", how="inner")

    if len(merged) != len(X_meta):
        # helpful diagnostics
        meta_ids = set(X_meta["row_id"])
        csv_ids = set(df1["row_id"])
        only_in_meta = list(meta_ids - csv_ids)[:5]
        only_in_csv = list(csv_ids - meta_ids)[:5]
        raise ValueError(
            f"row_id merge mismatch: meta rows={len(X_meta)} merged rows={len(merged)}.\n"
            f"Examples only in meta: {only_in_meta}\n"
            f"Examples only in csv:  {only_in_csv}\n"
            f"Likely your meta parquet was produced from a different stage1 dataset or filtered differently."
        )

    y = merged["y_pit"].astype(int)
    X = merged.drop(columns=["y_pit"])

    # Drop high-cardinality / unwanted columns
    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]
    for c in drop_cols:
        if c in X.columns:
            X = X.drop(columns=[c])

    # Never feed identifiers into the model
    if "row_id" in X.columns:
        X_model = X.drop(columns=["row_id"])
    else:
        X_model = X

    # ---- Rebuild the same race-grouped folds on df1 ----
    fb = make_race_group_folds(df1, target_col="y_pit", n_splits=args.n_splits, seed=args.seed)
    folds_df1: List[Tuple[np.ndarray, np.ndarray]] = fb.folds

    # Map df1 rows -> merged order via row_id
    merged_row_ids = merged["row_id"].astype(str).tolist()
    rowid_to_idx = {rid: i for i, rid in enumerate(merged_row_ids)}

    df1_row_ids = df1["row_id"].astype(str).tolist()

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for tr_idx_df1, va_idx_df1 in folds_df1:
        tr_ids = [df1_row_ids[i] for i in tr_idx_df1]
        va_ids = [df1_row_ids[i] for i in va_idx_df1]

        tr_idx = np.array([rowid_to_idx[r] for r in tr_ids if r in rowid_to_idx], dtype=int)
        va_idx = np.array([rowid_to_idx[r] for r in va_ids if r in rowid_to_idx], dtype=int)

        folds.append((tr_idx, va_idx))

    # ---- Preprocess: sparse one-hot -> SVD -> scale -> MLP ----
    num_cols = [c for c in X_model.columns if pd.api.types.is_numeric_dtype(X_model[c])]
    cat_cols = [c for c in X_model.columns if not pd.api.types.is_numeric_dtype(X_model[c])]

    pre = build_preprocessor(
        num_cols=num_cols,
        cat_cols=cat_cols,
        cfg=PreprocessConfig(scale_numeric=True, sparse_onehot=True),
    )

    cfg = ModelConfig(random_state=args.seed)
    mlp = make_meta_binary_mlp(cfg)

    pipe = Pipeline([
        ("pre", pre),
        ("svd", TruncatedSVD(n_components=args.svd_components, random_state=args.seed)),
        ("scale", StandardScaler()),
        ("model", mlp),
    ])

    fold_rows: List[Dict[str, Any]] = []
    oof_proba = np.zeros(len(X_model), dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(folds):
        X_tr, y_tr = X_model.iloc[tr_idx], y.iloc[tr_idx].to_numpy()
        X_va, y_va = X_model.iloc[va_idx], y.iloc[va_idx].to_numpy()

        est = clone(pipe)
        est.fit(X_tr, y_tr)

        proba_pos = est.predict_proba(X_va)[:, 1]
        oof_proba[va_idx] = proba_pos

        m = binary_metrics(y_va, proba_pos, threshold=args.threshold)
        fold_rows.append({"fold": int(fold), "n_valid": int(len(va_idx)), **m})

        print(f"[mlp-only][fold {fold}] " + " ".join(
            f"{k}={m[k]:.4f}" for k in ["accuracy", "f1", "roc_auc", "pr_auc", "logloss"]
        ))

        if args.save_models:
            dump(est, outdir / f"meta_mlp_pipe_fold_{fold}.joblib")

    np.save(outdir / "oof_meta_mlp_proba_pos.npy", oof_proba)

    summary = {
        "meta_model": "mlp",
        "n_folds": args.n_splits,
        "svd_components": args.svd_components,
        "threshold": args.threshold,
        **summarize(fold_rows, ["accuracy", "f1", "roc_auc", "pr_auc", "logloss"]),
    }

    (outdir / "fold_metrics.json").write_text(pd.DataFrame(fold_rows).to_json(orient="records", indent=2))
    (outdir / "summary.json").write_text(pd.Series(summary).to_json(indent=2))

    print("\nSummary:")
    print(summary)


if __name__ == "__main__":
    main()
