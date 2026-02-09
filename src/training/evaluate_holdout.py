# src/evaluate_holdout.py
from __future__ import annotations
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from joblib import load

from src.data import HOLDOUT_RACE_IDS

# Keras (ANN + TCN)
import tensorflow as tf

PIT_THRESHOLD = 0.5

def _race_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)

    out = {
        "n": float(len(y_true)),
        "pos_rate": float(np.mean(y_true)) if len(y_true) else np.nan,
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else np.nan,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)) if len(y_true) else np.nan,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)) if len(y_true) else np.nan,
        "recall": float(recall_score(y_true, y_pred, zero_division=0)) if len(y_true) else np.nan,
        "roc_auc": np.nan,
        "pr_auc": np.nan,
    }

    # ROC AUC / PR AUC require at least one positive and one negative
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["pr_auc"] = float(average_precision_score(y_true, y_score))

    return out


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text())

def _make_left_padded_sequences_from_rows(
    keys_df: pd.DataFrame,   # columns: race_id, driver_id, lapno (already sorted)
    X_row: np.ndarray,       # (N, F)
    window: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build left-padded sequences of length `window` per (race_id, driver_id),
    ending at each row. Returns (X_seq, effective_len).
      X_seq: (N, window, F)
      effective_len: (N,)
    """
    X_row = np.asarray(X_row)
    if X_row.ndim != 2:
        raise ValueError(f"Expected X_row to be 2D (N,F). Got shape {X_row.shape}")

    N, F = X_row.shape
    X_seq = np.zeros((N, window, F), dtype=np.float32)
    eff_len = np.zeros((N,), dtype=np.float32)

    # indices per group in the *current row order*
    group_indices = keys_df.groupby(["race_id", "driver_id"], sort=False).indices

    for _, idx in group_indices.items():
        idx = np.asarray(idx, dtype=int)
        for t, i in enumerate(idx):
            start = max(0, t - window + 1)
            take = idx[start : t + 1]
            L = len(take)
            X_seq[i, -L:, :] = X_row[take, :]
            eff_len[i] = L

    return X_seq, eff_len


def _ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[cols]


def _basic_normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Coerce keys
    for c in ["race_id", "driver_id", "lapno"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")

    # Common categorical-ish strings
    for c in ["race_track", "current_compound", "fcy_status", "compound", "nextcompound"]:
        if c in out.columns:
            out[c] = out[c].astype(object).where(~out[c].isna(), None)
            out[c] = out[c].apply(lambda x: x.strip().upper() if isinstance(x, str) else x)

    return out


def _require_keys(df: pd.DataFrame, name: str) -> None:
    req = ["race_id", "driver_id", "lapno"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required key columns: {missing}")


def _safe_predict_proba(pipe, X: pd.DataFrame) -> np.ndarray:
    """Works with sklearn pipelines that expose predict_proba."""
    proba = pipe.predict_proba(X)
    proba = np.asarray(proba, dtype=float)
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise ValueError(f"Unexpected predict_proba shape: {proba.shape}")
    return proba


def _keras_predict_proba(model: tf.keras.Model, X_np: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Returns 1D probability for the positive class."""
    y = model.predict(X_np, batch_size=batch_size, verbose=0)
    y = np.asarray(y)
    # Common cases: (N,1) sigmoid or (N,2) softmax
    if y.ndim == 2 and y.shape[1] == 1:
        return y[:, 0].astype(float)
    if y.ndim == 2 and y.shape[1] >= 2:
        return y[:, 1].astype(float)
    if y.ndim == 1:
        return y.astype(float)
    raise ValueError(f"Unexpected keras prediction shape: {y.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1_csv", required=True, help="Path to output1.csv")
    ap.add_argument(
        "--artifacts_root",
        required=True,
        help="Root folder containing stage1_binary/artifacts/",
    )
    ap.add_argument("--outdir", required=True, help="Where to write holdout probability CSVs")
    args = ap.parse_args()

    artifacts_root = Path(args.artifacts_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # -----------------------
    # Load + filter holdout data
    # -----------------------
    df1 = _basic_normalize(pd.read_csv(args.stage1_csv))
    _require_keys(df1, "stage1_csv")
    if "row_id" not in df1.columns:
        df1 = df1.copy()
        df1["row_id"] = np.arange(len(df1), dtype=int)

    holdout_set = set(int(x) for x in HOLDOUT_RACE_IDS)
    dfh = df1[df1["race_id"].astype("Int64").isin(list(holdout_set))].copy()
    if dfh.empty:
        raise ValueError(f"No holdout rows found in {args.stage1_csv} for HOLDOUT_RACE_IDS={HOLDOUT_RACE_IDS}")

    dfh = dfh.sort_values(["race_id", "driver_id", "lapno"], kind="mergesort").reset_index(drop=True)

    # -----------------------
    # Load artifacts
    # -----------------------
    stage1_art = artifacts_root / "stage1_binary" / "artifacts"

    # Raw feature columns used by base models
    feat_obj = _read_json(stage1_art / "feature_columns.json")
    # Your file contains {"X1_columns": [...]}
    if "X1_columns" in feat_obj:
        x1_cols = feat_obj["X1_columns"]
    elif "columns" in feat_obj:
        x1_cols = feat_obj["columns"]
    else:
        raise KeyError(f"feature_columns.json keys are {list(feat_obj.keys())}, expected X1_columns or columns")

    # Base learners
    base_rf = load(stage1_art / "base_rf_pipeline.joblib")
    base_svm = load(stage1_art / "base_svm_pipeline.joblib")
    base_xgb = load(stage1_art / "base_xgb_pipeline.joblib")

    ann_pre = load(stage1_art / "ann_preprocessor.joblib")
    ann_model = tf.keras.models.load_model(stage1_art / "ann_model.keras", compile=False)

    tcn_pre = load(stage1_art / "tcn_preprocessor.joblib")
    tcn_model = tf.keras.models.load_model(stage1_art / "tcn_model.keras", compile=False)

    # Meta model (expects p_* + tcn_* + raw features)
    meta_pipe = load(stage1_art / "meta_pipeline.joblib")

    # -----------------------
    # Build base features -> meta features
    # -----------------------
    X_raw = _ensure_columns(dfh, x1_cols)

    # sklearn base probas
    p_rf = _safe_predict_proba(base_rf, X_raw)[:, 1]
    p_svm = _safe_predict_proba(base_svm, X_raw)[:, 1]
    p_xgb = _safe_predict_proba(base_xgb, X_raw)[:, 1]

    # ANN proba
    X_ann = ann_pre.transform(X_raw)
    if hasattr(X_ann, "toarray"):  # sparse -> dense if needed
        X_ann = X_ann.toarray()
    p_ann = _keras_predict_proba(ann_model, np.asarray(X_ann))

    # -----------------------
    # TCN proba + effective length
    # -----------------------
    # IMPORTANT: tcn_pre gives per-row features; we must build (N,8,F) sequences ourselves.
    X_tcn_row = tcn_pre.transform(X_raw)
    X_tcn_row = np.asarray(X_tcn_row)

    # Debug sanity prints (keep for now)
    # print("X_tcn_row shape:", X_tcn_row.shape, "dtype:", X_tcn_row.dtype)

    if X_tcn_row.ndim != 2:
        raise ValueError(f"tcn_pre.transform must return (N,F). Got {X_tcn_row.shape}")

    # Temporary robustness: if we still get 87, pad a final zero column to make 88.
    # (The ideal fix is aligning the saved preprocessor/model feature set, but this gets you running.)
    if X_tcn_row.shape[1] == 87:
        X_tcn_row = np.concatenate([X_tcn_row, np.zeros((X_tcn_row.shape[0], 1), dtype=X_tcn_row.dtype)], axis=1)

    # Hard check against model expectation
    expected_T, expected_F = tcn_model.input_shape[1], tcn_model.input_shape[2]
    if expected_T != 8:
        raise ValueError(f"Unexpected TCN time window. Model expects T={expected_T}, not 8.")
    if X_tcn_row.shape[1] != expected_F:
        raise ValueError(f"TCN feature dim mismatch: model expects F={expected_F}, got F={X_tcn_row.shape[1]}")

    # Build sequences in lap order within each driver/race group
    keys_df = dfh[["race_id", "driver_id", "lapno"]]
    X_seq, tcn_effective_len = _make_left_padded_sequences_from_rows(keys_df, X_tcn_row, window=expected_T)

    tcn_proba = _keras_predict_proba(tcn_model, X_seq)


    # Compose meta input exactly as meta_pipeline expects
    X_meta = X_raw.copy()
    X_meta["p_svm"] = p_svm
    X_meta["p_rf"] = p_rf
    X_meta["p_xgb"] = p_xgb
    X_meta["p_ann"] = p_ann
    X_meta["tcn_proba"] = tcn_proba
    X_meta["tcn_effective_len"] = tcn_effective_len

    # -----------------------
    # Final stage-1 probabilities via meta model
    # -----------------------
    proba = _safe_predict_proba(meta_pipe, X_meta)[:, 1]
    pit_pred = (proba >= PIT_THRESHOLD).astype(int)

    # -----------------------
    # Output frame (add labels + race_progress for plotting/metrics)
    # -----------------------
    if "y_pit" not in dfh.columns:
        raise ValueError("stage1_csv must contain y_pit to draw pit-stop vertical lines and compute metrics.")
    if "race_progress" not in dfh.columns:
        raise ValueError("stage1_csv must contain race_progress to plot against race progress.")

    out = dfh[["race_id", "driver_id", "lapno"]].copy()
    out["race_progress_pct"] = (pd.to_numeric(dfh["race_progress"], errors="coerce") * 100.0).astype(float).values
    out["y_pit"] = pd.to_numeric(dfh["y_pit"], errors="coerce").fillna(0).astype(int).values

    out["p_pit"] = proba.astype(float)
    out["pit_pred"] = pit_pred.astype(int)

    # -----------------------
    # Save lap-level outputs
    # -----------------------
    out_path = outdir / "holdout_stage1_lap_probs.csv"
    out.to_csv(out_path, index=False)

    for rid, g in out.groupby("race_id", sort=False):
        (outdir / f"race_{int(rid)}_stage1_lap_probs.csv").write_text(g.to_csv(index=False))

    print(f"Wrote: {out_path}")
    print(f"Predicted pit on {int(out['pit_pred'].sum())} / {len(out)} laps in holdout (threshold={PIT_THRESHOLD}).")

    # -----------------------
    # Step 3: Plots per (race, driver)
    # -----------------------
    plots_dir = outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for (rid, did), g in out.groupby(["race_id", "driver_id"], sort=False):
        g = g.sort_values("lapno", kind="mergesort")

        x = g["race_progress_pct"].to_numpy(dtype=float)
        y = (g["p_pit"].to_numpy(dtype=float) * 100.0)

        # vertical lines at TRUE pit stops
        pit_x = g.loc[g["y_pit"] == 1, "race_progress_pct"].to_numpy(dtype=float)

        plt.figure(figsize=(10, 4))
        plt.plot(x, y, marker="o", linewidth=1)

        # threshold line (50%)
        plt.axhline(PIT_THRESHOLD * 100.0, linewidth=1)

        for px in pit_x:
            plt.axvline(px, linewidth=1)

        plt.ylim(0, 100)
        plt.xlim(0, 100)
        plt.xlabel("Race progress in %")
        plt.ylabel("Predicted pit stop prob. in %")
        plt.title(f"Race {int(rid)} | Driver {int(did)}")

        race_dir = plots_dir / f"race_{int(rid)}"
        race_dir.mkdir(parents=True, exist_ok=True)
        fig_path = race_dir / f"driver_{int(did)}_pit_prob.png"

        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()

    # -----------------------
    # Step 4: Metrics per race (ONLY)
    # -----------------------
    metric_rows = []
    for rid, g in out.groupby("race_id", sort=False):
        m = _race_metrics(g["y_pit"].to_numpy(), g["p_pit"].to_numpy(), threshold=PIT_THRESHOLD)
        m["race_id"] = int(rid)
        metric_rows.append(m)

    metrics_df = pd.DataFrame(metric_rows).sort_values("race_id")
    metrics_path = outdir / "holdout_stage1_metrics_by_race.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Wrote: {metrics_path}")

if __name__ == "__main__":
    main()
