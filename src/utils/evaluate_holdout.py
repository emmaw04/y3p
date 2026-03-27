
# note to self: this script is for evaluating the final model on the holdout races.
# it's been simplified to use hardcoded paths instead of command line arguments
# and the docstrings are just informal notes.

from __future__ import annotations
from typing import Dict, List, Tuple
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
import tensorflow as tf

from src.data.data import HOLDOUT_RACE_IDS

# note to self: hardcoded paths and the optimal threshold i found earlier
STAGE1_CSV = Path("data/processed/dataset1.csv")
ARTIFACTS_ROOT = Path("runs/final_run")
OUTDIR = Path("runs/holdout_run")
PIT_THRESHOLD = 0.3120

def _race_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    # note to self: calculate a bunch of metrics for a given race
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

    # note to self: auc scores need both classes to be present
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    return out

def _segments_from_bool_mask(x: np.ndarray, mask: np.ndarray) -> List[Tuple[float, float]]:
    # note to self: this finds contiguous true segments in a boolean mask for plotting
    x = np.asarray(x, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if not len(x):
        return []

    segs = []
    in_seg = False
    seg_start = 0.0
    for i, is_true in enumerate(mask):
        if is_true and not in_seg:
            in_seg = True
            seg_start = x[i]
        elif not is_true and in_seg:
            segs.append((seg_start, x[i - 1]))
            in_seg = False
    if in_seg:
        segs.append((seg_start, x[-1]))
    
    # note to self: make single point segments visible on the plot
    return [(a, min(100.0, b + 0.5)) if a == b else (a, b) for a, b in segs]

def _make_left_padded_sequences_from_rows(
    keys_df: pd.DataFrame, X_row: np.ndarray, window: int, add_timestep_mask: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    # note to self: builds the sequences needed for the rnn models
    X_row = np.asarray(X_row)
    N, F = X_row.shape
    
    if add_timestep_mask:
        X_seq = np.zeros((N, window, F + 1), dtype=np.float32)
    else:
        X_seq = np.zeros((N, window, F), dtype=np.float32)
        
    eff_len = np.zeros((N,), dtype=np.float32)
    group_indices = keys_df.groupby(["race_id", "driver_id"], sort=False).indices

    for _, idx in group_indices.items():
        idx = np.asarray(idx, dtype=int)
        for t, i in enumerate(idx):
            start = max(0, t - window + 1)
            take = idx[start : t + 1]
            L = len(take)
            
            if add_timestep_mask:
                X_seq[i, -L:, :F] = X_row[take, :]
                X_seq[i, -L:, F] = 1.0  # mask is 1 for real data, 0 for padding
            else:
                X_seq[i, -L:, :] = X_row[take, :]
                
            eff_len[i] = L
    return X_seq, eff_len

def _safe_predict_proba(pipe, X: pd.DataFrame) -> np.ndarray:
    # note to self: a wrapper for sklearn's predict_proba to make sure it returns the right shape
    proba = pipe.predict_proba(X)
    return np.asarray(proba, dtype=float)

def _keras_predict_proba(model: tf.keras.Model, X_np: np.ndarray) -> np.ndarray:
    # note to self: a wrapper for keras predict to get probabilities
    y = model.predict(X_np, batch_size=4096, verbose=0)
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        return y[:, 0].astype(float)
    if y.ndim == 2 and y.shape[1] >= 2:
        return y[:, 1].astype(float)
    if y.ndim == 1:
        return y.astype(float)
    raise ValueError(f"unexpected keras prediction shape: {y.shape}")

def main():
    #make sure the output directory exists
    OUTDIR.mkdir(parents=True, exist_ok=True)

    #load the main dataset and do some basic cleaning
    df1 = pd.read_csv(STAGE1_CSV)
    for c in ["race_id", "driver_id", "lapno"]:
        df1[c] = pd.to_numeric(df1[c], errors="coerce").astype("Int64")
    
    # note to self: clean up string columns, checking for type first
    for c in ["race_track", "current_compound", "fcy_status", "compound", "nextcompound"]:
        if c in df1.columns:
            df1[c] = df1[c].apply(lambda x: x.strip().upper() if isinstance(x, str) else x)

    # note to self: filter down to just the holdout races
    holdout_set = set(int(x) for x in HOLDOUT_RACE_IDS)
    dfh = df1[df1["race_id"].isin(list(holdout_set))].copy()
    dfh = dfh.sort_values(["race_id", "driver_id", "lapno"], kind="mergesort").reset_index(drop=True)
    if dfh.empty:
        raise ValueError(f"no holdout rows found in {STAGE1_CSV} for races {HOLDOUT_RACE_IDS}")

    # note to self: load all the model artifacts
    stage1_art = ARTIFACTS_ROOT / "stage1_binary" / "artifacts"
    x1_cols = json.loads((stage1_art / "feature_columns.json").read_text())["columns"]
    base_svm = load(stage1_art / "base_svm_pipeline.joblib")
    base_xgb = load(stage1_art / "base_xgb_pipeline.joblib")
    lstm_pre = load(stage1_art / "lstm_preprocessor.joblib")
    lstm_model = tf.keras.models.load_model(stage1_art / "lstm_model.keras", compile=False)
    tcn_gru_pre = load(stage1_art / "tcn_gru_preprocessor.joblib")
    tcn_gru_model = tf.keras.models.load_model(stage1_art / "tcn_gru_model.keras", compile=False)
    meta_pipe = load(stage1_art / "meta_pipeline.joblib")

    # note to self: prepare the raw feature set
    X_raw = dfh[x1_cols].copy()
    for c in ["is_wet_race", "is_raining", "minutes_rain"]:
        if c in X_raw.columns:
            X_raw[c] = pd.to_numeric(X_raw[c], errors="coerce").fillna(0.0)
        else:
            X_raw[c] = 0.0

    # note to self: generate predictions from all the base models
    p_svm = _safe_predict_proba(base_svm, X_raw)[:, 1]
    p_xgb = _safe_predict_proba(base_xgb, X_raw)[:, 1]
    keys_df = dfh[["race_id", "driver_id", "lapno"]]

    # note to self: generate predictions for the lstm model
    X_lstm_row = lstm_pre.transform(X_raw)
    if hasattr(X_lstm_row, "toarray"): X_lstm_row = X_lstm_row.toarray()
    expected_T_lstm, expected_F_lstm = lstm_model.input_shape[1], lstm_model.input_shape[2]
    X_seq_lstm, _ = _make_left_padded_sequences_from_rows(keys_df, X_lstm_row, window=expected_T_lstm, add_timestep_mask=True)
    p_lstm = _keras_predict_proba(lstm_model, X_seq_lstm)

    # note to self: generate predictions for the tcn_gru model
    X_tcn_gru_row = tcn_gru_pre.transform(X_raw)
    if hasattr(X_tcn_gru_row, "toarray"): X_tcn_gru_row = X_tcn_gru_row.toarray()
    expected_T_tcn_gru, expected_F_tcn_gru = tcn_gru_model.input_shape[1], tcn_gru_model.input_shape[2]
    X_seq_tcn_gru, tcn_effective_len = _make_left_padded_sequences_from_rows(keys_df, X_tcn_gru_row, window=expected_T_tcn_gru, add_timestep_mask=True)
    tcn_gru_proba = _keras_predict_proba(tcn_gru_model, X_seq_tcn_gru)

    # note to self: combine raw features and base model predictions for the meta model
    X_meta = X_raw.copy()
    X_meta["p_svm"] = p_svm
    X_meta["p_xgb"] = p_xgb
    X_meta["p_lstm"] = p_lstm
    X_meta["p_tcn_gru"] = tcn_gru_proba
    
    # note to self: get final predictions from the meta model
    proba = _safe_predict_proba(meta_pipe, X_meta)[:, 1]
    pit_pred = (proba >= PIT_THRESHOLD).astype(int)

    # note to self: build the final output dataframe
    out = dfh[["race_id", "driver_id", "lapno"]].copy()
    out["race_progress_pct"] = (pd.to_numeric(dfh["race_progress"], errors="coerce") * 100.0).fillna(0).astype(float)
    out["y_pit"] = pd.to_numeric(dfh["y_pit"], errors="coerce").fillna(0).astype(int)
    out["is_raining"] = pd.to_numeric(dfh.get("is_raining"), errors="coerce").fillna(0).astype(int)
    out["fcy_status"] = pd.to_numeric(dfh.get("fcy_status"), errors="coerce").fillna(0).astype(int)
    out["p_pit"] = proba.astype(float)
    out["pit_pred"] = pit_pred.astype(int)
    
    # note to self: save the lap-level predictions
    out_path = OUTDIR / "holdout_stage1_lap_probs.csv"
    out.to_csv(out_path, index=False)
    print(f"wrote: {out_path}")

    # note to self: now make the plots for each driver in each race
    plots_dir = OUTDIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for (rid, did), g in out.groupby(["race_id", "driver_id"], sort=False):
        g = g.sort_values("lapno", kind="mergesort")
        x = g["race_progress_pct"].to_numpy(dtype=float)
        y = g["p_pit"].to_numpy(dtype=float) * 100.0
        pit_x = g.loc[g["y_pit"] == 1, "race_progress_pct"].to_numpy(dtype=float)

        plt.figure(figsize=(10, 4))
        plt.plot(x, y, marker="o", linewidth=1)
        
        #add shaded backgrounds for rain and fcy
        rain_segs = _segments_from_bool_mask(x, g["is_raining"].to_numpy(dtype=int) == 1)
        fcy_segs = _segments_from_bool_mask(x, g["fcy_status"].to_numpy(dtype=int) != 0)
        for a, b in rain_segs: plt.axvspan(a, b, alpha=0.12, facecolor="blue", linewidth=0)
        for a, b in fcy_segs: plt.axvspan(a, b, alpha=0.12, facecolor="yellow", linewidth=0)

        plt.axhline(PIT_THRESHOLD * 100.0, linewidth=1, color='r', linestyle='--')
        for px in pit_x: plt.axvline(px, linewidth=1, color='g')

        plt.ylim(0, 100)
        plt.xlim(0, 100)
        plt.xlabel("race progress in %")
        plt.ylabel("predicted pit stop prob. in %")
        plt.title(f"race {int(rid)} | driver {int(did)}")
        
        race_dir = plots_dir / f"race_{int(rid)}"
        race_dir.mkdir(parents=True, exist_ok=True)
        fig_path = race_dir / f"driver_{int(did)}_pit_prob.png"

        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()

    # note to self: calculate and save metrics per race
    metric_rows = [
        {"race_id": int(rid), **_race_metrics(g["y_pit"].to_numpy(), g["p_pit"].to_numpy(), threshold=PIT_THRESHOLD)}
        for rid, g in out.groupby("race_id", sort=False)
    ]
    metrics_df = pd.DataFrame(metric_rows).sort_values("race_id")
    metrics_path = OUTDIR / "holdout_stage1_metrics_by_race.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"wrote: {metrics_path}")

if __name__ == "__main__":
    main()
