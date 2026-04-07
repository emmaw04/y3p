# this script evaluates the final stage 1 model on the holdout races.
# it writes lap-level probabilities, predicted positive events, per-driver plots,
# and per-race metrics.

from typing import List
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
from src.data.data import build_feature_sequences

HOLDOUT_RACE_IDS = [2, 24, 53, 73, 75]  # races being evaluated

# hardcoded paths and threshold
STAGE1_CSV = Path("data/processed/dataset1.csv")
ARTIFACTS_ROOT = Path("runs/final_run")
OUTDIR = Path("runs/holdout_run")
PIT_THRESHOLD = 0.264


def _race_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float):
    """
    calculate metrics for one race
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)  # convert probabilities to hard predictions

    out = {
        "n": float(len(y_true)),
        "pos_rate": float(np.mean(y_true)) if len(y_true) else np.nan,
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else np.nan,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)) if len(y_true) else np.nan,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)) if len(y_true) else np.nan,
        "recall": float(recall_score(y_true, y_pred, zero_division=0)) if len(y_true) else np.nan,
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
    }
    return out


def _segments_from_bool_mask(x: np.ndarray, mask: np.ndarray):
    # find contiguous true segments in a boolean mask for background shading (rain, fcy phases) in plots
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

    # make single-point segments visible by widening them by 0.5 (so if it rains for a single lap we can still see)
    return [(a, min(100.0, b + 0.5)) if a == b else (a, b) for a, b in segs]


def _keras_predict_proba(model: tf.keras.Model, X_np: np.ndarray):
    # wrapper around keras predict for binary probabilities
    y = model.predict(X_np, batch_size=4096, verbose=0)
    y = np.asarray(y)

    if y.ndim == 2 and y.shape[1] == 1:
        return y[:, 0].astype(float)
    if y.ndim == 2 and y.shape[1] >= 2:
        return y[:, 1].astype(float)
    if y.ndim == 1:
        return y.astype(float)

    raise ValueError(f"unexpected keras prediction shape: {y.shape}")


def _to_dense(x):
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)


def _prepare_raw_features(df_part: pd.DataFrame, x1_cols: List[str]):
    X_raw = df_part[x1_cols].copy()

    for c in ["is_wet_race", "is_raining", "minutes_rain"]:
        if c in X_raw.columns:
            X_raw[c] = pd.to_numeric(X_raw[c], errors="coerce").fillna(0.0)
        else:
            X_raw[c] = 0.0

    return X_raw


def _predict_sequence_model_from_builder(
    model: tf.keras.Model,
    keys_df: pd.DataFrame,
    X_row: np.ndarray,
    window: int,
    add_timestep_mask: bool = True,
):
    """
    builds padded sequences using build_feature_sequences and maps the
    predicted probabilities back to the original row order
    """
    dummy_y = pd.Series(np.zeros(len(keys_df), dtype=int))

    X_seq, _, idx_last, _, _ = build_feature_sequences(
        keys=keys_df,
        X_all=X_row,
        y_all=dummy_y,
        seq_len=window,
        pad_left=True,
        add_timestep_mask=add_timestep_mask,
    )

    p_seq = _keras_predict_proba(model, X_seq)

    p_full = np.full(len(keys_df), np.nan, dtype=float)
    p_full[idx_last] = p_seq
    return p_full


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # load full dataset
    df1 = pd.read_csv(STAGE1_CSV)

    for c in ["race_id", "driver_id", "lapno"]:
        df1[c] = pd.to_numeric(df1[c], errors="coerce").astype("Int64")

    # clean string columns
    for c in ["race_track", "current_compound", "fcy_status", "compound", "nextcompound"]:
        if c in df1.columns:
            df1[c] = df1[c].apply(lambda x: x.strip().upper() if isinstance(x, str) else x)

    # keep only holdout races
    holdout_set = set(int(x) for x in HOLDOUT_RACE_IDS)
    dfh = df1[df1["race_id"].isin(list(holdout_set))].copy()
    dfh = dfh.sort_values(["race_id", "driver_id", "lapno"], kind="mergesort").reset_index(drop=True)

    if dfh.empty:
        raise ValueError(f"no holdout rows found in {STAGE1_CSV} for races {HOLDOUT_RACE_IDS}")

    # load model artifacts
    stage1_art = ARTIFACTS_ROOT / "stage1_binary" / "artifacts"
    x1_cols = json.loads((stage1_art / "feature_columns.json").read_text())["columns"]

    base_svm = load(stage1_art / "base_svm_pipeline.joblib")
    base_xgb = load(stage1_art / "base_xgb_pipeline.joblib")
    lstm_pre = load(stage1_art / "lstm_preprocessor.joblib")
    lstm_model = tf.keras.models.load_model(stage1_art / "lstm_model.keras", compile=False)
    tcn_gru_pre = load(stage1_art / "tcn_gru_preprocessor.joblib")
    tcn_gru_model = tf.keras.models.load_model(stage1_art / "tcn_gru_model.keras", compile=False)
    meta_pipe = load(stage1_art / "meta_pipeline.joblib")

    # raw tabular features
    X_raw = _prepare_raw_features(dfh, x1_cols)
    keys_df = dfh[["race_id", "driver_id", "lapno"]]

    # base models
    p_svm = base_svm.predict_proba(X_raw)[:, 1].astype(float)
    p_xgb = base_xgb.predict_proba(X_raw)[:, 1].astype(float)

    # lstm base model
    X_lstm_row = _to_dense(lstm_pre.transform(X_raw))
    expected_T_lstm = lstm_model.input_shape[1]
    p_lstm = _predict_sequence_model_from_builder(
        lstm_model,
        keys_df,
        X_lstm_row,
        expected_T_lstm,
        True,
    )

    # tcn-gru base model
    X_tcn_gru_row = _to_dense(tcn_gru_pre.transform(X_raw))
    expected_T_tcn_gru = tcn_gru_model.input_shape[1]
    p_tcn_gru = _predict_sequence_model_from_builder(
        tcn_gru_model,
        keys_df,
        X_tcn_gru_row,
        expected_T_tcn_gru,
        True,
    )

    # meta input
    X_meta = X_raw.copy()
    X_meta["p_svm"] = p_svm
    X_meta["p_xgb"] = p_xgb
    X_meta["p_lstm"] = p_lstm
    X_meta["p_tcn_gru"] = p_tcn_gru

    # final probabilities and hard predictions
    proba = meta_pipe.predict_proba(X_meta)[:, 1].astype(float)
    pit_pred = (proba >= PIT_THRESHOLD).astype(int)

    # output dataframe
    out = dfh[["race_id", "driver_id", "lapno"]].copy()
    out["race_progress_pct"] = (pd.to_numeric(dfh["race_progress"], errors="coerce") * 100.0).fillna(0).astype(float)
    out["y_pit"] = pd.to_numeric(dfh["y_pit"], errors="coerce").fillna(0).astype(int)
    out["is_raining"] = (
        pd.to_numeric(dfh["is_raining"], errors="coerce").fillna(0).astype(int)
        if "is_raining" in dfh.columns else 0
    )
    out["fcy_status"] = (
        pd.to_numeric(dfh["fcy_status"], errors="coerce").fillna(0).astype(int)
        if "fcy_status" in dfh.columns else 0
    )
    out["p_pit"] = proba.astype(float)
    out["pit_pred"] = pit_pred.astype(int)
    out["event_type"] = np.where(
        out["pit_pred"] == 1,
        np.where(out["y_pit"] == 1, "TP", "FP"),
        "",
    )

    # save lap level probabilities
    out_path = OUTDIR / "holdout_stage1_lap_probs.csv"
    out.to_csv(out_path, index=False)
    print(f"wrote: {out_path}")

    # save predicted positive events only
    event_rows = out[out["pit_pred"] == 1].copy()
    event_rows_path = OUTDIR / "holdout_stage1_tp_fp_events.csv"
    event_rows.to_csv(event_rows_path, index=False)
    print(f"wrote: {event_rows_path}")

    # plot probability traces per driver
    plots_dir = OUTDIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for (rid, did), g in out.groupby(["race_id", "driver_id"], sort=False):
        g = g.sort_values("lapno", kind="mergesort")
        x = g["race_progress_pct"].to_numpy(dtype=float)
        y = g["p_pit"].to_numpy(dtype=float) * 100.0
        pit_x = g.loc[g["y_pit"] == 1, "race_progress_pct"].to_numpy(dtype=float)

        plt.figure(figsize=(10, 4))
        plt.plot(x, y, marker="o", linewidth=1)

        # background shading for rain and fcy
        rain_segs = _segments_from_bool_mask(x, g["is_raining"].to_numpy(dtype=int) == 1)
        fcy_segs = _segments_from_bool_mask(x, g["fcy_status"].to_numpy(dtype=int) != 0)

        for a, b in rain_segs:
            plt.axvspan(a, b, alpha=0.12, facecolor="blue", linewidth=0)
        for a, b in fcy_segs:
            plt.axvspan(a, b, alpha=0.12, facecolor="yellow", linewidth=0)

        plt.axhline(PIT_THRESHOLD * 100.0, linewidth=1, color="r", linestyle="--")
        for px in pit_x:
            plt.axvline(px, linewidth=1, color="g")

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

    # per race metrics
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