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
from sklearn.pipeline import Pipeline
import tensorflow as tf
import shap

HOLDOUT_RACE_IDS = [2, 24, 53, 73, 75]

# note to self: hardcoded paths and the optimal threshold i found earlier
STAGE1_CSV = Path("data/processed/dataset1.csv")
ARTIFACTS_ROOT = Path("runs/final_run")
OUTDIR = Path("runs/holdout_run")
PIT_THRESHOLD = 0.264

# note to self: shap settings
XGB_SHAP_BG_SIZE = 300
META_SHAP_BG_SIZE = 300
TCN_GRU_SHAP_BG_SIZE = 64
TOP_K = 10
RNG_SEED = 42

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

def _to_dense(x) -> np.ndarray:
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)

def _prepare_raw_features(df_part: pd.DataFrame, x1_cols: List[str]) -> pd.DataFrame:
    X_raw = df_part[x1_cols].copy()
    for c in ["is_wet_race", "is_raining", "minutes_rain"]:
        if c in X_raw.columns:
            X_raw[c] = pd.to_numeric(X_raw[c], errors="coerce").fillna(0.0)
        else:
            X_raw[c] = 0.0
    return X_raw

def _split_pipeline(pipe):
    if hasattr(pipe, "steps"):
        steps = pipe.steps
        est = steps[-1][1]
        pre = Pipeline(steps[:-1]) if len(steps) > 1 else None
        return pre, est
    return None, pipe

def _sample_indices(n_total: int, n_take: int, rng: np.random.Generator) -> np.ndarray:
    n_take = min(n_take, n_total)
    return rng.choice(n_total, size=n_take, replace=False)

def _get_feature_names_after_preprocessor(preprocessor, fallback_cols: List[str], X_transformed: np.ndarray) -> List[str]:
    if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
        try:
            return [str(x) for x in preprocessor.get_feature_names_out()]
        except Exception:
            pass
    if len(fallback_cols) == X_transformed.shape[1]:
        return [str(c) for c in fallback_cols]
    return [f"f{i}" for i in range(X_transformed.shape[1])]

def _make_tabular_explainer(estimator, X_bg: np.ndarray, feature_names: List[str]):
    cls_name = estimator.__class__.__name__.lower()
    tree_like = (
        hasattr(estimator, "get_booster")
        or "xgb" in cls_name
        or "forest" in cls_name
        or "tree" in cls_name
        or "boost" in cls_name
    )
    if tree_like:
        return shap.TreeExplainer(
            estimator,
            data=X_bg,
            model_output="probability",
            feature_names=feature_names,
        )

    masker = shap.maskers.Independent(X_bg)
    return shap.Explainer(
        lambda z: estimator.predict_proba(z)[:, 1],
        masker=masker,
        algorithm="permutation",
        feature_names=feature_names,
        seed=RNG_SEED,
    )

def _positive_class_from_explanation(expl: shap.Explanation) -> shap.Explanation:
    vals = np.asarray(expl.values)
    base = np.asarray(expl.base_values)

    if vals.ndim == 3 and vals.shape[-1] == 2:
        vals = vals[:, :, 1]
        if base.ndim == 2:
            base = base[:, 1]
    elif vals.ndim == 3 and vals.shape[-1] == 1:
        vals = vals[:, :, 0]
        if base.ndim == 2 and base.shape[1] == 1:
            base = base[:, 0]

    return shap.Explanation(
        values=vals,
        base_values=base,
        data=expl.data,
        feature_names=expl.feature_names,
    )

def _squeeze_binary_shap_array(vals) -> np.ndarray:
    if isinstance(vals, list):
        vals = vals[0]
    vals = np.asarray(vals)

    # common cases:
    # (n, T, F)
    # (n, T, F, 1)
    # (n, T, F, 2)
    if vals.ndim == 4 and vals.shape[-1] == 1:
        vals = vals[..., 0]
    elif vals.ndim == 4 and vals.shape[-1] == 2:
        vals = vals[..., 1]
    return vals

def _scalar_or_string(x):
    arr = np.asarray(x)
    if arr.ndim == 0:
        return arr.item()
    return str(arr.tolist())

def _write_tabular_long_csv(
    out_df: pd.DataFrame,
    explain_idx: np.ndarray,
    model_name: str,
    model_prob: np.ndarray,
    shap_expl: shap.Explanation,
    feature_names: List[str],
    out_csv: Path,
    plot_root: Path | None = None,
):
    rows = []

    for local_i, global_i in enumerate(explain_idx):
        row = out_df.iloc[global_i]
        vals = np.asarray(shap_expl.values[local_i], dtype=float)
        data_row = shap_expl.data[local_i] if shap_expl.data is not None else None
        order = np.argsort(np.abs(vals))[::-1][:TOP_K]

        for rank, j in enumerate(order, start=1):
            rows.append({
                "model": model_name,
                "event_type": str(row["event_type"]),
                "race_id": int(row["race_id"]),
                "driver_id": int(row["driver_id"]),
                "lapno": int(row["lapno"]),
                "final_p_pit": float(row["p_pit"]),
                "base_model_p_pit": float(model_prob[global_i]),
                "rank": rank,
                "feature": str(feature_names[j]),
                "shap_value": float(vals[j]),
                "abs_shap_value": float(abs(vals[j])),
                "feature_value": None if data_row is None else _scalar_or_string(data_row[j]),
            })

        if plot_root is not None:
            driver_dir = plot_root / f"race_{int(row['race_id'])}" / f"driver_{int(row['driver_id'])}"
            driver_dir.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(8, 6))
            shap.plots.waterfall(shap_expl[local_i], max_display=12, show=False)
            plt.title(f"{model_name.upper()} SHAP | {row['event_type']} | lap {int(row['lapno'])}")
            plt.tight_layout()
            plt.savefig(driver_dir / f"lap_{int(row['lapno'])}_waterfall.png", dpi=200, bbox_inches="tight")
            plt.close()

    df_long = pd.DataFrame(rows).sort_values(["race_id", "driver_id", "lapno", "rank"])
    df_long.to_csv(out_csv, index=False)
    print(f"wrote: {out_csv}")

    if not df_long.empty:
        summary = (
            df_long.groupby(["model", "event_type", "feature"], as_index=False)["abs_shap_value"]
            .mean()
            .rename(columns={"abs_shap_value": "mean_abs_shap"})
            .sort_values(["model", "event_type", "mean_abs_shap"], ascending=[True, True, False])
        )
        summary_path = out_csv.with_name(out_csv.stem + "_summary.csv")
        summary.to_csv(summary_path, index=False)
        print(f"wrote: {summary_path}")

def main():
    # make sure the output directory exists
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    # load the main dataset and do some basic cleaning
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

    # note to self: non-holdout rows for shap backgrounds
    df_bg = df1[~df1["race_id"].isin(list(holdout_set))].copy()
    df_bg = df_bg.sort_values(["race_id", "driver_id", "lapno"], kind="mergesort").reset_index(drop=True)
    if df_bg.empty:
        raise ValueError("no non-holdout rows available for SHAP background")

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
    X_raw = _prepare_raw_features(dfh, x1_cols)

    # note to self: generate predictions from all the base models
    p_svm = _safe_predict_proba(base_svm, X_raw)[:, 1]
    p_xgb = _safe_predict_proba(base_xgb, X_raw)[:, 1]
    keys_df = dfh[["race_id", "driver_id", "lapno"]]

    # note to self: generate predictions for the lstm model
    X_lstm_row = lstm_pre.transform(X_raw)
    X_lstm_row = _to_dense(X_lstm_row)
    expected_T_lstm = lstm_model.input_shape[1]
    X_seq_lstm, _ = _make_left_padded_sequences_from_rows(
        keys_df, X_lstm_row, window=expected_T_lstm, add_timestep_mask=True
    )
    p_lstm = _keras_predict_proba(lstm_model, X_seq_lstm)

    # note to self: generate predictions for the tcn_gru model
    X_tcn_gru_row = tcn_gru_pre.transform(X_raw)
    X_tcn_gru_row = _to_dense(X_tcn_gru_row)
    expected_T_tcn_gru = tcn_gru_model.input_shape[1]
    X_seq_tcn_gru, tcn_effective_len = _make_left_padded_sequences_from_rows(
        keys_df, X_tcn_gru_row, window=expected_T_tcn_gru, add_timestep_mask=True
    )
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

    # note to self: save the lap-level predictions
    out_path = OUTDIR / "holdout_stage1_lap_probs.csv"
    out.to_csv(out_path, index=False)
    print(f"wrote: {out_path}")

    # note to self: save just the tp/fp event laps
    event_rows = out[out["pit_pred"] == 1].copy()
    event_rows_path = OUTDIR / "holdout_stage1_tp_fp_events.csv"
    event_rows.to_csv(event_rows_path, index=False)
    print(f"wrote: {event_rows_path}")

    # note to self: shap only on final predicted positive laps (i.e. TP or FP)
    explain_idx = np.flatnonzero(out["pit_pred"].to_numpy() == 1)

    if len(explain_idx) > 0:
        shap_dir = OUTDIR / "shap_stage1"
        shap_dir.mkdir(parents=True, exist_ok=True)

        # =========================================================
        # XGB SHAP
        # =========================================================
        xgb_pre, xgb_est = _split_pipeline(base_xgb)

        X_bg_raw_full = _prepare_raw_features(df_bg, x1_cols)
        bg_xgb_idx = _sample_indices(len(X_bg_raw_full), XGB_SHAP_BG_SIZE, rng)
        X_bg_raw_xgb = X_bg_raw_full.iloc[bg_xgb_idx].copy()

        if xgb_pre is not None:
            X_bg_xgb = _to_dense(xgb_pre.transform(X_bg_raw_xgb))
            X_eval_xgb = _to_dense(xgb_pre.transform(X_raw.iloc[explain_idx]))
            xgb_feature_names = _get_feature_names_after_preprocessor(
                xgb_pre, list(X_raw.columns), X_bg_xgb
            )
        else:
            X_bg_xgb = X_bg_raw_xgb.to_numpy(dtype=float)
            X_eval_xgb = X_raw.iloc[explain_idx].to_numpy(dtype=float)
            xgb_feature_names = [str(c) for c in X_raw.columns]

        xgb_explainer = _make_tabular_explainer(xgb_est, X_bg_xgb, xgb_feature_names)
        xgb_expl = xgb_explainer(X_eval_xgb)
        xgb_expl = _positive_class_from_explanation(xgb_expl)

        _write_tabular_long_csv(
            out_df=out,
            explain_idx=explain_idx,
            model_name="xgb",
            model_prob=p_xgb,
            shap_expl=xgb_expl,
            feature_names=xgb_feature_names,
            out_csv=shap_dir / "xgb_tp_fp_shap_long.csv",
            plot_root=shap_dir / "xgb_plots",
        )

        # =========================================================
        # META SHAP
        # =========================================================
        # note to self: need background meta features too
        X_bg_raw_meta = _prepare_raw_features(df_bg, x1_cols)
        p_svm_bg = _safe_predict_proba(base_svm, X_bg_raw_meta)[:, 1]
        p_xgb_bg = _safe_predict_proba(base_xgb, X_bg_raw_meta)[:, 1]

        keys_bg = df_bg[["race_id", "driver_id", "lapno"]]

        X_lstm_row_bg = _to_dense(lstm_pre.transform(X_bg_raw_meta))
        X_seq_lstm_bg, _ = _make_left_padded_sequences_from_rows(
            keys_bg, X_lstm_row_bg, window=expected_T_lstm, add_timestep_mask=True
        )
        p_lstm_bg = _keras_predict_proba(lstm_model, X_seq_lstm_bg)

        X_tcn_gru_row_bg = _to_dense(tcn_gru_pre.transform(X_bg_raw_meta))
        X_seq_tcn_gru_bg, _ = _make_left_padded_sequences_from_rows(
            keys_bg, X_tcn_gru_row_bg, window=expected_T_tcn_gru, add_timestep_mask=True
        )
        p_tcn_gru_bg = _keras_predict_proba(tcn_gru_model, X_seq_tcn_gru_bg)

        X_meta_bg = X_bg_raw_meta.copy()
        X_meta_bg["p_svm"] = p_svm_bg
        X_meta_bg["p_xgb"] = p_xgb_bg
        X_meta_bg["p_lstm"] = p_lstm_bg
        X_meta_bg["p_tcn_gru"] = p_tcn_gru_bg

        bg_meta_idx = _sample_indices(len(X_meta_bg), META_SHAP_BG_SIZE, rng)
        X_meta_bg_sample = X_meta_bg.iloc[bg_meta_idx].copy()

        meta_pre, meta_est = _split_pipeline(meta_pipe)

        if meta_pre is not None:
            X_bg_meta = _to_dense(meta_pre.transform(X_meta_bg_sample))
            X_eval_meta = _to_dense(meta_pre.transform(X_meta.iloc[explain_idx]))
            meta_feature_names = _get_feature_names_after_preprocessor(
                meta_pre, list(X_meta.columns), X_bg_meta
            )
        else:
            X_bg_meta = X_meta_bg_sample.to_numpy(dtype=float)
            X_eval_meta = X_meta.iloc[explain_idx].to_numpy(dtype=float)
            meta_feature_names = [str(c) for c in X_meta.columns]

        meta_explainer = _make_tabular_explainer(meta_est, X_bg_meta, meta_feature_names)
        meta_expl = meta_explainer(X_eval_meta)
        meta_expl = _positive_class_from_explanation(meta_expl)

        _write_tabular_long_csv(
            out_df=out,
            explain_idx=explain_idx,
            model_name="meta",
            model_prob=proba,
            shap_expl=meta_expl,
            feature_names=meta_feature_names,
            out_csv=shap_dir / "meta_tp_fp_shap_long.csv",
            plot_root=shap_dir / "meta_plots",
        )

        # =========================================================
        # TCN-GRU SHAP
        # =========================================================
        # note to self: build background sequences from non-holdout rows first, then sample
        X_bg_tcn_row_full = _to_dense(tcn_gru_pre.transform(X_bg_raw_full))
        X_bg_seq_full, _ = _make_left_padded_sequences_from_rows(
            keys_bg, X_bg_tcn_row_full, window=expected_T_tcn_gru, add_timestep_mask=True
        )

        bg_tcn_idx = _sample_indices(len(X_bg_seq_full), TCN_GRU_SHAP_BG_SIZE, rng)
        X_bg_seq = X_bg_seq_full[bg_tcn_idx]

        if hasattr(tcn_gru_pre, "get_feature_names_out"):
            try:
                tcn_feature_names = [str(x) for x in tcn_gru_pre.get_feature_names_out()]
            except Exception:
                tcn_feature_names = [f"f{i}" for i in range(X_seq_tcn_gru.shape[2] - 1)]
        else:
            tcn_feature_names = [f"f{i}" for i in range(X_seq_tcn_gru.shape[2] - 1)]

        tcn_explainer = shap.GradientExplainer(tcn_gru_model, X_bg_seq)
        tcn_raw_vals = tcn_explainer.shap_values(X_seq_tcn_gru[explain_idx])
        tcn_raw_vals = _squeeze_binary_shap_array(tcn_raw_vals)  # shape -> (n_events, T, F_plus_mask)

        # note to self: drop the added timestep mask channel from interpretation
        tcn_vals = tcn_raw_vals[:, :, :-1]

        # aggregate over time so it becomes interpretable
        tcn_feat_signed = tcn_vals.sum(axis=1)          # (n_events, F)
        tcn_feat_abs = np.abs(tcn_vals).sum(axis=1)     # (n_events, F)
        tcn_time_abs = np.abs(tcn_vals).sum(axis=2)     # (n_events, T)

        tcn_feature_rows = []
        tcn_lag_rows = []

        for local_i, global_i in enumerate(explain_idx):
            row = out.iloc[global_i]
            feat_order = np.argsort(tcn_feat_abs[local_i])[::-1][:TOP_K]

            for rank, j in enumerate(feat_order, start=1):
                tcn_feature_rows.append({
                    "model": "tcn_gru",
                    "event_type": str(row["event_type"]),
                    "race_id": int(row["race_id"]),
                    "driver_id": int(row["driver_id"]),
                    "lapno": int(row["lapno"]),
                    "final_p_pit": float(row["p_pit"]),
                    "base_model_p_pit": float(tcn_gru_proba[global_i]),
                    "rank": rank,
                    "feature": str(tcn_feature_names[j]),
                    "signed_shap_sum_over_time": float(tcn_feat_signed[local_i, j]),
                    "abs_shap_sum_over_time": float(tcn_feat_abs[local_i, j]),
                })

            # note to self: also save which recent lags mattered most
            window = tcn_vals.shape[1]
            L = int(tcn_effective_len[global_i])
            valid_start = window - L
            valid_positions = np.arange(valid_start, window)
            valid_time_scores = tcn_time_abs[local_i, valid_start:]
            lag_order = np.argsort(valid_time_scores)[::-1][:min(5, len(valid_time_scores))]

            for rank, lag_local_idx in enumerate(lag_order, start=1):
                pos = valid_positions[lag_local_idx]
                lag = (window - 1) - pos   # 0=current lap, 1=one lap ago, etc.
                tcn_lag_rows.append({
                    "model": "tcn_gru",
                    "event_type": str(row["event_type"]),
                    "race_id": int(row["race_id"]),
                    "driver_id": int(row["driver_id"]),
                    "lapno": int(row["lapno"]),
                    "rank": rank,
                    "lag": int(lag),
                    "abs_shap_at_lag": float(tcn_time_abs[local_i, pos]),
                })

            # note to self: simple heatmap plot for each event
            top_plot_idx = feat_order[:min(8, len(feat_order))]
            heat = tcn_vals[local_i, :, top_plot_idx].T
            driver_dir = shap_dir / "tcn_gru_plots" / f"race_{int(row['race_id'])}" / f"driver_{int(row['driver_id'])}"
            driver_dir.mkdir(parents=True, exist_ok=True)

            plt.figure(figsize=(10, 4.5))
            plt.imshow(heat, aspect="auto", cmap="coolwarm")
            plt.colorbar(label="shap value")
            plt.yticks(range(len(top_plot_idx)), [tcn_feature_names[j] for j in top_plot_idx])
            plt.xticks(range(heat.shape[1]), [str((heat.shape[1] - 1) - i) for i in range(heat.shape[1])], rotation=0)
            plt.xlabel("lag (0 = current lap)")
            plt.ylabel("feature")
            plt.title(f"TCN-GRU SHAP heatmap | {row['event_type']} | lap {int(row['lapno'])}")
            plt.tight_layout()
            plt.savefig(driver_dir / f"lap_{int(row['lapno'])}_heatmap.png", dpi=200, bbox_inches="tight")
            plt.close()

        tcn_feat_df = pd.DataFrame(tcn_feature_rows).sort_values(["race_id", "driver_id", "lapno", "rank"])
        tcn_feat_path = shap_dir / "tcn_gru_tp_fp_shap_long.csv"
        tcn_feat_df.to_csv(tcn_feat_path, index=False)
        print(f"wrote: {tcn_feat_path}")

        if not tcn_feat_df.empty:
            tcn_feat_summary = (
                tcn_feat_df.groupby(["model", "event_type", "feature"], as_index=False)["abs_shap_sum_over_time"]
                .mean()
                .rename(columns={"abs_shap_sum_over_time": "mean_abs_shap_sum_over_time"})
                .sort_values(["model", "event_type", "mean_abs_shap_sum_over_time"], ascending=[True, True, False])
            )
            tcn_feat_summary_path = shap_dir / "tcn_gru_tp_fp_shap_long_summary.csv"
            tcn_feat_summary.to_csv(tcn_feat_summary_path, index=False)
            print(f"wrote: {tcn_feat_summary_path}")

        tcn_lag_df = pd.DataFrame(tcn_lag_rows).sort_values(["race_id", "driver_id", "lapno", "rank"])
        tcn_lag_path = shap_dir / "tcn_gru_tp_fp_top_lags.csv"
        tcn_lag_df.to_csv(tcn_lag_path, index=False)
        print(f"wrote: {tcn_lag_path}")

    else:
        print("no final predicted positive laps found, so no TP/FP SHAP was run")

    # note to self: now make the plots for each driver in each race
    plots_dir = OUTDIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for (rid, did), g in out.groupby(["race_id", "driver_id"], sort=False):
        g = g.sort_values("lapno", kind="mergesort")
        x = g["race_progress_pct"].to_numpy(dtype=float)
        y = g["p_pit"].to_numpy(dtype=float) * 100.0
        pit_x = g.loc[g["y_pit"] == 1, "race_progress_pct"].to_numpy(dtype=float)

        if int(rid) == 73:
            # temporary hard-coded fix:
            # ignore early historical pit markers in the first 10% of the race
            pit_x = pit_x[pit_x >= 10.0]

        plt.figure(figsize=(10, 4))
        plt.plot(x, y, marker="o", linewidth=1)

        # add shaded backgrounds for rain and fcy
        rain_segs = _segments_from_bool_mask(x, g["is_raining"].to_numpy(dtype=int) == 1)

        if int(rid) == 2:
            # temporary hard-coded FCY window for race 2
            fcy_segs = [(44.8, 53.4)]
        else:
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
    