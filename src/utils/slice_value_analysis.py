#!/usr/bin/env python3
"""
Slice-based value analysis for Stage-1.

Two modes:

1) META-PREDS mode (requires --oof_preds):
   Evaluate per-row OOF test predictions for meta ablations.
   Expected columns in oof preds parquet:
     race_id, row_id, fold_id, ablation_name, y_true, proba, threshold

2) BASE-STREAMS mode (no --oof_preds):
   Uses meta_table.parquet (which already contains OOF base probabilities):
     p_svm, p_rf, p_xgb, p_ann, tcn_proba
   Builds per-row OOF predictions by:
     - race-wise CV fold assignment
     - threshold tuned on train rows per fold per stream
     - predictions evaluated on test rows
   Then runs slice evaluation on those predictions.

Outputs (in --outdir):
  - slice_metrics_fold.csv
  - slice_metrics_summary.csv
  - slice_deltas_vs_ref.csv
  - slice_leaderboard.csv
  - (optional) generated_oof_preds.parquet (BASE-STREAMS mode only)

Usage (BASE-STREAMS; no oof preds file needed):
  python -m src.utils.slice_value_analysis \
    --meta_table runs/ablation/stage1_meta_ablation/cache/meta_table.parquet \
    --outdir    runs/ablation/stage1_meta_ablation/reports/slices_base_streams \
    --ref_ablation stream_tcn_proba \
    --n_splits 5 --seed 42 \
    --include_2d

Usage (META-PREDS; after you generate meta_oof_preds.parquet):
  python -m src.utils.slice_value_analysis \
    --meta_table runs/ablation/stage1_meta_ablation/cache/meta_table.parquet \
    --oof_preds  runs/ablation/stage1_meta_ablation/reports/meta_oof_preds.parquet \
    --outdir     runs/ablation/stage1_meta_ablation/reports/slices_meta \
    --ref_ablation base_only \
    --include_2d
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

# -----------------------
# Metrics
# -----------------------
def _safe_logloss(y_true: np.ndarray, proba_pos: np.ndarray) -> float:
    p = np.clip(proba_pos.astype(float), 1e-15, 1 - 1e-15)
    return float(log_loss(y_true.astype(int), p, labels=[0, 1]))


def _counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)}


def tune_threshold_max_f1(y_true: np.ndarray, proba_pos: np.ndarray, grid: Optional[np.ndarray] = None) -> float:
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        y_pred = (proba_pos >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def compute_metrics(y_true: np.ndarray, proba_pos: np.ndarray, thr: float) -> Dict[str, float]:
    y_true = y_true.astype(int)
    proba_pos = proba_pos.astype(float)

    if len(np.unique(y_true)) < 2:
        pr_auc = np.nan
        ll = np.nan
    else:
        pr_auc = float(average_precision_score(y_true, proba_pos))
        ll = _safe_logloss(y_true, proba_pos)

    y_pred = (proba_pos >= float(thr)).astype(int)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    c = _counts(y_true, y_pred)

    return {
        "PR_AUC": pr_auc,
        "LogLoss": ll,
        "F1": f1,
        "Precision": prec,
        "Recall": rec,
        **c,
        "n_rows": int(len(y_true)),
        "n_pos": int(np.sum(y_true == 1)),
        "pos_rate": float(np.mean(y_true)) if len(y_true) else np.nan,
    }


# -----------------------
# Slice builders
# -----------------------
def _ensure_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _as_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def add_default_slices(meta: pd.DataFrame) -> pd.DataFrame:
    out = meta.copy()

    # race progress -> early/mid/late
    if "race_progress" in out.columns:
        rp = _as_float(out["race_progress"])
    elif "lapno" in out.columns and "race_id" in out.columns:
        lap = _as_float(out["lapno"])
        maxlap = out.groupby("race_id")["lapno"].transform(lambda x: pd.to_numeric(x, errors="coerce").max())
        rp = lap / pd.to_numeric(maxlap, errors="coerce")
    else:
        rp = pd.Series(np.nan, index=out.index)

    out["slice_race_phase"] = pd.cut(
        rp, bins=[-np.inf, 1 / 3, 2 / 3, np.inf], labels=["early", "mid", "late"]
    ).astype("object")

    # FCY mapping
    if "fcy_status" in out.columns:
        fs = out["fcy_status"]
        if pd.api.types.is_numeric_dtype(fs):
            out["slice_fcy"] = fs.map({0: "GREEN", 1: "VSC", 2: "SC"}).fillna("OTHER").astype("object")
        else:
            fss = fs.astype(str).str.upper()
            out["slice_fcy"] = np.where(
                fss.str.contains("VSC", na=False), "VSC",
                np.where(fss.str.contains("SC", na=False), "SC", "GREEN")
            ).astype("object")
    else:
        out["slice_fcy"] = "UNKNOWN"

    # wetness
    is_rain = _as_float(out["is_raining"]) if "is_raining" in out.columns else pd.Series(0.0, index=out.index)
    rained_yet = _as_float(out["rained_yet"]) if "rained_yet" in out.columns else pd.Series(0.0, index=out.index)
    wetness = np.where(is_rain >= 0.5, "WET", np.where(rained_yet >= 0.5, "POST_RAIN", "DRY"))
    out["slice_wetness"] = pd.Series(wetness, index=out.index).astype("object")

    # traffic
    if "close_ahead" in out.columns:
        ca = out["close_ahead"]
        if pd.api.types.is_numeric_dtype(ca):
            out["slice_traffic"] = np.where(_as_float(ca) >= 0.5, "IN_TRAFFIC", "CLEAN_AIR").astype("object")
        else:
            cas = ca.astype(str).str.lower()
            out["slice_traffic"] = np.where(cas.isin(["1", "true", "t", "yes", "y"]), "IN_TRAFFIC", "CLEAN_AIR").astype("object")
    else:
        out["slice_traffic"] = "UNKNOWN"

    # tyre age bins
    if "tyre_age" in out.columns:
        ta = _as_float(out["tyre_age"])
        out["slice_tyre_age_bin"] = pd.cut(
            ta, bins=[-np.inf, 5, 15, np.inf], labels=["0_5", "6_15", "16_plus"]
        ).astype("object")
    else:
        out["slice_tyre_age_bin"] = "UNKNOWN"

    # pit stops so far
    if "pit_stops_so_far" in out.columns:
        ps = _as_float(out["pit_stops_so_far"])
        out["slice_pitstops"] = np.where(ps <= 0.5, "0", "1_plus").astype("object")
    else:
        out["slice_pitstops"] = "UNKNOWN"

    # undercut pressure
    if "tyre_change_pursuer" in out.columns:
        tcp = _as_float(out["tyre_change_pursuer"])
        out["slice_undercut_pressure"] = np.where(tcp >= 0.5, "PRESSURE", "NO_PRESSURE").astype("object")
    else:
        out["slice_undercut_pressure"] = "UNKNOWN"

    # track category
    if "track_category" in out.columns:
        out["slice_track_cat"] = out["track_category"].astype("object")
    else:
        out["slice_track_cat"] = "UNKNOWN"

    return out


def add_2d_slices(meta: pd.DataFrame) -> pd.DataFrame:
    out = meta.copy()
    out["slice_wetness_x_fcy"] = (out["slice_wetness"].astype(str) + "×" + out["slice_fcy"].astype(str)).astype("object")
    out["slice_phase_x_tyre"] = (out["slice_race_phase"].astype(str) + "×" + out["slice_tyre_age_bin"].astype(str)).astype("object")
    out["slice_traffic_x_undercut"] = (out["slice_traffic"].astype(str) + "×" + out["slice_undercut_pressure"].astype(str)).astype("object")
    out["slice_phase_x_traffic"] = (out["slice_race_phase"].astype(str) + "×" + out["slice_traffic"].astype(str)).astype("object")
    return out


# -----------------------
# Fold assignment (race-wise)
# -----------------------
def make_race_folds(df: pd.DataFrame, *, target_col: str, n_splits: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    # Prefer your project’s fold builder for consistency
    try:
        from src.data.data import make_race_group_folds  # type: ignore
        fb = make_race_group_folds(df, target_col=target_col, n_splits=n_splits, seed=seed)
        return fb.folds
    except Exception:
        from sklearn.model_selection import StratifiedGroupKFold
        y = df[target_col].astype(int).to_numpy()
        groups = df["race_id"].to_numpy()
        X_dummy = np.zeros((len(df), 1))
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return [(tr, te) for tr, te in sgkf.split(X_dummy, y, groups)]


# -----------------------
# Build OOF preds from base streams (no meta preds file needed)
# -----------------------
def build_oof_preds_from_streams(
    meta: pd.DataFrame,
    *,
    prob_cols: List[str],
    n_splits: int,
    seed: int,
) -> pd.DataFrame:
    _ensure_cols(meta, ["race_id", "row_id", "y_pit"])
    for c in prob_cols:
        if c not in meta.columns:
            raise ValueError(f"Probability column '{c}' not in meta_table.")

    folds = make_race_folds(meta[["race_id", "y_pit"]].copy(), target_col="y_pit", n_splits=n_splits, seed=seed)

    rows = []
    y_all = meta["y_pit"].astype(int).to_numpy()

    for fold_id, (tr_idx, te_idx) in enumerate(folds):
        y_tr = y_all[tr_idx]
        y_te = y_all[te_idx]

        for c in prob_cols:
            p_tr = meta.iloc[tr_idx][c].astype(float).to_numpy()
            p_te = meta.iloc[te_idx][c].astype(float).to_numpy()

            thr = tune_threshold_max_f1(y_tr, p_tr)

            tmp = meta.iloc[te_idx][["race_id", "row_id"]].copy()
            tmp["fold_id"] = int(fold_id)
            tmp["ablation_name"] = f"stream_{c}"
            tmp["y_true"] = y_te
            tmp["proba"] = p_te.astype(np.float32)
            tmp["threshold"] = float(thr)
            rows.append(tmp)

    return pd.concat(rows, ignore_index=True)


# -----------------------
# Slice eval core
# -----------------------
def compute_slice_metrics(merged: pd.DataFrame, slice_col: str) -> pd.DataFrame:
    out = []
    for (ab, fold, sval), g in merged.groupby(["ablation_name", "fold_id", slice_col], dropna=False):
        y = g["y_true"].to_numpy(dtype=int)
        p = g["proba"].to_numpy(dtype=float)
        thr = float(np.nanmedian(g["threshold"].to_numpy(dtype=float)))
        m = compute_metrics(y, p, thr=thr)
        out.append({
            "slice_col": slice_col,
            "slice_value": str(sval),
            "ablation_name": str(ab),
            "fold_id": int(fold),
            "threshold_used": thr,
            **m,
        })
    return pd.DataFrame(out)


def summarize_over_folds(fold_df: pd.DataFrame) -> pd.DataFrame:
    g = fold_df.groupby(["slice_col", "slice_value", "ablation_name"], dropna=False)
    return g.agg(
        mean_PR_AUC=("PR_AUC", "mean"),
        std_PR_AUC=("PR_AUC", "std"),
        mean_LogLoss=("LogLoss", "mean"),
        std_LogLoss=("LogLoss", "std"),
        mean_F1=("F1", "mean"),
        std_F1=("F1", "std"),
        mean_Precision=("Precision", "mean"),
        mean_Recall=("Recall", "mean"),
        mean_n_rows=("n_rows", "mean"),
        mean_n_pos=("n_pos", "mean"),
        mean_pos_rate=("pos_rate", "mean"),
    ).reset_index()


def deltas_vs_reference(summary_df: pd.DataFrame, ref_ablation: str) -> pd.DataFrame:
    ref = summary_df[summary_df["ablation_name"] == ref_ablation].copy()
    if ref.empty:
        raise ValueError(f"Reference ablation '{ref_ablation}' not found in summary.")
    ref = ref.rename(columns={
        "mean_PR_AUC": "ref_PR_AUC",
        "mean_LogLoss": "ref_LogLoss",
        "mean_F1": "ref_F1",
    })[["slice_col", "slice_value", "ref_PR_AUC", "ref_LogLoss", "ref_F1"]]

    out = summary_df.merge(ref, on=["slice_col", "slice_value"], how="left")
    out["delta_PR_AUC_vs_ref"] = out["mean_PR_AUC"] - out["ref_PR_AUC"]
    out["delta_LogLoss_vs_ref"] = out["mean_LogLoss"] - out["ref_LogLoss"]
    out["delta_F1_vs_ref"] = out["mean_F1"] - out["ref_F1"]
    return out


def build_leaderboard(deltas_df: pd.DataFrame, *, min_rows: int, min_pos: int, metric: str, topk: int = 3) -> pd.DataFrame:
    df = deltas_df.copy()
    df = df[(df["mean_n_rows"] >= min_rows) & (df["mean_n_pos"] >= min_pos)]
    rows = []
    for (sc, sv), g in df.groupby(["slice_col", "slice_value"], dropna=False):
        g2 = g.sort_values(metric, ascending=False)
        best = g2.head(topk)
        worst = g2.tail(topk)
        for _, r in best.iterrows():
            rows.append({"slice_col": sc, "slice_value": sv, "rank_group": "BEST", "ablation_name": r["ablation_name"], metric: r[metric]})
        for _, r in worst.iterrows():
            rows.append({"slice_col": sc, "slice_value": sv, "rank_group": "WORST", "ablation_name": r["ablation_name"], metric: r[metric]})
    return pd.DataFrame(rows)


# -----------------------
# CLI
# -----------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_table", required=True)
    ap.add_argument("--oof_preds", default="", help="Optional parquet with per-row OOF preds for meta ablations")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ref_ablation", default="", help="Reference ablation for deltas (default depends on mode)")
    ap.add_argument("--ablations", default="", help="Comma-separated allowlist of ablation_name values (optional)")

    # only used for BASE-STREAMS mode
    ap.add_argument("--streams", default="p_svm,p_rf,p_xgb,p_ann,tcn_proba", help="Comma-separated prob cols in meta_table")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--write_generated_preds", action="store_true", help="Write generated preds parquet in BASE-STREAMS mode")

    # slice controls
    ap.add_argument("--min_rows", type=int, default=2000)
    ap.add_argument("--min_pos", type=int, default=50)
    ap.add_argument("--include_2d", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_parquet(args.meta_table)
    _ensure_cols(meta, ["race_id", "row_id", "y_pit"])

    if args.oof_preds.strip():
        preds = pd.read_parquet(args.oof_preds)
        _ensure_cols(preds, ["race_id", "row_id", "fold_id", "ablation_name", "y_true", "proba", "threshold"])
        mode = "META-PREDS"
        if not args.ref_ablation.strip():
            args.ref_ablation = "base_only"
    else:
        prob_cols = [c.strip() for c in args.streams.split(",") if c.strip()]
        preds = build_oof_preds_from_streams(meta, prob_cols=prob_cols, n_splits=args.n_splits, seed=args.seed)
        mode = "BASE-STREAMS"
        if not args.ref_ablation.strip():
            args.ref_ablation = "stream_tcn_proba"
        if args.write_generated_preds:
            preds.to_parquet(outdir / "generated_oof_preds.parquet", index=False)

    if args.ablations.strip():
        allow = {a.strip() for a in args.ablations.split(",") if a.strip()}
        preds = preds[preds["ablation_name"].isin(allow)].copy()

    meta2 = add_default_slices(meta)
    if args.include_2d:
        meta2 = add_2d_slices(meta2)

    merged = preds.merge(meta2, on=["race_id", "row_id"], how="left", validate="many_to_one")

    slice_cols = [
        "slice_race_phase", "slice_fcy", "slice_wetness", "slice_traffic",
        "slice_tyre_age_bin", "slice_pitstops", "slice_undercut_pressure", "slice_track_cat",
    ]
    if args.include_2d:
        slice_cols += ["slice_wetness_x_fcy", "slice_phase_x_tyre", "slice_traffic_x_undercut", "slice_phase_x_traffic"]

    fold_parts = []
    for sc in slice_cols:
        if sc in merged.columns:
            fold_parts.append(compute_slice_metrics(merged, sc))
    fold_df = pd.concat(fold_parts, ignore_index=True)

    fold_df.to_csv(outdir / "slice_metrics_fold.csv", index=False)
    summary_df = summarize_over_folds(fold_df)
    summary_df.to_csv(outdir / "slice_metrics_summary.csv", index=False)

    deltas_df = deltas_vs_reference(summary_df, ref_ablation=args.ref_ablation)
    deltas_df.to_csv(outdir / "slice_deltas_vs_ref.csv", index=False)

    leaderboard = build_leaderboard(
        deltas_df, min_rows=args.min_rows, min_pos=args.min_pos, metric="delta_PR_AUC_vs_ref", topk=3
    )
    leaderboard.to_csv(outdir / "slice_leaderboard.csv", index=False)

    print(f"[slice_value_analysis] mode={mode}")
    print(f"[slice_value_analysis] wrote outputs to: {outdir}")
    print(f"[slice_value_analysis] ref_ablation={args.ref_ablation}")


if __name__ == "__main__":
    main()

'''
python -m src.utils.slice_value_analysis \
  --meta_table runs/ablation/stage1_meta_ablation/cache/meta_table.parquet \
  --outdir runs/ablation/stage1_meta_ablation/reports/slices_base_streams \
  --ref_ablation stream_tcn_proba \
  --n_splits 5 --seed 42 \
  --include_2d \
  --write_generated_preds
'''