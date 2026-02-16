#!/usr/bin/env python3
# src/utils/ablation_eval.py
"""
Ablation evaluation for Stage-1 (binary pit/no-pit) stacked ensemble using
a rigorous two-level CV procedure:

PART A (Base OOF):
  - 5-fold race-wise CV -> generate OOF probabilities for base learners
    (svm, rf, xgb, ann) + TCN (tcn_proba, tcn_effective_len).
  - Build a meta-table with:
      race_id, row_id (ID only), y_pit (label),
      p_* base OOF probs, tcn_* OOF probs,
      raw/context features (from get_stage1_xy()).

PART B (Meta OOF + Ablation):
  - 5-fold race-wise CV on the meta-table.
  - For each fold and each ablation setting:
      * train meta model (XGB) on meta-train rows ONLY
      * tune threshold on meta-train ONLY (maximize F1)
      * evaluate on meta-test ONLY
  - Report fold-level metrics + aggregated deltas vs baseline.

Outputs:
  1) ablation_results.csv (fold-level)
  2) ablation_summary.csv (aggregated across folds; deltas vs baseline)
  3) prints top-5 most damaging ablations by PR-AUC drop and LogLoss increase
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from src.data.data import (
    load_stage1_dataset,
    get_stage1_xy,
    make_race_group_folds,
    infer_feature_types,
    HOLDOUT_RACE_IDS,
    build_feature_sequences,
)

from src.data.preprocessing import build_preprocessor, PreprocessConfig, make_preprocessor_for_model
from src.models.models import (
    ModelConfig,
    build_model_pipeline,
    get_binary_base_learners,
    make_meta_binary_xgb,
    make_tcn_binary,
)

SEED_DEFAULT = 42


# -----------------------------
# Logging helpers
# -----------------------------
def vprint(verbose: bool, *args, **kwargs) -> None:
    if verbose:
        print(*args, **kwargs, flush=True)


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


# -----------------------------
# Metrics + threshold tuning
# -----------------------------
def _safe_logloss(y_true: np.ndarray, proba_pos: np.ndarray) -> float:
    # avoid logloss blow-ups on exact 0/1
    p = np.clip(proba_pos.astype(float), 1e-15, 1 - 1e-15)
    return float(log_loss(y_true, p, labels=[0, 1]))


def _binary_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)}


def tune_threshold_max_f1(
    y_true: np.ndarray,
    proba_pos: np.ndarray,
    *,
    grid: Optional[np.ndarray] = None,
) -> float:
    """
    Tune threshold to maximize F1 on TRAIN ONLY.
    Deterministic, same procedure for all ablations.

    grid default: 0.01..0.99 step 0.01
    """
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)

    best_t = 0.5
    best_f1 = -1.0

    # vectorized-ish: still cheap with 99 thresholds
    for t in grid:
        y_pred = (proba_pos >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    return best_t


def compute_fold_metrics(
    y_true: np.ndarray,
    proba_pos: np.ndarray,
    *,
    threshold: float,
) -> Dict[str, float]:
    y_true = y_true.astype(int)
    proba_pos = proba_pos.astype(float)

    y_pred = (proba_pos >= threshold).astype(int)

    pr_auc = float(average_precision_score(y_true, proba_pos))
    ll = _safe_logloss(y_true, proba_pos)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    counts = _binary_counts(y_true, y_pred)

    return {
        "PR_AUC": pr_auc,
        "LogLoss": ll,
        "F1": f1,
        "Precision": prec,
        "Recall": rec,
        **counts,
    }


# -----------------------------
# PART A: Base-level OOF probs
# -----------------------------
def collect_oof_base_probs_binary(
    X: pd.DataFrame,
    y: pd.Series,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    *,
    cfg: ModelConfig,
    base_keys: List[str],
    verbose: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Generate base OOF probabilities for tabular base learners, race-wise folds.
    Returns:
      df_probs: columns p_<name>
      used_names: list of base models in order
    """
    base_all = get_binary_base_learners(cfg)
    # keep only requested and present
    base_names = [k for k in base_keys if k in base_all]
    if not base_names:
        raise ValueError(f"No requested base learners found. Requested={base_keys}, available={list(base_all.keys())}")

    n = len(y)
    out = np.zeros((n, len(base_names)), dtype=float)

    # infer once on full X for consistent preprocessor config; make_preprocessor_for_model uses col lists
    num_cols, cat_cols = infer_feature_types(X)

    vprint(verbose, f"[PART A][tabular oof] n={n} folds={len(folds)} base={base_names}")

    for fold_id, (tr_idx, te_idx) in enumerate(folds):
        vprint(verbose, f"[PART A][tabular oof][fold {fold_id}] train={len(tr_idx)} test={len(te_idx)}")
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_te = X.iloc[te_idx]

        for j, name in enumerate(base_names):
            vprint(verbose, f"  -> fit base='{name}'")
            est = clone(base_all[name])

            # model-specific preprocessor (keeps behavior aligned with your evaluate_stack.py)
            if name == "ann":
                pre = make_preprocessor_for_model("ann", num_cols=num_cols, cat_cols=cat_cols)
            elif name == "svm":
                pre = make_preprocessor_for_model("svm", num_cols=num_cols, cat_cols=cat_cols)
            elif name == "rf":
                pre = make_preprocessor_for_model("rf", num_cols=num_cols, cat_cols=cat_cols)
            elif name == "xgb":
                pre = make_preprocessor_for_model("xgb", num_cols=num_cols, cat_cols=cat_cols)
            else:
                pre = make_preprocessor_for_model(name, num_cols=num_cols, cat_cols=cat_cols)

            pipe = build_model_pipeline(pre, est)
            pipe.fit(X_tr, y_tr)

            proba_pos = pipe.predict_proba(X_te)[:, 1].astype(float)
            out[te_idx, j] = proba_pos

            vprint(verbose, f"     wrote p_{name}: mean={float(np.mean(proba_pos)):.6f}")

    df_probs = pd.DataFrame(out, columns=[f"p_{n}" for n in base_names])
    return df_probs, base_names


def collect_oof_tcn_binary(
    df1: pd.DataFrame,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    *,
    cfg: ModelConfig,
    seq_len: int,
    pad_left: bool = True,
    add_timestep_mask: bool = True,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate OOF TCN predictions:
      - trains TCN per fold on train races only
      - predicts on test races only
      - writes predictions to the "last timestep row" (idx_last) for each sequence,
        which corresponds to each lap row in df1.

    Returns:
      tcn_oof_proba: (N,) float
      tcn_eff_scaled: (N,) float in [0,1] (effective_len / seq_len)
    """
    y_full = df1["y_pit"].astype(int)
    keys = df1[["race_id", "driver_id", "lapno"]].copy()

    # features for preprocessing (exclude label + identifiers)
    X_tab = df1.drop(
        columns=["y_pit", "y_compound", "race_id", "driver_id", "row_id"],
        errors="ignore",
    ).copy()
    num_cols, cat_cols = infer_feature_types(X_tab)
    pre_proto = make_preprocessor_for_model("tcn", num_cols=num_cols, cat_cols=cat_cols)

    N = len(df1)
    oof_pred = np.full(N, np.nan, dtype=float)
    oof_eff_len = np.zeros(N, dtype=float)

    vprint(verbose, f"[PART A][tcn oof] N={N} folds={len(folds)} seq_len={seq_len} pad_left={pad_left} tmask={add_timestep_mask}")

    for fold_id, (tr_idx, te_idx) in enumerate(folds):
        vprint(verbose, f"[PART A][tcn oof][fold {fold_id}] train={len(tr_idx)} test={len(te_idx)}")

        X_tr, y_tr = X_tab.iloc[tr_idx], y_full.iloc[tr_idx]

        pre = clone(pre_proto)
        pre.fit(X_tr, y_tr)

        Xt_all = pre.transform(X_tab)
        if hasattr(Xt_all, "toarray"):
            Xt_all = Xt_all.toarray()
        Xt_all = Xt_all.astype(np.float32)

        X_seq, y_seq, idx_last, seq_idx, eff_len = build_feature_sequences(
            keys,
            Xt_all,
            y_full,
            seq_len=seq_len,
            pad_left=pad_left,
            add_timestep_mask=add_timestep_mask,
        )

        # allow padded timesteps (-1) in membership checks
        if pad_left:
            tr_mask = np.all((seq_idx == -1) | np.isin(seq_idx, tr_idx), axis=1)
            te_mask = np.all((seq_idx == -1) | np.isin(seq_idx, te_idx), axis=1)
        else:
            tr_mask = np.all(np.isin(seq_idx, tr_idx), axis=1)
            te_mask = np.all(np.isin(seq_idx, te_idx), axis=1)

        X_seq_tr, y_seq_tr = X_seq[tr_mask], y_seq[tr_mask]
        X_seq_te = X_seq[te_mask]
        idx_last_te = idx_last[te_mask]
        eff_len_te = eff_len[te_mask]

        if len(y_seq_tr) == 0 or len(X_seq_te) == 0:
            vprint(verbose, f"[PART A][tcn oof][fold {fold_id}] skipping (no sequences after filtering)")
            continue

        n_pos = int(np.sum(y_seq_tr == 1))
        n_neg = int(np.sum(y_seq_tr == 0))
        if n_pos == 0 or n_neg == 0:
            const_p = 1.0 if n_neg == 0 else 0.0
            oof_pred[idx_last_te] = const_p
            oof_eff_len[idx_last_te] = eff_len_te
            vprint(verbose, f"[PART A][tcn oof][fold {fold_id}] degenerate y -> const_p={const_p}")
            continue

        pos_w = min(20.0, float(n_neg / n_pos))
        class_w = {0: 1.0, 1: pos_w}

        tcn = make_tcn_binary(cfg, seq_len=seq_len)
        tcn.fit(X_seq_tr, y_seq_tr, class_weight=class_w)

        proba_pos = tcn.predict_proba(X_seq_te)[:, 1].astype(float)
        oof_pred[idx_last_te] = proba_pos
        oof_eff_len[idx_last_te] = eff_len_te

        vprint(verbose, f"[PART A][tcn oof][fold {fold_id}] wrote {len(idx_last_te)} preds (pos_w={pos_w:.2f})")

    # fill any NaNs robustly (should be none with pad_left=True)
    oof_pred = np.nan_to_num(oof_pred, nan=0.0)
    eff_scaled = oof_eff_len / float(seq_len)
    return oof_pred, eff_scaled


def build_meta_table_stage1(
    df1: pd.DataFrame,
    X1: pd.DataFrame,
    y1: pd.Series,
    folds_base: List[Tuple[np.ndarray, np.ndarray]],
    *,
    cfg: ModelConfig,
    seq_len_tcn: int,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Builds meta-table with:
      race_id, row_id (ID only), y_pit,
      p_svm, p_rf, p_xgb, p_ann,
      tcn_proba, tcn_effective_len,
      raw/context features from X1 (with ID/label columns stripped out).
    """
    # --- Base OOF probs (tabular) ---
    base_keys = ["svm", "rf", "xgb", "ann"]
    df_base_probs, used_base = collect_oof_base_probs_binary(
        X1, y1, folds_base, cfg=cfg, base_keys=base_keys, verbose=verbose
    )

    # --- Base OOF probs (TCN) ---
    tcn_oof, tcn_eff_scaled = collect_oof_tcn_binary(
        df1,
        folds_base,
        cfg=cfg,
        seq_len=seq_len_tcn,
        pad_left=True,
        add_timestep_mask=True,
        verbose=verbose,
    )

    # --- Core meta-table (IDs + label) ---
    meta = pd.DataFrame(
        {
            "race_id": df1["race_id"].to_numpy(),
            "row_id": df1["row_id"].to_numpy() if "row_id" in df1.columns else np.arange(len(df1)),
            "y_pit": y1.to_numpy().astype(int),
        }
    )

    # --- Add OOF prob features ---
    meta = pd.concat([meta, df_base_probs.reset_index(drop=True)], axis=1)
    meta["tcn_proba"] = tcn_oof.astype(float)
    meta["tcn_effective_len"] = tcn_eff_scaled.astype(float)

    # --- Add raw/context features, but HARD DROP anything that must not appear / duplicate ---
    X1_safe = X1.copy()
    X1_safe = X1_safe.drop(columns=["race_id", "row_id", "y_pit", "driver_id"], errors="ignore")
    meta = pd.concat([meta, X1_safe.reset_index(drop=True)], axis=1)

    # --- Sanity checks (fail fast BEFORE saving parquet) ---
    if meta.columns.duplicated().any():
        dupes = meta.columns[meta.columns.duplicated()].tolist()
        raise RuntimeError(f"Duplicate columns in meta table (fix feature sources): {dupes}")

    if "y_pit" not in meta.columns:
        raise RuntimeError("meta-table missing y_pit")
    if "race_id" not in meta.columns:
        raise RuntimeError("meta-table missing race_id")
    if "row_id" not in meta.columns:
        raise RuntimeError("meta-table missing row_id")
    if "driver_id" in meta.columns:
        raise RuntimeError("driver_id leaked into meta-table")

    expected_base = [f"p_{k}" for k in used_base]
    missing = [c for c in expected_base if c not in meta.columns]
    if missing:
        raise RuntimeError(f"Missing expected base prob cols: {missing}")

    return meta


# -----------------------------
# PART B: Meta CV ablation eval
# -----------------------------
def make_ablation_specs(
    *,
    baseline_cols: List[str],
    base_prob_cols: List[str],
    tcn_cols: List[str],
    raw_cols: List[str],
    raw_groups: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """
    Returns mapping ablation_name -> list_of_columns_to_use (feature columns ONLY).
    """
    baseline_set = set(baseline_cols)

    def keep(cols: List[str]) -> List[str]:
        cols2 = [c for c in cols if c in baseline_set]
        if not cols2:
            raise ValueError("Ablation produced empty feature set.")
        return cols2

    def drop(cols_to_drop: List[str]) -> List[str]:
        drop_set = set(cols_to_drop)
        cols2 = [c for c in baseline_cols if c not in drop_set]
        if not cols2:
            raise ValueError("Ablation produced empty feature set.")
        return cols2

    specs: Dict[str, List[str]] = {}

    # (0) baseline_full
    specs["baseline_full"] = baseline_cols

    # TIER 1
    specs["base_only"] = keep(base_prob_cols)  # includes tcn cols if present in base_prob_cols
    specs["raw_only"] = keep(raw_cols)
    specs["classical_only"] = keep([c for c in base_prob_cols if c.startswith("p_")])  # p_* only
    specs["tcn_only"] = keep(tcn_cols)

    # TIER 2 leave-one-base-out
    for c in [x for x in base_prob_cols if x.startswith("p_")]:
        specs[f"drop_{c}"] = drop([c])
    # TCN leave-outs
    for c in tcn_cols:
        specs[f"drop_{c}"] = drop([c])

    # TIER 3 raw group LOGO
    for gname, gcols in raw_groups.items():
        specs[f"drop_{gname}"] = drop(gcols)

    # Optional targeted combos (max 3)
    if "G_tyre" in raw_groups and "G_race_control" in raw_groups:
        specs["drop_G_tyre_and_G_race_control"] = drop(raw_groups["G_tyre"] + raw_groups["G_race_control"])
    if "G_tyre" in raw_groups and "G_traffic" in raw_groups:
        specs["drop_G_tyre_and_G_traffic"] = drop(raw_groups["G_tyre"] + raw_groups["G_traffic"])
    if "G_weather" in raw_groups and "G_tyre" in raw_groups:
        specs["drop_G_weather_and_G_tyre"] = drop(raw_groups["G_weather"] + raw_groups["G_tyre"])

    return specs


def run_meta_ablation_cv(
    meta_df: pd.DataFrame,
    folds_meta: List[Tuple[np.ndarray, np.ndarray]],
    *,
    cfg: ModelConfig,
    ablation_specs: Dict[str, List[str]],
    verbose: bool = False,
) -> pd.DataFrame:
    """
    For each meta fold and ablation spec:
      - fit meta model on train fold (XGB) using selected cols
      - tune threshold on train fold ONLY (maximize F1)
      - evaluate on test fold ONLY
    Returns fold-level results dataframe.
    """
    y = meta_df["y_pit"].to_numpy().astype(int)

    results_rows: List[Dict[str, Any]] = []
    meta_model_proto = make_meta_binary_xgb(cfg)

    for fold_id, (tr_idx, te_idx) in enumerate(folds_meta):
        vprint(verbose, f"\n[PART B][meta cv][fold {fold_id}] train={len(tr_idx)} test={len(te_idx)}")
        y_tr = y[tr_idx]
        y_te = y[te_idx]

        for ab_name, cols in ablation_specs.items():
            X_tr = meta_df.iloc[tr_idx][cols]
            X_te = meta_df.iloc[te_idx][cols]

            # Preprocess inside fold (no leakage)
            num_cols, cat_cols = infer_feature_types(X_tr)
            pre = build_preprocessor(
                num_cols=num_cols,
                cat_cols=cat_cols,
                cfg=PreprocessConfig(scale_numeric=False, sparse_onehot=True),
            )

            meta_est = clone(meta_model_proto)
            pipe = build_model_pipeline(pre, meta_est)
            pipe.fit(X_tr, y_tr)

            # threshold tuning on meta-train only
            proba_tr = pipe.predict_proba(X_tr)[:, 1].astype(float)
            thr = tune_threshold_max_f1(y_tr, proba_tr)

            # evaluate on meta-test only
            proba_te = pipe.predict_proba(X_te)[:, 1].astype(float)
            m = compute_fold_metrics(y_te, proba_te, threshold=thr)

            row = {
                "ablation_name": ab_name,
                "meta_fold_id": int(fold_id),
                "n_test_rows": int(len(te_idx)),
                "threshold": float(thr),
                **m,
            }
            results_rows.append(row)

            if verbose:
                vprint(
                    verbose,
                    f"[PART B][{ab_name}][fold {fold_id}] "
                    f"PR_AUC={m['PR_AUC']:.4f} LogLoss={m['LogLoss']:.4f} "
                    f"F1={m['F1']:.4f} P={m['Precision']:.4f} R={m['Recall']:.4f} "
                    f"TP={m['TP']} FP={m['FP']} FN={m['FN']} TN={m['TN']}"
                )

    return pd.DataFrame(results_rows)


def summarize_ablation_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate mean/std across folds per ablation + deltas vs baseline_full.
    """
    agg = results_df.groupby("ablation_name").agg(
        mean_PR_AUC=("PR_AUC", "mean"),
        std_PR_AUC=("PR_AUC", "std"),
        mean_LogLoss=("LogLoss", "mean"),
        std_LogLoss=("LogLoss", "std"),
        mean_F1=("F1", "mean"),
        std_F1=("F1", "std"),
        mean_Precision=("Precision", "mean"),
        mean_Recall=("Recall", "mean"),
        mean_TP=("TP", "mean"),
        mean_FP=("FP", "mean"),
        mean_FN=("FN", "mean"),
        mean_TN=("TN", "mean"),
    ).reset_index()

    base = agg[agg["ablation_name"] == "baseline_full"]
    if len(base) != 1:
        raise RuntimeError("baseline_full summary missing or duplicated.")
    base_row = base.iloc[0]

    agg["delta_PR_AUC_vs_baseline"] = agg["mean_PR_AUC"] - float(base_row["mean_PR_AUC"])
    agg["delta_LogLoss_vs_baseline"] = agg["mean_LogLoss"] - float(base_row["mean_LogLoss"])
    agg["delta_F1_vs_baseline"] = agg["mean_F1"] - float(base_row["mean_F1"])

    # helpful “damage” measures (positive = worse)
    agg["PR_AUC_drop_vs_baseline"] = float(base_row["mean_PR_AUC"]) - agg["mean_PR_AUC"]
    agg["LogLoss_increase_vs_baseline"] = agg["mean_LogLoss"] - float(base_row["mean_LogLoss"])

    return agg


def print_top5_damage(summary_df: pd.DataFrame) -> None:
    def _top(df: pd.DataFrame, col: str) -> pd.DataFrame:
        return df[df["ablation_name"] != "baseline_full"].sort_values(col, ascending=False).head(5)

    top_pr = _top(summary_df, "PR_AUC_drop_vs_baseline")
    top_ll = _top(summary_df, "LogLoss_increase_vs_baseline")

    print("\nTop-5 most damaging ablations (by PR-AUC drop):")
    for _, r in top_pr.iterrows():
        print(
            f"  {r['ablation_name']}: PR_AUC_drop={r['PR_AUC_drop_vs_baseline']:.4f} "
            f"(mean_PR_AUC={r['mean_PR_AUC']:.4f})"
        )

    print("\nTop-5 most damaging ablations (by LogLoss increase):")
    for _, r in top_ll.iterrows():
        print(
            f"  {r['ablation_name']}: LogLoss_increase={r['LogLoss_increase_vs_baseline']:.4f} "
            f"(mean_LogLoss={r['mean_LogLoss']:.4f})"
        )


# -----------------------------
# CLI
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_stage1", required=True, help="Path to stage1 dataset (pit/no-pit), e.g. output1.csv")
    ap.add_argument("--outdir", default="results/runs/ablation", help="Output directory root")
    ap.add_argument("--run_name", default="stage1_meta_ablation", help="Subfolder under outdir")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--seq_len_tcn", type=int, default=8)
    ap.add_argument("--recompute_base_oof", action="store_true", help="Force recomputation of PART A base OOF cache")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--smoke_test", action="store_true", help="Run a quick 1-fold smoke test (base + meta).")
    ap.add_argument(
        "--smoke_schema_only",
        action="store_true",
        help="Fast smoke test: build a tiny meta table (no training) and try writing parquet.",
    )

    args = ap.parse_args()

    out_root = _ensure_dir(Path(args.outdir) / args.run_name)
    cache_dir = _ensure_dir(out_root / "cache")
    _ensure_dir(out_root / "reports")

    cfg = ModelConfig(random_state=int(args.seed))

    print(f"[ablation_eval] out_root: {out_root}")
    print(f"[ablation_eval] seed={args.seed} n_splits={args.n_splits} seq_len_tcn={args.seq_len_tcn}")
    print(f"[ablation_eval] HOLDOUT_RACE_IDS={list(map(int, HOLDOUT_RACE_IDS))}")

    # -----------------------------
    # Load data
    # -----------------------------
    vprint(args.verbose, "[ablation_eval] loading stage1 dataset...")
    df1 = load_stage1_dataset(args.data_stage1)
    X1, y1 = get_stage1_xy(df1)

    if args.smoke_schema_only:
        print("[smoke_schema_only] Running fast schema/parquet smoke test (NO training).")

        # pick 2 non-holdout races to keep it tiny
        all_races = sorted(df1["race_id"].unique().tolist())
        smoke_races = [r for r in all_races if int(r) not in set(map(int, HOLDOUT_RACE_IDS))][:2]
        df_smoke = df1[df1["race_id"].isin(smoke_races)].copy()

        # cap rows hard so it’s always fast
        df_smoke = df_smoke.head(2000).reset_index(drop=True)

        X_smoke, y_smoke = get_stage1_xy(df_smoke)

        # IMPORTANT: strip any IDs/labels so we can't create duplicates
        X_smoke_safe = X_smoke.drop(columns=["race_id", "row_id", "y_pit", "driver_id"], errors="ignore")

        # make dummy prob cols (no training)
        meta_smoke = pd.DataFrame(
            {
                "race_id": df_smoke["race_id"].to_numpy(),
                "row_id": df_smoke["row_id"].to_numpy() if "row_id" in df_smoke.columns else np.arange(len(df_smoke)),
                "y_pit": y_smoke.to_numpy().astype(int),
                "p_svm": np.zeros(len(df_smoke), dtype=float),
                "p_rf": np.zeros(len(df_smoke), dtype=float),
                "p_xgb": np.zeros(len(df_smoke), dtype=float),
                "p_ann": np.zeros(len(df_smoke), dtype=float),
                "tcn_proba": np.zeros(len(df_smoke), dtype=float),
                "tcn_effective_len": np.zeros(len(df_smoke), dtype=float),
            }
        )
        meta_smoke = pd.concat([meta_smoke, X_smoke_safe.reset_index(drop=True)], axis=1)

        # fail fast if duplicates exist (this is what killed your run)
        if meta_smoke.columns.duplicated().any():
            dupes = meta_smoke.columns[meta_smoke.columns.duplicated()].tolist()
            raise RuntimeError(f"[smoke_schema_only] Duplicate columns detected: {dupes}")

        smoke_path = out_root / "reports" / "smoke_meta_table.parquet"
        meta_smoke.to_parquet(smoke_path, index=False)
        print(f"[smoke_schema_only] OK: wrote {smoke_path} with {len(meta_smoke)} rows and {meta_smoke.shape[1]} cols.")
        return


    n_races = int(df1["race_id"].nunique())
    vprint(args.verbose, f"[ablation_eval] df1 rows={len(df1)} unique_races={n_races}")
    vprint(args.verbose, f"[ablation_eval] X1 shape={X1.shape} y1 pos_rate={float(np.mean(y1.to_numpy())):.4f}")

    # -----------------------------
    # PART A folds (base OOF)
    # -----------------------------
    vprint(args.verbose, "[PART A] building race-group folds (base OOF)...")
    fb_base = make_race_group_folds(df1, target_col="y_pit", n_splits=args.n_splits, seed=args.seed)
    folds_base = fb_base.folds

    # cache paths
    meta_table_path = cache_dir / "meta_table.parquet"
    meta_manifest_path = cache_dir / "meta_table_manifest.json"

    if meta_table_path.exists() and (not args.recompute_base_oof):
        print(f"[PART A] loading cached meta-table: {meta_table_path}")
        meta_df = pd.read_parquet(meta_table_path)
    else:
        print("[PART A] generating base-level OOF probabilities (tabular + TCN)...")
        meta_df = build_meta_table_stage1(
            df1,
            X1,
            y1,
            folds_base,
            cfg=cfg,
            seq_len_tcn=int(args.seq_len_tcn),
            verbose=args.verbose,
        )
        meta_df.to_parquet(meta_table_path, index=False)
        _write_json(
            meta_manifest_path,
            {
                "data_stage1": str(args.data_stage1),
                "n_rows": int(len(meta_df)),
                "n_races": int(meta_df["race_id"].nunique()),
                "n_splits_base": int(args.n_splits),
                "seed": int(args.seed),
                "seq_len_tcn": int(args.seq_len_tcn),
                "holdout_race_ids": list(map(int, HOLDOUT_RACE_IDS)),
                "model_config": asdict(cfg),
            },
        )
        print(f"[PART A] wrote meta-table cache: {meta_table_path}")

    # -----------------------------
    # Define feature sets
    # -----------------------------
    # label / identifiers (NEVER used as features)
    id_cols = {"race_id", "row_id"}
    label_col = "y_pit"

    # base prob cols (as per your manifest)
    base_prob_cols = ["p_svm", "p_rf", "p_xgb", "p_ann", "tcn_proba", "tcn_effective_len"]
    base_prob_cols = [c for c in base_prob_cols if c in meta_df.columns]

    tcn_cols = [c for c in ["tcn_proba", "tcn_effective_len"] if c in meta_df.columns]
    p_cols = [c for c in base_prob_cols if c.startswith("p_")]

    # raw/context cols = everything except ids, label, and prob cols
    raw_cols = [c for c in meta_df.columns if c not in (id_cols | {label_col} | set(base_prob_cols))]
    # safety: never allow any forbidden cols to sneak in
    raw_cols = [c for c in raw_cols if c not in {"driver_id"}]

    baseline_cols = base_prob_cols + raw_cols

    # raw groups
    raw_groups: Dict[str, List[str]] = {
        "G_race_phase": [c for c in ["lapno", "race_progress"] if c in meta_df.columns],
        "G_race_control": [c for c in ["fcy_status"] if c in meta_df.columns],
        "G_track": [c for c in ["race_track", "track_category"] if c in meta_df.columns],
        "G_tyre": [c for c in ["current_compound", "tyre_age", "fulfilled_second_compound"] if c in meta_df.columns],
        "G_traffic": [c for c in ["position", "interval", "close_ahead"] if c in meta_df.columns],
        "G_strategy_history": [c for c in ["pit_stops_so_far", "tyre_change_pursuer"] if c in meta_df.columns],
        "G_pace": [c for c in ["lap_time"] if c in meta_df.columns],
        "G_weather": [c for c in ["rained_yet", "is_raining", "minutes_rain"] if c in meta_df.columns],
    }

    # -----------------------------
    # PART B folds (meta OOF ablation)
    # -----------------------------
    vprint(args.verbose, "[PART B] building race-group folds (meta CV)...")
    # Use make_race_group_folds on the meta table as well (race-wise)
    fb_meta = make_race_group_folds(meta_df[["race_id", "y_pit"]].copy(), target_col="y_pit", n_splits=args.n_splits, seed=args.seed)
    folds_meta = fb_meta.folds
    if args.smoke_test:
        folds_base = folds_base[:1]
        folds_meta = folds_meta[:1]

    # Ablation specs
    ablation_specs = make_ablation_specs(
        baseline_cols=baseline_cols,
        base_prob_cols=base_prob_cols,
        tcn_cols=tcn_cols,
        raw_cols=raw_cols,
        raw_groups=raw_groups,
    )
    _write_json(out_root / "reports" / "ablation_specs.json", {"ablations": list(ablation_specs.keys())})
    _write_json(out_root / "reports" / "feature_sets.json", {
        "baseline_cols": baseline_cols,
        "base_prob_cols": base_prob_cols,
        "raw_cols": raw_cols,
        "raw_groups": raw_groups,
    })

    print(f"[PART B] running meta ablation CV: n_ablations={len(ablation_specs)} n_folds={len(folds_meta)}")
    results_df = run_meta_ablation_cv(
        meta_df,
        folds_meta,
        cfg=cfg,
        ablation_specs=ablation_specs,
        verbose=args.verbose,
    )

    # Write fold-level results
    results_path = out_root / "reports" / "ablation_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"[reports] wrote: {results_path}")

    # Aggregate + deltas
    summary_df = summarize_ablation_results(results_df)
    summary_path = out_root / "reports" / "ablation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[reports] wrote: {summary_path}")

    # Rankings
    ranking_pr = summary_df.sort_values("PR_AUC_drop_vs_baseline", ascending=False)
    ranking_ll = summary_df.sort_values("LogLoss_increase_vs_baseline", ascending=False)
    ranking_pr.to_csv(out_root / "reports" / "ranking_by_pr_auc_drop.csv", index=False)
    ranking_ll.to_csv(out_root / "reports" / "ranking_by_logloss_increase.csv", index=False)

    print_top5_damage(summary_df)

    print(f"\n[ablation_eval] done. Outputs in: {out_root / 'reports'}")


if __name__ == "__main__":
    main()

"""
python -m src.utils.ablation_eval \
  --data_stage1 data/processed/output1.csv \
  --outdir runs/ablation \
  --run_name stage1_meta_ablation \
  --n_splits 5 \
  --seed 42 \
  --seq_len_tcn 8 \
  --verbose
"""