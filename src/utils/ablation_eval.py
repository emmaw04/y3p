#!/usr/bin/env python3
# src/utils/ablation_eval.py
"""
Ablation evaluation for Stage-1 (binary pit/no-pit) using a rigorous two-level CV procedure.

MODE A: META (stacked ensemble ablation)
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

MODE B: SEQ (single sequence model ablation: TCN / TCN-GRU)
  - 5-fold race-wise CV on lap rows.
  - For each fold and each ablation setting:
      * fit sequence model on train sequences ONLY
      * tune threshold on train sequences ONLY
      * evaluate on test sequences ONLY
  - Ablation is applied to the *sequence input tensors* by zeroing selected feature dims.

Why "zeroing"?
  - After preprocessing, we no longer have raw columns; we have transformed feature dims.
  - Zeroing selected dims is a clean “remove this information” intervention.

Outputs (for each mode):
  - reports/<mode>_ablation_results.csv (fold-level)
  - reports/<mode>_ablation_summary.csv (aggregated; deltas vs baseline)
  - reports/<mode>_ablation_specs.json
  - reports/<mode>_feature_sets.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
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
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)

    best_t = 0.5
    best_f1 = -1.0
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
# PART A: Base-level OOF probs (META mode)
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
    base_all = get_binary_base_learners(cfg)
    base_names = [k for k in base_keys if k in base_all]
    if not base_names:
        raise ValueError(f"No requested base learners found. Requested={base_keys}, available={list(base_all.keys())}")

    n = len(y)
    out = np.zeros((n, len(base_names)), dtype=float)

    num_cols, cat_cols = infer_feature_types(X)

    vprint(verbose, f"[META][PART A][tabular oof] n={n} folds={len(folds)} base={base_names}")

    for fold_id, (tr_idx, te_idx) in enumerate(folds):
        vprint(verbose, f"[META][PART A][tabular oof][fold {fold_id}] train={len(tr_idx)} test={len(te_idx)}")
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_te = X.iloc[te_idx]

        for j, name in enumerate(base_names):
            vprint(verbose, f"  -> fit base='{name}'")
            est = clone(base_all[name])

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
    model_kind: str = "tcn",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    OOF predictions from a sequence model. model_kind:
      - "tcn": uses make_tcn_binary(cfg, seq_len)
      - "tcn_gru": tries to call make_tcn_gru_binary(cfg, seq_len) if present
                  (you must implement/import it in src.models.models).
    """
    y_full = df1["y_pit"].astype(int)
    keys = df1[["race_id", "driver_id", "lapno"]].copy()

    X_tab = df1.drop(
        columns=["y_pit", "y_compound", "race_id", "driver_id", "row_id"],
        errors="ignore",
    ).copy()

    num_cols, cat_cols = infer_feature_types(X_tab)
    pre_proto = make_preprocessor_for_model("tcn", num_cols=num_cols, cat_cols=cat_cols)

    N = len(df1)
    oof_pred = np.full(N, np.nan, dtype=float)
    oof_eff_len = np.zeros(N, dtype=float)

    vprint(verbose, f"[SEQ-OFF][{model_kind}] N={N} folds={len(folds)} seq_len={seq_len} pad_left={pad_left} tmask={add_timestep_mask}")

    for fold_id, (tr_idx, te_idx) in enumerate(folds):
        vprint(verbose, f"[SEQ-OFF][{model_kind}][fold {fold_id}] train={len(tr_idx)} test={len(te_idx)}")

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

        # membership filtering
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
            vprint(verbose, f"[SEQ-OFF][{model_kind}][fold {fold_id}] skipping (no sequences after filtering)")
            continue

        n_pos = int(np.sum(y_seq_tr == 1))
        n_neg = int(np.sum(y_seq_tr == 0))
        if n_pos == 0 or n_neg == 0:
            const_p = 1.0 if n_neg == 0 else 0.0
            oof_pred[idx_last_te] = const_p
            oof_eff_len[idx_last_te] = eff_len_te
            vprint(verbose, f"[SEQ-OFF][{model_kind}][fold {fold_id}] degenerate y -> const_p={const_p}")
            continue

        pos_w = min(20.0, float(n_neg / n_pos))
        class_w = {0: 1.0, 1: pos_w}

        if model_kind == "tcn":
            model = make_tcn_binary(cfg, seq_len=seq_len)
        elif model_kind == "tcn_gru":
            from src.models.models import make_tcn_gru_binary  # type: ignore
            model = make_tcn_gru_binary(cfg)  # seq_len inferred from X shape by SciKeras
        else:
            raise ValueError(f"Unknown model_kind={model_kind}")

        model.fit(X_seq_tr, y_seq_tr, class_weight=class_w)
        proba_pos = model.predict_proba(X_seq_te)[:, 1].astype(float)

        oof_pred[idx_last_te] = proba_pos
        oof_eff_len[idx_last_te] = eff_len_te
        vprint(verbose, f"[SEQ-OFF][{model_kind}][fold {fold_id}] wrote {len(idx_last_te)} preds (pos_w={pos_w:.2f})")

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
    base_keys = ["svm", "rf", "xgb", "ann"]
    df_base_probs, used_base = collect_oof_base_probs_binary(
        X1, y1, folds_base, cfg=cfg, base_keys=base_keys, verbose=verbose
    )

    tcn_oof, tcn_eff_scaled = collect_oof_tcn_binary(
        df1,
        folds_base,
        cfg=cfg,
        seq_len=seq_len_tcn,
        pad_left=True,
        add_timestep_mask=True,
        verbose=verbose,
        model_kind="tcn",
    )

    meta = pd.DataFrame(
        {
            "race_id": df1["race_id"].to_numpy(),
            "row_id": df1["row_id"].to_numpy() if "row_id" in df1.columns else np.arange(len(df1)),
            "y_pit": y1.to_numpy().astype(int),
        }
    )

    meta = pd.concat([meta, df_base_probs.reset_index(drop=True)], axis=1)
    meta["tcn_proba"] = tcn_oof.astype(float)
    meta["tcn_effective_len"] = tcn_eff_scaled.astype(float)

    X1_safe = X1.copy().drop(columns=["race_id", "row_id", "y_pit", "driver_id"], errors="ignore")
    meta = pd.concat([meta, X1_safe.reset_index(drop=True)], axis=1)

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
# Feature groups (your engineered intent)
# -----------------------------
def build_stage1_feature_groups(cols: List[str]) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Returns (raw_groups, engineered_cols_present)
    """
    colset = set(cols)

    engineered_all = [
        "fcy_status",
        "track_category",
        "tyre_change_pursuer",
        "close_ahead",
        "gap_behind_s",
        "n_cars_within_5s_ahead",
        "tyre_age_diff_to_ahead",
        "rejoin_gap_ahead_est_s",
        "rejoin_gap_behind_est_s",
    ]
    engineered_all = [c for c in engineered_all if c in colset]

    undercut_defence = [c for c in ["tyre_change_pursuer", "gap_behind_s"] if c in colset]
    undercut_window = [c for c in [
        "n_cars_within_5s_ahead",
        "tyre_age_diff_to_ahead",
        "rejoin_gap_ahead_est_s",
        "rejoin_gap_behind_est_s",
    ] if c in colset]

    raw_groups: Dict[str, List[str]] = {
        "G_race_phase": [c for c in ["lapno", "race_progress"] if c in colset],
        "G_race_control": [c for c in ["fcy_status"] if c in colset],
        "G_track": [c for c in ["race_track", "track_category"] if c in colset],
        "G_tyre_state": [c for c in ["current_compound", "tyre_age", "fulfilled_second_compound"] if c in colset],
        "G_strategy_history": [c for c in ["pit_stops_so_far"] if c in colset],
        "G_pace": [c for c in ["lap_time"] if c in colset],
        "G_traffic_proximity": [c for c in ["position", "interval", "close_ahead"] if c in colset],
        "G_undercut_defence": undercut_defence,
        "G_undercut_window": undercut_window,
        "G_weather": [c for c in "is_wet_race", "is_raining", "minutes_rain"] if c in colset],
    }

    # remove empty groups
    raw_groups = {k: v for k, v in raw_groups.items() if len(v) > 0}
    return raw_groups, engineered_all


# -----------------------------
# Ablation specs (column-level for META; group specs for SEQ too)
# -----------------------------
def make_ablation_specs(
    *,
    baseline_cols: List[str],
    base_prob_cols: List[str],
    tcn_cols: List[str],
    raw_cols: List[str],
    raw_groups: Dict[str, List[str]],
    engineered_cols: List[str],
) -> Dict[str, List[str]]:
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
    specs["baseline_full"] = baseline_cols

    # TIER 1
    specs["base_only"] = keep(base_prob_cols)
    specs["raw_only"] = keep(raw_cols)
    specs["classical_only"] = keep([c for c in base_prob_cols if c.startswith("p_")])
    specs["tcn_only"] = keep(tcn_cols)

    # TIER 2 leave-one-base-out (p_* only)
    for c in [x for x in base_prob_cols if x.startswith("p_")]:
        specs[f"drop_{c}"] = drop([c])
    for c in tcn_cols:
        specs[f"drop_{c}"] = drop([c])

    # TIER 3 raw group LOGO
    for gname, gcols in raw_groups.items():
        specs[f"drop_{gname}"] = drop(gcols)

    # Engineered specific
    engineered_cols = [c for c in engineered_cols if c in baseline_set]
    if engineered_cols:
        specs["drop_ENGINEERED_ALL"] = drop(engineered_cols)
        for c in engineered_cols:
            specs[f"drop_{c}"] = drop([c])

    # Optional: all undercut pack
    undercut_pack = [c for c in [
        "tyre_change_pursuer", "gap_behind_s",
        "n_cars_within_5s_ahead", "tyre_age_diff_to_ahead",
        "rejoin_gap_ahead_est_s", "rejoin_gap_behind_est_s"
    ] if c in baseline_set]
    if undercut_pack:
        specs["drop_UNDERCUT_PACK"] = drop(undercut_pack)

    return specs


# -----------------------------
# PART B: Meta CV ablation eval (META mode)
# -----------------------------
def run_meta_ablation_cv(
    meta_df: pd.DataFrame,
    folds_meta: List[Tuple[np.ndarray, np.ndarray]],
    *,
    cfg: ModelConfig,
    ablation_specs: Dict[str, List[str]],
    verbose: bool = False,
) -> pd.DataFrame:
    y = meta_df["y_pit"].to_numpy().astype(int)

    results_rows: List[Dict[str, Any]] = []
    meta_model_proto = make_meta_binary_xgb(cfg)

    for fold_id, (tr_idx, te_idx) in enumerate(folds_meta):
        vprint(verbose, f"\n[META][fold {fold_id}] train={len(tr_idx)} test={len(te_idx)}")
        y_tr = y[tr_idx]
        y_te = y[te_idx]

        for ab_name, cols in ablation_specs.items():
            X_tr = meta_df.iloc[tr_idx][cols]
            X_te = meta_df.iloc[te_idx][cols]

            num_cols, cat_cols = infer_feature_types(X_tr)
            pre = build_preprocessor(
                num_cols=num_cols,
                cat_cols=cat_cols,
                cfg=PreprocessConfig(scale_numeric=False, sparse_onehot=True),
            )

            meta_est = clone(meta_model_proto)
            pipe = build_model_pipeline(pre, meta_est)
            pipe.fit(X_tr, y_tr)

            proba_tr = pipe.predict_proba(X_tr)[:, 1].astype(float)
            thr = tune_threshold_max_f1(y_tr, proba_tr)

            proba_te = pipe.predict_proba(X_te)[:, 1].astype(float)
            m = compute_fold_metrics(y_te, proba_te, threshold=thr)

            results_rows.append({
                "mode": "meta",
                "ablation_name": ab_name,
                "fold_id": int(fold_id),
                "n_test_rows": int(len(te_idx)),
                "threshold": float(thr),
                **m,
            })

            if verbose:
                vprint(
                    verbose,
                    f"[META][{ab_name}][fold {fold_id}] PR_AUC={m['PR_AUC']:.4f} "
                    f"LogLoss={m['LogLoss']:.4f} F1={m['F1']:.4f}"
                )

    return pd.DataFrame(results_rows)


# -----------------------------
# SEQ mode: sequence tensor ablation
# -----------------------------
def _feature_dim_groups_from_preprocessor(pre, X_cols: List[str]) -> Dict[str, List[int]]:
    """
    Map raw column groups -> transformed feature indices using pre.get_feature_names_out().
    We rely on sklearn's naming convention: <transformer>__<col>... for one-hot.
    """
    names = list(pre.get_feature_names_out(X_cols))
    # strip to strings
    names = [str(n) for n in names]

    def dims_for_col(col: str) -> List[int]:
        idxs: List[int] = []
        for i, nm in enumerate(names):
            base = nm.split("__", 1)[1] if "__" in nm else nm
            # base may be "col" or "col_value"
            if base == col or base.startswith(col + "_") or base.startswith(col + "="):
                idxs.append(i)
        return idxs

    # Build raw groups based on the columns present
    raw_groups, engineered_cols = build_stage1_feature_groups(X_cols)

    dim_groups: Dict[str, List[int]] = {}

    for gname, cols in raw_groups.items():
        dims: List[int] = []
        for c in cols:
            dims.extend(dims_for_col(c))
        dim_groups[gname] = sorted(set(dims))

    # Add engineered packs (dims)
    if engineered_cols:
        dims: List[int] = []
        for c in engineered_cols:
            dims.extend(dims_for_col(c))
        dim_groups["ENGINEERED_ALL"] = sorted(set(dims))

    # Undercut pack
    undercut_pack_cols = [c for c in [
        "tyre_change_pursuer", "gap_behind_s",
        "n_cars_within_5s_ahead", "tyre_age_diff_to_ahead",
        "rejoin_gap_ahead_est_s", "rejoin_gap_behind_est_s"
    ] if c in X_cols]
    if undercut_pack_cols:
        dims: List[int] = []
        for c in undercut_pack_cols:
            dims.extend(dims_for_col(c))
        dim_groups["UNDERCUT_PACK"] = sorted(set(dims))

    # Also include individual-col groups (optional but super informative)
    for c in engineered_cols:
        dim_groups[f"COL_{c}"] = sorted(set(dims_for_col(c)))

    # Drop empty
    dim_groups = {k: v for k, v in dim_groups.items() if len(v) > 0}
    return dim_groups


def _apply_dim_ablation(X_seq: np.ndarray, dims: List[int]) -> np.ndarray:
    """
    Zero out selected feature dims for every timestep.
    X_seq: (N, T, D)
    dims: list of feature indices into D
    """
    X2 = np.array(X_seq, copy=True)
    if len(dims) == 0:
        return X2
    X2[:, :, dims] = 0.0
    return X2

# -----------------------------
# TABULAR base-model ablation (SVM / XGB etc.)
# -----------------------------
def make_tabular_ablation_specs(
    X_cols: List[str],
) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
    """
    Mirror SEQ-style ablations, but at *raw column level*.

    Returns:
      ablation_specs: dict name -> list of columns to KEEP
      feature_sets: metadata (baseline_cols, raw_groups, engineered_cols, undercut_pack_cols)
    """
    colset = set(X_cols)
    baseline_cols = list(X_cols)

    raw_groups, engineered_cols = build_stage1_feature_groups(baseline_cols)

    # undercut pack columns (same list as elsewhere)
    undercut_pack_cols = [c for c in [
        "tyre_change_pursuer", "gap_behind_s",
        "n_cars_within_5s_ahead", "tyre_age_diff_to_ahead",
        "rejoin_gap_ahead_est_s", "rejoin_gap_behind_est_s"
    ] if c in colset]

    def drop(cols_to_drop: List[str]) -> List[str]:
        dset = set(cols_to_drop)
        kept = [c for c in baseline_cols if c not in dset]
        if not kept:
            raise ValueError("Ablation produced empty feature set.")
        return kept

    ablations: Dict[str, List[str]] = {"baseline_full": baseline_cols}

    # drop each raw group (same names as SEQ)
    for gname, gcols in raw_groups.items():
        ablations[f"drop_{gname}"] = drop(gcols)

    # engineered pack + individual engineered cols (same as SEQ)
    engineered_cols = [c for c in engineered_cols if c in colset]
    if engineered_cols:
        ablations["drop_ENGINEERED_ALL"] = drop(engineered_cols)
        for c in engineered_cols:
            ablations[f"drop_COL_{c}"] = drop([c])

    # undercut pack
    if undercut_pack_cols:
        ablations["drop_UNDERCUT_PACK"] = drop(undercut_pack_cols)

    feature_sets = {
        "baseline_cols": baseline_cols,
        "raw_groups": raw_groups,
        "engineered_cols": engineered_cols,
        "undercut_pack_cols": undercut_pack_cols,
    }
    return ablations, feature_sets


def run_tabular_ablation_cv(
    X: pd.DataFrame,
    y: pd.Series,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    *,
    cfg: ModelConfig,
    model_name: str,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, List[str]], Dict[str, Any]]:
    """
    Per-fold ablation for a *single* tabular base learner.
    Uses:
      - train fold only to fit
      - train fold only to tune threshold (max F1)
      - test fold only to evaluate

    Returns: (results_df, ablation_specs, feature_sets)
    """
    X_cols = list(X.columns)
    ablation_specs, feature_sets = make_tabular_ablation_specs(X_cols)

    y_np = y.to_numpy().astype(int)

    # prototype estimator from your registry
    base_all = get_binary_base_learners(cfg)
    if model_name not in base_all:
        raise ValueError(f"Unknown base model '{model_name}'. Available={list(base_all.keys())}")

    results_rows: List[Dict[str, Any]] = []

    vprint(verbose, f"[BASE][{model_name}] n={len(y_np)} folds={len(folds)} n_ablations={len(ablation_specs)}")

    for fold_id, (tr_idx, te_idx) in enumerate(folds):
        vprint(verbose, f"\n[BASE][{model_name}][fold {fold_id}] train={len(tr_idx)} test={len(te_idx)}")

        y_tr = y_np[tr_idx]
        y_te = y_np[te_idx]

        for ab_name, cols_keep in ablation_specs.items():
            X_tr = X.iloc[tr_idx][cols_keep]
            X_te = X.iloc[te_idx][cols_keep]

            num_cols, cat_cols = infer_feature_types(X_tr)

            # IMPORTANT: use your model-specific preprocessor rules
            pre = make_preprocessor_for_model(model_name, num_cols=num_cols, cat_cols=cat_cols)

            est = clone(base_all[model_name])
            pipe = build_model_pipeline(pre, est)
            pipe.fit(X_tr, y_tr)

            proba_tr = pipe.predict_proba(X_tr)[:, 1].astype(float)
            thr = tune_threshold_max_f1(y_tr, proba_tr)

            proba_te = pipe.predict_proba(X_te)[:, 1].astype(float)
            m = compute_fold_metrics(y_te, proba_te, threshold=thr)

            results_rows.append({
                "mode": "base",
                "model_kind": model_name,
                "ablation_name": ab_name,
                "fold_id": int(fold_id),
                "n_test_rows": int(len(te_idx)),
                "threshold": float(thr),
                **m,
            })

            if verbose:
                vprint(
                    verbose,
                    f"[BASE][{model_name}][{ab_name}][fold {fold_id}] "
                    f"PR_AUC={m['PR_AUC']:.4f} LogLoss={m['LogLoss']:.4f} F1={m['F1']:.4f}"
                )

    return pd.DataFrame(results_rows), ablation_specs, feature_sets

def run_seq_ablation_cv(
    df1: pd.DataFrame,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    *,
    cfg: ModelConfig,
    model_kind: str,
    seq_len: int,
    pad_left: bool,
    add_timestep_mask: bool,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Train/eval a single sequence model with ablations applied to input tensors.
    Ablations: baseline_full + drop_<group> (zero group dims)
    """
    y_full = df1["y_pit"].astype(int)
    keys = df1[["race_id", "driver_id", "lapno"]].copy()

    # raw feature table for preprocessing
    X_tab = df1.drop(
        columns=["y_pit", "y_compound", "race_id", "driver_id", "row_id"],
        errors="ignore",
    ).copy()
    X_cols = list(X_tab.columns)

    # Fit preprocessor prototype
    num_cols, cat_cols = infer_feature_types(X_tab)
    pre_proto = make_preprocessor_for_model("tcn", num_cols=num_cols, cat_cols=cat_cols)

    results: List[Dict[str, Any]] = []

    for fold_id, (tr_idx, te_idx) in enumerate(folds):
        vprint(verbose, f"\n[SEQ][{model_kind}][fold {fold_id}] train_rows={len(tr_idx)} test_rows={len(te_idx)}")

        # fit preprocessor on train rows only
        pre = clone(pre_proto)
        pre.fit(X_tab.iloc[tr_idx], y_full.iloc[tr_idx])

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

        # split sequences by fold membership
        if pad_left:
            tr_mask = np.all((seq_idx == -1) | np.isin(seq_idx, tr_idx), axis=1)
            te_mask = np.all((seq_idx == -1) | np.isin(seq_idx, te_idx), axis=1)
        else:
            tr_mask = np.all(np.isin(seq_idx, tr_idx), axis=1)
            te_mask = np.all(np.isin(seq_idx, te_idx), axis=1)

        X_tr, y_tr = X_seq[tr_mask], y_seq[tr_mask]
        X_te, y_te = X_seq[te_mask], y_seq[te_mask]

        if len(y_tr) == 0 or len(y_te) == 0:
            vprint(verbose, f"[SEQ][{model_kind}][fold {fold_id}] skipping (no sequences after filtering)")
            continue

        # build dim groups for ablation
        dim_groups = _feature_dim_groups_from_preprocessor(pre, X_cols=X_cols)

        # baseline + drop groups
        ablations: Dict[str, Optional[List[int]]] = {"baseline_full": None}
        for gname, dims in dim_groups.items():
            ablations[f"drop_{gname}"] = dims

        # helper to build model
        def _make_model():
            if model_kind == "tcn":
                return make_tcn_binary(cfg, seq_len=seq_len)
            elif model_kind == "tcn_gru":
                from src.models.models import make_tcn_gru_binary  # type: ignore
                return make_tcn_gru_binary(cfg)  # seq_len inferred from X shape by SciKeras
            else:
                raise ValueError(f"Unknown model_kind={model_kind}")

        # class weights on training sequences
        n_pos = int(np.sum(y_tr == 1))
        n_neg = int(np.sum(y_tr == 0))
        if n_pos == 0 or n_neg == 0:
            # degenerate: skip fold
            vprint(verbose, f"[SEQ][{model_kind}][fold {fold_id}] degenerate y in train sequences; skipping")
            continue
        pos_w = min(20.0, float(n_neg / n_pos))
        class_w = {0: 1.0, 1: pos_w}

        for ab_name, dims in ablations.items():
            X_tr_ab = X_tr if dims is None else _apply_dim_ablation(X_tr, dims)
            X_te_ab = X_te if dims is None else _apply_dim_ablation(X_te, dims)

            model = _make_model()
            model.fit(X_tr_ab, y_tr, class_weight=class_w)

            proba_tr = model.predict_proba(X_tr_ab)[:, 1].astype(float)
            thr = tune_threshold_max_f1(y_tr, proba_tr)

            proba_te = model.predict_proba(X_te_ab)[:, 1].astype(float)
            m = compute_fold_metrics(y_te, proba_te, threshold=thr)

            results.append({
                "mode": "seq",
                "model_kind": model_kind,
                "ablation_name": ab_name,
                "fold_id": int(fold_id),
                "n_test_seqs": int(len(y_te)),
                "threshold": float(thr),
                **m,
            })

            if verbose:
                vprint(verbose, f"[SEQ][{model_kind}][{ab_name}][fold {fold_id}] PR_AUC={m['PR_AUC']:.4f} LogLoss={m['LogLoss']:.4f} F1={m['F1']:.4f}")

    return pd.DataFrame(results)


# -----------------------------
# Summaries
# -----------------------------
def summarize_ablation_results(results_df: pd.DataFrame, baseline_name: str = "baseline_full") -> pd.DataFrame:
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

    base = agg[agg["ablation_name"] == baseline_name]
    if len(base) != 1:
        raise RuntimeError(f"{baseline_name} summary missing or duplicated.")
    base_row = base.iloc[0]

    agg["delta_PR_AUC_vs_baseline"] = agg["mean_PR_AUC"] - float(base_row["mean_PR_AUC"])
    agg["delta_LogLoss_vs_baseline"] = agg["mean_LogLoss"] - float(base_row["mean_LogLoss"])
    agg["delta_F1_vs_baseline"] = agg["mean_F1"] - float(base_row["mean_F1"])

    agg["PR_AUC_drop_vs_baseline"] = float(base_row["mean_PR_AUC"]) - agg["mean_PR_AUC"]
    agg["LogLoss_increase_vs_baseline"] = agg["mean_LogLoss"] - float(base_row["mean_LogLoss"])
    return agg


def print_top5_damage(summary_df: pd.DataFrame, title: str) -> None:
    def _top(df: pd.DataFrame, col: str) -> pd.DataFrame:
        return df[df["ablation_name"] != "baseline_full"].sort_values(col, ascending=False).head(5)

    top_pr = _top(summary_df, "PR_AUC_drop_vs_baseline")
    top_ll = _top(summary_df, "LogLoss_increase_vs_baseline")

    print(f"\n{title}")
    print("Top-5 most damaging ablations (by PR-AUC drop):")
    for _, r in top_pr.iterrows():
        print(f"  {r['ablation_name']}: PR_AUC_drop={r['PR_AUC_drop_vs_baseline']:.4f} (mean_PR_AUC={r['mean_PR_AUC']:.4f})")

    print("\nTop-5 most damaging ablations (by LogLoss increase):")
    for _, r in top_ll.iterrows():
        print(f"  {r['ablation_name']}: LogLoss_increase={r['LogLoss_increase_vs_baseline']:.4f} (mean_LogLoss={r['mean_LogLoss']:.4f})")


# -----------------------------
# CLI
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_stage1", required=True, help="Path to stage1 dataset (pit/no-pit), e.g. output1.csv")
    ap.add_argument("--outdir", default="runs/ablation", help="Output directory root")
    ap.add_argument("--run_name", default="stage1_ablation", help="Subfolder under outdir")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)

    # sequence settings (used in META-partA TCN and SEQ mode)
    ap.add_argument("--seq_len", type=int, default=8)
    ap.add_argument("--pad_left", action="store_true", help="Pad left to allow early laps.")
    ap.add_argument("--add_timestep_mask", action="store_true", help="Append timestep mask feature.")

    ap.add_argument("--recompute_base_oof", action="store_true", help="Force recomputation of META base OOF cache")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--smoke_test", action="store_true", help="Run a quick 1-fold smoke test.")

    ap.add_argument(
        "--mode",
        type=str,
        default="meta",
        choices=["meta", "seq", "base", "both", "all"],
        help=(
            "meta: stacked ensemble ablation; "
            "seq: single sequence model ablation; "
            "base: tabular base-model ablation (svm/xgb); "
            "both: meta+seq; "
            "all: meta+seq+base."
        ),
    )

    ap.add_argument(
        "--base_models",
        type=str,
        default="svm,xgb",
        help="Comma-separated base models to ablate in BASE mode (e.g. 'svm,xgb').",
    )

    ap.add_argument(
        "--seq_model",
        type=str,
        default="tcn",
        choices=["tcn", "tcn_gru"],
        help="Which sequence model factory to use for SEQ mode. Requires make_tcn_gru_binary for tcn_gru.",
    )

    args = ap.parse_args()

    out_root = _ensure_dir(Path(args.outdir) / args.run_name)
    cache_dir = _ensure_dir(out_root / "cache")
    reports_dir = _ensure_dir(out_root / "reports")

    cfg = ModelConfig(random_state=int(args.seed))

    print(f"[ablation_eval] out_root: {out_root}")
    print(f"[ablation_eval] seed={args.seed} n_splits={args.n_splits} seq_len={args.seq_len} pad_left={args.pad_left} tmask={args.add_timestep_mask}")
    print(f"[ablation_eval] HOLDOUT_RACE_IDS={list(map(int, HOLDOUT_RACE_IDS))}")

    # Load data
    vprint(args.verbose, "[ablation_eval] loading stage1 dataset...")
    df1 = load_stage1_dataset(args.data_stage1)

    # folds are on lap rows (race-wise)
    fb = make_race_group_folds(df1, target_col="y_pit", n_splits=args.n_splits, seed=args.seed)
    folds = fb.folds
    if args.smoke_test:
        folds = folds[:1]

    # -----------------------------
    # MODE: META
    # -----------------------------
    if args.mode in ("meta", "both"):
        X1, y1 = get_stage1_xy(df1)

        # cache paths
        meta_table_path = cache_dir / "meta_table.parquet"
        meta_manifest_path = cache_dir / "meta_table_manifest.json"

        if meta_table_path.exists() and (not args.recompute_base_oof):
            print(f"[META][PART A] loading cached meta-table: {meta_table_path}")
            meta_df = pd.read_parquet(meta_table_path)
        else:
            print("[META][PART A] generating base-level OOF probabilities (tabular + TCN)...")
            meta_df = build_meta_table_stage1(
                df1,
                X1,
                y1,
                folds,
                cfg=cfg,
                seq_len_tcn=int(args.seq_len),
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
                    "seq_len": int(args.seq_len),
                    "pad_left": bool(args.pad_left),
                    "add_timestep_mask": bool(args.add_timestep_mask),
                    "holdout_race_ids": list(map(int, HOLDOUT_RACE_IDS)),
                    "model_config": asdict(cfg),
                },
            )
            print(f"[META][PART A] wrote meta-table cache: {meta_table_path}")

        # Define feature sets
        id_cols = {"race_id", "row_id"}
        label_col = "y_pit"

        base_prob_cols = ["p_svm", "p_rf", "p_xgb", "p_ann", "tcn_proba", "tcn_effective_len"]
        base_prob_cols = [c for c in base_prob_cols if c in meta_df.columns]
        tcn_cols = [c for c in ["tcn_proba", "tcn_effective_len"] if c in meta_df.columns]

        raw_cols = [c for c in meta_df.columns if c not in (id_cols | {label_col} | set(base_prob_cols))]
        raw_cols = [c for c in raw_cols if c not in {"driver_id"}]

        baseline_cols = base_prob_cols + raw_cols

        raw_groups, engineered_cols = build_stage1_feature_groups(raw_cols)

        ablation_specs = make_ablation_specs(
            baseline_cols=baseline_cols,
            base_prob_cols=base_prob_cols,
            tcn_cols=tcn_cols,
            raw_cols=raw_cols,
            raw_groups=raw_groups,
            engineered_cols=engineered_cols,
        )

        _write_json(reports_dir / "meta_ablation_specs.json", {"ablations": list(ablation_specs.keys())})
        _write_json(reports_dir / "meta_feature_sets.json", {
            "baseline_cols": baseline_cols,
            "base_prob_cols": base_prob_cols,
            "tcn_cols": tcn_cols,
            "raw_cols": raw_cols,
            "raw_groups": raw_groups,
            "engineered_cols": engineered_cols,
        })

        # meta folds should be race-wise too; we reuse folds by indexing meta_df rows (same ordering)
        fb_meta = make_race_group_folds(meta_df[["race_id", "y_pit"]].copy(), target_col="y_pit", n_splits=args.n_splits, seed=args.seed)
        folds_meta = fb_meta.folds
        if args.smoke_test:
            folds_meta = folds_meta[:1]

        print(f"[META][PART B] running meta ablation CV: n_ablations={len(ablation_specs)} n_folds={len(folds_meta)}")
        meta_results = run_meta_ablation_cv(
            meta_df,
            folds_meta,
            cfg=cfg,
            ablation_specs=ablation_specs,
            verbose=args.verbose,
        )

        meta_results_path = reports_dir / "meta_ablation_results.csv"
        meta_results.to_csv(meta_results_path, index=False)
        print(f"[META][reports] wrote: {meta_results_path}")

        meta_summary = summarize_ablation_results(meta_results, baseline_name="baseline_full")
        meta_summary_path = reports_dir / "meta_ablation_summary.csv"
        meta_summary.to_csv(meta_summary_path, index=False)
        print(f"[META][reports] wrote: {meta_summary_path}")

        print_top5_damage(meta_summary, title="[META] Damage rankings")

        # -----------------------------
    # MODE: BASE (tabular SVM/XGB etc.)
    # -----------------------------
    if args.mode in ("base", "all"):
        X1, y1 = get_stage1_xy(df1)

        # match what you avoid leaking elsewhere
        X_base = X1.copy().drop(columns=["race_id", "row_id", "driver_id", "y_pit"], errors="ignore")

        # folds reuse df1 row indices -> same ordering in X_base/y1
        base_models = [s.strip() for s in str(args.base_models).split(",") if s.strip()]
        if not base_models:
            raise ValueError("--base_models parsed to empty list")

        for bm in base_models:
            print(f"[BASE] running tabular ablation for model={bm}")

            base_results, base_specs, base_feature_sets = run_tabular_ablation_cv(
                X=X_base,
                y=y1,
                folds=folds,
                cfg=cfg,
                model_name=bm,
                verbose=args.verbose,
            )

            # write per-model reports
            _write_json(reports_dir / f"base_{bm}_ablation_specs.json", {"ablations": list(base_specs.keys())})
            _write_json(reports_dir / f"base_{bm}_feature_sets.json", base_feature_sets)

            base_results_path = reports_dir / f"base_{bm}_ablation_results.csv"
            base_results.to_csv(base_results_path, index=False)
            print(f"[BASE][reports] wrote: {base_results_path}")

            base_summary = summarize_ablation_results(base_results, baseline_name="baseline_full")
            base_summary_path = reports_dir / f"base_{bm}_ablation_summary.csv"
            base_summary.to_csv(base_summary_path, index=False)
            print(f"[BASE][reports] wrote: {base_summary_path}")

            print_top5_damage(base_summary, title=f"[BASE:{bm}] Damage rankings")

    # -----------------------------
    # MODE: SEQ (TCN / TCN-GRU)
    # -----------------------------
    if args.mode in ("seq", "both"):
        print(f"[SEQ] running single-model ablation for model={args.seq_model}")

        seq_results = run_seq_ablation_cv(
            df1=df1,
            folds=folds,
            cfg=cfg,
            model_kind=args.seq_model,
            seq_len=int(args.seq_len),
            pad_left=bool(args.pad_left),
            add_timestep_mask=bool(args.add_timestep_mask),
            verbose=args.verbose,
        )

        seq_results_path = reports_dir / f"seq_{args.seq_model}_ablation_results.csv"
        seq_results.to_csv(seq_results_path, index=False)
        print(f"[SEQ][reports] wrote: {seq_results_path}")

        seq_summary = summarize_ablation_results(seq_results, baseline_name="baseline_full")
        seq_summary_path = reports_dir / f"seq_{args.seq_model}_ablation_summary.csv"
        seq_summary.to_csv(seq_summary_path, index=False)
        print(f"[SEQ][reports] wrote: {seq_summary_path}")

        print_top5_damage(seq_summary, title=f"[SEQ:{args.seq_model}] Damage rankings")

    print(f"\n[ablation_eval] done. Outputs in: {reports_dir}")


if __name__ == "__main__":
    main()

'''
python3 -m src.utils.ablation_eval \
  --data_stage1 data/processed/output1.csv \
  --outdir runs/ablation \
  --run_name stage1_seq_tcn_ablation \
  --mode seq \
  --seq_model tcn \
  --n_splits 5 \
  --seed 42 \
  --seq_len 8 \
  --pad_left \
  --add_timestep_mask \
  --verbose


python3 -m src.utils.ablation_eval \
  --data_stage1 data/processed/output1.csv \
  --outdir runs/ablation \
  --run_name stage1_seq_tcn_gru_ablation \
  --mode seq \
  --seq_model tcn_gru \
  --n_splits 5 \
  --seed 42 \
  --seq_len 8 \
  --pad_left \
  --add_timestep_mask \
  --verbose
'''


'''
python3 -m src.utils.ablation_eval \
  --data_stage1 path/to/output1.csv \
  --mode base \
  --base_models svm,xgb \
  --n_splits 5 \
  --seed 42 \
  --verbose
  
  
python3 -m src.utils.ablation_eval \
  --data_stage1 path/to/output1.csv \
  --mode all \
  --base_models svm,xgb \
  --seq_model tcn \
  --seq_len 8 \
  --pad_left \
  --add_timestep_mask
  '''