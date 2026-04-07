#!/usr/bin/env python3
# src/utils/ablation_eval.py
"""
Ablation evaluation for Stage-1 and Stage-2 meta learners using OOF base learner probabilities.
This script generates OOF probabilities for all specified base learners and excludes one base learner at a time to see how the xgb meta learners performance drops

Outputs:
reports/meta_ablation_results_stageX.csv (fold-level results)
reports/meta_ablation_summary_stageX.csv (average across the fold results)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import label_binarize

from src.data.data import (
    COMPOUND_CLASSES,
    HOLDOUT_RACE_IDS,
    build_feature_sequences,
    encode_y_compound,
    get_stage1_xy,
    get_stage2_xy,
    infer_feature_types,
    load_stage1_dataset,
    load_stage2_dataset,
    make_race_group_folds,
)
from src.data.preprocessing import (
    PreprocessConfig,
    build_preprocessor,
    make_preprocessor_for_model,
)
from src.models.models import (
    ModelConfig,
    build_model_pipeline,
    get_binary_base_learners,
    get_multiclass_base_learners,
    get_stage1_sequential_models,
    get_stage2_sequential_models,
    make_meta_binary_xgb,
    make_meta_multiclass_xgb,
)

SEED_DEFAULT = 42


def vprint(verbose: bool, *args, **kwargs) -> None:
    if verbose:
        print(*args, **kwargs, flush=True)


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def _safe_logloss_binary(y_true: np.ndarray, proba_pos: np.ndarray) -> float:
    p = np.clip(proba_pos.astype(float), 1e-15, 1 - 1e-15)
    return float(log_loss(y_true, p, labels=[0, 1]))


def _safe_logloss_multi(y_true: np.ndarray, proba: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(proba.astype(float), 1e-15, 1 - 1e-15)
    return float(log_loss(y_true, p, labels=labels))


def compute_fold_metrics_binary(
    y_true: np.ndarray,
    proba_pos: np.ndarray,
    *,
    threshold: float,
) -> Dict[str, float]:
    y_true = y_true.astype(int)
    proba_pos = proba_pos.astype(float)
    y_pred = (proba_pos >= threshold).astype(int)

    pr_auc = float(average_precision_score(y_true, proba_pos))
    ll = _safe_logloss_binary(y_true, proba_pos)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))

    return {
        "PR_AUC": pr_auc,
        "LogLoss": ll,
        "F1": f1,
        "Precision": prec,
        "Recall": rec,
    }


def compute_fold_metrics_multi(
    y_true: np.ndarray,
    proba: np.ndarray,
) -> Dict[str, float]:
    y_true = y_true.astype(int)
    proba = proba.astype(float)
    y_pred = np.argmax(proba, axis=1)
    labels = np.arange(proba.shape[1])

    y_true_bin = label_binarize(y_true, classes=labels)
    pr_auc = float(average_precision_score(y_true_bin, proba, average="macro"))
    ll = _safe_logloss_multi(y_true, proba, labels)
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    return {
        "PR_AUC": pr_auc,
        "LogLoss": ll,
        "F1": f1,
        "Precision": prec,
        "Recall": rec,
    }


def collect_oof_base_probs_tabular(
    X: pd.DataFrame,
    y: pd.Series,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    *,
    cfg: ModelConfig,
    stage: int,
    n_classes: Optional[int] = None,
    verbose: bool = False,
) -> pd.DataFrame:

    if stage == 1:
        base_all = get_binary_base_learners(cfg)
        is_binary = True
    else:
        base_all = get_multiclass_base_learners(cfg, n_classes=n_classes)
        is_binary = False

    names = list(base_all.keys())
    n = len(y)

    if is_binary:
        out = np.zeros((n, len(names)), dtype=float)
        cols = [f"p_{name}" for name in names]
    else:
        out = np.zeros((n, len(names) * n_classes), dtype=float)
        cols = []
        for name in names:
            for k in range(n_classes):
                cols.append(f"p_{name}_c{k}")

    num_cols, cat_cols = infer_feature_types(X)
    vprint(verbose, f"[META][PART A][tabular oof] stage={stage} n={n} base={names}")

    for fold_id, (tr_idx, te_idx) in enumerate(folds):
        vprint(
            verbose,
            f"[META][PART A][tabular oof][fold {fold_id}] train={len(tr_idx)} test={len(te_idx)}",
        )
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_te = X.iloc[te_idx]

        for j, name in enumerate(names):
            vprint(verbose, f"  -> fit base='{name}'")
            est = clone(base_all[name])
            pre = make_preprocessor_for_model(name, num_cols=num_cols, cat_cols=cat_cols)
            pipe = build_model_pipeline(pre, est)
            pipe.fit(X_tr, y_tr)

            if is_binary:
                out[te_idx, j] = pipe.predict_proba(X_te)[:, 1].astype(float)
            else:
                proba_fold = pipe.predict_proba(X_te)
                classes_seen = (
                    pipe.named_steps["model"].classes_
                    if hasattr(pipe.named_steps["model"], "classes_")
                    else getattr(pipe, "classes_")
                )
                proba_full = np.zeros((len(te_idx), n_classes), dtype=float)
                for c_idx, cls in enumerate(classes_seen):
                    proba_full[:, int(cls)] = proba_fold[:, c_idx]
                start = j * n_classes
                out[te_idx, start : start + n_classes] = proba_full

    return pd.DataFrame(out, columns=cols)


def collect_oof_seq_models(
    df: pd.DataFrame,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    *,
    cfg: ModelConfig,
    stage: int,
    n_classes: Optional[int] = None,
    seq_len: int,
    pad_left: bool = True,
    add_timestep_mask: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:

    if stage == 1:
        target_col = "y_pit"
        seq_models = get_stage1_sequential_models(cfg)
        is_binary = True
    else:
        target_col = "y_compound_encoded"
        seq_models = get_stage2_sequential_models(cfg, n_classes=n_classes)
        is_binary = False

    names = list(seq_models.keys())
    y_full = df[target_col].astype(int)
    keys = df[["race_id", "driver_id", "lapno"]].copy()

    X_tab = df.drop(
        columns=["y_pit", "y_compound", "y_compound_encoded", "race_id", "driver_id", "row_id"],
        errors="ignore",
    ).copy()

    num_cols, cat_cols = infer_feature_types(X_tab)
    pre_proto = make_preprocessor_for_model("tcn", num_cols=num_cols, cat_cols=cat_cols)

    N = len(df)

    if is_binary:
        out = np.full((N, len(names)), np.nan, dtype=float)
        cols = [f"seq_{name}_proba" for name in names]
        eff_len_cols = [f"seq_{name}_eff_len" for name in names]
    else:
        out = np.full((N, len(names) * n_classes), np.nan, dtype=float)
        cols = []
        for name in names:
            for k in range(n_classes):
                cols.append(f"seq_{name}_c{k}")
        eff_len_cols = [f"seq_{name}_eff_len" for name in names]

    oof_eff_len = np.zeros((N, len(names)), dtype=float)

    vprint(verbose, f"[SEQ-OFF] stage={stage} N={N} seq_models={names}")

    for fold_id, (tr_idx, te_idx) in enumerate(folds):
        vprint(verbose, f"[SEQ-OFF][fold {fold_id}] train={len(tr_idx)} test={len(te_idx)}")
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
            vprint(verbose, f"[SEQ-OFF][fold {fold_id}] skipping (no sequences after filtering)")
            continue

        if is_binary:
            n_pos = int(np.sum(y_seq_tr == 1))
            n_neg = int(np.sum(y_seq_tr == 0))
            if n_pos == 0 or n_neg == 0:
                const_p = 1.0 if n_neg == 0 else 0.0
                for j in range(len(names)):
                    out[idx_last_te, j] = const_p
                    oof_eff_len[idx_last_te, j] = eff_len_te
                vprint(verbose, f"[SEQ-OFF][fold {fold_id}] degenerate y -> const_p={const_p}")
                continue
            pos_w = min(20.0, float(n_neg / n_pos))
            class_w = {0: 1.0, 1: pos_w}
        else:
            class_w = None

        for j, name in enumerate(names):
            vprint(verbose, f"  -> fit seq='{name}'")
            model = clone(seq_models[name])
            if class_w is not None:
                model.fit(X_seq_tr, y_seq_tr, class_weight=class_w)
            else:
                model.fit(X_seq_tr, y_seq_tr)

            if is_binary:
                proba = model.predict_proba(X_seq_te)[:, 1].astype(float)
                out[idx_last_te, j] = proba
            else:
                proba_fold = model.predict_proba(X_seq_te)
                classes_seen = (
                    model.model_.classes_
                    if hasattr(model, "model_") and hasattr(model.model_, "classes_")
                    else np.unique(y_seq_tr)
                )
                proba_full = np.zeros((len(idx_last_te), n_classes), dtype=float)
                for c_idx, cls in enumerate(classes_seen):
                    proba_full[:, int(cls)] = proba_fold[:, c_idx]
                start = j * n_classes
                out[idx_last_te, start : start + n_classes] = proba_full

            oof_eff_len[idx_last_te, j] = eff_len_te

    out = np.nan_to_num(out, nan=0.0)
    eff_scaled = oof_eff_len / float(seq_len)

    df_out = pd.DataFrame(out, columns=cols)
    df_eff = pd.DataFrame(eff_scaled, columns=eff_len_cols)

    return pd.concat([df_out, df_eff], axis=1)


def build_meta_table(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    *,
    cfg: ModelConfig,
    stage: int,
    n_classes: Optional[int] = None,
    seq_len: int,
    verbose: bool = False,
) -> pd.DataFrame:

    target_col = "y_pit" if stage == 1 else "y_compound_encoded"

    df_base_probs = collect_oof_base_probs_tabular(
        X,
        y,
        folds,
        cfg=cfg,
        stage=stage,
        n_classes=n_classes,
        verbose=verbose,
    )

    df_seq_probs = collect_oof_seq_models(
        df,
        folds,
        cfg=cfg,
        stage=stage,
        n_classes=n_classes,
        seq_len=seq_len,
        pad_left=True,
        add_timestep_mask=True,
        verbose=verbose,
    )

    meta = pd.DataFrame(
        {
            "race_id": df["race_id"].to_numpy(),
            "row_id": df["row_id"].to_numpy() if "row_id" in df.columns else np.arange(len(df)),
            target_col: y.to_numpy().astype(int),
        }
    )

    meta = pd.concat(
        [
            meta,
            df_base_probs.reset_index(drop=True),
            df_seq_probs.reset_index(drop=True),
        ],
        axis=1,
    )

    X_safe = X.copy().drop(
        columns=["race_id", "row_id", "y_pit", "y_compound", "y_compound_encoded", "driver_id"],
        errors="ignore",
    )
    # this concat ensures that the contextual features from the race are passed to the meta learner
    meta = pd.concat([meta, X_safe.reset_index(drop=True)], axis=1)

    if meta.columns.duplicated().any():
        dupes = meta.columns[meta.columns.duplicated()].tolist()
        raise RuntimeError(f"Duplicate columns in meta table: {dupes}")

    return meta


def make_ablation_specs_meta(
    *,
    baseline_cols: List[str],
    prob_cols_dict: Dict[str, List[str]],
    tabular_cols: List[str],
    seq_cols: List[str],
    raw_cols: List[str],
) -> Dict[str, List[str]]:

    specs: Dict[str, List[str]] = {}
    specs["baseline_full"] = baseline_cols

    def drop(cols_to_drop: List[str]) -> List[str]:
        drop_set = set(cols_to_drop)
        cols2 = [c for c in baseline_cols if c not in drop_set]
        if not cols2:
            raise ValueError("Ablation dropped all columns!")
        return cols2

    # 1. Leave-One-Base-Out (LOBO) for every individual model
    for model_name, pcols in prob_cols_dict.items():
        specs[f"drop_{model_name}"] = drop(pcols)

    # 2. Broader Group Exclusions
    if tabular_cols:
        specs["drop_all_tabular_bases"] = drop(tabular_cols)
    if seq_cols:
        specs["drop_all_sequential_bases"] = drop(seq_cols)
    if raw_cols:
        specs["drop_all_race_context_KEEP_ONLY_PREDS"] = drop(raw_cols)

    return specs


def run_meta_ablation_cv(
    meta_df: pd.DataFrame,
    folds_meta: List[Tuple[np.ndarray, np.ndarray]],
    *,
    cfg: ModelConfig,
    stage: int,
    n_classes: Optional[int],
    ablation_specs: Dict[str, List[str]],
    verbose: bool = False,
) -> pd.DataFrame:

    target_col = "y_pit" if stage == 1 else "y_compound_encoded"
    y = meta_df[target_col].to_numpy().astype(int)

    results_rows: List[Dict[str, Any]] = []

    if stage == 1:
        meta_model_proto = make_meta_binary_xgb(cfg)
        is_binary = True
    else:
        meta_model_proto = make_meta_multiclass_xgb(cfg, n_classes=n_classes)
        is_binary = False

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

            if is_binary:
                proba_tr = pipe.predict_proba(X_tr)[:, 1].astype(float)
                proba_te = pipe.predict_proba(X_te)[:, 1].astype(float)
                m = compute_fold_metrics_binary(y_te, proba_te, threshold=0.5)
                m["threshold"] = float(0.5)
            else:
                proba_te = pipe.predict_proba(X_te)
                classes_seen = (
                    pipe.named_steps["model"].classes_
                    if hasattr(pipe.named_steps["model"], "classes_")
                    else getattr(pipe, "classes_")
                )
                proba_full = np.zeros((len(te_idx), n_classes), dtype=float)
                for c_idx, cls in enumerate(classes_seen):
                    proba_full[:, int(cls)] = proba_te[:, c_idx]

                m = compute_fold_metrics_multi(y_te, proba_full)
                m["threshold"] = np.nan

            results_rows.append(
                {
                    "mode": "meta",
                    "ablation_name": ab_name,
                    "fold_id": int(fold_id),
                    "n_test_rows": int(len(te_idx)),
                    **m,
                }
            )

            if verbose:
                vprint(
                    verbose,
                    f"[META][{ab_name}][fold {fold_id}] PR_AUC={m['PR_AUC']:.4f} "
                    f"LogLoss={m['LogLoss']:.4f} F1={m['F1']:.4f}",
                )

    return pd.DataFrame(results_rows)


def summarize_ablation_results(
    results_df: pd.DataFrame,
    baseline_name: str = "baseline_full",
) -> pd.DataFrame:
    agg = results_df.groupby("ablation_name").agg(
        mean_PR_AUC=("PR_AUC", "mean"),
        std_PR_AUC=("PR_AUC", "std"),
        mean_LogLoss=("LogLoss", "mean"),
        std_LogLoss=("LogLoss", "std"),
        mean_F1=("F1", "mean"),
        std_F1=("F1", "std"),
        mean_Precision=("Precision", "mean"),
        mean_Recall=("Recall", "mean"),
    ).reset_index()

    base = agg[agg["ablation_name"] == baseline_name]
    if len(base) != 1:
        raise RuntimeError(f"{baseline_name} summary missing or duplicated.")
    base_row = base.iloc[0]

    agg["delta_PR_AUC_vs_baseline"] = agg["mean_PR_AUC"] - float(base_row["mean_PR_AUC"])
    agg["delta_LogLoss_vs_baseline"] = agg["mean_LogLoss"] - float(base_row["mean_LogLoss"])
    agg["delta_F1_vs_baseline"] = agg["mean_F1"] - float(base_row["mean_F1"])
    agg["delta_Precision_vs_baseline"] = agg["mean_Precision"] - float(base_row["mean_Precision"])
    agg["delta_Recall_vs_baseline"] = agg["mean_Recall"] - float(base_row["mean_Recall"])

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


def main() -> None:
    """
    hardcoded entry point for meta-ablation evaluation.
    edit the values below directly instead of passing cli arguments.
    """

    data_path = "data/processed/dataset1.csv"  # dataset1.csv or dataset2.csv
    stage = 1  # 1 = binary, 2 = multiclass
    outdir = "runs/ablation"
    run_name = "meta_ablation"
    n_splits = 5
    seed = SEED_DEFAULT
    seq_len = 8

    out_root = _ensure_dir(Path(outdir) / run_name)
    cache_dir = _ensure_dir(out_root / "cache")
    reports_dir = _ensure_dir(out_root / "reports")

    cfg = ModelConfig(random_state=int(seed))

    print(f"[ablation_eval] stage={stage} data={data_path}")

    n_classes = len(COMPOUND_CLASSES) if stage == 2 else None

    if stage == 1:
        df = load_stage1_dataset(data_path)
        X, y = get_stage1_xy(df)
        target_col = "y_pit"
    else:
        df_raw = load_stage2_dataset(data_path)
        df = encode_y_compound(df_raw, col="y_compound", out_col="y_compound_encoded")
        df = df.loc[~df["y_compound_encoded"].isna()].copy()
        X, y = get_stage2_xy(df)
        target_col = "y_compound_encoded"

    fb = make_race_group_folds(df, n_splits=n_splits, seed=seed)
    folds = fb.folds

    meta_table_path = cache_dir / f"meta_table_stage{stage}.parquet"
    meta_manifest_path = cache_dir / f"meta_table_stage{stage}_manifest.json"

    print("[META] generating base-level OOF probabilities for ALL base models...")
    meta_df = build_meta_table(
        df,
        X,
        y,
        folds,
        cfg=cfg,
        stage=stage,
        n_classes=n_classes,
        seq_len=int(seq_len),
        verbose=False,
    )
    meta_df.to_parquet(meta_table_path, index=False)
    _write_json(
        meta_manifest_path,
        {
            "data_path": str(data_path),
            "stage": stage,
            "n_rows": int(len(meta_df)),
            "n_races": int(meta_df["race_id"].nunique()),
            "n_splits": int(n_splits),
            "seed": int(seed),
            "seq_len": int(seq_len),
            "holdout_race_ids": list(map(int, HOLDOUT_RACE_IDS)),
        },
    )
    print(f"[META] wrote meta-table cache: {meta_table_path}")

    if stage == 1:
        base_models = list(get_binary_base_learners(cfg).keys())
        seq_models = list(get_stage1_sequential_models(cfg).keys())
    else:
        base_models = list(get_multiclass_base_learners(cfg, n_classes=n_classes).keys())
        seq_models = list(get_stage2_sequential_models(cfg, n_classes=n_classes).keys())

    prob_cols_dict = {}
    tabular_cols = []

    for bm in base_models:
        if stage == 1:
            cols = [f"p_{bm}"]
        else:
            cols = [f"p_{bm}_c{k}" for k in range(n_classes)]
        cols = [c for c in cols if c in meta_df.columns]
        if cols:
            prob_cols_dict[bm] = cols
            tabular_cols.extend(cols)

    seq_cols = []
    for sm in seq_models:
        if stage == 1:
            cols = [f"seq_{sm}_proba", f"seq_{sm}_eff_len"]
        else:
            cols = [f"seq_{sm}_c{k}" for k in range(n_classes)] + [f"seq_{sm}_eff_len"]
        cols = [c for c in cols if c in meta_df.columns]
        if cols:
            prob_cols_dict[sm] = cols
            seq_cols.extend(cols)

    id_cols = {"race_id", "row_id", target_col}
    baseline_cols = [c for c in meta_df.columns if c not in id_cols]
    raw_cols = [c for c in baseline_cols if c not in tabular_cols and c not in seq_cols]

    ablation_specs = make_ablation_specs_meta(
        baseline_cols=baseline_cols,
        prob_cols_dict=prob_cols_dict,
        tabular_cols=tabular_cols,
        seq_cols=seq_cols,
        raw_cols=raw_cols,
    )

    _write_json(
        reports_dir / f"meta_ablation_specs_stage{stage}.json",
        {"ablations": list(ablation_specs.keys())},
    )

    fb_meta = make_race_group_folds(meta_df, n_splits=n_splits, seed=seed)
    folds_meta = fb_meta.folds

    print(f"[META] running meta ablation CV: n_ablations={len(ablation_specs)} n_folds={len(folds_meta)}")
    meta_results = run_meta_ablation_cv(
        meta_df,
        folds_meta,
        cfg=cfg,
        stage=stage,
        n_classes=n_classes,
        ablation_specs=ablation_specs,
        verbose=False,
    )

    meta_results_path = reports_dir / f"meta_ablation_results_stage{stage}.csv"
    meta_results.to_csv(meta_results_path, index=False)

    meta_summary = summarize_ablation_results(meta_results, baseline_name="baseline_full")
    meta_summary_path = reports_dir / f"meta_ablation_summary_stage{stage}.csv"
    meta_summary.to_csv(meta_summary_path, index=False)

    print_top5_damage(meta_summary, title=f"[META Stage {stage}] Damage rankings")
    print(f"\n[ablation_eval] done. Outputs in: {reports_dir}")


if __name__ == "__main__":
    main()