#!/usr/bin/env python3
# src/utils/base_ablation_eval.py
"""
Base learner feature ablation evaluation for Stage-1 and Stage-2 base learners.
The script groups features and, for each group, trains the individual base models
without a single feature group, recording the drop in performance.

Outputs:
reports/base_ablation_results_{model}_stageX.csv (per fold results)
reports/base_ablation_summary_{model}_stageX.csv (averaged across folds results)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.data.data import (
    COMPOUND_CLASSES,
    build_feature_sequences,
    encode_y_compound,
    get_stage1_xy,
    get_stage2_xy,
    infer_feature_types,
    load_stage1_dataset,
    load_stage2_dataset,
    make_race_group_folds,
)
from src.data.preprocessing import make_preprocessor_for_model
from src.models.models import (
    ModelConfig,
    build_model_pipeline,
    get_stage1_sequential_models,
    get_stage1_tabular_models,
    get_stage2_sequential_models,
    get_stage2_tabular_models,
)
from src.utils.meta_ablation_eval import (
    _ensure_dir,
    _write_json,
    compute_fold_metrics_binary,
    compute_fold_metrics_multi,
    print_top5_damage,
    summarize_ablation_results,
    tune_threshold_max_f1_binary,
    vprint,
)

SEED_DEFAULT = 42

STAGE1_FEATURE_GROUPS = {
    "G_race_phase": ["lapno", "race_progress"],
    "G_strategy_history": ["pit_stops_so_far", "fulfilled_second_compound"],
    "G_tyre_state": ["current_compound", "fulfilled_second_compound", "tyre_age", "tyre_age_missing"],
    "G_track_context": ["race_track", "track_category"],
    "G_race_control_weather": ["is_wet_race", "is_raining", "minutes_rain", "fcy_status"],
    "G_traffic_gap_context": ["gap_behind_s", "gap_behind_s_missing", "n_cars_within_5s_ahead", "close_ahead"],
    "G_rejoin_window": [
        "tyre_age_diff_to_ahead",
        "tyre_age_diff_to_ahead_missing",
        "rejoin_gap_ahead_est_s",
        "rejoin_gap_ahead_est_s_missing",
        "rejoin_gap_behind_est_s",
        "rejoin_gap_behind_est_s_missing",
    ],
    "G_pace_position": ["lap_time", "position", "interval", "tyre_change_pursuer", "tyre_age", "current_compound"],
}

STAGE2_FEATURE_GROUPS = {
    "G_race_phase": ["lapno", "race_progress"],
    "G_strategy_history": ["pit_stops_so_far", "fulfilled_second_compound"],
    "G_tyre_state": ["current_compound", "fulfilled_second_compound"],
    "G_track_context": ["race_track"],
    "G_race_control_weather": ["is_wet_race", "is_raining", "minutes_rain"],
    "G_traffic_gap_context": ["gap_behind_s", "gap_behind_s_missing", "n_cars_within_5s_ahead"],
    "G_rejoin_window": [
        "tyre_age_diff_to_ahead",
        "tyre_age_diff_to_ahead_missing",
        "rejoin_gap_ahead_est_s",
        "rejoin_gap_ahead_est_s_missing",
        "rejoin_gap_behind_est_s",
        "rejoin_gap_behind_est_s_missing",
    ],
}


def make_ablation_specs(
    available_cols: List[str],
    feature_groups: Dict[str, List[str]],
):
    specs = {"baseline_full": available_cols.copy()}

    for group_name, cols_to_drop in feature_groups.items():
        valid_drop = [c for c in cols_to_drop if c in available_cols]
        if valid_drop:
            specs[f"drop_{group_name}"] = [c for c in available_cols if c not in valid_drop]

    return specs


def run_tabular_ablation(
    X: pd.DataFrame,
    y: pd.Series,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    model_name: str,
    cfg: ModelConfig,
    stage: int,
    n_classes: Optional[int],
    ablation_specs: Dict[str, List[str]],
    verbose: bool,
):
    if stage == 1:
        base_models = get_stage1_tabular_models(cfg)
        is_binary = True
    else:
        base_models = get_stage2_tabular_models(cfg, n_classes=n_classes)
        is_binary = False

    if model_name not in base_models:
        raise ValueError(f"Model '{model_name}' not found in stage {stage} tabular models.")

    model_proto = base_models[model_name]
    results = []

    for ab_name, keep_cols in ablation_specs.items():
        vprint(verbose, f"\n[TABULAR][{model_name}] Ablation: {ab_name}")
        X_sub = X[keep_cols]

        for fold_id, (tr_idx, te_idx) in enumerate(folds):
            X_tr, y_tr = X_sub.iloc[tr_idx], y.iloc[tr_idx]
            X_te, y_te = X_sub.iloc[te_idx], y.iloc[te_idx]

            num_cols, cat_cols = infer_feature_types(X_tr)
            pre = make_preprocessor_for_model(model_name, num_cols=num_cols, cat_cols=cat_cols)
            pipe = build_model_pipeline(pre, clone(model_proto))

            pipe.fit(X_tr, y_tr)

            if is_binary:
                proba_tr = pipe.predict_proba(X_tr)[:, 1].astype(float)
                thr = tune_threshold_max_f1_binary(y_tr.values, proba_tr)
                proba_te = pipe.predict_proba(X_te)[:, 1].astype(float)
                m = compute_fold_metrics_binary(y_te.values, proba_te, threshold=thr)
                m["threshold"] = float(thr)
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

                m = compute_fold_metrics_multi(y_te.values, proba_full)
                m["threshold"] = np.nan

            results.append(
                {
                    "model": model_name,
                    "ablation_name": ab_name,
                    "fold_id": int(fold_id),
                    "n_test_rows": int(len(te_idx)),
                    **m,
                }
            )
            if verbose:
                vprint(
                    verbose,
                    f"  Fold {fold_id} | PR_AUC={m['PR_AUC']:.4f} "
                    f"LogLoss={m['LogLoss']:.4f} F1={m['F1']:.4f}",
                )

    return pd.DataFrame(results)


def run_seq_ablation(
    df: pd.DataFrame,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    model_name: str,
    cfg: ModelConfig,
    stage: int,
    n_classes: Optional[int],
    seq_len: int,
    ablation_specs: Dict[str, List[str]],
    verbose: bool,
):
    if stage == 1:
        target_col = "y_pit"
        seq_models = get_stage1_sequential_models(cfg)
        is_binary = True
    else:
        target_col = "y_compound_encoded"
        seq_models = get_stage2_sequential_models(cfg, n_classes=n_classes)
        is_binary = False

    if model_name not in seq_models:
        raise ValueError(f"Model '{model_name}' not found in stage {stage} sequential models.")

    model_proto = seq_models[model_name]
    y_full = df[target_col].astype(int)

    key_cols = []
    if "race_id" in df.columns:
        key_cols.append("race_id")
    if "driver_id" in df.columns:
        key_cols.append("driver_id")
    if "lapno" in df.columns:
        key_cols.append("lapno")

    keys = df[key_cols].copy()
    if "race_id" not in keys.columns:
        keys["race_id"] = 1
    if "driver_id" not in keys.columns:
        keys["driver_id"] = 1
    if "lapno" not in keys.columns:
        keys["lapno"] = df["race_progress"] if "race_progress" in df.columns else np.arange(len(df))

    X_full = df.drop(
        columns=["y_pit", "y_compound", "y_compound_encoded", "race_id", "driver_id", "row_id"],
        errors="ignore",
    ).copy()

    results = []

    for ab_name, keep_cols in ablation_specs.items():
        vprint(verbose, f"\n[SEQ][{model_name}] Ablation: {ab_name}")
        X_sub = X_full[[c for c in keep_cols if c in X_full.columns]]

        for fold_id, (tr_idx, te_idx) in enumerate(folds):
            X_tr, y_tr = X_sub.iloc[tr_idx], y_full.iloc[tr_idx]

            num_cols, cat_cols = infer_feature_types(X_tr)
            pre = make_preprocessor_for_model("tcn", num_cols=num_cols, cat_cols=cat_cols)
            pre.fit(X_tr, y_tr)

            Xt_all = pre.transform(X_sub)
            if hasattr(Xt_all, "toarray"):
                Xt_all = Xt_all.toarray()
            Xt_all = Xt_all.astype(np.float32)

            X_seq, y_seq, idx_last, seq_idx, _ = build_feature_sequences(
                keys,
                Xt_all,
                y_full,
                seq_len=seq_len,
                pad_left=True,
                add_timestep_mask=True,
            )

            tr_mask = np.all((seq_idx == -1) | np.isin(seq_idx, tr_idx), axis=1)
            te_mask = np.all((seq_idx == -1) | np.isin(seq_idx, te_idx), axis=1)

            X_seq_tr, y_seq_tr = X_seq[tr_mask], y_seq[tr_mask]
            X_seq_te, y_seq_te = X_seq[te_mask], y_seq[te_mask]

            if len(y_seq_tr) == 0 or len(X_seq_te) == 0:
                vprint(verbose, f"  Fold {fold_id} | Skipping, no sequences.")
                continue

            class_w = None
            if is_binary:
                n_pos = int(np.sum(y_seq_tr == 1))
                n_neg = int(np.sum(y_seq_tr == 0))
                if n_pos > 0 and n_neg > 0:
                    pos_w = min(20.0, float(n_neg / n_pos))
                    class_w = {0: 1.0, 1: pos_w}

            model = clone(model_proto)
            if class_w is not None:
                model.fit(X_seq_tr, y_seq_tr, class_weight=class_w)
            else:
                model.fit(X_seq_tr, y_seq_tr)

            if is_binary:
                proba_tr = model.predict_proba(X_seq_tr)[:, 1].astype(float)
                thr = tune_threshold_max_f1_binary(y_seq_tr, proba_tr)
                proba_te = model.predict_proba(X_seq_te)[:, 1].astype(float)
                m = compute_fold_metrics_binary(y_seq_te, proba_te, threshold=thr)
                m["threshold"] = float(thr)
            else:
                proba_te_fold = model.predict_proba(X_seq_te)
                classes_seen = (
                    model.model_.classes_
                    if hasattr(model, "model_") and hasattr(model.model_, "classes_")
                    else np.unique(y_seq_tr)
                )
                proba_full = np.zeros((len(X_seq_te), n_classes), dtype=float)
                for c_idx, cls in enumerate(classes_seen):
                    proba_full[:, int(cls)] = proba_te_fold[:, c_idx]
                m = compute_fold_metrics_multi(y_seq_te, proba_full)
                m["threshold"] = np.nan

            results.append(
                {
                    "model": model_name,
                    "ablation_name": ab_name,
                    "fold_id": int(fold_id),
                    "n_test_rows": int(len(X_seq_te)),
                    **m,
                }
            )
            if verbose:
                vprint(
                    verbose,
                    f"  Fold {fold_id} | PR_AUC={m['PR_AUC']:.4f} "
                    f"LogLoss={m['LogLoss']:.4f} F1={m['F1']:.4f}",
                )

    return pd.DataFrame(results)

def main():
    data_path = "data/processed/dataset1.csv"
    stage = 1
    # data_path = "data/processed/dataset2.csv"
    # stage = 2
    models_to_run = ["xgb", "lstm"]  # ["xgb", "lstm", "svm", "rf"]
    outdir = "runs/ablation"
    run_name = "base_ablation"
    n_splits = 5
    seed = 42
    seq_len = 8
    verbose = False

    cfg = ModelConfig(random_state=int(seed))

    out_root = _ensure_dir(Path(outdir) / run_name)
    reports_dir = _ensure_dir(out_root / "reports")

    print(f"[base_ablation_eval] stage={stage} models={models_to_run} data={data_path}")

    n_classes = len(COMPOUND_CLASSES) if stage == 2 else None

    if stage == 1:
        df = load_stage1_dataset(data_path)
        X, y = get_stage1_xy(df)
        target_col = "y_pit"
        feature_groups = STAGE1_FEATURE_GROUPS
        tabular_models = list(get_stage1_tabular_models(cfg).keys())
        seq_models = list(get_stage1_sequential_models(cfg).keys())
    else:
        df_raw = load_stage2_dataset(data_path)
        df = encode_y_compound(df_raw, col="y_compound", out_col="y_compound_encoded")
        df = df.loc[~df["y_compound_encoded"].isna()].copy()
        X, y = get_stage2_xy(df)
        target_col = "y_compound_encoded"
        feature_groups = STAGE2_FEATURE_GROUPS
        tabular_models = list(get_stage2_tabular_models(cfg, n_classes=n_classes).keys())
        seq_models = list(get_stage2_sequential_models(cfg, n_classes=n_classes).keys())

    folds = make_race_group_folds(
        df,
        n_splits=n_splits,
        seed=seed,
    ).folds

    available_cols = [
        c
        for c in X.columns
        if c not in {"race_id", "driver_id", "row_id", "y_pit", "y_compound", "y_compound_encoded"}
    ]
    ablation_specs = make_ablation_specs(available_cols, feature_groups)
    _write_json(
        reports_dir / f"base_ablation_specs_stage{stage}.json",
        {"ablations": list(ablation_specs.keys())},
    )

    for model_name in models_to_run:
        print(f"\nRunning ablation for base model: {model_name}")

        if model_name not in tabular_models and model_name not in seq_models:
            print(f"Warning: model '{model_name}' not found in stage {stage} registries. Skipping.")
            continue

        if model_name in tabular_models:
            results = run_tabular_ablation(X,y,folds,model_name,cfg,stage,n_classes,ablation_specs,verbose)
        else:
            results = run_seq_ablation(df,folds, model_name, cfg, stage, n_classes, seq_len, ablation_specs, verbose)

        results.to_csv(
            reports_dir / f"base_ablation_results_{model_name}_stage{stage}.csv",
            index=False,
        )

        summary = summarize_ablation_results(results, baseline_name="baseline_full")
        summary.to_csv(
            reports_dir / f"base_ablation_summary_{model_name}_stage{stage}.csv",
            index=False,
        )

        print_top5_damage(summary, title=f"[{model_name.upper()} Stage {stage}] Damage rankings")

    print(f"\n[base_ablation_eval] done. Outputs in: {reports_dir}")

if __name__ == "__main__":
    main()