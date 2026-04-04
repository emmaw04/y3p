"""
evaluates different meta learners for the stacking ensemble.
we collect out of fold predictions from the base models and use them to train a meta model
for either stage 1 pit decision or stage 2 compound decision.
"""
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from src.data.data import (
    COMPOUND_CLASSES,
    COMPOUND_TO_INT,
    build_feature_sequences,
    encode_y_compound,
    get_stage1_xy,
    get_stage2_xy,
    infer_feature_types,
    load_stage1_dataset,
    load_stage2_dataset,
    make_race_group_folds,
)
from src.data.preprocessing import build_preprocessor, PreprocessConfig, make_preprocessor_for_model
from src.models.models import (
    build_model_pipeline,
    get_binary_base_learners,
    get_meta_binary_learners,
    get_meta_multiclass_learners,
    get_multiclass_base_learners,
    get_stage1_sequential_models,
)

def compute_binary_metrics(y_true: np.ndarray, proba_pos: np.ndarray) -> dict[str, float]:
    """calc standard binary metrics"""
    y_pred = (proba_pos >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba_pos)),
        "pr_auc": float(average_precision_score(y_true, proba_pos)),
        "logloss": float(log_loss(y_true, proba_pos, labels=[0, 1])),
    }

def compute_multiclass_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """calc standard multiclass metrics"""
    y_pred = np.argmax(proba, axis=1)
    labels = np.arange(proba.shape[1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "logloss": float(log_loss(y_true, proba, labels=labels)),
    }

def summarise(metrics_list: list[dict[str, Any]], keys: list[str]) -> dict[str, float]:
    """get the mean of each metric across all folds"""
    if not metrics_list:
        return {f"{k}_mean": float("nan") for k in keys}
    return {f"{k}_mean": float(np.mean([m[k] for m in metrics_list])) for k in keys}

def get_tcn_oof_preds(df1: pd.DataFrame, folds: list[tuple[np.ndarray, np.ndarray]], cfg: ModelConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    grabs out of fold predictions for the tcn model.
    we have to process it specially since it needs sequence data rather than just tabular rows.
    returns the predictions and effective length.
    """
    seq_len = 8
    y_full = df1["y_pit"].astype(int)
    keys = df1[["race_id", "driver_id", "lapno"]].copy()

    oof_pred = np.full(len(df1), np.nan, dtype=float)
    oof_eff_len = np.zeros(len(df1), dtype=float)

    x_tab = df1.drop(columns=["y_pit", "y_compound", "race_id", "driver_id"], errors="ignore").copy()
    num_cols, cat_cols = infer_feature_types(x_tab)
    base_pre = make_preprocessor_for_model("tcn", num_cols=num_cols, cat_cols=cat_cols)

    for fold, (tr_idx, va_idx) in enumerate(folds):
        print(f"tcn fold {fold} processing")
        x_tr, y_tr = x_tab.iloc[tr_idx], y_full.iloc[tr_idx]

        pre = clone(base_pre)
        pre.fit(x_tr, y_tr)

        xt_all = pre.transform(x_tab)
        if hasattr(xt_all, "toarray"):
            xt_all = xt_all.toarray()
        xt_all = xt_all.astype(np.float32)

        x_seq, y_seq, idx_last, seq_idx, eff_len = build_feature_sequences(
            keys, xt_all, y_full, seq_len=seq_len, pad_left=True, add_timestep_mask=True
        )

        tr_mask = np.all((seq_idx == -1) | np.isin(seq_idx, tr_idx), axis=1)
        va_mask = np.all((seq_idx == -1) | np.isin(seq_idx, va_idx), axis=1)

        x_seq_tr, y_seq_tr = x_seq[tr_mask], y_seq[tr_mask]
        x_seq_va = x_seq[va_mask]
        idx_last_va = idx_last[va_mask]
        eff_len_va = eff_len[va_mask]

        if len(y_seq_tr) == 0 or len(x_seq_va) == 0:
            continue

        n_pos = int(np.sum(y_seq_tr == 1))
        n_neg = int(np.sum(y_seq_tr == 0))

        if n_pos == 0 or n_neg == 0:
            const_p = 1.0 if n_neg == 0 else 0.0
            oof_pred[idx_last_va] = const_p
            oof_eff_len[idx_last_va] = eff_len_va
            continue

        pos_w = min(20.0, float(n_neg / n_pos))
        class_w = {0: 1.0, 1: pos_w}

        tcn = get_stage1_sequential_models(cfg)["tcn"]
        tcn.fit(x_seq_tr, y_seq_tr, class_weight=class_w)

        proba = tcn.predict_proba(x_seq_va)[:, 1]
        oof_pred[idx_last_va] = proba
        oof_eff_len[idx_last_va] = eff_len_va

    return oof_pred, oof_eff_len

def collect_tabular_oof_preds(x, y, folds, models_dict, is_binary: bool) -> np.ndarray:
    """
    trains base tabular models on the folds and grabs their out of fold predictions
    to be used as features by the meta model.
    """
    n = len(y)
    names = list(models_dict.keys())
    
    if is_binary:
        meta_x = np.zeros((n, len(names)), dtype=float)
    else:
        n_classes = len(COMPOUND_CLASSES)
        meta_x = np.zeros((n, len(names) * n_classes), dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(folds):
        print(f"collecting tabular oof preds for fold {fold}")
        x_tr, y_tr = x.iloc[tr_idx], y.iloc[tr_idx]
        x_va = x.iloc[va_idx]

        for j, name in enumerate(names):
            est = clone(models_dict[name])
            num_cols, cat_cols = infer_feature_types(x)
            pre = make_preprocessor_for_model(name, num_cols=num_cols, cat_cols=cat_cols)
            
            pipe = build_model_pipeline(pre, est)
            pipe.fit(x_tr, y_tr)

            if is_binary:
                meta_x[va_idx, j] = pipe.predict_proba(x_va)[:, 1]
            else:
                proba_fold = pipe.predict_proba(x_va)
                classes_seen = pipe.named_steps["model"].classes_ if hasattr(pipe.named_steps["model"], "classes_") else getattr(pipe, "classes_")
                proba_full = np.zeros((len(va_idx), n_classes), dtype=float)
                for c_idx, cls in enumerate(classes_seen):
                    proba_full[:, int(cls)] = proba_fold[:, c_idx]
                
                start = j * n_classes
                meta_x[va_idx, start:start + n_classes] = proba_full

    return meta_x

def run_stage1_stacking():
    """handles the whole pipeline for stage 1 pit decision stacking"""
    outdir.mkdir(parents=True, exist_ok=True)
    
    base_models = get_binary_base_learners(cfg)
    names = list(base_models.keys())
    
    print("starting stage 1 tabular base models")
    meta_x_tabular = collect_tabular_oof_preds(x, y, folds, base_models, is_binary=True)
    
    print("starting stage 1 tcn base model")
    tcn_oof, tcn_eff_len = get_tcn_oof_preds(df1, folds, cfg)
    
    tcn_oof_filled = np.nan_to_num(tcn_oof, nan=0.0)
    tcn_eff_len_scaled = tcn_eff_len / 8.0
    
    meta_x_all = np.column_stack([meta_x_tabular, tcn_oof_filled, tcn_eff_len_scaled])
    
    prob_cols = [f"p_{name}" for name in names] + ["tcn_proba", "tcn_effective_len"]
    (outdir / "base_oof_columns.json").write_text(json.dumps({"columns": prob_cols}, indent=2))
    np.save(outdir / "oof_base_preds.npy", meta_x_all)
    
    df_probs = pd.DataFrame(meta_x_all, columns=prob_cols)
    x_meta_df = pd.concat([df_probs.reset_index(drop=True), x.reset_index(drop=True)], axis=1)
    
    (outdir / "meta_features_names.json").write_text(json.dumps({"columns": list(x_meta_df.columns)}, indent=2))
    x_meta_df.to_parquet(outdir / "meta_features_oof.parquet", index=False)
    
    fold_rows = []
    oof_meta_proba = np.zeros(len(y), dtype=float)
    
    print(f"training stage 1 meta model: {meta_model_name}")
    for fold, (tr_idx, va_idx) in enumerate(folds):
        x_meta_tr = x_meta_df.iloc[tr_idx]
        x_meta_va = x_meta_df.iloc[va_idx]
        y_tr = y.iloc[tr_idx].to_numpy()
        y_va = y.iloc[va_idx].to_numpy()
        
        meta_est = clone(get_meta_binary_learners(cfg)[meta_model_name])
        num_m, cat_m = infer_feature_types(x_meta_df)
        meta_pre = build_preprocessor(num_cols=num_m, cat_cols=cat_m, cfg=PreprocessConfig(scale_numeric=False, sparse_onehot=True))
        
        meta_pipe = build_model_pipeline(meta_pre, meta_est)
        meta_pipe.fit(x_meta_tr, y_tr)
        
        proba = meta_pipe.predict_proba(x_meta_va)[:, 1]
        oof_meta_proba[va_idx] = proba
        
        metrics = compute_binary_metrics(y_va, proba)
        fold_rows.append({"fold": fold, "n_valid": len(va_idx), **metrics})
        
        dump(meta_pipe, outdir / f"meta_pipe_fold_{fold}.joblib")
        print(f"stage 1 fold {fold} done. metrics: accuracy {metrics['accuracy']:.4f} f1 {metrics['f1']:.4f}")
        
    np.save(outdir / "oof_meta_proba.npy", oof_meta_proba)
    
    summary = {
        "stage": "stage1_binary",
        "meta_model": meta_model_name,
        "n_folds": len(folds),
        **summarize(fold_rows, ["accuracy", "f1", "roc_auc", "pr_auc", "logloss"])
    }
    
    (outdir / "fold_metrics.json").write_text(json.dumps(fold_rows, indent=2))
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


    """handles the pipeline for stage 2 compound decision stacking"""
    outdir.mkdir(parents=True, exist_ok=True)
    n_classes = len(COMPOUND_CLASSES)
    
    base_models = get_multiclass_base_learners(cfg, n_classes=n_classes)
    names = list(base_models.keys())
    
    print("starting stage 2 tabular base models")
    meta_x = collect_tabular_oof_preds(x, y, folds, base_models, is_binary=False)
    
    prob_cols = []
    for name in names:
        for k in range(n_classes):
            prob_cols.append(f"p_{name}_c{k}")
            
    df_probs = pd.DataFrame(meta_x, columns=prob_cols)
    x_meta_df = pd.concat([df_probs.reset_index(drop=True), x.reset_index(drop=True)], axis=1)
    
    np.save(outdir / "oof_base_preds.npy", meta_x)
    x_meta_df.to_parquet(outdir / "meta_features_oof.parquet", index=False)
    
    fold_rows = []
    oof_meta_proba = np.zeros((len(y), n_classes), dtype=float)
    
    print(f"training stage 2 meta model: {meta_model_name}")
    for fold, (tr_idx, va_idx) in enumerate(folds):
        x_meta_tr = x_meta_df.iloc[tr_idx]
        x_meta_va = x_meta_df.iloc[va_idx]
        y_tr = y.iloc[tr_idx].to_numpy()
        y_va = y.iloc[va_idx].to_numpy()
        
        meta_est = clone(get_meta_multiclass_learners(cfg, n_classes=n_classes)[meta_model_name])
        num_m, cat_m = infer_feature_types(x_meta_df)
        meta_pre = build_preprocessor(num_cols=num_m, cat_cols=cat_m, cfg=PreprocessConfig(scale_numeric=False, sparse_onehot=True))
        
        meta_pipe = build_model_pipeline(meta_pre, meta_est)
        meta_pipe.fit(x_meta_tr, y_tr)
        
        proba_fold = meta_pipe.predict_proba(x_meta_va)
        
        classes_seen = meta_pipe.named_steps["model"].classes_ if hasattr(meta_pipe.named_steps["model"], "classes_") else getattr(meta_pipe, "classes_")
        proba_full = np.zeros((len(va_idx), n_classes), dtype=float)
        for c_idx, cls in enumerate(classes_seen):
            proba_full[:, int(cls)] = proba_fold[:, c_idx]
            
        oof_meta_proba[va_idx] = proba_full
        metrics = compute_multiclass_metrics(y_va, proba_full)
        fold_rows.append({"fold": fold, "n_valid": len(va_idx), **metrics})
        
        dump(meta_pipe, outdir / f"meta_pipe_fold_{fold}.joblib")
        print(f"stage 2 fold {fold} done. metrics: accuracy {metrics['accuracy']:.4f} f1_macro {metrics['f1_macro']:.4f}")
        
    np.save(outdir / "oof_meta_proba.npy", oof_meta_proba)
    
    summary = {
        "stage": "stage2_multiclass",
        "meta_model": meta_model_name,
        "n_folds": len(folds),
        "n_classes": n_classes,
        **summarize(fold_rows, ["accuracy", "f1_macro", "logloss"])
    }
    
    (outdir / "fold_metrics.json").write_text(json.dumps(fold_rows, indent=2))
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    (outdir / "classes.json").write_text(json.dumps({"classes": list(COMPOUND_CLASSES)}, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser(description="evaluate stack meta models for stage1 and stage2")
    ap.add_argument("--data_stage1", help="path to stage 1 data")
    ap.add_argument("--data_stage2", help="path to stage 2 data")
    ap.add_argument("--outdir", default="runs/stack", help="where to save models and metrics")
    ap.add_argument("--meta_model_stage1", choices=["xgb", "lr", "mlp"], default="xgb", help="meta model to use for pit decision")
    ap.add_argument("--meta_model_stage2", choices=["xgb", "lr", "mlp"], default="xgb", help="meta model to use for compound decision")
    ap.add_argument("--only_stage", choices=["all", "stage1", "stage2"], default="all", help="which stage to run")
    args = ap.parse_args()

    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=True)

    seed = 42
    n_splits = 5
    

    if args.only_stage in {"all", "stage1"}:
        if not args.data_stage1:
            raise ValueError("need stage 1 data path")
        df1 = load_stage1_dataset(args.data_stage1)
        x1, y1 = get_stage1_xy(df1)
        fb1 = make_race_group_folds(df1, target_col="y_pit", n_splits=n_splits, seed=seed)
        
        print("running stage 1 stacking evaluation")
        s1_out = root / "stage1_binary"
        s1_summary = run_stage1_stacking(df1, x1, y1, cfg, fb1.folds, args.meta_model_stage1, s1_out)
        print("stage 1 summary:", json.dumps(s1_summary, indent=2))

    if args.only_stage in {"all", "stage2"}:
        if not args.data_stage2:
            raise ValueError("need stage 2 data path")
        df2 = load_stage2_dataset(args.data_stage2, strict=True)
        x2, y2 = get_stage2_xy(df2)
        
        # map classes to integers before grouping folds to ensure balance
        df2_tmp = encode_y_compound(df2, col="y_compound", out_col="y_compound_encoded")
        fb2 = make_race_group_folds(df2_tmp, target_col="y_compound_encoded", n_splits=n_splits, seed=seed)
        
        print("running stage 2 stacking evaluation")
        s2_out = root / "stage2_multiclass"
        s2_summary = run_stage2_stacking(x2, y2, cfg, fb2.folds, args.meta_model_stage2, s2_out)
        print("stage 2 summary:", json.dumps(s2_summary, indent=2))

if __name__ == "__main__":
    main()
