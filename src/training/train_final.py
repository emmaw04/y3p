"""
trains and freezes the final models for both stages.
we use out of fold predictions to train the meta learner, then retrain
all base models on the full dataset before saving them for inference.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import clone

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
from src.data.preprocessing import build_preprocessor, PreprocessConfig, make_preprocessor_for_model
from src.models.models import (
    ModelConfig,
    build_model_pipeline,
    get_binary_base_learners,
    get_meta_binary_learners,
    get_meta_multiclass_learners,
    get_multiclass_base_learners,
    get_stage1_sequential_models,
)


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


def get_tcn_oof_preds(df1: pd.DataFrame, folds: list[tuple[np.ndarray, np.ndarray]], cfg: ModelConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    grabs out of fold predictions for the tcn model.
    we process it specially since it needs sequence data.
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


def train_stage1_final(df1, x, y, folds, outdir: Path, cfg: ModelConfig):
    """handles the final training pipeline for stage 1 pit decision"""
    art_dir = outdir / "stage1_binary" / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    
    base_models = get_binary_base_learners(cfg)
    names = list(base_models.keys())
    
    print("starting stage 1 tabular base models oof collection")
    meta_x_tabular = collect_tabular_oof_preds(x, y, folds, base_models, is_binary=True)
    
    print("starting stage 1 tcn base model oof collection")
    tcn_oof, tcn_eff_len = get_tcn_oof_preds(df1, folds, cfg)
    
    tcn_oof_filled = np.nan_to_num(tcn_oof, nan=0.0)
    tcn_eff_len_scaled = tcn_eff_len / 8.0
    
    meta_x_all = np.column_stack([meta_x_tabular, tcn_oof_filled, tcn_eff_len_scaled])
    
    prob_cols = [f"p_{name}" for name in names] + ["tcn_proba", "tcn_effective_len"]
    df_probs = pd.DataFrame(meta_x_all, columns=prob_cols)
    x_meta_df = pd.concat([df_probs.reset_index(drop=True), x.reset_index(drop=True)], axis=1)
    
    print("fitting final stage 1 meta model on full dataset")
    meta_est = get_meta_binary_learners(cfg)["xgb"]
    num_m, cat_m = infer_feature_types(x_meta_df)
    meta_pre = build_preprocessor(num_cols=num_m, cat_cols=cat_m, cfg=PreprocessConfig(scale_numeric=False, sparse_onehot=True))
    
    meta_pipe = build_model_pipeline(meta_pre, meta_est)
    meta_pipe.fit(x_meta_df, y.to_numpy())
    dump(meta_pipe, art_dir / "meta_pipeline.joblib")
    
    base_frozen = {}
    print("retraining tabular base models on full dataset")
    num_cols, cat_cols = infer_feature_types(x)
    
    for name in names:
        est = clone(base_models[name])
        pre = make_preprocessor_for_model(name, num_cols=num_cols, cat_cols=cat_cols)
        
        pipe = build_model_pipeline(pre, est)
        pipe.fit(x, y)
        path = art_dir / f"base_{name}_pipeline.joblib"
        dump(pipe, path)
        base_frozen[name] = path.name
        
    print("retraining tcn on full dataset")
    y_full = df1["y_pit"].astype(int)
    keys = df1[["race_id", "driver_id", "lapno"]].copy()
    x_tab = df1.drop(columns=["y_pit", "y_compound", "race_id", "driver_id"], errors="ignore").copy()
    
    num_t, cat_t = infer_feature_types(x_tab)
    tcn_pre = make_preprocessor_for_model("tcn", num_cols=num_t, cat_cols=cat_t)
    tcn_pre.fit(x_tab, y_full)

    xt_all = tcn_pre.transform(x_tab)
    if hasattr(xt_all, "toarray"):
        xt_all = xt_all.toarray()
    xt_all = xt_all.astype(np.float32)

    x_seq, y_seq, _, _, _ = build_feature_sequences(
        keys, xt_all, y_full, seq_len=8, pad_left=True, add_timestep_mask=True
    )

    n_pos = int(np.sum(y_seq == 1))
    n_neg = int(np.sum(y_seq == 0))
    pos_w = min(20.0, float(n_neg / n_pos)) if n_pos > 0 else 1.0
    
    tcn = get_stage1_sequential_models(cfg)["tcn"]
    tcn.fit(x_seq, y_seq, class_weight={0: 1.0, 1: float(pos_w)})

    dump(tcn_pre, art_dir / "tcn_preprocessor.joblib")
    tcn.model_.save(art_dir / "tcn_model.keras")

    (art_dir / "feature_columns.json").write_text(json.dumps({"columns": list(x.columns)}, indent=2))
    (art_dir / "meta_features.json").write_text(json.dumps({"columns": list(x_meta_df.columns)}, indent=2))
    
    return {
        "stage": "stage1_binary",
        "base_models": names,
        "meta_model": "xgb",
        "tcn_enabled": True,
        "artifacts": {
            "meta_pipeline": "meta_pipeline.joblib",
            "base_pipelines": base_frozen,
            "tcn_preprocessor": "tcn_preprocessor.joblib",
            "tcn_model": "tcn_model.keras"
        }
    }


def train_stage2_final(x, y, folds, outdir: Path, cfg: ModelConfig):
    """handles the final training pipeline for stage 2 compound decision"""
    art_dir = outdir / "stage2_multiclass" / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    n_classes = len(COMPOUND_CLASSES)
    
    base_models = get_multiclass_base_learners(cfg, n_classes=n_classes)
    names = list(base_models.keys())
    
    print("starting stage 2 tabular base models oof collection")
    meta_x = collect_tabular_oof_preds(x, y, folds, base_models, is_binary=False)
    
    prob_cols = []
    for name in names:
        for k in range(n_classes):
            prob_cols.append(f"p_{name}_c{k}")
            
    df_probs = pd.DataFrame(meta_x, columns=prob_cols)
    x_meta_df = pd.concat([df_probs.reset_index(drop=True), x.reset_index(drop=True)], axis=1)
    
    print("fitting final stage 2 meta model on full dataset")
    meta_est = get_meta_multiclass_learners(cfg, n_classes=n_classes)["xgb"]
    num_m, cat_m = infer_feature_types(x_meta_df)
    meta_pre = build_preprocessor(num_cols=num_m, cat_cols=cat_m, cfg=PreprocessConfig(scale_numeric=False, sparse_onehot=True))
    
    meta_pipe = build_model_pipeline(meta_pre, meta_est)
    meta_pipe.fit(x_meta_df, y.to_numpy())
    dump(meta_pipe, art_dir / "meta_pipeline.joblib")
    
    base_frozen = {}
    print("retraining tabular base models on full dataset")
    num_cols, cat_cols = infer_feature_types(x)
    
    for name in names:
        est = clone(base_models[name])
        pre = make_preprocessor_for_model(name, num_cols=num_cols, cat_cols=cat_cols)
        
        pipe = build_model_pipeline(pre, est)
        pipe.fit(x, y)
        path = art_dir / f"base_{name}_pipeline.joblib"
        dump(pipe, path)
        base_frozen[name] = path.name
        
    (art_dir / "feature_columns.json").write_text(json.dumps({"columns": list(x.columns)}, indent=2))
    (art_dir / "meta_features.json").write_text(json.dumps({"columns": list(x_meta_df.columns)}, indent=2))
    (art_dir / "classes.json").write_text(json.dumps({"classes": list(COMPOUND_CLASSES)}, indent=2))
    
    return {
        "stage": "stage2_multiclass",
        "n_classes": n_classes,
        "base_models": names,
        "meta_model": "xgb",
        "artifacts": {
            "meta_pipeline": "meta_pipeline.joblib",
            "base_pipelines": base_frozen
        }
    }


def main():
    ap = argparse.ArgumentParser(description="train final frozen models for both stages")
    ap.add_argument("--data_stage1", help="path to stage 1 data")
    ap.add_argument("--data_stage2", help="path to stage 2 data")
    ap.add_argument("--only_stage", choices=["all", "stage1", "stage2"], default="all", help="which stage to run")
    args = ap.parse_args()

    root = Path("runs_final") / "final_run"
    root.mkdir(parents=True, exist_ok=True)
    
    seed = 42
    n_splits = 5
    cfg = ModelConfig(random_state=seed)

    manifest = {
        "seed": seed,
        "n_splits": n_splits,
        "holdout_race_ids": list(HOLDOUT_RACE_IDS)
    }

    if args.only_stage in {"all", "stage1"}:
        if not args.data_stage1:
            raise ValueError("need stage 1 data path")
        df1 = load_stage1_dataset(args.data_stage1)
        x1, y1 = get_stage1_xy(df1)
        fb1 = make_race_group_folds(df1, target_col="y_pit", n_splits=n_splits, seed=seed)
        
        print("training final stage 1 models")
        manifest["stage1"] = train_stage1_final(df1, x1, y1, fb1.folds, root, cfg)

    if args.only_stage in {"all", "stage2"}:
        if not args.data_stage2:
            raise ValueError("need stage 2 data path")
        df2 = load_stage2_dataset(args.data_stage2, strict=True)
        x2, y2 = get_stage2_xy(df2)
        
        df2_tmp = encode_y_compound(df2, col="y_compound", out_col="y_compound_encoded")
        fb2 = make_race_group_folds(df2_tmp, target_col="y_compound_encoded", n_splits=n_splits, seed=seed)
        
        print("training final stage 2 models")
        manifest["stage2"] = train_stage2_final(x2, y2, fb2.folds, root, cfg)
        
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"done. all artifacts saved to {root}")


if __name__ == "__main__":
    main()
