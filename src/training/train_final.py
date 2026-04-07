"""
trains and freezes the final models for both stages.
we use out of fold predictions to train the meta learner, then retrain
all base models on the full dataset before saving them for inference.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import clone
from sklearn.utils.class_weight import compute_class_weight

from src.data.data import (
    COMPOUND_CLASSES,
    HOLDOUT_RACE_IDS,
    build_feature_sequences,
    build_feature_sequences_from_reference,
    encode_y_compound,
    get_stage1_xy,
    get_stage2_xy,
    infer_feature_types,
    load_stage1_dataset,
    load_stage2_dataset,
    make_race_group_folds,
)
from src.data.preprocessing import (
    build_preprocessor,
    PreprocessConfig,
    make_preprocessor_for_model,
)
from src.models.models import (
    build_model_pipeline,
    make_stage1_xgb,
    make_stage1_svm,
    make_lstm_binary,
    make_tcn_gru_binary,
    make_meta_binary_xgb,
    make_stage2_rf,
    make_stage2_xgb,
    make_stage2_svm,
    make_tcn_gru_multiclass,
    make_meta_multiclass_xgb,
    ModelConfig,
)


def collect_tabular_oof_preds(x, y, folds, models_dict, is_binary: bool) -> np.ndarray:
    """
    trains base tabular models on the folds and grabs their out of fold predictions
    to be used as features by the meta model.
    """
    n = len(y)
    names = list(models_dict.keys())

    #creates a storage array, if the base learner is binary it creates one probability column per model, if multiclass it creates one full probability vector per model
    if is_binary:
        meta_x = np.zeros((n, len(names)), dtype=float)
    else:
        n_classes = len(COMPOUND_CLASSES)
        meta_x = np.zeros((n, len(names) * n_classes), dtype=float)

    #loop through each fold
    for fold, (tr_idx, va_idx) in enumerate(folds):
        print(f"collecting tabular oof preds for fold {fold}")
        x_tr, y_tr = x.iloc[tr_idx], y.iloc[tr_idx] #split into training and validation rows
        x_va = x.iloc[va_idx]

        for j, name in enumerate(names):
            est = clone(models_dict[name])
            num_cols, cat_cols = infer_feature_types(x_tr)
            pre = make_preprocessor_for_model(name, num_cols=num_cols, cat_cols=cat_cols) #build preprocessor

            pipe = build_model_pipeline(pre, est) #combine preprocessor and model into one pipeline
            pipe.fit(x_tr, y_tr) #fit on training fold

            if is_binary: #for binary models only the positive class probability is saved
                meta_x[va_idx, j] = pipe.predict_proba(x_va)[:, 1]
            else: #ensures probabilities are given for all classes
                n_classes = len(COMPOUND_CLASSES)
                proba_fold = pipe.predict_proba(x_va)
                classes_seen = (
                    pipe.named_steps["model"].classes_
                    if hasattr(pipe.named_steps["model"], "classes_")
                    else getattr(pipe, "classes_")
                )
                proba_full = np.zeros((len(va_idx), n_classes), dtype=float)
                for c_idx, cls in enumerate(classes_seen):
                    proba_full[:, int(cls)] = proba_fold[:, c_idx]

                start = j * n_classes
                meta_x[va_idx, start:start + n_classes] = proba_full

    return meta_x


def get_stage1_seq_oof_preds(
    df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    model_name: str,
    model_instance,
) -> np.ndarray:
    """
    gets out of fold predictions for stage 1 sequential models
    """
    seq_len = 8
    keys = df[["race_id", "driver_id", "lapno"]].copy() #unique identifiers
    y_full = df["y_pit"].astype(int) #target variable
    x_tab = df.drop(columns=["y_pit", "race_id", "driver_id"], errors="ignore").copy()
    oof_pred = np.full(len(df), np.nan, dtype=float)

    num_cols, cat_cols = infer_feature_types(x_tab)
    base_pre = make_preprocessor_for_model(model_name, num_cols=num_cols, cat_cols=cat_cols) #make preprocessor 

    #loop through each fold
    for fold, (tr_idx, va_idx) in enumerate(folds):
        print(f"{model_name} fold {fold} processing")
        x_tr = x_tab.iloc[tr_idx]

        pre = clone(base_pre)
        pre.fit(x_tr, y_full.iloc[tr_idx]) #fit preprocessor only on training rows 

        xt_all = pre.transform(x_tab) #transform the full table
        if hasattr(xt_all, "toarray"):
            xt_all = xt_all.toarray()
        xt_all = xt_all.astype(np.float32)

        #build sequences
        x_seq, y_seq, idx_last, seq_idx, _ = build_feature_sequences(
            keys,
            xt_all,
            y_full,
            seq_len=seq_len,
            pad_left=True,
            add_timestep_mask=True,
        )

        tr_mask = np.all((seq_idx == -1) | np.isin(seq_idx, tr_idx), axis=1)
        va_mask = np.all((seq_idx == -1) | np.isin(seq_idx, va_idx), axis=1)

        #splits into training and validation sequences
        x_seq_tr, y_seq_tr = x_seq[tr_mask], y_seq[tr_mask]
        x_seq_va = x_seq[va_mask]
        idx_last_va = idx_last[va_mask]

        model = clone(model_instance)

        #computes binary class weights
        n_pos = int(np.sum(y_seq_tr == 1))
        n_neg = int(np.sum(y_seq_tr == 0))

        if n_pos == 0 or n_neg == 0:
            const_p = 1.0 if n_neg == 0 else 0.0
            oof_pred[idx_last_va] = const_p
            continue

        pos_w = min(20.0, float(n_neg / n_pos))
        class_w = {0: 1.0, 1: pos_w}

        #fits the model and gets its predictions
        model.fit(x_seq_tr, y_seq_tr, class_weight=class_w)
        proba = model.predict_proba(x_seq_va)[:, 1]
        oof_pred[idx_last_va] = proba

    return oof_pred

def _get_stage2_reference_x(df1_ref: pd.DataFrame, stage2_feature_cols: list[str]) -> pd.DataFrame:
    """
    helper for stage 2 sequence modelling, checks that all the attributes in dataset 2 exist in dataset 1, and returns these attributes
    """
    missing = [c for c in stage2_feature_cols if c not in df1_ref.columns]
    if missing:
        raise ValueError(
            "stage 1 reference dataframe is missing stage 2 feature columns: "
            f"{missing}"
        )
    return df1_ref[stage2_feature_cols].copy()


def get_stage2_seq_oof_preds(
    df1_ref: pd.DataFrame,
    df2_target: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    model_name: str,
    model_instance,
    stage2_feature_cols: list[str],
) -> np.ndarray:
    """
    out of fold predictions for stage 2 sequential models

    sequences are built from the full lap-level reference dataframe from dataset1,
    but labels and oof alignment come from the stage 2 pit-event dataframe df2_target.
    only stage 2 feature columns are used from dataset1
    """
    seq_len = 8
    n_classes = len(COMPOUND_CLASSES)

    ref_keys = df1_ref[["race_id", "driver_id", "lapno"]].copy()
    ref_x = _get_stage2_reference_x(df1_ref, stage2_feature_cols)

    target_keys = df2_target[["race_id", "driver_id", "lapno"]].copy()
    y_target = df2_target["y_compound_encoded"].astype(int)

    oof_pred = np.full((len(df2_target), n_classes), np.nan, dtype=float)

    num_cols, cat_cols = infer_feature_types(ref_x)
    base_pre = make_preprocessor_for_model(model_name, num_cols=num_cols, cat_cols=cat_cols)

    for fold, (tr_idx, va_idx) in enumerate(folds):
        print(f"{model_name} fold {fold} processing")

        train_race_ids = set(df2_target.iloc[tr_idx]["race_id"].astype(int).tolist())
        ref_train_mask = df1_ref["race_id"].isin(train_race_ids).to_numpy()

        x_ref_tr = ref_x.loc[ref_train_mask].copy()

        pre = clone(base_pre)
        pre.fit(x_ref_tr)

        xt_ref_all = pre.transform(ref_x)
        if hasattr(xt_ref_all, "toarray"):
            xt_ref_all = xt_ref_all.toarray()
        xt_ref_all = xt_ref_all.astype(np.float32)

        x_seq, y_seq, idx_target, _, _ = build_feature_sequences_from_reference(
            reference_keys=ref_keys,
            X_reference=xt_ref_all,
            target_keys=target_keys,
            y_target=y_target,
            seq_len=seq_len,
            pad_left=True,
            add_timestep_mask=True,
            strict_match=True,
        )

        tr_mask = np.isin(idx_target, tr_idx)
        va_mask = np.isin(idx_target, va_idx)

        x_seq_tr, y_seq_tr = x_seq[tr_mask], y_seq[tr_mask]
        x_seq_va = x_seq[va_mask]
        idx_target_va = idx_target[va_mask]

        if len(y_seq_tr) == 0 or len(x_seq_va) == 0:
            continue

        model = clone(model_instance)

        present = np.unique(y_seq_tr)
        w = compute_class_weight(class_weight="balanced", classes=present, y=y_seq_tr)
        class_w = {int(c): float(wi) for c, wi in zip(present, w)}
        for c in range(n_classes):
            class_w.setdefault(c, 1.0)

        model.fit(x_seq_tr, y_seq_tr, class_weight=class_w)
        proba_fold = model.predict_proba(x_seq_va)

        classes_seen = (
            model.model_.classes_
            if hasattr(model, "model_") and hasattr(model.model_, "classes_")
            else (model.classes_ if hasattr(model, "classes_") else np.arange(n_classes))
        )

        proba_full = np.zeros((len(x_seq_va), n_classes), dtype=float)
        for c_idx, cls in enumerate(classes_seen):
            proba_full[:, int(cls)] = proba_fold[:, c_idx]

        oof_pred[idx_target_va] = proba_full

    return oof_pred


def train_stage1_final(df1, x, y, folds, outdir: Path, cfg: ModelConfig):
    """handles the final training pipeline for stage 1 pit decision"""
    art_dir = outdir / "stage1_binary" / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    tab_models = {
        "xgb": make_stage1_xgb(cfg),
        "svm": make_stage1_svm(cfg),
    }
    seq_models = {
        "lstm": make_lstm_binary(cfg),
        "tcn_gru": make_tcn_gru_binary(cfg),
    }

    tab_names = list(tab_models.keys())
    seq_names = list(seq_models.keys())
    all_names = tab_names + seq_names

    print("starting stage 1 tabular base models oof collection")
    meta_x_tabular = collect_tabular_oof_preds(x, y, folds, tab_models, is_binary=True)

    seq_oofs = []
    for s_name, s_model in seq_models.items():
        print(f"starting stage 1 {s_name} base model oof collection")
        seq_oof = get_stage1_seq_oof_preds(df1, folds, s_name, s_model)
        seq_oofs.append(np.nan_to_num(seq_oof, nan=0.0))

    meta_x_all = np.column_stack([meta_x_tabular] + seq_oofs)

    prob_cols = [f"p_{name}" for name in all_names]
    df_probs = pd.DataFrame(meta_x_all, columns=prob_cols)

    # meta learner just gets base probabilities
    x_meta_df = pd.concat([x.reset_index(drop=True),df_probs.reset_index(drop=True)], axis=1)

    print("generating OOF predictions for stage 1 meta learner")
    meta_oof_proba = np.zeros(len(y), dtype=float)
    fold_assignment = np.full(len(y), -1, dtype=int)

    for fold, (tr_idx, va_idx) in enumerate(folds):
        fold_assignment[va_idx] = fold
        x_meta_tr = x_meta_df.iloc[tr_idx]
        x_meta_va = x_meta_df.iloc[va_idx]
        y_tr = y.iloc[tr_idx].to_numpy()

        meta_est = make_meta_binary_xgb(cfg)
        num_m, cat_m = infer_feature_types(x_meta_tr)
        meta_pre = build_preprocessor(
            num_cols=num_m,
            cat_cols=cat_m,
            cfg=PreprocessConfig(scale_numeric=False, sparse_onehot=True),
        )

        meta_pipe = build_model_pipeline(meta_pre, meta_est)
        meta_pipe.fit(x_meta_tr, y_tr)
        meta_oof_proba[va_idx] = meta_pipe.predict_proba(x_meta_va)[:, 1]

    print("saving stage 1 OOF predictions")
    oof_df = pd.DataFrame({
        "row_id": df1.index,
        "race_id": df1["race_id"],
        "driver_id": df1["driver_id"],
        "lapno": df1["lapno"],
        "fold": fold_assignment,
        "y_pit": df1["y_pit"],
        "is_wet_race": df1["is_wet_race"],
        "fcy_status": df1["fcy_status"],
        "race_track": df1["race_track"],
        "current_compound": df1["current_compound"],
    })

    for i, name in enumerate(all_names):
        oof_df[f"p_{name}"] = meta_x_all[:, i]
    oof_df["meta_proba"] = meta_oof_proba

    oof_df.to_csv(art_dir / "oof_predictions.csv", index=False)

    print("fitting final stage 1 meta model on full dataset")
    meta_est = make_meta_binary_xgb(cfg)
    num_m, cat_m = infer_feature_types(x_meta_df)
    meta_pre = build_preprocessor(
        num_cols=num_m,
        cat_cols=cat_m,
        cfg=PreprocessConfig(scale_numeric=False, sparse_onehot=True),
    )

    meta_pipe = build_model_pipeline(meta_pre, meta_est)
    meta_pipe.fit(x_meta_df, y.to_numpy())
    dump(meta_pipe, art_dir / "meta_pipeline.joblib")

    base_frozen = {}
    print("retraining tabular base models on full dataset")
    num_cols, cat_cols = infer_feature_types(x)

    for name, est in tab_models.items():
        pre = make_preprocessor_for_model(name, num_cols=num_cols, cat_cols=cat_cols)
        pipe = build_model_pipeline(pre, est)
        pipe.fit(x, y)
        path = art_dir / f"base_{name}_pipeline.joblib"
        dump(pipe, path)
        base_frozen[name] = path.name

    print("retraining sequential base models on full dataset")
    y_full = df1["y_pit"].astype(int)
    keys = df1[["race_id", "driver_id", "lapno"]].copy()
    x_tab = df1.drop(columns=["y_pit", "y_compound", "race_id", "driver_id"], errors="ignore").copy()

    for name, model in seq_models.items():
        num_t, cat_t = infer_feature_types(x_tab)
        pre = make_preprocessor_for_model(name, num_cols=num_t, cat_cols=cat_t)
        pre.fit(x_tab, y_full)

        xt_all = pre.transform(x_tab)
        if hasattr(xt_all, "toarray"):
            xt_all = xt_all.toarray()
        xt_all = xt_all.astype(np.float32)

        x_seq, y_seq, _, _, _ = build_feature_sequences(
            keys,
            xt_all,
            y_full,
            seq_len=8,
            pad_left=True,
            add_timestep_mask=True,
        )

        n_pos = int(np.sum(y_seq == 1))
        n_neg = int(np.sum(y_seq == 0))
        if n_pos == 0 or n_neg == 0:
            raise RuntimeError(f"Degenerate sequence labels for {name}: n_pos={n_pos}, n_neg={n_neg}")

        pos_w = min(20.0, float(n_neg / n_pos))
        model.fit(x_seq, y_seq, class_weight={0: 1.0, 1: float(pos_w)})

        dump(pre, art_dir / f"{name}_preprocessor.joblib")
        model.model_.save(art_dir / f"{name}_model.keras")
        base_frozen[name] = {
            "preprocessor": f"{name}_preprocessor.joblib",
            "model": f"{name}_model.keras",
        }

    (art_dir / "feature_columns.json").write_text(json.dumps({"columns": list(x.columns)}, indent=2))
    (art_dir / "meta_features.json").write_text(json.dumps({"columns": list(x_meta_df.columns)}, indent=2))

    return {
        "stage": "stage1_binary",
        "tab_models": tab_names,
        "seq_models": seq_names,
        "meta_model": "xgb",
        "meta_uses_context": True,
        "artifacts": {
            "meta_pipeline": "meta_pipeline.joblib",
            "base_pipelines": base_frozen,
            "oof_predictions": "oof_predictions.csv",
        },
    }


def train_stage2_final(df1_ref, df2, x, y, folds, outdir: Path, cfg: ModelConfig):
    """handles the final training pipeline for stage 2 compound decision"""
    art_dir = outdir / "stage2_multiclass" / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    n_classes = len(COMPOUND_CLASSES)

    tab_models = {
        "rf": make_stage2_rf(cfg),
        "xgb": make_stage2_xgb(cfg, n_classes=n_classes),
        "svm": make_stage2_svm(cfg),
    }
    seq_models = {
        "tcn_gru": make_tcn_gru_multiclass(cfg, n_classes=n_classes),
    }

    tab_names = list(tab_models.keys())
    seq_names = list(seq_models.keys())
    all_names = tab_names + seq_names

    print("starting stage 2 tabular base models oof collection")
    meta_x_tabular = collect_tabular_oof_preds(x, y, folds, tab_models, is_binary=False)

    stage2_feature_cols = list(x.columns)

    seq_oofs = []
    for s_name, s_model in seq_models.items():
        print(f"starting stage 2 {s_name} base model oof collection")
        seq_oof = get_stage2_seq_oof_preds(
            df1_ref=df1_ref,
            df2_target=df2,
            folds=folds,
            model_name=s_name,
            model_instance=s_model,
            stage2_feature_cols=stage2_feature_cols,
        )
        seq_oof = np.nan_to_num(seq_oof, nan=1.0 / n_classes)
        seq_oofs.append(seq_oof)

    meta_x_all = np.column_stack([meta_x_tabular] + seq_oofs)

    prob_cols = []
    for name in all_names:
        for k in range(n_classes):
            prob_cols.append(f"p_{name}_c{k}")

    df_probs = pd.DataFrame(meta_x_all, columns=prob_cols)
    x_meta_df = df_probs.copy()

    print("generating OOF predictions for stage 2 meta learner")
    meta_oof_proba = np.zeros((len(y), n_classes), dtype=float)
    fold_assignment = np.full(len(y), -1, dtype=int)

    for fold, (tr_idx, va_idx) in enumerate(folds):
        fold_assignment[va_idx] = fold
        x_meta_tr = x_meta_df.iloc[tr_idx]
        x_meta_va = x_meta_df.iloc[va_idx]
        y_tr = y.iloc[tr_idx].to_numpy()

        meta_est = make_meta_multiclass_xgb(cfg, n_classes=n_classes)
        num_m, cat_m = infer_feature_types(x_meta_tr)
        meta_pre = build_preprocessor(
            num_cols=num_m,
            cat_cols=cat_m,
            cfg=PreprocessConfig(scale_numeric=False, sparse_onehot=True),
        )

        meta_pipe = build_model_pipeline(meta_pre, meta_est)
        meta_pipe.fit(x_meta_tr, y_tr)

        proba_fold = meta_pipe.predict_proba(x_meta_va)
        classes_seen = (
            meta_pipe.named_steps["model"].classes_
            if hasattr(meta_pipe.named_steps["model"], "classes_")
            else getattr(meta_pipe, "classes_")
        )
        proba_full = np.zeros((len(va_idx), n_classes), dtype=float)
        for c_idx, cls in enumerate(classes_seen):
            proba_full[:, int(cls)] = proba_fold[:, c_idx]

        meta_oof_proba[va_idx] = proba_full

    print("saving stage 2 OOF predictions")
    oof_df = pd.DataFrame({
        "row_id": df2.index,
        "race_id": df2["race_id"],
        "driver_id": df2["driver_id"],
        "lapno": df2["lapno"],
        "fold": fold_assignment,
        "y_compound": df2["y_compound"],
        "is_wet_race": df2["is_wet_race"],
        "race_track": df2["race_track"],
        "current_compound": df2["current_compound"],
    })

    for j, name in enumerate(all_names):
        for k in range(n_classes):
            oof_df[f"p_{name}_c{k}"] = meta_x_all[:, j * n_classes + k]

    for k in range(n_classes):
        oof_df[f"meta_proba_c{k}"] = meta_oof_proba[:, k]

    oof_df.to_csv(art_dir / "oof_predictions.csv", index=False)

    print("fitting final stage 2 meta model on full dataset")
    meta_est = make_meta_multiclass_xgb(cfg, n_classes=n_classes)
    num_m, cat_m = infer_feature_types(x_meta_df)
    meta_pre = build_preprocessor(
        num_cols=num_m,
        cat_cols=cat_m,
        cfg=PreprocessConfig(scale_numeric=False, sparse_onehot=True),
    )

    meta_pipe = build_model_pipeline(meta_pre, meta_est)
    meta_pipe.fit(x_meta_df, y.to_numpy())
    dump(meta_pipe, art_dir / "meta_pipeline.joblib")

    base_frozen = {}
    print("retraining tabular base models on full dataset")
    num_cols, cat_cols = infer_feature_types(x)

    for name, est in tab_models.items():
        pre = make_preprocessor_for_model(name, num_cols=num_cols, cat_cols=cat_cols)
        pipe = build_model_pipeline(pre, est)
        pipe.fit(x, y)
        path = art_dir / f"base_{name}_pipeline.joblib"
        dump(pipe, path)
        base_frozen[name] = path.name

    print("retraining sequential base models on full dataset")
    y_full = df2["y_compound_encoded"].astype(int)
    ref_keys = df1_ref[["race_id", "driver_id", "lapno"]].copy()
    ref_x = _get_stage2_reference_x(df1_ref, stage2_feature_cols)
    target_keys = df2[["race_id", "driver_id", "lapno"]].copy()

    for name, model in seq_models.items():
        num_t, cat_t = infer_feature_types(ref_x)
        pre = make_preprocessor_for_model(name, num_cols=num_t, cat_cols=cat_t)
        pre.fit(ref_x)

        xt_ref_all = pre.transform(ref_x)
        if hasattr(xt_ref_all, "toarray"):
            xt_ref_all = xt_ref_all.toarray()
        xt_ref_all = xt_ref_all.astype(np.float32)

        x_seq, y_seq, _, _, _ = build_feature_sequences_from_reference(
            reference_keys=ref_keys,
            X_reference=xt_ref_all,
            target_keys=target_keys,
            y_target=y_full,
            seq_len=8,
            pad_left=True,
            add_timestep_mask=True,
            strict_match=True,
        )

        present = np.unique(y_seq)
        w = compute_class_weight(class_weight="balanced", classes=present, y=y_seq)
        class_w = {int(c): float(wi) for c, wi in zip(present, w)}
        for c in range(n_classes):
            class_w.setdefault(c, 1.0)

        model.fit(x_seq, y_seq, class_weight=class_w)

        dump(pre, art_dir / f"{name}_preprocessor.joblib")
        model.model_.save(art_dir / f"{name}_model.keras")
        base_frozen[name] = {
            "preprocessor": f"{name}_preprocessor.joblib",
            "model": f"{name}_model.keras",
        }

    (art_dir / "feature_columns.json").write_text(json.dumps({"columns": list(x.columns)}, indent=2))
    (art_dir / "meta_features.json").write_text(json.dumps({"columns": list(x_meta_df.columns)}, indent=2))
    (art_dir / "classes.json").write_text(json.dumps({"classes": list(COMPOUND_CLASSES)}, indent=2))

    return {
        "stage": "stage2_multiclass",
        "n_classes": n_classes,
        "tab_models": tab_names,
        "seq_models": seq_names,
        "meta_model": "xgb",
        "meta_uses_context": False,
        "artifacts": {
            "meta_pipeline": "meta_pipeline.joblib",
            "base_pipelines": base_frozen,
            "oof_predictions": "oof_predictions.csv",
        },
    }

def main():
    root = Path("runs") / "final_run"
    root.mkdir(parents=True, exist_ok=True)

    seed = 42
    n_splits = 5
    cfg = ModelConfig(random_state=seed)

    # hardcoded run settings
    only_stage = "all"  # either "all", "stage1", "stage2"
    data_stage1 = "data/processed/dataset1.csv"
    data_stage2 = "data/processed/dataset2.csv"

    manifest = { #create manifest for final model
        "seed": seed,
        "n_splits": n_splits,
        "holdout_race_ids": list(HOLDOUT_RACE_IDS),
    }

    df1 = None

    if only_stage in {"all", "stage1"}:
        df1 = load_stage1_dataset(data_stage1)
        x1, y1 = get_stage1_xy(df1)
        fb1 = make_race_group_folds(df1, n_splits=n_splits, seed=seed)

        print("training final stage 1 models")
        manifest["stage1"] = train_stage1_final(df1, x1, y1, fb1.folds, root, cfg)

    if only_stage in {"all", "stage2"}:
        if df1 is None:
            df1 = load_stage1_dataset(data_stage1)

        df2 = load_stage2_dataset(data_stage2, strict=True)
        df2 = encode_y_compound(df2, col="y_compound", out_col="y_compound_encoded")
        df2 = df2.loc[~df2["y_compound_encoded"].isna()].copy()

        x2, y2 = get_stage2_xy(df2)
        fb2 = make_race_group_folds(df2, n_splits=n_splits, seed=seed)

        print("training final stage 2 models")
        manifest["stage2"] = train_stage2_final(df1, df2, x2, y2, fb2.folds, root, cfg)

    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"done. all artifacts saved to {root}")

if __name__ == "__main__":
    main()