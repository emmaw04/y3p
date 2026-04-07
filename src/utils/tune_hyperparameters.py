"""
single unified script to tune all models using optuna.
combines the logic from tune_8_optuna and tune_optuna into one clean flow.
handles both tabular and sequential models for both stages.
"""

import json
from pathlib import Path
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow import keras
from src.data.data import (
    COMPOUND_CLASSES,
    HOLDOUT_RACE_IDS,
    build_feature_sequences,
    build_feature_sequences_from_reference,
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
    build_stage1_rf,
    build_stage1_xgb,
    build_stage1_svm,
    build_stage1_ann,
    build_stage2_rf,
    build_stage2_xgb,
    build_stage2_svm,
    build_stage2_ffnn,
    build_meta_binary_lr,
    build_meta_binary_mlp,
    build_meta_binary_xgb,
    build_meta_multiclass_lr,
    build_meta_multiclass_mlp,
    build_meta_multiclass_xgb,
    _build_tcn_binary,
    _build_tcn_gru_binary,
    _build_lstm_binary,
    _build_gru_binary,
    _build_vse_hybrid_binary,
    _build_tcn_multiclass,
    _build_tcn_gru_multiclass,
    _build_lstm_multiclass,
    _build_gru_multiclass,
    make_stage1_svm,
    make_stage1_xgb,
    make_tcn_gru_binary,
    make_stage2_svm,
    make_stage2_rf,
    make_stage2_xgb,
    make_lstm_binary,
    make_tcn_gru_multiclass,
)
from sklearn.utils.class_weight import compute_class_weight
from typing import Optional

#hardcoded tuning values
DATA_STAGE1 = "data/processed/dataset1.csv"
DATA_STAGE2 = "data/processed/dataset2.csv"
TASK = "binary" #binary or multiclass
MODEL = "meta_lr" #rf, xgb, svm, ann, tcn, gry, lstm, tcn_gru, hybrid_vse, meta_lr, meta_xgb, meta_mlp
N_TRIALS = 2
OUTDIR = "runs/tuning"
SEED = 42
N_SPLITS = 5

def compute_metrics(y_true: np.ndarray, proba: np.ndarray, is_binary: bool):
    """calculates evaluation metrics"""
    if is_binary:
        proba = np.asarray(proba)

        #gets positive class probability
        if proba.ndim == 2 and proba.shape[1] == 2:
            proba_pos = proba[:, 1]
        elif proba.ndim == 2 and proba.shape[1] == 1:
            proba_pos = proba.reshape(-1)
        else:
            proba_pos = proba.reshape(-1)

        y_pred = (proba_pos >= 0.5).astype(int) #turns probabilities into class predictions
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, proba_pos)) if len(np.unique(y_true)) > 1 else float("nan"),
            "pr_auc": float(average_precision_score(y_true, proba_pos)) if len(np.unique(y_true)) > 1 else float("nan"),
            "logloss": float(log_loss(y_true, proba_pos, labels=[0, 1])),
        }
    else:
        eps = 1e-15
        proba_norm = np.clip(proba, eps, 1.0) #clips probabilities slightly so log loss doesn't break
        proba_norm = proba_norm / proba_norm.sum(axis=1, keepdims=True) #renormalises rows so probabilities sum to 1

        y_pred = np.argmax(proba_norm, axis=1) #gets predicted class with argmax
        labels = np.arange(proba_norm.shape[1])

        y_bin = label_binarize(y_true, classes=labels)

        try:
            roc_auc = float(roc_auc_score(y_bin, proba_norm, average="macro", multi_class="ovr"))
        except ValueError:
            roc_auc = float("nan")

        pr_aucs = []
        for c in range(proba_norm.shape[1]):
            if np.unique(y_bin[:, c]).size > 1:
                pr_aucs.append(average_precision_score(y_bin[:, c], proba_norm[:, c]))
        pr_auc = float(np.mean(pr_aucs)) if pr_aucs else float("nan")

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "logloss": float(log_loss(y_true, proba_norm, labels=labels)),
        }

def mean_metric_dict(metric_dicts: list[dict[str, float]]):
    """
    computes average metrics across folds
    """
    keys = metric_dicts[0].keys()
    return {
        k: float(np.nanmean([m[k] for m in metric_dicts]))
        for k in keys
    }

def suggest_rf_params(trial: optuna.Trial):
    """
    search space for random forest model
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
        "max_depth": trial.suggest_categorical("max_depth", [None, 6, 10, 14, 20]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.8]),
        "bootstrap": True,
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }

def suggest_xgb_params(trial: optuna.Trial, is_binary: bool):
    """
    search space for xgboost model
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.12, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 0.5, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
        "max_bin": trial.suggest_categorical("max_bin", [128, 256]),
    }

def suggest_svm_params(trial: optuna.Trial):
    """
    search space for svm model
    """
    return {
        "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 1e-1, log=True),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }

def suggest_ann_params(trial: optuna.Trial):
    """
    search space for ann model
    """
    return {
        "n_layers": trial.suggest_int("n_layers", 1, 2),
        "hidden_units": trial.suggest_categorical("hidden_units", [32, 64, 128]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "l2": trial.suggest_float("l2", 1e-6, 1e-3, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
    }

def build_cached_folds(x: pd.DataFrame, y: pd.Series, folds: list, pre_name: str):
    """precomputes all tabular preprocessing so trials run much faster"""
    num_cols, cat_cols = infer_feature_types(x)
    base_pre = make_preprocessor_for_model(pre_name, num_cols=num_cols, cat_cols=cat_cols) #fits preprocessor on training data only

    cached = []
    y_np = y.to_numpy()

    #transforms train and validation data
    for tr_idx, va_idx in folds:
        x_tr, x_va = x.iloc[tr_idx], x.iloc[va_idx]
        y_tr, y_va = y_np[tr_idx], y_np[va_idx]

        pre = clone(base_pre)
        pre.fit(x_tr, y_tr)

        xt_tr = pre.transform(x_tr)
        xt_va = pre.transform(x_va)

        cached.append((xt_tr, y_tr, xt_va, y_va, tr_idx, va_idx))

    return cached

def objective_tabular(
    trial: optuna.Trial,
    cached_folds: list,
    model_name: str,
    is_binary: bool,
    n_classes: int,
    seed: int,
):
    """the function that optuna optimises for tabular models"""
    #samples hyperparameters for the chosen model
    if model_name == "rf":
        params = suggest_rf_params(trial)
    elif model_name == "xgb":
        params = suggest_xgb_params(trial, is_binary)
    elif model_name == "svm":
        params = suggest_svm_params(trial)
    elif model_name == "ann":
        params = suggest_ann_params(trial)
        tf.keras.utils.set_random_seed(seed)
    else:
        raise ValueError(f"unsupported tabular model: {model_name}")

    #creates a ModelConfig
    cfg = ModelConfig(random_state=seed, n_jobs=-1, use_class_weight=True)

    losses = []
    fold_metrics = []

    #loops through the folds
    for fold, (xt_tr, y_tr, xt_va, y_va, _, _) in enumerate(cached_folds):
        #builds the correct model using the builder functions from models.py
        if model_name == "rf":
            if is_binary:
                model = build_stage1_rf(cfg, **params)
            else:
                model = build_stage2_rf(cfg, **params)

            model.fit(xt_tr, y_tr) #fits the model on thr training fold
            proba = model.predict_proba(xt_va) #predict probabilities on the validation fold

            if not is_binary:
                proba = _expand_multiclass_proba(proba, model.classes_, n_classes)

        elif model_name == "xgb":
            kwargs = dict(params)

            if is_binary:
                n_pos = int(np.sum(y_tr == 1))
                n_neg = int(np.sum(y_tr == 0))
                kwargs["scale_pos_weight"] = float(n_neg / n_pos) if n_pos > 0 else 1.0
                model = build_stage1_xgb(cfg, **kwargs)
            else:
                kwargs["n_classes"] = n_classes
                model = build_stage2_xgb(cfg, **kwargs)

            model.set_params(early_stopping_rounds=20)
            model.fit(xt_tr, y_tr, eval_set=[(xt_va, y_va)], verbose=False)
            proba = model.predict_proba(xt_va)

            if not is_binary:
                proba = _expand_multiclass_proba(proba, model.classes_, n_classes)

        elif model_name == "svm":
            if is_binary:
                model = build_stage1_svm(cfg, **params)
            else:
                model = build_stage2_svm(cfg, **params)

            model.fit(xt_tr, y_tr)
            proba = model.predict_proba(xt_va)

            if not is_binary:
                proba = _expand_multiclass_proba(proba, model.classes_, n_classes)

        elif model_name == "ann":
            xt_tr_dense = _as_dense(xt_tr)
            xt_va_dense = _as_dense(xt_va)

            if is_binary:
                model = build_stage1_ann(
                    cfg,
                    **params,
                    epochs=60,
                    patience=5,
                )
            else:
                model = build_stage2_ffnn(
                    cfg,
                    n_classes=n_classes,
                    **params,
                    epochs=60,
                    patience=5,
                )

            model.fit(xt_tr_dense, y_tr)
            proba = model.predict_proba(xt_va_dense)
            tf.keras.backend.clear_session()

        #computes metrics as a result of the trial
        m = compute_metrics(y_va, proba, is_binary)
        losses.append(m["logloss"])
        fold_metrics.append(m)

        trial.report(float(np.mean(losses)), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    #averages metrics across folds
    mean_metrics = mean_metric_dict(fold_metrics)
    trial.set_user_attr("cv_metrics", mean_metrics)
    return mean_metrics["logloss"] #returns mean log loss

def suggest_seq_params(trial: optuna.Trial, model_name: str, is_binary: bool):
    """
    search space for sequence models
    """
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 2e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "epochs": 60,
        "patience": 6,
    }

    if model_name in {"tcn", "tcn_gru"}:
        params.update({
            "filters": trial.suggest_categorical("filters", [16, 32, 64]),
            "kernel_size": trial.suggest_categorical("kernel_size", [2, 3, 4]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        })

    if model_name == "tcn":
        params["pooling"] = trial.suggest_categorical("pooling", ["gap", "last"])
        if is_binary:
            params["focal_gamma"] = trial.suggest_categorical("focal_gamma", [1.0, 2.0, 3.0])
            params["focal_alpha"] = trial.suggest_categorical("focal_alpha", [0.5, 0.75])

    if model_name in {"lstm", "gru", "tcn_gru", "hybrid_vse"}:
        params["rnn_units"] = trial.suggest_categorical("rnn_units", [16, 32, 64])
        params["rnn_dropout"] = trial.suggest_float("rnn_dropout", 0.0, 0.4)

    return params

def build_cached_seq_folds(df: pd.DataFrame, x: pd.DataFrame, y: pd.Series, folds: list, seq_len: int):
    """precomputes and caches lap sequences for our sequential models so optuna runs much faster without recalculating every trial
    sequences are for stage 1 models
    """
    num_cols, cat_cols = infer_feature_types(x)
    base_pre = make_preprocessor_for_model("tcn", num_cols=num_cols, cat_cols=cat_cols)

    cached = []
    has_seq_keys = "lapno" in df.columns and "driver_id" in df.columns

    for tr_idx, va_idx in folds:
        pre = clone(base_pre)
        pre.fit(x.iloc[tr_idx], y.iloc[tr_idx])

        xt_all = pre.transform(x)
        if hasattr(xt_all, "toarray"):
            xt_all = xt_all.toarray()
        xt_all = xt_all.astype(np.float32)

        if has_seq_keys:
            keys = df[["race_id", "driver_id", "lapno"]].copy()
            x_seq, y_seq, _, seq_idx, _ = build_feature_sequences(keys, xt_all, y, seq_len=seq_len, pad_left=True, add_timestep_mask=True) #builds lap sequences

            def all_in(s_idx, allowed):
                return np.all((s_idx == -1) | np.isin(s_idx, allowed), axis=1)

            tr_mask = all_in(seq_idx, tr_idx)
            va_mask = all_in(seq_idx, va_idx)

            x_tr, y_tr = x_seq[tr_mask], y_seq[tr_mask]
            x_va, y_va = x_seq[va_mask], y_seq[va_mask]
        else:
            # fallback for tabular-only datasets (like dataset2.csv or smote)
            x_tr, x_va = xt_all[tr_idx], xt_all[va_idx]
            y_tr, y_va = y.iloc[tr_idx].to_numpy(), y.iloc[va_idx].to_numpy()
            x_tr = np.expand_dims(x_tr, axis=1)
            x_va = np.expand_dims(x_va, axis=1)

        cached.append((x_tr.astype(np.float32), y_tr.astype(int), x_va.astype(np.float32), y_va.astype(int), tr_idx, va_idx))

    return cached

def build_cached_stage2_seq_folds_from_reference(
    reference_df: pd.DataFrame,
    target_df: pd.DataFrame,
    x_target: pd.DataFrame,
    y_target: pd.Series,
    folds: list,
    seq_len: int,
):
    """
    builds cached sequence folds for stage 2 sequence models

    sequences come from the full dataset (dataset1)
    but labels and fold membership come from the pit-event target data (dataset2)

    each target row is a pit lap, and each sequence ends on that pit lap
    """
    feature_cols = list(x_target.columns)
    missing = [c for c in feature_cols if c not in reference_df.columns]
    if missing:
        raise ValueError(
            f"reference_df is missing stage 2 feature columns needed for sequence construction: {missing}"
        )

    x_reference = reference_df[feature_cols].copy()

    num_cols, cat_cols = infer_feature_types(x_target)
    base_pre = make_preprocessor_for_model("tcn", num_cols=num_cols, cat_cols=cat_cols)

    reference_keys = reference_df[["race_id", "driver_id", "lapno"]].reset_index(drop=True)
    target_keys = target_df[["race_id", "driver_id", "lapno"]].reset_index(drop=True)

    cached = []

    for tr_idx, va_idx in folds:
        # fit preprocessing on reference rows from the training races only
        train_race_ids = set(target_df.iloc[tr_idx]["race_id"].astype(int).tolist())
        ref_train_idx = np.flatnonzero(reference_df["race_id"].isin(train_race_ids).to_numpy())

        pre = clone(base_pre)
        pre.fit(x_reference.iloc[ref_train_idx])

        xt_reference = pre.transform(x_reference)
        if hasattr(xt_reference, "toarray"):
            xt_reference = xt_reference.toarray()
        xt_reference = xt_reference.astype(np.float32)

        # build all target aligned sequences from the full lap-level reference table
        x_seq_all, y_seq_all, idx_target_all, _, _ = build_feature_sequences_from_reference(
            reference_keys=reference_keys,
            X_reference=xt_reference,
            target_keys=target_keys,
            y_target=y_target,
            seq_len=seq_len,
            pad_left=True,
            add_timestep_mask=True,
            strict_match=True,
        )

        idx_target_all = np.asarray(idx_target_all, dtype=int)

        tr_mask = np.isin(idx_target_all, tr_idx)
        va_mask = np.isin(idx_target_all, va_idx)

        x_tr = x_seq_all[tr_mask]
        y_tr = y_seq_all[tr_mask]
        idx_tr = idx_target_all[tr_mask]

        x_va = x_seq_all[va_mask]
        y_va = y_seq_all[va_mask]
        idx_va = idx_target_all[va_mask]

        # reorder sequences so they match the original target fold order
        tr_pos = {int(idx): pos for pos, idx in enumerate(tr_idx)}
        va_pos = {int(idx): pos for pos, idx in enumerate(va_idx)}

        if len(idx_tr) > 0:
            tr_order = np.argsort([tr_pos[int(i)] for i in idx_tr])
            x_tr = x_tr[tr_order]
            y_tr = y_tr[tr_order]

        if len(idx_va) > 0:
            va_order = np.argsort([va_pos[int(i)] for i in idx_va])
            x_va = x_va[va_order]
            y_va = y_va[va_order]

        cached.append((
            x_tr.astype(np.float32),
            y_tr.astype(int),
            x_va.astype(np.float32),
            y_va.astype(int),
            tr_idx,
            va_idx,
        ))

    return cached

def _as_dense(x):
    """
    converts sparse matrixes to dense arrays
    """
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def _binary_pos_col(proba: np.ndarray):
    """
    standardises binary probabilities to a single positive class column
    """
    proba = np.asarray(proba)
    if proba.ndim == 1:
        return proba.reshape(-1, 1)
    if proba.ndim == 2 and proba.shape[1] == 2:
        return proba[:, [1]]
    if proba.ndim == 2 and proba.shape[1] == 1:
        return proba
    raise ValueError(f"unexpected binary proba shape: {proba.shape}")


def _expand_multiclass_proba(proba: np.ndarray, classes_: np.ndarray, n_classes: int):
    """
    in the case of the compound class, expands probability outputs back to the full class set if a model was trained on a fold with missing classes (the wet compound)
    """
    proba = np.asarray(proba)
    full = np.zeros((proba.shape[0], n_classes), dtype=float)
    for j, cls in enumerate(classes_):
        full[:, int(cls)] = proba[:, j]
    return full

def suggest_meta_lr_params(trial: optuna.Trial, is_binary: bool):
    """
    search space for logistic regression meta learner
    """
    return {
        "C": trial.suggest_float("C", 1e-3, 50.0, log=True),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        "max_iter": 6000 if not is_binary else 4000,
    }


def suggest_meta_mlp_params(trial: optuna.Trial):
    """
    search space for MLP meta learner
    """
    return {
        "hidden_layer_sizes": trial.suggest_categorical(
            "hidden_layer_sizes",
            [(32,), (64,), (128,), (64, 32), (128, 64)]
        ),
        "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 3e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
    }


def suggest_meta_xgb_params(trial: optuna.Trial, is_binary: bool):
    """
    search space for xgboost meta learner
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }

def build_meta_folds(
    df: pd.DataFrame,
    x: pd.DataFrame,
    y: pd.Series,
    folds: list,
    is_binary: bool,
    n_classes: int,
    seed: int,
    reference_df: Optional[pd.DataFrame] = None):
    """
    builds the training data for stacked models using the OOF probabilities given by the base learners

    Binary stage 1 bases: xgb, svm, tcn_gru, lstm
    Multiclass stage 2 bases: xgb, svm, rf, tcn_gru
    base learners selected through individual performance and how much the meta learner relied on their probabilities which was found out through ablation
    """
    cfg = ModelConfig(random_state=seed, n_jobs=-1, use_class_weight=True)
    fold_val_sets: list[tuple[np.ndarray, np.ndarray]] = []

    if is_binary:
        svm_cached = build_cached_folds(x, y, folds, "svm")
        xgb_cached = build_cached_folds(x, y, folds, "xgb")
        seq_cached = build_cached_seq_folds(df, x, y, folds, seq_len=8)

        for fold_idx in range(len(folds)):
            cols = []

            # svm
            xt_tr, y_tr, xt_va, y_va, _, _ = svm_cached[fold_idx]
            model = make_stage1_svm(cfg)
            model.fit(xt_tr, y_tr)
            cols.append(_binary_pos_col(model.predict_proba(xt_va)))

            # xgb
            xt_tr, y_tr, xt_va, y_va2, _, _ = xgb_cached[fold_idx]
            if not np.array_equal(y_va, y_va2):
                raise ValueError("binary meta fold misalignment between svm and xgb cached folds")
            model = make_stage1_xgb(cfg)
            model.fit(xt_tr, y_tr)
            cols.append(_binary_pos_col(model.predict_proba(xt_va)))

            # sequence models: tcn_gru + lstm
            x_tr_seq, y_tr_seq, x_va_seq, y_va_seq, _, _ = seq_cached[fold_idx]
            if len(y_va_seq) != len(y_va):
                raise ValueError("binary meta fold misalignment between tabular and sequence folds")

            model = make_tcn_gru_binary(cfg)
            model.fit(x_tr_seq, y_tr_seq)
            cols.append(_binary_pos_col(model.predict_proba(x_va_seq)))
            tf.keras.backend.clear_session()

            model = make_lstm_binary(cfg)
            model.fit(x_tr_seq, y_tr_seq)
            cols.append(_binary_pos_col(model.predict_proba(x_va_seq)))
            tf.keras.backend.clear_session()

            z_va = np.hstack(cols)
            fold_val_sets.append((z_va, y_va))

    else:
        svm_cached = build_cached_folds(x, y, folds, "svm")
        rf_cached = build_cached_folds(x, y, folds, "rf")
        xgb_cached = build_cached_folds(x, y, folds, "xgb")
        seq_cached = build_cached_stage2_seq_folds_from_reference(reference_df=reference_df, target_df=df, x_target=x, y_target=y, folds=folds, seq_len=8)

        for fold_idx in range(len(folds)):
            cols = []

            # svm
            xt_tr, y_tr, xt_va, y_va, _, _ = svm_cached[fold_idx]
            model = make_stage2_svm(cfg)
            model.fit(xt_tr, y_tr)
            cols.append(_expand_multiclass_proba(model.predict_proba(xt_va), model.classes_, n_classes))

            # rf
            xt_tr, y_tr, xt_va, y_va2, _, _ = rf_cached[fold_idx]
            if not np.array_equal(y_va, y_va2):
                raise ValueError("multiclass meta fold misalignment between svm and rf cached folds")
            model = make_stage2_rf(cfg)
            model.fit(xt_tr, y_tr)
            cols.append(_expand_multiclass_proba(model.predict_proba(xt_va), model.classes_, n_classes))

            # xgb
            xt_tr, y_tr, xt_va, y_va3, _, _ = xgb_cached[fold_idx]
            if not np.array_equal(y_va, y_va3):
                raise ValueError("multiclass meta fold misalignment between rf and xgb cached folds")
            model = make_stage2_xgb(cfg, n_classes=n_classes)
            model.fit(xt_tr, y_tr)
            cols.append(_expand_multiclass_proba(model.predict_proba(xt_va), model.classes_, n_classes))

            #tcn_gru
            x_tr_seq, y_tr_seq, x_va_seq, y_va_seq, _, _ = seq_cached[fold_idx]
            if len(y_va_seq) != len(y_va):
                raise ValueError("multiclass meta fold misalignment between tabular and sequence folds")

            model = make_tcn_gru_multiclass(cfg, n_classes=n_classes)
            model.fit(x_tr_seq, y_tr_seq)
            cols.append(_expand_multiclass_proba(model.predict_proba(x_va_seq), model.classes_, n_classes))
            tf.keras.backend.clear_session()

            z_va = np.hstack(cols)
            fold_val_sets.append((z_va, y_va))

    meta_folds = []
    for i in range(len(fold_val_sets)):
        x_va, y_va = fold_val_sets[i]
        x_tr = np.vstack([fold_val_sets[j][0] for j in range(len(fold_val_sets)) if j != i])
        y_tr = np.concatenate([fold_val_sets[j][1] for j in range(len(fold_val_sets)) if j != i])
        meta_folds.append((x_tr, y_tr, x_va, y_va))

    return meta_folds


def objective_meta(
    trial: optuna.Trial,
    meta_folds: list,
    model_name: str,
    is_binary: bool,
    n_classes: int,
    seed: int,
):
    """
    the function that optuna optimises for meta models
    """

    #sample meta model hyperparameters
    if model_name == "meta_lr":
        params = suggest_meta_lr_params(trial, is_binary)
    elif model_name == "meta_mlp":
        params = suggest_meta_mlp_params(trial)
    elif model_name == "meta_xgb":
        params = suggest_meta_xgb_params(trial, is_binary)
    else:
        raise ValueError(f"unsupported meta model: {model_name}")

    cfg = ModelConfig(random_state=seed, n_jobs=-1, use_class_weight=True)

    losses = []
    fold_metrics = []

    #loops through the folds, build the meta learner, fit on OOF base learner probabilities, make predictions, and compute metrics
    for fold, (x_tr, y_tr, x_va, y_va) in enumerate(meta_folds):
        if model_name == "meta_lr":
            if is_binary:
                model = build_meta_binary_lr(cfg, **params)
            else:
                model = build_meta_multiclass_lr(cfg, **params)

        elif model_name == "meta_mlp":
            if is_binary:
                model = build_meta_binary_mlp(cfg, **params)
            else:
                model = build_meta_multiclass_mlp(cfg, **params)

        elif model_name == "meta_xgb":
            if is_binary:
                model = build_meta_binary_xgb(cfg, **params)
            else:
                model = build_meta_multiclass_xgb(cfg, n_classes=n_classes, **params)

        model.fit(x_tr, y_tr)
        proba = model.predict_proba(x_va)

        if not is_binary:
            proba = _expand_multiclass_proba(proba, model.classes_, n_classes)

        m = compute_metrics(y_va, proba, is_binary)
        losses.append(m["logloss"])
        fold_metrics.append(m)

        trial.report(float(np.mean(losses)), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    #compute average metrics
    mean_metrics = mean_metric_dict(fold_metrics)
    trial.set_user_attr("cv_metrics", mean_metrics)
    return mean_metrics["logloss"]

def objective_seq(trial: optuna.Trial, cached_folds: list, model_name: str, is_binary: bool, n_classes: int, seed: int):
    """
    the function that optuna optimises for sequence models
    """
    #sample sequence model hyperparameters
    params = suggest_seq_params(trial, model_name, is_binary)
    tf.keras.utils.set_random_seed(seed)

    losses = []
    fold_metrics = []

    #for each fold
    for fold, (x_tr, y_tr, x_va, y_va, _, _) in enumerate(cached_folds):
        #remove sequences that don't have a valid class label
        if not is_binary:
            valid_tr = y_tr >= 0
            valid_va = y_va >= 0
            x_tr, y_tr = x_tr[valid_tr], y_tr[valid_tr]
            x_va, y_va = x_va[valid_va], y_va[valid_va]

        seq_len = x_tr.shape[1]
        n_features = x_tr.shape[2]

        # safely extract args to pass to model builder
        builder_kwargs = {k: v for k, v in params.items() if k not in ["batch_size", "epochs", "patience"]}

        #build sequence model
        if is_binary:
            if model_name == "tcn":
                model = _build_tcn_binary(seq_len, n_features, **builder_kwargs)
            elif model_name == "tcn_gru":
                model = _build_tcn_gru_binary(seq_len, n_features, **builder_kwargs)
            elif model_name == "lstm":
                model = _build_lstm_binary(seq_len, n_features, **builder_kwargs)
            elif model_name == "gru":
                model = _build_gru_binary(seq_len, n_features, **builder_kwargs)
            elif model_name == "hybrid_vse":
                model = _build_vse_hybrid_binary(seq_len, n_features, **builder_kwargs)

            n_pos = int(np.sum(y_tr == 1))
            n_neg = int(np.sum(y_tr == 0))
            pos_w = min(20.0, float(n_neg / n_pos)) if n_pos > 0 else 1.0 #cap positive class weight at 20
            class_w = {0: 1.0, 1: pos_w}

        else:
            if model_name == "tcn":
                model = _build_tcn_multiclass(seq_len, n_features, n_classes, **builder_kwargs)
            elif model_name == "tcn_gru":
                model = _build_tcn_gru_multiclass(seq_len, n_features, n_classes, **builder_kwargs)
            elif model_name == "lstm":
                model = _build_lstm_multiclass(seq_len, n_features, n_classes, **builder_kwargs)
            elif model_name == "gru":
                model = _build_gru_multiclass(seq_len, n_features, n_classes, **builder_kwargs)

            present = np.unique(y_tr)
            w = compute_class_weight("balanced", classes=present, y=y_tr)
            class_w = {int(c): float(wi) for c, wi in zip(present, w)}
            for c in range(n_classes):
                class_w.setdefault(c, 1.0)

        #train with early stopping
        cb = keras.callbacks.EarlyStopping(monitor="val_pr_auc" if is_binary else "val_acc", mode="max", patience=params["patience"], restore_best_weights=True)
        model.fit(x_tr, y_tr, validation_data=(x_va, y_va), epochs=params["epochs"], batch_size=params["batch_size"], class_weight=class_w, verbose=0, callbacks=[cb])

        proba = model.predict(x_va, batch_size=params["batch_size"], verbose=0)
        m = compute_metrics(y_va, proba, is_binary)
        losses.append(m["logloss"])
        fold_metrics.append(m)

        #reports mean loss so far to optuna and prunes bad trials early
        trial.report(float(np.mean(losses)), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

        tf.keras.backend.clear_session()

    #return mean log loss
    mean_metrics = mean_metric_dict(fold_metrics)
    trial.set_user_attr("cv_metrics", mean_metrics)
    return mean_metrics["logloss"]

def filter_stage2_to_reference(
    target_df: pd.DataFrame,
    reference_df: pd.DataFrame,
):
    """
    keep only stage 2 pit-event rows whose (race_id, driver_id, lapno)
    exist in the lap-level reference dataframe
    """
    valid_keys = reference_df[["race_id", "driver_id", "lapno"]].drop_duplicates()

    before = len(target_df)

    filtered = target_df.merge(
        valid_keys,
        on=["race_id", "driver_id", "lapno"],
        how="inner",
    ).copy()

    after = len(filtered)
    dropped = before - after

    if dropped > 0:
        print(f"dropped {dropped} stage 2 rows not present in reference_df")

    return filtered

def main():
    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    seq_models = {"tcn", "gru", "lstm", "tcn_gru", "hybrid_vse"}
    meta_models = {"meta_lr", "meta_xgb", "meta_mlp"}

    valid_tasks = {"binary", "multiclass"}
    valid_models = {"rf", "xgb", "svm", "ann","tcn", "gru", "lstm", "tcn_gru", "hybrid_vse","meta_lr", "meta_xgb", "meta_mlp",}

    if TASK not in valid_tasks:
        raise ValueError(f"invalid TASK={TASK!r}; must be one of {sorted(valid_tasks)}")
    if MODEL not in valid_models:
        raise ValueError(f"invalid MODEL={MODEL!r}; must be one of {sorted(valid_models)}")

    is_seq = MODEL in seq_models
    is_meta = MODEL in meta_models
    is_binary = TASK == "binary"

    if is_binary:

        df = load_stage1_dataset(DATA_STAGE1)
        df = df.loc[~df["race_id"].isin(HOLDOUT_RACE_IDS)].copy()
        x, y = get_stage1_xy(df)
        fb = make_race_group_folds(df, group_col="race_id", n_splits=N_SPLITS, seed=SEED)
        n_classes = 2

    else:  # note hybrid_vse isn't a multiclass model

        reference_df = None

        # stage 2 targets always come from dataset2
        df = load_stage2_dataset(DATA_STAGE2, strict=True).reset_index(drop=True)

        # stage 2 sequence models and stage 2 stacked models with sequential bases also need lap-level reference histories from dataset1
        if is_seq or is_meta:
            reference_df = load_stage1_dataset(DATA_STAGE1)
            reference_df = reference_df.sort_values(
                ["race_id", "driver_id", "lapno"],
                kind="mergesort",
            ).reset_index(drop=True)

            # temporary workaround: drop stage 2 rows missing from reference NOT IDEAL
            df = filter_stage2_to_reference(df, reference_df).reset_index(drop=True)

        x, y = get_stage2_xy(df)
        fb = make_race_group_folds(df, n_splits=N_SPLITS, seed=SEED)
        n_classes = len(COMPOUND_CLASSES)

    sampler = TPESampler(seed=SEED)
    pruner = MedianPruner(n_startup_trials=10)
    study_name = f"{TASK}_{MODEL}"

    #create optuna study, optimise by minimising mean logloss, use TPE sampling, and use median pruning
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
    )

    if is_meta:
        #if the model is a meta leaner build folds with oof probabilities then create the study
        meta_folds = build_meta_folds(df, x, y, fb.folds, is_binary, n_classes, SEED, None if is_binary else reference_df)
        study.optimize(
            lambda t: objective_meta(t, meta_folds, MODEL, is_binary, n_classes, SEED),
            n_trials=N_TRIALS,
            show_progress_bar=True,
        )

    elif not is_seq:
        #if the model is tabular, build folds then create the study
        pre_name = MODEL
        cached_folds = build_cached_folds(x, y, fb.folds, pre_name)
        study.optimize(
            lambda t: objective_tabular(t, cached_folds, MODEL, is_binary, n_classes, SEED),
            n_trials=N_TRIALS,
            show_progress_bar=True,
        )

    else:
        seq_len = 8
        #if the model is sequential build the folds with lap sequences, then create the study
        if is_binary:
            cached_folds = build_cached_seq_folds(df, x, y, fb.folds, seq_len)
        else:
            cached_folds = build_cached_stage2_seq_folds_from_reference(
                reference_df=reference_df,
                target_df=df,
                x_target=x,
                y_target=y,
                folds=fb.folds,
                seq_len=seq_len,
            )

        study.optimize(
            lambda t: objective_seq(t, cached_folds, MODEL, is_binary, n_classes, SEED),
            n_trials=N_TRIALS,
            show_progress_bar=True,
        )

    #write the best results to file
    best_obj = {
        "study": study_name,
        "best_logloss": float(study.best_value),
        "best_metrics": study.best_trial.user_attrs.get("cv_metrics", {}),
        "best_params": study.best_params,
        "n_trials": len(study.trials),
    }

    (outdir / f"{study_name}_best.json").write_text(json.dumps(best_obj, indent=2))
    print(json.dumps(best_obj, indent=2))

if __name__ == "__main__":
    main()