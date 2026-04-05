"""
single unified script to tune all models using optuna.
combines the logic from tune_8_optuna and tune_optuna into one clean flow.
handles both tabular and sequential models for both stages.
"""

import json
from pathlib import Path
from typing import Any
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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
from src.data.preprocessing import make_preprocessor_for_model
from src.models.models import (
    _build_gru_binary,
    _build_gru_multiclass,
    _build_lstm_binary,
    _build_lstm_multiclass,
    _build_tcn_binary,
    _build_tcn_gru_binary,
    _build_tcn_gru_multiclass,
    _build_tcn_multiclass,
    _build_vse_hybrid_binary,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from src.models.models import (
    make_stage1_svm,
    make_stage1_xgb,
    make_tcn_binary,
    make_tcn_gru_binary,
    make_stage2_svm,
    make_stage2_rf,
    make_stage2_xgb,
    make_stage2_ffnn,
    _build_gru_binary,
    _build_gru_multiclass,
    _build_lstm_binary,
    _build_lstm_multiclass,
    _build_tcn_binary,
    _build_tcn_gru_binary,
    _build_tcn_gru_multiclass,
    _build_tcn_multiclass,
    _build_vse_hybrid_binary,
)

#hardcoded tuning values
DATA_STAGE1 = "../../data/processed/dataset1.csv"
DATA_STAGE2 = "../../data/processed/dataset2.csv"
TASK = "binary" #binary or multiclass
MODEL = "xgb" #rf, xgb, svm, ann, tcn, gry, lstm, tcn_gru, hybrid_vse, meta_lr, meta_xgb, meta_mlp
N_TRIALS = 100
OUTDIR = "runs/tuning"
SEED = 42
N_SPLITS = 5

def compute_metrics(y_true: np.ndarray, proba: np.ndarray, is_binary: bool) -> dict[str, float]:
    """unified metric computation for both binary and multiclass tasks"""
    if is_binary:
        proba = np.asarray(proba)

        if proba.ndim == 2 and proba.shape[1] == 2:
            proba_pos = proba[:, 1]
        elif proba.ndim == 2 and proba.shape[1] == 1:
            proba_pos = proba.reshape(-1)
        else:
            proba_pos = proba.reshape(-1)

        y_pred = (proba_pos >= 0.5).astype(int)
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
        proba_norm = np.clip(proba, eps, 1.0)
        proba_norm = proba_norm / proba_norm.sum(axis=1, keepdims=True)

        y_pred = np.argmax(proba_norm, axis=1)
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

def mean_metric_dict(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
    keys = metric_dicts[0].keys()
    return {
        k: float(np.nanmean([m[k] for m in metric_dicts]))
        for k in keys
    }

def suggest_rf_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
        "max_depth": trial.suggest_categorical("max_depth", [None, 6, 10, 14, 20]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.8]),
        "bootstrap": True,
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        "n_jobs": -1,
    }

def suggest_xgb_params(trial: optuna.Trial, is_binary: bool) -> dict[str, Any]:
    params = {
        "tree_method": "hist",
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
    if is_binary:
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "logloss"
    else:
        params["objective"] = "multi:softprob"
        params["eval_metric"] = "mlogloss"

    return params


def suggest_svm_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 1e-1, log=True),
        "kernel": "rbf",
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }

def suggest_ann_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_layers": trial.suggest_int("n_layers", 1, 2),
        "hidden_units": trial.suggest_categorical("hidden_units", [32, 64, 128]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "l2": trial.suggest_float("l2", 1e-6, 1e-3, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
    }

def build_ann_model(params: dict, input_dim: int, n_classes: int) -> keras.Model:
    reg = keras.regularizers.l2(params["l2"])
    x_in = layers.Input(shape=(input_dim,))
    x = x_in
    
    for _ in range(params["n_layers"]):
        x = layers.Dense(params["hidden_units"], activation="relu", kernel_regularizer=reg)(x)
        if params["dropout"] > 0:
            x = layers.Dropout(params["dropout"])(x)
            
    y_out = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(x_in, y_out)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    return model

def build_cached_folds(x: pd.DataFrame, y: pd.Series, folds: list, pre_name: str) -> list:
    """precomputes all tabular preprocessing so trials run much faster"""
    num_cols, cat_cols = infer_feature_types(x)
    base_pre = make_preprocessor_for_model(pre_name, num_cols=num_cols, cat_cols=cat_cols)

    cached = []
    y_np = y.to_numpy()

    for tr_idx, va_idx in folds:
        x_tr, x_va = x.iloc[tr_idx], x.iloc[va_idx]
        y_tr, y_va = y_np[tr_idx], y_np[va_idx]

        pre = clone(base_pre)
        pre.fit(x_tr, y_tr)

        xt_tr = pre.transform(x_tr)
        xt_va = pre.transform(x_va)

        cached.append((xt_tr, y_tr, xt_va, y_va, tr_idx, va_idx))

    return cached

def objective_tabular(trial: optuna.Trial, cached_folds: list, model_name: str, is_binary: bool, n_classes: int, seed: int) -> float:
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

    losses = []
    fold_metrics = []

    for fold, (xt_tr, y_tr, xt_va, y_va, _, _) in enumerate(cached_folds):
        if model_name == "rf":
            model = RandomForestClassifier(**params, random_state=seed)
            model.fit(xt_tr, y_tr)
            proba = model.predict_proba(xt_va)
            
            if not is_binary:
                proba_full = np.zeros((len(y_va), n_classes), dtype=float)
                for j, cls in enumerate(model.classes_):
                    proba_full[:, int(cls)] = proba[:, j]
                proba = proba_full
                
        elif model_name == "xgb":
            kwargs = dict(params)
            if is_binary:
                n_pos = int(np.sum(y_tr == 1))
                n_neg = int(np.sum(y_tr == 0))
                spw = float(n_neg / n_pos) if n_pos > 0 else 1.0
                kwargs["scale_pos_weight"] = spw
            else:
                kwargs["num_class"] = n_classes
                
            model = XGBClassifier(**kwargs, random_state=seed, early_stopping_rounds=20)
            model.fit(xt_tr, y_tr, eval_set=[(xt_va, y_va)], verbose=False)
            proba = model.predict_proba(xt_va)
            
            if not is_binary:
                proba_full = np.zeros((len(y_va), n_classes), dtype=float)
                for j, cls in enumerate(model.classes_):
                    proba_full[:, int(cls)] = proba[:, j]
                proba = proba_full
                
        elif model_name == "svm":
            base = SVC(**params, probability=False, random_state=seed)
            model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
            model.fit(xt_tr, y_tr)
            proba = model.predict_proba(xt_va)
            
            if not is_binary:
                proba_full = np.zeros((len(y_va), n_classes), dtype=float)
                for j, cls in enumerate(model.classes_):
                    proba_full[:, int(cls)] = proba[:, j]
                proba = proba_full
                
        elif model_name == "ann":
            xt_tr_dense = xt_tr.toarray() if hasattr(xt_tr, "toarray") else np.asarray(xt_tr)
            xt_va_dense = xt_va.toarray() if hasattr(xt_va, "toarray") else np.asarray(xt_va)
            
            model = build_ann_model(params, xt_tr_dense.shape[1], n_classes)
            cb = keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=5, restore_best_weights=True)
            model.fit(xt_tr_dense, y_tr, validation_data=(xt_va_dense, y_va), epochs=60, batch_size=params["batch_size"], verbose=0, callbacks=[cb])
            proba = model.predict(xt_va_dense, batch_size=params["batch_size"], verbose=0)
            tf.keras.backend.clear_session()

        m = compute_metrics(y_va, proba, is_binary)
        losses.append(m["logloss"])
        fold_metrics.append(m)

        trial.report(float(np.mean(losses)), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    mean_metrics = mean_metric_dict(fold_metrics)
    trial.set_user_attr("cv_metrics", mean_metrics)
    return mean_metrics["logloss"]

def suggest_seq_params(trial: optuna.Trial, model_name: str, is_binary: bool) -> dict[str, Any]:
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

def build_cached_seq_folds(df: pd.DataFrame, x: pd.DataFrame, y: pd.Series, folds: list, seq_len: int) -> list:
    """precomputes and caches sequence windows so optuna runs much faster without recalculating every trial"""
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
            x_seq, y_seq, _, seq_idx, _ = build_feature_sequences(keys, xt_all, y, seq_len=seq_len, pad_left=True, add_timestep_mask=True)
            
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

def _as_dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def _binary_pos_col(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba)
    if proba.ndim == 1:
        return proba.reshape(-1, 1)
    if proba.ndim == 2 and proba.shape[1] == 2:
        return proba[:, [1]]
    if proba.ndim == 2 and proba.shape[1] == 1:
        return proba
    raise ValueError(f"unexpected binary proba shape: {proba.shape}")


def _expand_multiclass_proba(proba: np.ndarray, classes_: np.ndarray, n_classes: int) -> np.ndarray:
    proba = np.asarray(proba)
    full = np.zeros((proba.shape[0], n_classes), dtype=float)
    for j, cls in enumerate(classes_):
        full[:, int(cls)] = proba[:, j]
    return full


def suggest_meta_lr_params(trial: optuna.Trial, is_binary: bool) -> dict[str, Any]:
    params = {
        "C": trial.suggest_float("C", 1e-3, 50.0, log=True),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        "solver": "lbfgs",
        "max_iter": 6000 if not is_binary else 4000,
    }
    if not is_binary:
        params["multi_class"] = "multinomial"
    return params


def suggest_meta_mlp_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "hidden_layer_sizes": trial.suggest_categorical(
            "hidden_layer_sizes",
            [(32,), (64,), (128,), (64, 32), (128, 64)]
        ),
        "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 3e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
    }


def suggest_meta_xgb_params(trial: optuna.Trial, is_binary: bool) -> dict[str, Any]:
    params = {
        "tree_method": "hist",
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
    if is_binary:
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "logloss"
    else:
        params["objective"] = "multi:softprob"
        params["eval_metric"] = "mlogloss"
    return params


def build_meta_folds(
    df: pd.DataFrame,
    x: pd.DataFrame,
    y: pd.Series,
    folds: list,
    *,
    is_binary: bool,
    n_classes: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Build OOF stacked features for the meta learner.

    Binary stage 1 bases: tcn, tcn_gru, svm, xgb
    Multiclass stage 2 bases: svm, rf, xgb, ann
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

            # tcn + tcn_gru
            x_tr_seq, y_tr_seq, x_va_seq, y_va_seq, _, _ = seq_cached[fold_idx]
            if len(y_va_seq) != len(y_va):
                raise ValueError("binary meta fold misalignment between tabular and sequence folds")

            model = make_tcn_binary(cfg)
            model.fit(x_tr_seq, y_tr_seq)
            cols.append(_binary_pos_col(model.predict_proba(x_va_seq)))
            tf.keras.backend.clear_session()

            model = make_tcn_gru_binary(cfg)
            model.fit(x_tr_seq, y_tr_seq)
            cols.append(_binary_pos_col(model.predict_proba(x_va_seq)))
            tf.keras.backend.clear_session()

            z_va = np.hstack(cols)
            fold_val_sets.append((z_va, y_va))

    else:
        svm_cached = build_cached_folds(x, y, folds, "svm")
        rf_cached = build_cached_folds(x, y, folds, "rf")
        xgb_cached = build_cached_folds(x, y, folds, "xgb")
        ann_cached = build_cached_folds(x, y, folds, "ann")

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

            # ann
            xt_tr, y_tr, xt_va, y_va4, _, _ = ann_cached[fold_idx]
            if not np.array_equal(y_va, y_va4):
                raise ValueError("multiclass meta fold misalignment between xgb and ann cached folds")
            xt_tr = _as_dense(xt_tr)
            xt_va = _as_dense(xt_va)
            model = make_stage2_ffnn(cfg, n_classes=n_classes)
            model.fit(xt_tr, y_tr)
            cols.append(_expand_multiclass_proba(model.predict_proba(xt_va), model.classes_, n_classes))
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
) -> float:
    if model_name == "meta_lr":
        params = suggest_meta_lr_params(trial, is_binary)
    elif model_name == "meta_mlp":
        params = suggest_meta_mlp_params(trial)
    elif model_name == "meta_xgb":
        params = suggest_meta_xgb_params(trial, is_binary)
    else:
        raise ValueError(f"unsupported meta model: {model_name}")

    losses = []
    fold_metrics = []

    for fold, (x_tr, y_tr, x_va, y_va) in enumerate(meta_folds):
        if model_name == "meta_lr":
            model = LogisticRegression(**params, random_state=seed)

        elif model_name == "meta_mlp":
            model = MLPClassifier(
                hidden_layer_sizes=params["hidden_layer_sizes"],
                alpha=params["alpha"],
                learning_rate_init=params["learning_rate_init"],
                batch_size=params["batch_size"],
                activation="relu",
                max_iter=300,
                early_stopping=True,
                random_state=seed,
            )

        elif model_name == "meta_xgb":
            kwargs = dict(params)
            if not is_binary:
                kwargs["num_class"] = n_classes
            model = XGBClassifier(**kwargs, n_jobs=-1, random_state=seed)

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

    mean_metrics = mean_metric_dict(fold_metrics)
    trial.set_user_attr("cv_metrics", mean_metrics)
    return mean_metrics["logloss"]

def objective_seq(trial: optuna.Trial, cached_folds: list, model_name: str, is_binary: bool, n_classes: int, seed: int) -> float:
    params = suggest_seq_params(trial, model_name, is_binary)
    tf.keras.utils.set_random_seed(seed)
    
    losses = []
    fold_metrics = []
    
    for fold, (x_tr, y_tr, x_va, y_va, _, _) in enumerate(cached_folds):
        if not is_binary:
            valid_tr = y_tr >= 0
            valid_va = y_va >= 0
            x_tr, y_tr = x_tr[valid_tr], y_tr[valid_tr]
            x_va, y_va = x_va[valid_va], y_va[valid_va]
            
        if len(y_tr) == 0 or len(y_va) == 0:
            m = {
                "accuracy": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "roc_auc": float("nan"),
                "pr_auc": float("nan"),
                "logloss": 1.0,
            }
            losses.append(m["logloss"])
            fold_metrics.append(m)
            continue
            
        seq_len = x_tr.shape[1]
        n_features = x_tr.shape[2]
        
        # safely extract args to pass to model builder
        builder_kwargs = {k: v for k, v in params.items() if k not in ["batch_size", "epochs", "patience"]}
        
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
            pos_w = min(20.0, float(n_neg / n_pos)) if n_pos > 0 else 1.0
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
            
            from sklearn.utils.class_weight import compute_class_weight
            present = np.unique(y_tr)
            w = compute_class_weight("balanced", classes=present, y=y_tr)
            class_w = {int(c): float(wi) for c, wi in zip(present, w)}
            for c in range(n_classes):
                class_w.setdefault(c, 1.0)

        cb = keras.callbacks.EarlyStopping(monitor="val_pr_auc" if is_binary else "val_acc", mode="max", patience=params["patience"], restore_best_weights=True)
        model.fit(x_tr, y_tr, validation_data=(x_va, y_va), epochs=params["epochs"], batch_size=params["batch_size"], class_weight=class_w, verbose=0, callbacks=[cb])
        
        proba = model.predict(x_va, batch_size=params["batch_size"], verbose=0)
        m = compute_metrics(y_va, proba, is_binary)
        losses.append(m["logloss"])
        fold_metrics.append(m)
        
        trial.report(float(np.mean(losses)), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
            
        tf.keras.backend.clear_session()
        
    mean_metrics = mean_metric_dict(fold_metrics)
    trial.set_user_attr("cv_metrics", mean_metrics)
    return mean_metrics["logloss"]

def main() -> None:
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
        if not DATA_STAGE1:
            raise ValueError("DATA_STAGE1 must be set for binary tuning")

        df = load_stage1_dataset(DATA_STAGE1)
        df = df.loc[~df["race_id"].isin(HOLDOUT_RACE_IDS)].copy()
        x, y = get_stage1_xy(df)
        fb = make_race_group_folds(df, target_col="y_pit", n_splits=N_SPLITS, seed=SEED)
        n_classes = 2

    else:
        if MODEL == "hybrid_vse":
            raise ValueError("hybrid_vse is only valid for binary stage 1")

        if DATA_STAGE1 and is_seq:
            # build stage 2 sequence data directly from stage 1 lap-level data
            df = load_stage1_dataset(DATA_STAGE1)
            df = df.loc[~df["race_id"].isin(HOLDOUT_RACE_IDS)].copy()

            df = df.sort_values(["race_id", "driver_id", "lapno"], kind="mergesort").reset_index(drop=True)

            next_comp = df.groupby(["race_id", "driver_id"], sort=False)["current_compound"].shift(-1)
            df["y_compound"] = np.where(df["y_pit"].astype(int).to_numpy() == 1, next_comp, np.nan)
            df = encode_y_compound(df, col="y_compound", out_col="y_compound_encoded")

            x = df.drop(
                columns=["y_pit", "y_compound", "y_compound_encoded", "race_id", "driver_id"],
                errors="ignore",
            ).copy()
            y = df["y_compound_encoded"].fillna(-1).astype(int)

            fb = make_race_group_folds(
                df,
                target_col="y_compound_encoded",
                n_splits=N_SPLITS,
                seed=SEED,
            )

        elif DATA_STAGE2:
            # normal stage 2 tabular data, or fallback for tuning sequence models on SMOTE data
            df = load_stage2_dataset(DATA_STAGE2, strict=True)
            df = df.loc[~df["race_id"].isin(HOLDOUT_RACE_IDS)].copy()
            df = encode_y_compound(df, col="y_compound", out_col="y_compound_encoded")
            x, y = get_stage2_xy(df)
            fb = make_race_group_folds(
                df,
                target_col="y_compound_encoded",
                n_splits=N_SPLITS,
                seed=SEED,
            )
        else:
            raise ValueError("for multiclass tuning, set DATA_STAGE1 or DATA_STAGE2")

        n_classes = len(COMPOUND_CLASSES)

    sampler = TPESampler(seed=SEED)
    pruner = MedianPruner(n_startup_trials=10)
    study_name = f"{TASK}_{MODEL}"

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
    )

    if is_meta:
        meta_folds = build_meta_folds(
            df,
            x,
            y,
            fb.folds,
            is_binary=is_binary,
            n_classes=n_classes,
            seed=SEED,
        )
        study.optimize(
            lambda t: objective_meta(t, meta_folds, MODEL, is_binary, n_classes, SEED),
            n_trials=N_TRIALS,
            show_progress_bar=True,
        )

    elif not is_seq:
        pre_name = MODEL
        cached_folds = build_cached_folds(x, y, fb.folds, pre_name)
        study.optimize(
            lambda t: objective_tabular(t, cached_folds, MODEL, is_binary, n_classes, SEED),
            n_trials=N_TRIALS,
            show_progress_bar=True,
        )

    else:
        seq_len = 8 if is_binary else 12
        cached_folds = build_cached_seq_folds(df, x, y, fb.folds, seq_len)
        study.optimize(
            lambda t: objective_seq(t, cached_folds, MODEL, is_binary, n_classes, SEED),
            n_trials=N_TRIALS,
            show_progress_bar=True,
        )

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