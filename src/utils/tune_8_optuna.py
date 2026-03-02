# src/tune_8_optuna.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from src.data.data import (
    infer_feature_types, COMPOUND_CLASSES,
    load_stage1_dataset, load_stage2_dataset,
    get_stage1_xy, get_stage2_xy,
    encode_y_compound,
    make_race_group_folds,
    build_feature_sequences,
)
from src.data.preprocessing import make_preprocessor_for_model

try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    from tensorflow import keras
    from tensorflow.keras import layers
    _HAS_TF = True
except Exception:
    _HAS_TF = False


HOLDOUT_RACE_IDS = {53, 73, 24, 75, 2}

# -------------------------
# Metrics
# -------------------------
def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))

def _softmax(Z: np.ndarray) -> np.ndarray:
    Z = Z - np.max(Z, axis=1, keepdims=True)
    expZ = np.exp(np.clip(Z, -50, 50))
    return expZ / np.sum(expZ, axis=1, keepdims=True)

def compute_binary_metrics(y_true: np.ndarray, proba_pos: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (proba_pos >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba_pos)) if len(np.unique(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, proba_pos)) if len(np.unique(y_true)) > 1 else float("nan"),
        "logloss": float(log_loss(y_true, proba_pos, labels=[0, 1])),
    }


def compute_multiclass_metrics(y_true: np.ndarray, proba_full: np.ndarray) -> Dict[str, float]:
    y_pred = np.argmax(proba_full, axis=1)
    K = proba_full.shape[1]
    labels = np.arange(K)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "logloss": float(log_loss(y_true, proba_full, labels=labels)),
    }


def mean_std_summary(fold_rows: List[Dict[str, Any]], metric_keys: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in metric_keys:
        vals = [r[k] for r in fold_rows if k in r and np.isfinite(r[k])]
        out[f"{k}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
        out[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
    return out


# -------------------------
# Race stratified sampling (same idea as your existing file)
# -------------------------
def _race_level_strata(df: pd.DataFrame) -> pd.Series:
    def _has(col: str) -> bool:
        return col in df.columns

    # rain flag
    if _has("rained_yet"):
        rain_num = pd.to_numeric(df["rained_yet"].astype(str), errors="coerce").fillna(0)
        rain = rain_num.groupby(df["race_id"]).max().astype(int)
    elif _has("is_raining"):
        rain_num = pd.to_numeric(df["is_raining"].astype(str), errors="coerce").fillna(0)
        rain = rain_num.groupby(df["race_id"]).max().astype(int)
    elif _has("minutes_rain"):
        rain_num = pd.to_numeric(df["minutes_rain"].astype(str), errors="coerce").fillna(0)
        rain = (rain_num.groupby(df["race_id"]).max() > 0).astype(int)
    else:
        rain = pd.Series(0, index=df["race_id"].unique())

    # FCY flag
    if _has("fcy_status"):
        fcy_num = pd.to_numeric(df["fcy_status"].astype(str), errors="coerce").fillna(0)
        fcy = (fcy_num.groupby(df["race_id"]).max() > 0).astype(int)
    else:
        fcy = pd.Series(0, index=rain.index)

    strata = pd.Series(
        [f"rain{int(rain.loc[r])}_fcy{int(fcy.loc[r])}" for r in rain.index],
        index=rain.index,
        name="strata",
    )
    return strata

def sample_races_stratified(
    df: pd.DataFrame,
    *,
    rng: np.random.Generator,
    max_races: int = 0,
    sample_frac: float = 1.0,
) -> pd.DataFrame:
    race_ids = np.array(sorted(df["race_id"].unique()))
    if len(race_ids) == 0:
        return df

    strata = _race_level_strata(df)
    n_target = len(race_ids)
    if sample_frac < 1.0:
        n_target = max(4, int(np.ceil(sample_frac * len(race_ids))))
    if max_races and max_races > 0:
        n_target = min(n_target, int(max_races))
    n_target = min(n_target, len(race_ids))

    by_stratum: Dict[str, List[int]] = {}
    for r in race_ids:
        s = strata.loc[r] if r in strata.index else "rain0_fcy0"
        by_stratum.setdefault(s, []).append(int(r))

    chosen: List[int] = []
    total = len(race_ids)
    for s, rs in by_stratum.items():
        k = int(np.round(n_target * (len(rs) / total)))
        k = max(1, min(k, len(rs)))
        rs = np.array(rs)
        rng.shuffle(rs)
        chosen.extend(rs[:k].tolist())

    chosen = np.array(sorted(set(chosen)))
    if len(chosen) > n_target:
        rng.shuffle(chosen)
        chosen = chosen[:n_target]
    elif len(chosen) < n_target:
        remaining = np.setdiff1d(race_ids, chosen)
        rng.shuffle(remaining)
        need = n_target - len(chosen)
        chosen = np.concatenate([chosen, remaining[:need]])

    return df[df["race_id"].isin(chosen)].copy()


# -------------------------
# Cached folds for tabular models (SVM/RF/XGB/VSE-ANN)
# -------------------------
CachedFold = Tuple[Any, np.ndarray, Any, np.ndarray, np.ndarray, np.ndarray]  # Xt_tr, y_tr, Xt_va, y_va, tr_idx, va_idx


def build_cached_folds(
    X: pd.DataFrame,
    y: pd.Series,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    *,
    pre_name: str,
) -> List[CachedFold]:
    num_cols, cat_cols = infer_feature_types(X)
    base_pre = make_preprocessor_for_model(pre_name, num_cols=num_cols, cat_cols=cat_cols)

    cached: List[CachedFold] = []
    y_np_all = y.to_numpy()

    for (tr_idx, va_idx) in folds:
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y_np_all[tr_idx], y_np_all[va_idx]

        pre = clone(base_pre)
        pre.fit(X_tr, y_tr)

        Xt_tr = pre.transform(X_tr)
        Xt_va = pre.transform(X_va)

        if hasattr(Xt_tr, "tocsr"):
            Xt_tr = Xt_tr.tocsr()
            Xt_va = Xt_va.tocsr()

        cached.append((Xt_tr, y_tr, Xt_va, y_va, tr_idx, va_idx))

    return cached


# -------------------------
# Search spaces (8 models)
# -------------------------
def suggest_svm_params(trial: optuna.Trial, *, task: str) -> Dict[str, Any]:
    # kernel: categorical ["linear", "rbf"]
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
    params: Dict[str, Any] = {"kernel": kernel}

    if task == "binary":
        cw_mode = trial.suggest_categorical("class_weight_mode", ["balanced", "custom"])
        if cw_mode == "balanced":
            params["class_weight"] = "balanced"
        else:
            pos_w = trial.suggest_float("pos_weight", 8.0, 120.0, log=True)
            params["class_weight"] = {0: 1.0, 1: float(pos_w)}
    else:
        params["class_weight"] = trial.suggest_categorical("class_weight", [None, "balanced"])

    if kernel == "linear":
        params["C"] = trial.suggest_float("C", 1e-3, 1e2, log=True)
        params["tol"] = trial.suggest_float("tol", 1e-6, 1e-2, log=True)
        params["max_iter"] = trial.suggest_int("max_iter", 2000, 20000)
        params["calib_method"] = trial.suggest_categorical("calib_method", ["sigmoid", "isotonic"])
        params["calib_cv"] = trial.suggest_categorical("calib_cv", [3, 5])
    else:
        params["C"] = trial.suggest_float("C", 1e-2, 1e2, log=True)
        params["gamma"] = trial.suggest_float("gamma", 1e-5, 1e0, log=True)
        params["calib_method"] = trial.suggest_categorical("calib_method", ["sigmoid", "isotonic"])
        params["calib_cv"] = trial.suggest_categorical("calib_cv", [3, 5])

    return params


def suggest_xgb_params(trial: optuna.Trial, *, task: str) -> Dict[str, Any]:
    if not _HAS_XGB:
        raise ImportError("XGBoost not installed")

    if task == "binary":
        return {
            "tree_method": "hist",
            "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-3, 8.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 12.0, 80.0, log=True),
            "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
        }
    else:
        return {
            "tree_method": "hist",
            "n_estimators": trial.suggest_int("n_estimators", 600, 6000),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-3, 6.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            # optional but included (you said "within reason" and a couple weeks is fine)
            "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
        }


def suggest_rf_params(trial: optuna.Trial) -> Dict[str, Any]:
    # Multiclass RF only (bootstrap fixed True)
    crit_choices = ["gini", "entropy"]
    # log_loss exists in newer sklearn; if unsupported, sklearn will raise -> keep it optional by try/except
    # We include it because it's part of your desired search.
    crit_choices.append("log_loss")

    return {
        "n_estimators": trial.suggest_int("n_estimators", 300, 2500),
        "max_depth": trial.suggest_categorical("max_depth", [None, 8, 12, 16, 20, 26, 32]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 60),
        "max_features": trial.suggest_float("max_features", 0.05, 0.8),
        "bootstrap": True,
        "max_samples": trial.suggest_float("max_samples", 0.50, 1.00),
        "criterion": trial.suggest_categorical("criterion", crit_choices),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }


def suggest_tcn_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "seq_len": trial.suggest_categorical("seq_len", [4, 6, 8, 10, 12]),
        "filters": trial.suggest_categorical("filters", [32, 64, 96, 128]),
        "kernel_size": trial.suggest_categorical("kernel_size", [2, 3, 5, 7]),
        "n_blocks": trial.suggest_int("n_blocks", 3, 6),
        "pooling": trial.suggest_categorical("pooling", ["gap", "last"]),
        "dropout_residual": trial.suggest_float("dropout_residual", 0.0, 0.35),
        "dropout_head": trial.suggest_float("dropout_head", 0.0, 0.45),
        "dense_units": trial.suggest_categorical("dense_units", [32, 64, 128]),
        "learning_rate": trial.suggest_float("learning_rate", 3e-5, 3e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
        "focal_alpha": trial.suggest_float("focal_alpha", 0.10, 0.90),
        "focal_gamma": trial.suggest_float("focal_gamma", 0.5, 5.0, log=True),
        "epochs": 60,
        "patience": 6,
    }


def suggest_tcn_gru_params(trial: optuna.Trial) -> Dict[str, Any]:
    loss_type = trial.suggest_categorical("loss_type", ["bce", "focal"])
    params: Dict[str, Any] = {
        "seq_len": trial.suggest_categorical("seq_len", [4, 6, 8, 10, 12]),
        "filters": trial.suggest_categorical("filters", [32, 64, 96, 128]),
        "kernel_size": trial.suggest_categorical("kernel_size", [2, 3, 5]),
        "n_blocks": trial.suggest_int("n_blocks", 3, 6),
        "pooling": trial.suggest_categorical("pooling", ["gap", "last"]),
        "dropout_residual": trial.suggest_float("dropout_residual", 0.0, 0.35),

        "gru_units": trial.suggest_categorical("gru_units", [16, 32, 64, 96, 128]),
        "gru_dropout": trial.suggest_float("gru_dropout", 0.0, 0.40),
        "gru_l2": trial.suggest_float("gru_l2", 1e-6, 3e-3, log=True),

        "dropout_head": trial.suggest_float("dropout_head", 0.0, 0.45),
        "dense_units": trial.suggest_categorical("dense_units", [32, 64, 128]),
        "learning_rate": trial.suggest_float("learning_rate", 3e-5, 3e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
        "loss_type": loss_type,
        "epochs": 60,
        "patience": 6,
    }
    if loss_type == "focal":
        params["focal_alpha"] = trial.suggest_float("focal_alpha", 0.10, 0.90)
        params["focal_gamma"] = trial.suggest_float("focal_gamma", 0.5, 5.0, log=True)
    return params


def suggest_vse_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "n_layers": trial.suggest_int("n_layers", 1, 3),
        "hidden_units": trial.suggest_categorical("hidden_units", [32, 64, 128, 256]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "l2": trial.suggest_float("l2", 1e-6, 1e-2, log=True),
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "nadam"]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.15),
        "epochs": 200,
        "patience": 8,
    }


# -------------------------
# CV eval: SVM (cached folds)
# -------------------------
def cv_eval_svm_cached(
    cached_folds: List[CachedFold],
    *,
    params: Dict[str, Any],
    task: str,
    n_classes: Optional[int],
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Calibrated SVM probabilities (for logloss), with a robust fallback when
    CalibratedClassifierCV cannot run because a fold has too few examples
    for at least one class.

    Fallback: fit the base SVM and convert decision_function outputs into
    pseudo-probabilities (sigmoid for binary, softmax for multiclass).
    """
    rows: List[Dict[str, Any]] = []

    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -50, 50)
        return 1.0 / (1.0 + np.exp(-z))

    def _softmax(Z: np.ndarray) -> np.ndarray:
        Z = Z - np.max(Z, axis=1, keepdims=True)
        expZ = np.exp(np.clip(Z, -50, 50))
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    kernel = params["kernel"]
    C = float(params["C"])
    class_weight = params.get("class_weight", None)
    calib_method = params.get("calib_method", "sigmoid")
    calib_cv = int(params.get("calib_cv", 3))

    for fold, (Xt_tr, y_tr, Xt_va, y_va, tr_idx, va_idx) in enumerate(cached_folds):
        # build base estimator
        if kernel == "linear":
            base = LinearSVC(
                C=C,
                class_weight=class_weight,
                random_state=seed,
                dual="auto",
                tol=float(params.get("tol", 1e-4)),
                max_iter=int(params.get("max_iter", 10000)),
            )
        else:
            gamma = float(params["gamma"])
            base = SVC(
                C=C,
                kernel="rbf",
                gamma=gamma,
                probability=False,  # use explicit calibration
                class_weight=class_weight,
                random_state=seed,
                decision_function_shape="ovr",  # helps multiclass fallback
            )

        # choose calibration cv that is feasible for this fold
        counts = np.bincount(y_tr.astype(int))
        present = counts[counts > 0]
        min_count = int(present.min()) if len(present) else 0
        cv_eff = int(min(calib_cv, min_count))

        try:
            if cv_eff < 2:
                raise ValueError("Too few examples per class for calibration")

            model = CalibratedClassifierCV(base, method=calib_method, cv=cv_eff)
            model.fit(Xt_tr, y_tr)
            proba = model.predict_proba(Xt_va)

        except ValueError:
            # Fallback: fit base model and convert decision scores to probabilities
            base.fit(Xt_tr, y_tr)

            if task == "binary":
                # LinearSVC decision_function -> (n,), SVC -> (n,)
                scores = np.asarray(base.decision_function(Xt_va)).reshape(-1)
                proba_pos = _sigmoid(scores)
                proba = np.vstack([1.0 - proba_pos, proba_pos]).T
            else:
                K = int(n_classes) if n_classes is not None else int(len(np.unique(y_tr)))
                scores = np.asarray(base.decision_function(Xt_va))
                if scores.ndim == 1:
                    # edge case: can't recover per-class scores; return uniform probs
                    proba = np.full((scores.shape[0], K), 1.0 / K, dtype=float)
                else:
                    # scores is (n, K) for ovr; softmax gives a usable distribution
                    proba = _softmax(scores)
                    # if K differs (rare), pad/truncate safely
                    if proba.shape[1] != K:
                        proba_full = np.full((proba.shape[0], K), 1.0 / K, dtype=float)
                        cols = min(K, proba.shape[1])
                        proba_full[:, :cols] = proba[:, :cols]
                        proba = proba_full

        # metrics
        if task == "binary":
            proba_pos = proba[:, 1]
            m = compute_binary_metrics(y_va, proba_pos, threshold=0.5)
            rows.append({"fold": int(fold), "n_valid": int(len(va_idx)), **m})
        else:
            K = int(n_classes) if n_classes is not None else int(len(np.unique(y_tr)))
            # ensure K columns + normalise
            proba_full = np.zeros((len(va_idx), K), dtype=float)
            cols = min(K, proba.shape[1])
            proba_full[:, :cols] = proba[:, :cols]

            eps = 1e-15
            row_sums = proba_full.sum(axis=1, keepdims=True)
            zero_mask = (row_sums[:, 0] == 0)
            if np.any(zero_mask):
                proba_full[zero_mask] = 1.0 / K
                row_sums = proba_full.sum(axis=1, keepdims=True)
            proba_full = (proba_full + eps) / (row_sums + eps * K)

            m = compute_multiclass_metrics(y_va, proba_full)
            rows.append({"fold": int(fold), "n_valid": int(len(va_idx)), **m})

    return rows


def objective_svm(
    trial: optuna.Trial,
    cached_folds: List[CachedFold],
    *,
    task: str,
    seed: int,
    n_classes: Optional[int],
) -> float:
    params = suggest_svm_params(trial, task=task)

    fold_rows = cv_eval_svm_cached(
        cached_folds,
        params=params,
        task=task,
        n_classes=n_classes,
        seed=seed,
    )

    losses = [r["logloss"] for r in fold_rows]
    for i in range(len(losses)):
        trial.report(float(np.mean(losses[: i + 1])), step=i)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(losses))


# -------------------------
# CV eval: RF (cached folds)
# -------------------------
def cv_eval_rf_cached(
    cached_folds: List[CachedFold],
    *,
    params: Dict[str, Any],
    task: str,
    n_classes: Optional[int],
    seed: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fold, (Xt_tr, y_tr, Xt_va, y_va, tr_idx, va_idx) in enumerate(cached_folds):
        rf = RandomForestClassifier(**params, random_state=seed, n_jobs=-1)

        # criterion="log_loss" may not exist in older sklearn; fallback
        try:
            rf.fit(Xt_tr, y_tr)
        except ValueError as e:
            if "log_loss" in str(e) and params.get("criterion") == "log_loss":
                p2 = dict(params)
                p2["criterion"] = "gini"
                rf = RandomForestClassifier(**p2, random_state=seed, n_jobs=-1)
                rf.fit(Xt_tr, y_tr)
            else:
                raise

        proba = rf.predict_proba(Xt_va)

        if task == "binary":
            m = compute_binary_metrics(y_va, proba[:, 1], threshold=0.5)
            rows.append({"fold": int(fold), "n_valid": int(len(va_idx)), **m})
        else:
            K = int(n_classes) if n_classes is not None else int(len(np.unique(y_tr)))
            classes_seen = rf.classes_
            proba_full = np.zeros((len(va_idx), K), dtype=float)
            for j, cls in enumerate(classes_seen):
                if int(cls) < K:
                    proba_full[:, int(cls)] = proba[:, j]

            eps = 1e-15
            row_sums = proba_full.sum(axis=1, keepdims=True)
            zero_mask = (row_sums[:, 0] == 0)
            if np.any(zero_mask):
                proba_full[zero_mask] = 1.0 / K
                row_sums = proba_full.sum(axis=1, keepdims=True)
            proba_full = (proba_full + eps) / (row_sums + eps * K)

            m = compute_multiclass_metrics(y_va, proba_full)
            rows.append({"fold": int(fold), "n_valid": int(len(va_idx)), **m})

    return rows


def objective_rf(
    trial: optuna.Trial,
    cached_folds: List[CachedFold],
    *,
    task: str,
    seed: int,
    n_classes: Optional[int],
) -> float:
    params = suggest_rf_params(trial)
    fold_rows = cv_eval_rf_cached(
        cached_folds,
        params=params,
        task=task,
        n_classes=n_classes,
        seed=seed,
    )
    losses = [r["logloss"] for r in fold_rows]
    for i in range(len(losses)):
        trial.report(float(np.mean(losses[: i + 1])), step=i)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(losses))


# -------------------------
# CV eval: XGB (cached folds)
# -------------------------
def cv_eval_xgb_cached(
    cached_folds: List[CachedFold],
    *,
    params: Dict[str, Any],
    task: str,
    n_classes: Optional[int],
    seed: int,
    early_stopping_rounds: int,
    n_jobs: int,
) -> List[Dict[str, Any]]:
    if not _HAS_XGB:
        raise ImportError("XGBoost not installed")

    rows: List[Dict[str, Any]] = []
    callbacks = [xgb.callback.EarlyStopping(rounds=early_stopping_rounds, save_best=False)] if early_stopping_rounds > 0 else None

    for fold, (Xt_tr, y_tr, Xt_va, y_va, tr_idx, va_idx) in enumerate(cached_folds):
        if task == "binary":
            model = XGBClassifier(
                **{k: v for k, v in params.items() if k != "scale_pos_weight"},
                scale_pos_weight=float(params.get("scale_pos_weight", 1.0)),
                objective="binary:logistic",
                eval_metric="logloss",
                n_jobs=n_jobs,
                random_state=seed,
                callbacks=callbacks,
            )
            model.fit(Xt_tr, y_tr, eval_set=[(Xt_va, y_va)], verbose=False)
            proba_pos = model.predict_proba(Xt_va)[:, 1]
            m = compute_binary_metrics(y_va, proba_pos, threshold=0.5)
            bi = getattr(model, "best_iteration", None)
            rows.append({"fold": int(fold), "n_valid": int(len(va_idx)), "best_iteration": int(bi) if bi is not None else -1, **m})
        else:
            K = int(n_classes) if n_classes is not None else int(len(np.unique(y_tr)))
            model = XGBClassifier(
                **params,
                objective="multi:softprob",
                num_class=K,
                eval_metric="mlogloss",
                n_jobs=n_jobs,
                random_state=seed,
                callbacks=callbacks,
            )
            model.fit(Xt_tr, y_tr, eval_set=[(Xt_va, y_va)], verbose=False)
            proba = model.predict_proba(Xt_va)

            # Ensure full K columns (XGB usually returns K, but be safe)
            proba_full = np.zeros((len(va_idx), K), dtype=float)
            cols = min(K, proba.shape[1])
            proba_full[:, :cols] = proba[:, :cols]

            eps = 1e-15
            row_sums = proba_full.sum(axis=1, keepdims=True)
            zero_mask = (row_sums[:, 0] == 0)
            if np.any(zero_mask):
                proba_full[zero_mask] = 1.0 / K
                row_sums = proba_full.sum(axis=1, keepdims=True)
            proba_full = (proba_full + eps) / (row_sums + eps * K)

            m = compute_multiclass_metrics(y_va, proba_full)
            bi = getattr(model, "best_iteration", None)
            rows.append({"fold": int(fold), "n_valid": int(len(va_idx)), "best_iteration": int(bi) if bi is not None else -1, **m})

    return rows


def objective_xgb(
    trial: optuna.Trial,
    cached_folds: List[CachedFold],
    *,
    task: str,
    seed: int,
    n_classes: Optional[int],
    early_stopping_rounds: int,
    n_jobs: int,
) -> float:
    params = suggest_xgb_params(trial, task=task)

    fold_rows = cv_eval_xgb_cached(
        cached_folds,
        params=params,
        task=task,
        n_classes=n_classes,
        seed=seed,
        early_stopping_rounds=early_stopping_rounds,
        n_jobs=n_jobs,
    )
    losses = [r["logloss"] for r in fold_rows]
    for i in range(len(losses)):
        trial.report(float(np.mean(losses[: i + 1])), step=i)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(losses))


# -------------------------
# Sequence models: TCN + TCN-GRU
# -------------------------
def _tcn_residual_block(x, *, filters: int, kernel_size: int, dilation: int, dropout: float):
    shortcut = x
    x = layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout)(x)

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding="same")(shortcut)

    return layers.Add()([shortcut, x])


def build_tcn_binary_model(
    *,
    seq_len: int,
    n_features: int,
    filters: int,
    kernel_size: int,
    n_blocks: int,
    pooling: str,
    dropout_residual: float,
    dropout_head: float,
    dense_units: int,
    learning_rate: float,
    focal_alpha: float,
    focal_gamma: float,
) -> "keras.Model":
    x_in = layers.Input(shape=(seq_len, n_features))
    x = layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=1)(x_in)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)

    dilations = [1, 2, 4, 8, 16, 32]
    for d in dilations[:n_blocks]:
        x = _tcn_residual_block(x, filters=filters, kernel_size=kernel_size, dilation=d, dropout=dropout_residual)

    if pooling == "gap":
        x = layers.GlobalAveragePooling1D()(x)
    elif pooling == "last":
        x = layers.Lambda(lambda z: z[:, -1, :])(x)
    else:
        raise ValueError("pooling must be 'gap' or 'last'")

    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout_head)(x)
    y_out = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(x_in, y_out)

    loss = keras.losses.BinaryFocalCrossentropy(gamma=focal_gamma, alpha=focal_alpha)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[
            keras.metrics.AUC(curve="PR", name="pr_auc"),
            keras.metrics.AUC(curve="ROC", name="roc_auc"),
        ],
    )
    return model


def build_tcn_gru_binary_model(
    *,
    seq_len: int,
    n_features: int,
    filters: int,
    kernel_size: int,
    n_blocks: int,
    pooling: str,
    dropout_residual: float,
    gru_units: int,
    gru_dropout: float,
    gru_l2: float,
    dense_units: int,
    dropout_head: float,
    learning_rate: float,
    loss_type: str,
    focal_alpha: Optional[float],
    focal_gamma: Optional[float],
) -> "keras.Model":
    x_in = layers.Input(shape=(seq_len, n_features))
    x = layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=1)(x_in)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)

    dilations = [1, 2, 4, 8, 16, 32]
    for d in dilations[:n_blocks]:
        x = _tcn_residual_block(x, filters=filters, kernel_size=kernel_size, dilation=d, dropout=dropout_residual)

    if pooling == "gap":
        x = layers.GlobalAveragePooling1D()(x)
        # after GAP, sequence dimension removed; GRU needs sequence -> so for pooling="gap" we skip GRU
        # but your spec includes GRU, so enforce pooling="last" or apply GRU before pooling.
        # Here: apply GRU BEFORE pooling always for tcn_gru.
        # (We’ll rebuild a safe path below)
        raise ValueError("For tcn_gru, use pooling='last' (GRU consumes sequence).")
    elif pooling == "last":
        pass
    else:
        raise ValueError("pooling must be 'gap' or 'last'")

    # For tcn_gru: feed sequence output into GRU (no pooling before GRU)
    # So instead: take x as (batch, time, channels) and apply GRU
    x = layers.GRU(
        gru_units,
        return_sequences=False,
        dropout=gru_dropout,
        recurrent_dropout=0.0,
        kernel_regularizer=keras.regularizers.l2(gru_l2),
    )(x)

    x = layers.Dense(dense_units, activation="relu", kernel_regularizer=keras.regularizers.l2(gru_l2))(x)
    x = layers.Dropout(dropout_head)(x)
    y_out = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(x_in, y_out)

    if loss_type == "focal":
        assert focal_alpha is not None and focal_gamma is not None
        loss = keras.losses.BinaryFocalCrossentropy(gamma=float(focal_gamma), alpha=float(focal_alpha))
    else:
        loss = "binary_crossentropy"

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[
            keras.metrics.AUC(curve="PR", name="pr_auc"),
            keras.metrics.AUC(curve="ROC", name="roc_auc"),
        ],
    )
    return model


def _prep_sequences_for_fold(
    *,
    df: pd.DataFrame,
    X: pd.DataFrame,
    y_full: pd.Series,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    seq_len: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Fit preprocessor on train rows only; transform all rows; then build sequences
    keys = df[["race_id", "driver_id", "lapno"]].copy()

    num_cols, cat_cols = infer_feature_types(X)
    base_pre = make_preprocessor_for_model("tcn", num_cols=num_cols, cat_cols=cat_cols)

    pre = clone(base_pre)
    pre.fit(X.iloc[tr_idx], y_full.iloc[tr_idx])

    Xt_all = pre.transform(X)
    if hasattr(Xt_all, "toarray"):
        Xt_all = Xt_all.toarray()
    Xt_all = Xt_all.astype(np.float32)

    X_seq, y_seq, idx_last, seq_idx, eff_len = build_feature_sequences(
        keys, Xt_all, y_full,
        seq_len=seq_len,
        pad_left=True,
        add_timestep_mask=True,
    )

    tr_mask = np.all((seq_idx == -1) | np.isin(seq_idx, tr_idx), axis=1)
    va_mask = np.all((seq_idx == -1) | np.isin(seq_idx, va_idx), axis=1)

    X_seq_tr, y_seq_tr = X_seq[tr_mask], y_seq[tr_mask]
    X_seq_va, y_seq_va = X_seq[va_mask], y_seq[va_mask]

    # Standardize across features using TRAIN ONLY, over all timesteps
    scaler = StandardScaler()
    X_tr_2d = X_seq_tr.reshape(-1, X_seq_tr.shape[-1])
    X_va_2d = X_seq_va.reshape(-1, X_seq_va.shape[-1])
    scaler.fit(X_tr_2d)

    X_seq_tr = scaler.transform(X_tr_2d).reshape(X_seq_tr.shape).astype(np.float32)
    X_seq_va = scaler.transform(X_va_2d).reshape(X_seq_va.shape).astype(np.float32)

    return X_seq_tr, y_seq_tr.astype(int), X_seq_va, y_seq_va.astype(int)


def objective_tcn(
    trial: optuna.Trial,
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int,
    seed: int,
) -> float:
    if not _HAS_TF:
        raise ImportError("TensorFlow/Keras not installed")

    params = suggest_tcn_params(trial)
    seq_len = int(params["seq_len"])

    fold_bundle = make_race_group_folds(df, target_col="y_pit", n_splits=n_splits, seed=seed)
    folds = fold_bundle.folds

    y_full = df["y_pit"].astype(int)

    losses: List[float] = []

    keras.utils.set_random_seed(seed)

    for fold, (tr_idx, va_idx) in enumerate(folds):
        X_seq_tr, y_seq_tr, X_seq_va, y_seq_va = _prep_sequences_for_fold(
            df=df, X=X, y_full=y_full,
            tr_idx=tr_idx, va_idx=va_idx,
            seq_len=seq_len, seed=seed,
        )

        if len(y_seq_tr) == 0 or len(y_seq_va) == 0:
            losses.append(1.0)
            continue

        model = build_tcn_binary_model(
            seq_len=seq_len,
            n_features=X_seq_tr.shape[2],
            filters=int(params["filters"]),
            kernel_size=int(params["kernel_size"]),
            n_blocks=int(params["n_blocks"]),
            pooling=str(params["pooling"]),
            dropout_residual=float(params["dropout_residual"]),
            dropout_head=float(params["dropout_head"]),
            dense_units=int(params["dense_units"]),
            learning_rate=float(params["learning_rate"]),
            focal_alpha=float(params["focal_alpha"]),
            focal_gamma=float(params["focal_gamma"]),
        )

        cb = keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=int(params["patience"]),
            restore_best_weights=True,
        )

        model.fit(
            X_seq_tr, y_seq_tr,
            validation_data=(X_seq_va, y_seq_va),
            epochs=int(params["epochs"]),
            batch_size=int(params["batch_size"]),
            verbose=0,
            callbacks=[cb],
        )

        proba_pos = model.predict(X_seq_va, batch_size=int(params["batch_size"]), verbose=0).reshape(-1)
        m = compute_binary_metrics(y_seq_va, proba_pos, threshold=0.5)

        losses.append(float(m["logloss"]))

        trial.report(float(np.mean(losses)), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

        keras.backend.clear_session()

    return float(np.mean(losses))


def objective_tcn_gru(
    trial: optuna.Trial,
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int,
    seed: int,
) -> float:
    if not _HAS_TF:
        raise ImportError("TensorFlow/Keras not installed")

    params = suggest_tcn_gru_params(trial)
    seq_len = int(params["seq_len"])

    fold_bundle = make_race_group_folds(df, target_col="y_pit", n_splits=n_splits, seed=seed)
    folds = fold_bundle.folds

    y_full = df["y_pit"].astype(int)

    losses: List[float] = []

    keras.utils.set_random_seed(seed)

    for fold, (tr_idx, va_idx) in enumerate(folds):
        X_seq_tr, y_seq_tr, X_seq_va, y_seq_va = _prep_sequences_for_fold(
            df=df, X=X, y_full=y_full,
            tr_idx=tr_idx, va_idx=va_idx,
            seq_len=seq_len, seed=seed,
        )

        if len(y_seq_tr) == 0 or len(y_seq_va) == 0:
            losses.append(1.0)
            continue

        # Enforce pooling='last' for tcn_gru (GRU consumes sequence output)
        pooling = str(params["pooling"])
        if pooling != "last":
            # prune quickly if sampled "gap"
            losses.append(1.0)
            trial.report(float(np.mean(losses)), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
            continue

        model = build_tcn_gru_binary_model(
            seq_len=seq_len,
            n_features=X_seq_tr.shape[2],
            filters=int(params["filters"]),
            kernel_size=int(params["kernel_size"]),
            n_blocks=int(params["n_blocks"]),
            pooling="last",
            dropout_residual=float(params["dropout_residual"]),
            gru_units=int(params["gru_units"]),
            gru_dropout=float(params["gru_dropout"]),
            gru_l2=float(params["gru_l2"]),
            dense_units=int(params["dense_units"]),
            dropout_head=float(params["dropout_head"]),
            learning_rate=float(params["learning_rate"]),
            loss_type=str(params["loss_type"]),
            focal_alpha=float(params["focal_alpha"]) if params["loss_type"] == "focal" else None,
            focal_gamma=float(params["focal_gamma"]) if params["loss_type"] == "focal" else None,
        )

        cb = keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=int(params["patience"]),
            restore_best_weights=True,
        )

        model.fit(
            X_seq_tr, y_seq_tr,
            validation_data=(X_seq_va, y_seq_va),
            epochs=int(params["epochs"]),
            batch_size=int(params["batch_size"]),
            verbose=0,
            callbacks=[cb],
        )

        proba_pos = model.predict(X_seq_va, batch_size=int(params["batch_size"]), verbose=0).reshape(-1)
        m = compute_binary_metrics(y_seq_va, proba_pos, threshold=0.5)

        losses.append(float(m["logloss"]))

        trial.report(float(np.mean(losses)), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

        keras.backend.clear_session()

    return float(np.mean(losses))


# -------------------------
# VSE Compound ANN (multiclass) - cached folds
# -------------------------
def _build_vse_mlp(
    input_dim: int,
    n_classes: int,
    *,
    n_layers: int,
    hidden_units: int,
    dropout: float,
    l2: float,
    optimizer: str,
    learning_rate: float,
    label_smoothing: float,
) -> "keras.Model":
    reg = keras.regularizers.l2(l2)
    x_in = layers.Input(shape=(input_dim,))
    x = x_in
    for _ in range(int(n_layers)):
        x = layers.Dense(int(hidden_units), activation="relu", kernel_regularizer=reg)(x)
        if float(dropout) > 0:
            x = layers.Dropout(float(dropout))(x)
    y_out = layers.Dense(int(n_classes), activation="softmax")(x)

    model = keras.Model(x_in, y_out)

    if optimizer == "nadam":
        opt = keras.optimizers.Nadam(learning_rate=float(learning_rate))
    else:
        opt = keras.optimizers.Adam(learning_rate=float(learning_rate))

    try:
        loss = keras.losses.SparseCategoricalCrossentropy(label_smoothing=float(label_smoothing))
    except TypeError:
        # Older TF/Keras: SparseCategoricalCrossentropy has no label_smoothing
        loss = keras.losses.SparseCategoricalCrossentropy()

    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    return model


def cv_eval_vse_cached(
    cached_folds: List[CachedFold],
    *,
    params: Dict[str, Any],
    n_classes: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if not _HAS_TF:
        raise ImportError("TensorFlow/Keras not installed")

    rows: List[Dict[str, Any]] = []
    keras.utils.set_random_seed(seed)

    for fold, (Xt_tr, y_tr, Xt_va, y_va, tr_idx, va_idx) in enumerate(cached_folds):
        Xtr = Xt_tr.toarray() if hasattr(Xt_tr, "toarray") else np.asarray(Xt_tr)
        Xva = Xt_va.toarray() if hasattr(Xt_va, "toarray") else np.asarray(Xt_va)

        model = _build_vse_mlp(
            input_dim=int(Xtr.shape[1]),
            n_classes=int(n_classes),
            n_layers=int(params["n_layers"]),
            hidden_units=int(params["hidden_units"]),
            dropout=float(params["dropout"]),
            l2=float(params["l2"]),
            optimizer=str(params["optimizer"]),
            learning_rate=float(params["learning_rate"]),
            label_smoothing=float(params["label_smoothing"]),
        )

        cb = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=int(params["patience"]),
            restore_best_weights=True,
        )

        model.fit(
            Xtr, y_tr,
            validation_data=(Xva, y_va),
            epochs=int(params["epochs"]),
            batch_size=int(params["batch_size"]),
            verbose=0,
            callbacks=[cb],
        )

        proba = model.predict(Xva, batch_size=int(params["batch_size"]), verbose=0)
        # ensure normalised
        eps = 1e-15
        proba = np.clip(proba, eps, 1.0)
        proba = proba / proba.sum(axis=1, keepdims=True)

        m = compute_multiclass_metrics(y_va, proba)
        rows.append({"fold": int(fold), "n_valid": int(len(va_idx)), **m})

        keras.backend.clear_session()

    return rows


def objective_vse(
    trial: optuna.Trial,
    cached_folds: List[CachedFold],
    *,
    seed: int,
    n_classes: int,
) -> float:
    params = suggest_vse_params(trial)
    fold_rows = cv_eval_vse_cached(cached_folds, params=params, n_classes=n_classes, seed=seed)

    losses = [r["logloss"] for r in fold_rows]
    for i in range(len(losses)):
        trial.report(float(np.mean(losses[: i + 1])), step=i)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(losses))


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_stage1", required=False)
    ap.add_argument("--data_stage2", required=False)
    ap.add_argument("--stage2", action="store_true")
    ap.add_argument("--task", choices=["binary", "multiclass"], required=True)

    ap.add_argument(
        "--model",
        choices=["svm", "xgb", "tcn", "tcn_gru", "rf", "vse_compound_ann"],
        required=True,
    )

    ap.add_argument("--n_trials", type=int, default=200)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="runs/tuning_8")

    # speed controls
    ap.add_argument("--sample_frac", type=float, default=1.0)
    ap.add_argument("--max_races", type=int, default=0)
    ap.add_argument("--timeout_sec", type=int, default=0)

    # XGB controls
    ap.add_argument("--early_stopping_rounds", type=int, default=50)
    ap.add_argument("--n_jobs", type=int, default=-1)

    # SVM safety cap (especially RBF)
    ap.add_argument("--svm_max_rows_rbf", type=int, default=20000, help="Hard cap rows for SVM runs via race subsample.")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # -------------------------
    # Enforce model/stage consistency (only the 8 cases)
    # -------------------------
    if args.stage2:
        if args.task != "multiclass":
            raise ValueError("--stage2 requires --task multiclass")
        if args.model not in {"rf", "svm", "xgb", "vse_compound_ann"}:
            raise ValueError("Stage2 (multiclass) supports only: rf, svm, xgb, vse_compound_ann")
        if not args.data_stage2:
            raise ValueError("Pass --data_stage2")
    else:
        if args.task != "binary":
            raise ValueError("Stage1 requires --task binary (for these 4 binary models)")
        if args.model not in {"svm", "xgb", "tcn", "tcn_gru"}:
            raise ValueError("Stage1 (binary) supports only: svm, xgb, tcn, tcn_gru")
        if not args.data_stage1:
            raise ValueError("Pass --data_stage1")

    # -------------------------
    # Load dataset + folds
    # -------------------------
    if args.stage2:
        df = load_stage2_dataset(args.data_stage2, strict=True)
        df = df.loc[~df["race_id"].isin(HOLDOUT_RACE_IDS)].copy()

        df = encode_y_compound(df, col="y_compound", out_col="y_compound_encoded")
        df = df.loc[~df["y_compound_encoded"].isna()].copy()

        df = sample_races_stratified(df, rng=rng, max_races=args.max_races, sample_frac=args.sample_frac)

        # SVM RBF safety cap via race downsample (preserves strata)
        if args.model == "svm" and args.svm_max_rows_rbf and args.svm_max_rows_rbf > 0 and len(df) > args.svm_max_rows_rbf:
            avg_rows_per_race = len(df) / max(1, df["race_id"].nunique())
            approx_max_races = max(5, int(args.svm_max_rows_rbf / max(1.0, avg_rows_per_race)))
            approx_max_races = min(approx_max_races, df["race_id"].nunique())
            df = sample_races_stratified(df, rng=rng, max_races=approx_max_races, sample_frac=1.0)

        fold_bundle = make_race_group_folds(df, target_col="y_compound_encoded", n_splits=args.n_splits, seed=args.seed)
        folds = fold_bundle.folds

        X, y = get_stage2_xy(df)
        n_classes = int(len(COMPOUND_CLASSES))
        stage_tag = "stage2"
    else:
        df = load_stage1_dataset(args.data_stage1)
        df = df.loc[~df["race_id"].isin(HOLDOUT_RACE_IDS)].copy()

        df = sample_races_stratified(df, rng=rng, max_races=args.max_races, sample_frac=args.sample_frac)

        if args.model == "svm" and args.svm_max_rows_rbf and args.svm_max_rows_rbf > 0 and len(df) > args.svm_max_rows_rbf:
            avg_rows_per_race = len(df) / max(1, df["race_id"].nunique())
            approx_max_races = max(5, int(args.svm_max_rows_rbf / max(1.0, avg_rows_per_race)))
            approx_max_races = min(approx_max_races, df["race_id"].nunique())
            df = sample_races_stratified(df, rng=rng, max_races=approx_max_races, sample_frac=1.0)

        fold_bundle = make_race_group_folds(df, target_col="y_pit", n_splits=args.n_splits, seed=args.seed)
        folds = fold_bundle.folds

        X, y = get_stage1_xy(df)
        n_classes = None
        stage_tag = "stage1"

    # -------------------------
    # Cache folds for tabular models
    # -------------------------
    cached_folds: Optional[List[CachedFold]] = None
    if args.model in {"svm", "rf", "xgb", "vse_compound_ann"}:
        pre_name = "svm" if args.model in {"svm", "vse_compound_ann"} else args.model
        cached_folds = build_cached_folds(X, y, folds, pre_name=pre_name)

    # -------------------------
    # Optuna study
    # -------------------------
    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(n_startup_trials=15)
    study_name = f"{stage_tag}_{args.task}_{args.model}"
    storage_path = outdir / f"optuna_{study_name}.db"
    storage = f"sqlite:///{storage_path}"
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )

    timeout = args.timeout_sec if args.timeout_sec and args.timeout_sec > 0 else None

    # -------------------------
    # Run tuning + evaluate best on folds
    # -------------------------
    if args.model == "svm":
        assert cached_folds is not None
        study.optimize(
            lambda t: objective_svm(t, cached_folds, task=args.task, seed=args.seed, n_classes=n_classes),
            n_trials=args.n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )
        best_params = study.best_params
        fold_rows = cv_eval_svm_cached(cached_folds, params=best_params, task=args.task, n_classes=n_classes, seed=args.seed)

    elif args.model == "xgb":
        assert cached_folds is not None
        if not _HAS_XGB:
            raise ImportError("XGBoost not installed")
        study.optimize(
            lambda t: objective_xgb(
                t, cached_folds,
                task=args.task,
                seed=args.seed,
                n_classes=n_classes,
                early_stopping_rounds=args.early_stopping_rounds,
                n_jobs=args.n_jobs,
            ),
            n_trials=args.n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )
        best_params = study.best_params
        fold_rows = cv_eval_xgb_cached(
            cached_folds,
            params=best_params,
            task=args.task,
            n_classes=n_classes,
            seed=args.seed,
            early_stopping_rounds=args.early_stopping_rounds,
            n_jobs=args.n_jobs,
        )

    elif args.model == "rf":
        assert cached_folds is not None
        study.optimize(
            lambda t: objective_rf(t, cached_folds, task=args.task, seed=args.seed, n_classes=n_classes),
            n_trials=args.n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )
        best_params = study.best_params
        fold_rows = cv_eval_rf_cached(cached_folds, params=best_params, task=args.task, n_classes=n_classes, seed=args.seed)

    elif args.model == "vse_compound_ann":
        assert cached_folds is not None
        if not _HAS_TF:
            raise ImportError("TensorFlow/Keras not installed")
        if args.task != "multiclass" or not args.stage2:
            raise ValueError("vse_compound_ann is stage2 multiclass only")
        study.optimize(
            lambda t: objective_vse(t, cached_folds, seed=args.seed, n_classes=int(len(COMPOUND_CLASSES))),
            n_trials=args.n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )
        best_params = study.best_params
        fold_rows = cv_eval_vse_cached(cached_folds, params=best_params, n_classes=int(len(COMPOUND_CLASSES)), seed=args.seed)

    elif args.model == "tcn":
        if not _HAS_TF:
            raise ImportError("TensorFlow/Keras not installed")
        if args.task != "binary" or args.stage2:
            raise ValueError("TCN tuning is stage1 binary only")
        study.optimize(
            lambda t: objective_tcn(t, df, X, y, n_splits=args.n_splits, seed=args.seed),
            n_trials=args.n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )
        best_params = study.best_params

        # re-evaluate best across folds by running objective logic again but collecting fold rows
        # (keeps it consistent with the objective)
        fold_bundle = make_race_group_folds(df, target_col="y_pit", n_splits=args.n_splits, seed=args.seed)
        folds = fold_bundle.folds
        y_full = df["y_pit"].astype(int)

        fold_rows = []
        keras.utils.set_random_seed(args.seed)
        for fold, (tr_idx, va_idx) in enumerate(folds):
            X_seq_tr, y_seq_tr, X_seq_va, y_seq_va = _prep_sequences_for_fold(
                df=df, X=X, y_full=y_full,
                tr_idx=tr_idx, va_idx=va_idx,
                seq_len=int(best_params["seq_len"]), seed=args.seed,
            )

            model = build_tcn_binary_model(
                seq_len=int(best_params["seq_len"]),
                n_features=X_seq_tr.shape[2],
                filters=int(best_params["filters"]),
                kernel_size=int(best_params["kernel_size"]),
                n_blocks=int(best_params["n_blocks"]),
                pooling=str(best_params["pooling"]),
                dropout_residual=float(best_params["dropout_residual"]),
                dropout_head=float(best_params["dropout_head"]),
                dense_units=int(best_params["dense_units"]),
                learning_rate=float(best_params["learning_rate"]),
                focal_alpha=float(best_params["focal_alpha"]),
                focal_gamma=float(best_params["focal_gamma"]),
            )

            cb = keras.callbacks.EarlyStopping(
                monitor="val_pr_auc",
                mode="max",
                patience=int(best_params.get("patience", 6)),
                restore_best_weights=True,
            )
            model.fit(
                X_seq_tr, y_seq_tr,
                validation_data=(X_seq_va, y_seq_va),
                epochs=int(best_params.get("epochs", 60)),
                batch_size=int(best_params["batch_size"]),
                verbose=0,
                callbacks=[cb],
            )

            proba_pos = model.predict(X_seq_va, batch_size=int(best_params["batch_size"]), verbose=0).reshape(-1)
            m = compute_binary_metrics(y_seq_va, proba_pos, threshold=0.5)
            fold_rows.append({"fold": int(fold), "n_valid": int(len(y_seq_va)), **m})
            keras.backend.clear_session()

    elif args.model == "tcn_gru":
        if not _HAS_TF:
            raise ImportError("TensorFlow/Keras not installed")
        if args.task != "binary" or args.stage2:
            raise ValueError("tcn_gru tuning is stage1 binary only")
        study.optimize(
            lambda t: objective_tcn_gru(t, df, X, y, n_splits=args.n_splits, seed=args.seed),
            n_trials=args.n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )
        best_params = study.best_params

        # re-evaluate best across folds with fold rows
        fold_bundle = make_race_group_folds(df, target_col="y_pit", n_splits=args.n_splits, seed=args.seed)
        folds = fold_bundle.folds
        y_full = df["y_pit"].astype(int)

        fold_rows = []
        keras.utils.set_random_seed(args.seed)
        for fold, (tr_idx, va_idx) in enumerate(folds):
            X_seq_tr, y_seq_tr, X_seq_va, y_seq_va = _prep_sequences_for_fold(
                df=df, X=X, y_full=y_full,
                tr_idx=tr_idx, va_idx=va_idx,
                seq_len=int(best_params["seq_len"]), seed=args.seed,
            )

            # enforce pooling last
            if str(best_params.get("pooling", "last")) != "last":
                fold_rows.append({"fold": int(fold), "n_valid": int(len(y_seq_va)), "logloss": 1.0})
                continue

            model = build_tcn_gru_binary_model(
                seq_len=int(best_params["seq_len"]),
                n_features=X_seq_tr.shape[2],
                filters=int(best_params["filters"]),
                kernel_size=int(best_params["kernel_size"]),
                n_blocks=int(best_params["n_blocks"]),
                pooling="last",
                dropout_residual=float(best_params["dropout_residual"]),
                gru_units=int(best_params["gru_units"]),
                gru_dropout=float(best_params["gru_dropout"]),
                gru_l2=float(best_params["gru_l2"]),
                dense_units=int(best_params["dense_units"]),
                dropout_head=float(best_params["dropout_head"]),
                learning_rate=float(best_params["learning_rate"]),
                loss_type=str(best_params["loss_type"]),
                focal_alpha=float(best_params["focal_alpha"]) if best_params["loss_type"] == "focal" else None,
                focal_gamma=float(best_params["focal_gamma"]) if best_params["loss_type"] == "focal" else None,
            )

            cb = keras.callbacks.EarlyStopping(
                monitor="val_pr_auc",
                mode="max",
                patience=int(best_params.get("patience", 6)),
                restore_best_weights=True,
            )
            model.fit(
                X_seq_tr, y_seq_tr,
                validation_data=(X_seq_va, y_seq_va),
                epochs=int(best_params.get("epochs", 60)),
                batch_size=int(best_params["batch_size"]),
                verbose=0,
                callbacks=[cb],
            )

            proba_pos = model.predict(X_seq_va, batch_size=int(best_params["batch_size"]), verbose=0).reshape(-1)
            m = compute_binary_metrics(y_seq_va, proba_pos, threshold=0.5)
            fold_rows.append({"fold": int(fold), "n_valid": int(len(y_seq_va)), **m})
            keras.backend.clear_session()

    else:
        raise ValueError(f"Unsupported model: {args.model}")

    # -------------------------
    # Deliverables
    # -------------------------
    metric_keys = (
        ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "logloss"]
        if args.task == "binary"
        else ["accuracy", "precision_macro", "recall_macro", "f1_macro", "logloss"]
    )

    best_obj = {
        "study": study_name,
        "optimised_objective": "logloss",
        "best_value_logloss_mean": float(study.best_value),
        "best_params": best_params,
        "n_trials": int(len(study.trials)),
        "n_splits": int(args.n_splits),
        "holdout_race_ids_excluded": sorted(list(HOLDOUT_RACE_IDS)),
        "sample_frac": float(args.sample_frac),
        "max_races": int(args.max_races),
        "early_stopping_rounds": int(args.early_stopping_rounds),
    }

    summary = {
        "study": study_name,
        "task": args.task,
        "model": args.model,
        "metrics": mean_std_summary(fold_rows, metric_keys),
    }

    (outdir / f"{study_name}_best.json").write_text(json.dumps(best_obj, indent=2))
    (outdir / f"{study_name}_cv_folds.json").write_text(json.dumps(fold_rows, indent=2))
    (outdir / f"{study_name}_cv_summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(fold_rows).to_csv(outdir / f"{study_name}_cv_table.csv", index=False)
    study.trials_dataframe().to_csv(outdir / f"{study_name}_trials.csv", index=False)

    print(json.dumps(best_obj, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""
common vars to be set in terminal:
SEED=42
FOLDS=5
TIMEOUT=170000

STAGE1=data/processed/output1.csv
STAGE2=data/processed/output2.csv

RACES1=80
RACES2=80
"""