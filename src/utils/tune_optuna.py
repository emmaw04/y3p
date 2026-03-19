# src/tune_optuna.py
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
from sklearn.preprocessing import StandardScaler

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss, average_precision_score
)


from src.data.data import (
    infer_feature_types, COMPOUND_CLASSES,
    load_stage1_dataset, load_stage2_dataset,
    get_stage1_xy, get_stage2_xy,
    encode_y_compound,
    make_race_group_folds,  # FoldBundle(folds=[(tr_idx, va_idx), ...], fold_race_ids=[...])
    build_feature_sequences
)
from src.data.preprocessing import make_preprocessor_for_model
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV


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

def _race_level_strata(df: pd.DataFrame) -> pd.Series:
    """
    Return a race_id-indexed Series of strata labels like:
      "rain0_fcy0", "rain1_fcy0", "rain0_fcy1", "rain1_fcy1"
    Uses columns if present; otherwise falls back to 0.
    """
    rid = df["race_id"]

    def _has(col: str) -> bool:
        return col in df.columns

    # rain flag
    if _has("is_wet_race"):
        rain = df.groupby("race_id")["is_wet_race"].max().fillna(0).astype(int)
    elif _has("is_raining"):
        rain = df.groupby("race_id")["is_raining"].max().fillna(0).astype(int)
    elif _has("minutes_rain"):
        rain = (df.groupby("race_id")["minutes_rain"].max().fillna(0) > 0).astype(int)
    else:
        rain = pd.Series(0, index=df["race_id"].unique())

    # FCY flag
    if _has("fcy_status"):
        fcy = (df.groupby("race_id")["fcy_status"].max().fillna(0) > 0).astype(int)
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
    """
    Choose races (not rows), roughly preserving rain/FCY strata proportions.
    """
    race_ids = np.array(sorted(df["race_id"].unique()))
    if len(race_ids) == 0:
        return df

    strata = _race_level_strata(df)  # index=race_id
    # target count
    n_target = len(race_ids)
    if sample_frac < 1.0:
        n_target = max(4, int(np.ceil(sample_frac * len(race_ids))))
    if max_races and max_races > 0:
        n_target = min(n_target, int(max_races))
    n_target = min(n_target, len(race_ids))

    # stratified allocate
    by_stratum = {}
    for r in race_ids:
        s = strata.loc[r] if r in strata.index else "rain0_fcy0"
        by_stratum.setdefault(s, []).append(r)

    chosen = []
    # proportional allocation
    total = len(race_ids)
    for s, rs in by_stratum.items():
        k = int(np.round(n_target * (len(rs) / total)))
        k = max(1, min(k, len(rs)))  # at least 1 if stratum exists
        rs = np.array(rs)
        rng.shuffle(rs)
        chosen.extend(rs[:k].tolist())

    chosen = np.array(sorted(set(chosen)))
    # fix overshoot/undershoot
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
# Search spaces
# -------------------------
def suggest_rf_params(trial: optuna.Trial) -> Dict[str, Any]:
    max_depth_mode = trial.suggest_categorical("max_depth_mode", ["none", "bounded"])
    max_depth = None if max_depth_mode == "none" else trial.suggest_int("max_depth", 3, 40)

    max_features_mode = trial.suggest_categorical("max_features_mode", ["sqrt", "fraction"])
    max_features = "sqrt" if max_features_mode == "sqrt" else trial.suggest_float("max_features_frac", 0.1, 1.0)

    return {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
        "max_depth": max_depth,
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
        "max_features": max_features,
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        "max_samples": trial.suggest_float("max_samples", 0.5, 1.0),  # only if bootstrap True
    }


def suggest_xgb_params_common(trial: optuna.Trial, *, max_estimators: int) -> Dict[str, Any]:
    if not _HAS_XGB:
        raise ImportError("XGBoost not installed")

    # Tighten space to reduce “bad long” trials
    return {
        "n_estimators": trial.suggest_int("n_estimators", 300, max_estimators),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "tree_method": "hist",
        "max_delta_step": trial.suggest_int("max_delta_step", 0, 5),
    }

def suggest_ann_params(trial):
    return {
        "n_layers": trial.suggest_int("n_layers", 1, 3),
        "units": trial.suggest_categorical("units", [64, 128, 256]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "l2": trial.suggest_float("l2", 1e-6, 1e-3, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "epochs": 60,
        "patience": 6,
        "activation": trial.suggest_categorical("activation", ["relu", "elu"]),
    }

###tcn stuff

def _suggest_scale_pos_weight(trial: optuna.Trial, y_all: np.ndarray) -> float:
    n_pos = int(np.sum(y_all == 1))
    n_neg = int(np.sum(y_all == 0))
    ratio = (n_neg / n_pos) if n_pos > 0 else 1.0
    low = max(1.0, ratio / 2.0)
    high = min(50.0, ratio * 2.0)
    if low >= high:
        return float(min(50.0, max(1.0, ratio)))
    return float(trial.suggest_float("scale_pos_weight", low, high))


def make_group_folds_from_groups(groups: np.ndarray, n_splits: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    # deterministic shuffle of unique groups, then chunk
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)

    # simple split of races into folds
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    fold_races = np.array_split(uniq, n_splits)
    idx = np.arange(len(groups))
    for fr in fold_races:
        va_mask = np.isin(groups, fr)
        va_idx = idx[va_mask]
        tr_idx = idx[~va_mask]
        folds.append((tr_idx, va_idx))
    return folds

def build_tcn_model(
    *,
    seq_len: int,
    n_features: int,
    filters: int,
    kernel_size: int,
    dropout: float,
    learning_rate: float,
    pooling: str,
    focal_gamma: float,
    focal_alpha: float,
) -> "keras.Model":
    x_in = layers.Input(shape=(seq_len, n_features))

    x = layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=1)(x_in)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)

    def residual_block(x, dilation: int):
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

    for d in [1, 2, 4, 8]:
        x = residual_block(x, dilation=d)

    if pooling == "gap":
        x = layers.GlobalAveragePooling1D()(x)
    elif pooling == "last":
        x = layers.Lambda(lambda z: z[:, -1, :])(x)
    else:
        raise ValueError("pooling must be 'gap' or 'last'")

    x = layers.Dense(filters, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
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

def suggest_tcn_params(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "seq_len": trial.suggest_categorical("seq_len", [4, 6, 8, 10, 12]),
        "filters": trial.suggest_categorical("filters", [32, 64, 96]),
        "kernel_size": trial.suggest_categorical("kernel_size", [2, 3, 5]),
        "dropout": trial.suggest_float("dropout", 0.05, 0.35),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
        "focal_gamma": trial.suggest_categorical("focal_gamma", [1.0, 2.0, 3.0]),
        "focal_alpha": trial.suggest_categorical("focal_alpha", [0.5, 0.75]),
        "pooling": trial.suggest_categorical("pooling", ["gap", "last"]),
        # keep epochs fixed and rely on early stopping for speed
        "epochs": 60,
        "patience": 6,
    }

def suggest_svm_params(trial: optuna.Trial, *, kernel: str) -> Dict[str, Any]:
    if kernel == "linear":
        return {
            "kernel": "linear",
            "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
        }
    elif kernel == "rbf":
        return {
            "kernel": "rbf",
            "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 1e0, log=True),
        }
    else:
        raise ValueError("kernel must be 'linear' or 'rbf'")

CachedFold = Tuple[Any, np.ndarray, Any, np.ndarray, np.ndarray, np.ndarray]  # Xt_tr, y_tr, Xt_va, y_va, tr_idx, va_idx

def cv_eval_svm_cached(
    cached_folds: List[CachedFold],
    *,
    params: Dict[str, Any],
    task: str,
    threshold: float,
    n_classes: Optional[int],
    seed: int,
    use_class_weight: bool,
    calibrate_cv: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    kernel = params["kernel"]
    C = float(params["C"])

    for fold, (Xt_tr, y_tr, Xt_va, y_va, tr_idx, va_idx) in enumerate(cached_folds):
        if kernel == "linear":
            base = LinearSVC(
                C=C,
                class_weight="balanced" if use_class_weight else None,
                random_state=seed,
                dual="auto",
                max_iter=10000,
            )
            # Calibration is needed for predict_proba (stacking + logloss)
            model = CalibratedClassifierCV(base, method="sigmoid", cv=calibrate_cv)
        else:
            gamma = params["gamma"]
            # Option A (simple, potentially slower): probability=True
            # model = SVC(
            #     C=C, kernel="rbf", gamma=gamma,
            #     probability=True,
            #     class_weight="balanced" if use_class_weight else None,
            #     random_state=seed,
            # )

            # Option B (often faster + consistent): calibrate SVC without probability=True
            base = SVC(
                C=C, kernel="rbf", gamma=gamma,
                probability=False,
                class_weight="balanced" if use_class_weight else None,
                random_state=seed,
            )
            model = CalibratedClassifierCV(base, method="sigmoid", cv=calibrate_cv)

        model.fit(Xt_tr, y_tr)

        if task == "binary":
            proba_pos = model.predict_proba(Xt_va)[:, 1]
            m = compute_binary_metrics(y_va, proba_pos, threshold=threshold)
            rows.append({"fold": int(fold), "n_valid": int(len(va_idx)), **m})
        else:
            K = int(n_classes) if n_classes is not None else int(len(np.unique(y_tr)))
            proba = model.predict_proba(Xt_va)

            # Some calibrators can drop unseen classes; map safely
            classes_seen = model.classes_
            proba_full = np.zeros((len(va_idx), K), dtype=float)
            for j, cls in enumerate(classes_seen):
                proba_full[:, int(cls)] = proba[:, j]

            # normalise rows
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
    svm_kernel: str,
    use_class_weight: bool,
    calibrate_cv: int,
) -> float:
    params = suggest_svm_params(trial, kernel=svm_kernel)

    fold_rows = cv_eval_svm_cached(
        cached_folds,
        params=params,
        task=task,
        threshold=0.5,
        n_classes=n_classes,
        seed=seed,
        use_class_weight=use_class_weight,
        calibrate_cv=calibrate_cv,
    )

    losses = [r["logloss"] for r in fold_rows]
    for i in range(len(losses)):
        trial.report(float(np.mean(losses[: i + 1])), step=i)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(losses))


def cv_eval_tcn(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    *,
    params: Dict[str, Any],
    seed: int,
) -> List[Dict[str, Any]]:
    if not _HAS_TF:
        raise ImportError("TensorFlow/Keras not installed")

    rows: List[Dict[str, Any]] = []

    # Determinism-ish (TF still not perfectly deterministic)
    keras.utils.set_random_seed(seed)

    for fold, (tr_idx, va_idx) in enumerate(folds):
        X_tr, y_tr = X_seq[tr_idx], y_seq[tr_idx]
        X_va, y_va = X_seq[va_idx], y_seq[va_idx]

        # Standardize features using TRAIN ONLY, across all timesteps
        scaler = StandardScaler()
        X_tr_2d = X_tr.reshape(-1, X_tr.shape[-1])
        X_va_2d = X_va.reshape(-1, X_va.shape[-1])

        scaler.fit(X_tr_2d)
        X_tr = scaler.transform(X_tr_2d).reshape(X_tr.shape).astype(np.float32)
        X_va = scaler.transform(X_va_2d).reshape(X_va.shape).astype(np.float32)

        model = build_tcn_model(
            seq_len=X_tr.shape[1],
            n_features=X_tr.shape[2],
            filters=int(params["filters"]),
            kernel_size=int(params["kernel_size"]),
            dropout=float(params["dropout"]),
            learning_rate=float(params["learning_rate"]),
            pooling=str(params["pooling"]),
            focal_gamma=float(params["focal_gamma"]),
            focal_alpha=float(params["focal_alpha"]),
        )

        cb = keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=int(params["patience"]),
            restore_best_weights=True,
        )

        model.fit(
            X_tr, y_tr,
            validation_data=(X_va, y_va),
            epochs=int(params["epochs"]),
            batch_size=int(params["batch_size"]),
            verbose=0,
            callbacks=[cb],
        )

        proba_pos = model.predict(X_va, batch_size=int(params["batch_size"]), verbose=0).reshape(-1)
        m = compute_binary_metrics(y_va.astype(int), proba_pos.astype(float), threshold=0.5)

        rows.append({"fold": int(fold), "n_valid": int(len(va_idx)), **m})

        # free memory between folds
        keras.backend.clear_session()

    return rows

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

    # folds are on ROWS (not sequences) like everywhere else
    fold_bundle = make_race_group_folds(df, target_col="y_pit", n_splits=n_splits, seed=seed)
    folds = fold_bundle.folds

    # keys needed for build_feature_sequences (aligned with X/y row order)
    keys = df[["race_id", "driver_id", "lapno"]].copy()
    y_full = df["y_pit"].astype(int)

    # Tabular preprocessor for TCN: encodes categoricals -> numeric
    num_cols, cat_cols = infer_feature_types(X)
    base_pre = make_preprocessor_for_model("tcn", num_cols=num_cols, cat_cols=cat_cols)

    fold_losses: List[float] = []

    keras.utils.set_random_seed(seed)

    for fold, (tr_idx, va_idx) in enumerate(folds):
        # ---- Fit preprocessor on TRAIN ROWS ONLY ----
        X_tr = X.iloc[tr_idx]
        y_tr = y_full.iloc[tr_idx]

        pre = clone(base_pre)
        pre.fit(X_tr, y_tr)

        # Transform ALL rows using train-fitted preprocessor (so sequences can include early laps)
        Xt_all = pre.transform(X)
        if hasattr(Xt_all, "toarray"):
            Xt_all = Xt_all.toarray()
        Xt_all = Xt_all.astype(np.float32)

        # ---- Build sequences exactly like evaluate_stack ----
        X_seq, y_seq, idx_last, seq_idx, eff_len = build_feature_sequences(
            keys, Xt_all, y_full,
            seq_len=seq_len,
            pad_left=True,
            add_timestep_mask=True,   # keep consistent with your stacking TCN feature
        )

        # ---- Split sequences into train/val by membership of their underlying rows ----
        tr_mask = np.all((seq_idx == -1) | np.isin(seq_idx, tr_idx), axis=1)
        va_mask = np.all((seq_idx == -1) | np.isin(seq_idx, va_idx), axis=1)

        X_seq_tr, y_seq_tr = X_seq[tr_mask], y_seq[tr_mask]
        X_seq_va, y_seq_va = X_seq[va_mask], y_seq[va_mask]

        if len(y_seq_tr) == 0 or len(y_seq_va) == 0:
            # if a fold ends up empty (rare), penalise and continue
            fold_losses.append(1.0)
            continue

        # ---- Build model ----
        model = build_tcn_model(
            seq_len=seq_len,
            n_features=X_seq_tr.shape[2],
            filters=int(params["filters"]),
            kernel_size=int(params["kernel_size"]),
            dropout=float(params["dropout"]),
            learning_rate=float(params["learning_rate"]),
            pooling=str(params["pooling"]),
            focal_gamma=float(params["focal_gamma"]),
            focal_alpha=float(params["focal_alpha"]),
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
        m = compute_binary_metrics(y_seq_va.astype(int), proba_pos.astype(float), threshold=0.5)

        fold_losses.append(float(m["logloss"]))

        # pruning
        trial.report(float(np.mean(fold_losses)), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

        keras.backend.clear_session()

    return float(np.mean(fold_losses))



# -------------------------
# FAST: cache preprocessed fold matrices ONCE
# -------------------------

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

        # XGB likes CSR; RF can handle dense/sparse via sklearn
        if hasattr(Xt_tr, "tocsr"):
            Xt_tr = Xt_tr.tocsr()
            Xt_va = Xt_va.tocsr()

        cached.append((Xt_tr, y_tr, Xt_va, y_va, tr_idx, va_idx))

    return cached


# -------------------------
# CV evaluation on cached folds
# -------------------------
def cv_eval_rf_cached(
    cached_folds: List[CachedFold],
    *,
    params: Dict[str, Any],
    task: str,
    threshold: float,
    n_classes: Optional[int],
    seed: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fold, (Xt_tr, y_tr, Xt_va, y_va, tr_idx, va_idx) in enumerate(cached_folds):
        rf = RandomForestClassifier(**params, random_state=seed, n_jobs=-1)
        rf.fit(Xt_tr, y_tr)

        proba = rf.predict_proba(Xt_va)

        if task == "binary":
            m = compute_binary_metrics(y_va, proba[:, 1], threshold=threshold)
            rows.append({"fold": int(fold), "n_valid": int(len(va_idx)), **m})
        else:
            K = int(n_classes) if n_classes is not None else int(len(np.unique(y_tr)))
            classes_seen = rf.classes_
            proba_full = np.zeros((len(va_idx), K), dtype=float)
            for j, cls in enumerate(classes_seen):
                proba_full[:, int(cls)] = proba[:, j]

            # ensure rows sum to 1 (prevents sklearn log_loss warnings when a fold lacks a class)
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


def cv_eval_xgb_cached(
    cached_folds: List[CachedFold],
    *,
    params_common: Dict[str, Any],
    task: str,
    threshold: float,
    n_classes: Optional[int],
    seed: int,
    early_stopping_rounds: int,
    n_jobs: int,
) -> List[Dict[str, Any]]:
    if not _HAS_XGB:
        raise ImportError("XGBoost not installed")

    rows: List[Dict[str, Any]] = []

    # Put callbacks in CONSTRUCTOR (avoids the deprecated fit(callbacks=...) warning)
    callbacks = [xgb.callback.EarlyStopping(rounds=early_stopping_rounds, save_best=False)] if early_stopping_rounds > 0 else None


    for fold, (Xt_tr, y_tr, Xt_va, y_va, tr_idx, va_idx) in enumerate(cached_folds):
        if task == "binary":
            spw = params_common.get("scale_pos_weight", 1.0)
            model = XGBClassifier(
                **{k: v for k, v in params_common.items() if k != "scale_pos_weight"},
                scale_pos_weight=spw,
                objective="binary:logistic",
                eval_metric="logloss",
                n_jobs=n_jobs,
                random_state=seed,
                callbacks=callbacks,
            )
            model.fit(Xt_tr, y_tr, eval_set=[(Xt_va, y_va)], verbose=False)
            proba_pos = model.predict_proba(Xt_va)[:, 1]
            m = compute_binary_metrics(y_va, proba_pos, threshold=threshold)
            bi = getattr(model, "best_iteration", None)
            rows.append({"fold": int(fold), "n_valid": int(len(va_idx)), "best_iteration": int(bi) if bi is not None else -1, **m})
        else:
            K = int(n_classes) if n_classes is not None else int(len(np.unique(y_tr)))
            model = XGBClassifier(
                **params_common,
                objective="multi:softprob",
                num_class=K,
                eval_metric="mlogloss",
                n_jobs=n_jobs,
                random_state=seed,
                callbacks=callbacks,
            )
            model.fit(Xt_tr, y_tr, eval_set=[(Xt_va, y_va)], verbose=False)
            proba = model.predict_proba(Xt_va)
            m = compute_multiclass_metrics(y_va, proba)
            bi = getattr(model, "best_iteration", None)
            rows.append({"fold": int(fold), "n_valid": int(len(va_idx)), "best_iteration": int(bi) if bi is not None else -1, **m})

    return rows


# -------------------------
# Optuna objectives (cached folds)
# -------------------------
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
        threshold=0.5,
        n_classes=n_classes,
        seed=seed,
    )
    losses = [r["logloss"] for r in fold_rows]
    for i in range(len(losses)):
        trial.report(float(np.mean(losses[: i + 1])), step=i)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(losses))


def objective_xgb(
    trial: optuna.Trial,
    cached_folds: List[CachedFold],
    *,
    task: str,
    seed: int,
    n_classes: Optional[int],
    early_stopping_rounds: int,
    n_jobs: int,
    max_estimators: int,
    y_all_for_spw: Optional[np.ndarray],
) -> float:
    common = suggest_xgb_params_common(trial, max_estimators=max_estimators)

    if task == "binary":
        # stable single spw per trial
        common["scale_pos_weight"] = _suggest_scale_pos_weight(trial, y_all_for_spw)

    fold_rows = cv_eval_xgb_cached(
        cached_folds,
        params_common=common,
        task=task,
        threshold=0.5,
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
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_stage1", required=False)
    ap.add_argument("--data_stage2", required=False)
    ap.add_argument("--stage2", action="store_true")
    ap.add_argument("--task", choices=["binary", "multiclass"], required=True)
    ap.add_argument("--model", choices=["rf", "xgb", "svm", "ann", "tcn", "gru", "lstm", "tcn_gru"], required=True)
    ap.add_argument("--svm_kernel", choices=["linear", "rbf"], default="linear")
    ap.add_argument("--svm_calibrate_cv", type=int, default=3)
    ap.add_argument("--svm_max_rows_rbf", type=int, default=20000, help="Hard cap rows if svm_kernel=rbf (0 disables).")

    ap.add_argument("--n_trials", type=int, default=100)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="runs/tuning")

    # SPEED CONTROLS
    ap.add_argument("--sample_frac", type=float, default=1.0, help="Subsample rows for tuning (e.g. 0.25).")
    ap.add_argument("--max_races", type=int, default=0, help="Cap number of races used for tuning (0 = no cap).")
    ap.add_argument("--early_stopping_rounds", type=int, default=20, help="XGB early stopping rounds (0 disables).")
    ap.add_argument("--max_estimators", type=int, default=2500, help="Upper bound for XGB n_estimators search.")
    ap.add_argument("--timeout_sec", type=int, default=0, help="Stop Optuna after N seconds (0 disables).")
    ap.add_argument("--n_jobs", type=int, default=-1, help="XGB n_jobs.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # -------------------------
    # Load + prepare dataset + folds
    # -------------------------
    if args.stage2:
        if args.task != "multiclass":
            raise ValueError("--stage2 requires --task multiclass")
        if not args.data_stage2:
            raise ValueError("Pass --data_stage2")

        df = load_stage2_dataset(args.data_stage2, strict=True)
        df = df.loc[~df["race_id"].isin(HOLDOUT_RACE_IDS)].copy()

        df = encode_y_compound(df, col="y_compound", out_col="y_compound_encoded")
        df = df.loc[~df["y_compound_encoded"].isna()].copy()

        df = sample_races_stratified(df, rng=rng, max_races=args.max_races, sample_frac=args.sample_frac)
        # --- Extra safety for SVM RBF: cap via races, not rows (preserves stratification) ---
        if args.model == "svm" and args.svm_kernel == "rbf" and args.svm_max_rows_rbf and args.svm_max_rows_rbf > 0:
            if len(df) > args.svm_max_rows_rbf:
                avg_rows_per_race = len(df) / max(1, df["race_id"].nunique())
                approx_max_races = max(5, int(args.svm_max_rows_rbf / max(1.0, avg_rows_per_race)))
                approx_max_races = min(approx_max_races, df["race_id"].nunique())

                df = sample_races_stratified(df, rng=rng, max_races=approx_max_races, sample_frac=1.0)

        fold_bundle = make_race_group_folds(df, target_col="y_compound_encoded", n_splits=args.n_splits, seed=args.seed)
        folds = fold_bundle.folds

        X, y = get_stage2_xy(df)
        n_classes = len(COMPOUND_CLASSES)
        stage_tag = "stage2"

        y_all_for_spw = None  # not used

    else:
        if not args.data_stage1:
            raise ValueError("Pass --data_stage1")

        df = load_stage1_dataset(args.data_stage1)
        df = df.loc[~df["race_id"].isin(HOLDOUT_RACE_IDS)].copy()

        df = sample_races_stratified(df, rng=rng, max_races=args.max_races, sample_frac=args.sample_frac)
        # --- Extra safety for SVM RBF: cap via races, not rows (preserves stratification) ---
        if args.model == "svm" and args.svm_kernel == "rbf" and args.svm_max_rows_rbf and args.svm_max_rows_rbf > 0:
            if len(df) > args.svm_max_rows_rbf:
                avg_rows_per_race = len(df) / max(1, df["race_id"].nunique())
                approx_max_races = max(5, int(args.svm_max_rows_rbf / max(1.0, avg_rows_per_race)))
                approx_max_races = min(approx_max_races, df["race_id"].nunique())

                df = sample_races_stratified(df, rng=rng, max_races=approx_max_races, sample_frac=1.0)

        fold_bundle = make_race_group_folds(df, target_col="y_pit", n_splits=args.n_splits, seed=args.seed)
        folds = fold_bundle.folds

        X, y = get_stage1_xy(df)
        n_classes = None
        stage_tag = "stage1"
        y_all_for_spw = y.to_numpy()

    # -------------------------
    # Cache folds ONCE (major speedup)
    # -------------------------
    pre_name = args.model  # routes inside make_preprocessor_for_model
    cached_folds = None
    if args.model in {"rf", "xgb", "svm", "ann"}:
        pre_name = "svm" if args.model in {"svm", "ann"} else args.model
        cached_folds = build_cached_folds(X, y, folds, pre_name=pre_name)


    # -------------------------
    # Optuna study
    # -------------------------
    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(n_startup_trials=15)
    study_name = (
        f"{stage_tag}_{args.task}_{args.model}_{args.svm_kernel}"
        if args.model == "svm"
        else f"{stage_tag}_{args.task}_{args.model}"
    )
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner, study_name=study_name)

    timeout = args.timeout_sec if args.timeout_sec and args.timeout_sec > 0 else None

    # -------------------------
    # TCN branch (Stage 1 binary only)
    # -------------------------
    if args.model == "tcn":
        if args.stage2 or args.task != "binary":
            raise ValueError("TCN tuning currently implemented for Stage 1 binary only.")
        if not _HAS_TF:
            raise ImportError("TensorFlow/Keras not installed.")

        study.optimize(
            lambda t: objective_tcn(
                t,
                df, X, y,
                n_splits=args.n_splits,
                seed=args.seed,
            ),
            n_trials=args.n_trials,
            timeout=timeout,
            show_progress_bar=True,
        )

        best_params = study.best_params

        # Re-evaluate with best params to produce fold table for saving
        # Re-evaluate with best params using the SAME logic as objective_tcn / evaluate_stack
        seq_len = int(best_params["seq_len"])
        batch_size = int(best_params.get("batch_size", 512))
        patience = int(best_params.get("patience", 6))
        epochs = int(best_params.get("epochs", 60))
        fold_bundle = make_race_group_folds(df, target_col="y_pit", n_splits=args.n_splits, seed=args.seed)
        folds = fold_bundle.folds

        keys = df[["race_id", "driver_id", "lapno"]].copy()
        y_full = df["y_pit"].astype(int)

        num_cols, cat_cols = infer_feature_types(X)
        base_pre = make_preprocessor_for_model("tcn", num_cols=num_cols, cat_cols=cat_cols)

        fold_rows = []
        keras.utils.set_random_seed(args.seed)

        for fold, (tr_idx, va_idx) in enumerate(folds):
            # fit preprocessor on train rows only
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

            if len(y_seq_tr) == 0 or len(y_seq_va) == 0:
                fold_rows.append({"fold": int(fold), "n_valid": int(len(y_seq_va)), "logloss": 1.0})
                continue

            model = build_tcn_model(
                seq_len=seq_len,
                n_features=X_seq_tr.shape[2],
                filters=int(best_params["filters"]),
                kernel_size=int(best_params["kernel_size"]),
                dropout=float(best_params["dropout"]),
                learning_rate=float(best_params["learning_rate"]),
                pooling=str(best_params["pooling"]),
                focal_gamma=float(best_params["focal_gamma"]),
                focal_alpha=float(best_params["focal_alpha"]),
            )

            cb = keras.callbacks.EarlyStopping(
                monitor="val_pr_auc",
                mode="max",
                patience=patience,
                restore_best_weights=True,
            )

            model.fit(
                X_seq_tr, y_seq_tr,
                validation_data=(X_seq_va, y_seq_va),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                callbacks=[cb],
            )

            proba_pos = model.predict(X_seq_va, batch_size=batch_size, verbose=0).reshape(-1)

            m = compute_binary_metrics(y_seq_va.astype(int), proba_pos.astype(float), threshold=0.5)
            fold_rows.append({"fold": int(fold), "n_valid": int(len(y_seq_va)), **m})

            keras.backend.clear_session()


    else:

        if args.model == "rf":
            study.optimize(
                lambda t: objective_rf(
                    t, cached_folds,
                    task=args.task,
                    seed=args.seed,
                    n_classes=n_classes,
                ),
                n_trials=args.n_trials,
                timeout=timeout,
                show_progress_bar=True,
            )
            raw_best_params = study.best_params

            def decode_rf(best: Dict[str, Any]) -> Dict[str, Any]:
                max_depth = None if best.get("max_depth_mode") == "none" else int(best["max_depth"])
                max_features = "sqrt" if best.get("max_features_mode") == "sqrt" else float(best["max_features_frac"])
                return {
                    "n_estimators": int(best["n_estimators"]),
                    "max_depth": max_depth,
                    "min_samples_leaf": int(best["min_samples_leaf"]),
                    "min_samples_split": int(best["min_samples_split"]),
                    "max_features": max_features,
                    "bootstrap": bool(best["bootstrap"]),
                    "class_weight": best["class_weight"],
                }

            best_params = decode_rf(raw_best_params)

            fold_rows = cv_eval_rf_cached(
                cached_folds,
                params=best_params,
                task=args.task,
                threshold=0.5,
                n_classes=n_classes,
                seed=args.seed,
            )
        elif args.model == "svm":
            study.optimize(
                lambda t: objective_svm(
                    t, cached_folds,
                    task=args.task,
                    seed=args.seed,
                    n_classes=n_classes,
                    svm_kernel=args.svm_kernel,
                    use_class_weight=True,  # or cfg.use_class_weight if you pass it through
                    calibrate_cv=args.svm_calibrate_cv,
                ),
                n_trials=args.n_trials,
                timeout=timeout,
                show_progress_bar=True,
            )

            best_params = {"kernel": args.svm_kernel, **study.best_params}

            fold_rows = cv_eval_svm_cached(
                cached_folds,
                params=best_params,
                task=args.task,
                threshold=0.5,
                n_classes=n_classes,
                seed=args.seed,
                use_class_weight=True,
                calibrate_cv=args.svm_calibrate_cv,
            )

        else:
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
                    max_estimators=args.max_estimators,
                    y_all_for_spw=y_all_for_spw,
                ),
                n_trials=args.n_trials,
                timeout=timeout,
                show_progress_bar=True,
            )

            best_params = study.best_params

            fold_rows = cv_eval_xgb_cached(
                cached_folds,
                params_common=best_params,
                task=args.task,
                threshold=0.5,
                n_classes=n_classes,
                seed=args.seed,
                early_stopping_rounds=args.early_stopping_rounds,
                n_jobs=args.n_jobs,
            )

    # -------------------------
    # Deliverables
    # -------------------------
    if args.task == "binary":
        metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "logloss"]
    else:
        metric_keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "logloss"]

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
commands to run:
# shared knobs
DATA=data/processed/output1.csv
OUT=runs/tuning_coarse
SEED=42
RACES=60
FOLDS=3
TRIALS=200
TIME=1200   # seconds

python -m src.tune_optuna --data_stage1 $DATA --task binary --model rf  --n_splits $FOLDS --max_races $RACES --n_trials $TRIALS --timeout_sec $TIME --seed $SEED --outdir $OUT
python -m src.tune_optuna --data_stage1 $DATA --task binary --model xgb --n_splits $FOLDS --max_races $RACES --n_trials $TRIALS --timeout_sec $TIME --seed $SEED --outdir $OUT --early_stopping_rounds 20 --max_estimators 1500
python -m src.tune_optuna --data_stage1 $DATA --task binary --model svm --svm_kernel linear --n_splits $FOLDS --max_races $RACES --n_trials $TRIALS --timeout_sec $TIME --seed $SEED --outdir $OUT
python -m src.tune_optuna --data_stage1 $DATA --task binary --model ann --n_splits $FOLDS --max_races $RACES --n_trials $TRIALS --timeout_sec $TIME --seed $SEED --outdir $OUT

python -m src.tune_optuna --data_stage1 $DATA --task binary --model tcn      --n_splits $FOLDS --max_races $RACES --n_trials $TRIALS --timeout_sec $TIME --seed $SEED --outdir $OUT
python -m src.tune_optuna --data_stage1 $DATA --task binary --model gru      --n_splits $FOLDS --max_races $RACES --n_trials $TRIALS --timeout_sec $TIME --seed $SEED --outdir $OUT
python -m src.tune_optuna --data_stage1 $DATA --task binary --model lstm     --n_splits $FOLDS --max_races $RACES --n_trials $TRIALS --timeout_sec $TIME --seed $SEED --outdir $OUT
python -m src.tune_optuna --data_stage1 $DATA --task binary --model tcn_gru  --n_splits $FOLDS --max_races $RACES --n_trials $TRIALS --timeout_sec $TIME --seed $SEED --outdir $OUT
"""