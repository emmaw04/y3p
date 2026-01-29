from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC

# Optional deps
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    from scikeras.wrappers import KerasClassifier
    from tensorflow import keras
    from tensorflow.keras import layers
    _HAS_KERAS = True
except Exception:
    _HAS_KERAS = False


@dataclass(frozen=True)
class ModelConfig:
    random_state: int = 42
    n_jobs: int = -1
    use_class_weight: bool = False

def build_model_pipeline(preprocessor, estimator: BaseEstimator) -> Pipeline:
    """Preprocess + model together to avoid CV leakage."""
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])

# -------------------------
# Meta learners (explicit)
# -------------------------

def make_meta_binary_lr(cfg: ModelConfig) -> BaseEstimator:
    return LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=4000,
        class_weight="balanced" if cfg.use_class_weight else None,
        random_state=cfg.random_state,
    )


def make_meta_multinomial_lr(cfg: ModelConfig) -> BaseEstimator:
    return LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=6000,
        multi_class="multinomial",  # softmax
        n_jobs=cfg.n_jobs,
        class_weight="balanced" if cfg.use_class_weight else None,
        random_state=cfg.random_state,
    )


#stage 1 new meta learners

from sklearn.neural_network import MLPClassifier

def make_meta_binary_xgb(cfg: ModelConfig) -> BaseEstimator:
    if not _HAS_XGB:
        raise ImportError("XGBoost not installed")
    return XGBClassifier(
        n_estimators=600,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_state,
    )

def make_meta_binary_mlp(cfg: ModelConfig) -> BaseEstimator:
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=200,
        early_stopping=True,
        random_state=cfg.random_state,
    )

#stage 2 meta learners

def make_meta_multiclass_xgb(cfg: ModelConfig, *, n_classes: int) -> BaseEstimator:
    if not _HAS_XGB:
        raise ImportError("XGBoost not installed")
    return XGBClassifier(
        n_estimators=800,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_state,
    )

def make_meta_multiclass_mlp(cfg: ModelConfig) -> BaseEstimator:
    return MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=250,
        early_stopping=True,
        random_state=cfg.random_state,
    )


# -------------------------
# SVM (binary + multiclass)
# -------------------------

def make_svm_binary(cfg: ModelConfig, *, kernel: str = "rbf") -> BaseEstimator:
    if kernel == "linear":
        base = LinearSVC(
            C=1.0,
            class_weight="balanced" if cfg.use_class_weight else None,
            random_state=cfg.random_state,
        )
        return CalibratedClassifierCV(base, method="sigmoid", cv=3)

    return SVC(
        C=2.0,
        kernel="rbf",
        gamma="scale",
        probability=True,
        class_weight="balanced" if cfg.use_class_weight else None,
        random_state=cfg.random_state,
    )


def make_svm_multiclass(cfg: ModelConfig, *, kernel: str = "rbf") -> BaseEstimator:
    if kernel == "linear":
        base = LinearSVC(
            C=1.0,
            class_weight="balanced" if cfg.use_class_weight else None,
            random_state=cfg.random_state,
        )
        # calibration gives predict_proba needed for stacking
        return CalibratedClassifierCV(base, method="sigmoid", cv=3)

    return SVC(
        C=2.0,
        kernel="rbf",
        gamma="scale",
        probability=True,
        decision_function_shape="ovr",
        class_weight="balanced" if cfg.use_class_weight else None,
        random_state=cfg.random_state,
    )


# -------------------------
# RF (binary + multiclass)
# -------------------------

def make_rf_binary(cfg: ModelConfig) -> BaseEstimator:
    return RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        max_features="sqrt",
        class_weight="balanced" if cfg.use_class_weight else None,
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_state,
    )


def make_rf_multiclass(cfg: ModelConfig) -> BaseEstimator:
    # Same estimator works; kept separate for clarity/config divergence later.
    return RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        max_features="sqrt",
        class_weight="balanced" if cfg.use_class_weight else None,
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_state,
    )


# -------------------------
# XGBoost (binary + multiclass)
# -------------------------

def make_xgb_binary(cfg: ModelConfig) -> BaseEstimator:
    if not _HAS_XGB:
        raise ImportError("XGBoost not installed. pip install xgboost")
    return XGBClassifier(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_state,
    )


def make_xgb_multiclass(cfg: ModelConfig, *, n_classes: int) -> BaseEstimator:
    if not _HAS_XGB:
        raise ImportError("XGBoost not installed. pip install xgboost")
    if n_classes < 3:
        raise ValueError("n_classes must be >= 3 for multiclass XGB")
    return XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_state,
    )

# -------------------------
# VSE-style ANN for compound choice (multiclass)
# -------------------------

def _build_keras_vse_compound_mlp(input_dim: int, n_classes: int = 5) -> "keras.Model":
    """
    VSE compound NN (paper: 1 hidden layer, 32 neurons, ReLU->softmax,
    L2=0.001, Nadam, sparse categorical crossentropy, batch size 32).
    Adapted to n_classes=5 for {HARD, MEDIUM, SOFT, WET, INTERMEDIATE} (or whatever
    order your COMPOUND_TO_INT uses).
    """
    reg = keras.regularizers.l2(0.001)

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation="relu", kernel_regularizer=reg),
        layers.Dense(n_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Nadam(),
        loss="sparse_categorical_crossentropy",
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    return model


def make_vse_compound_ann(cfg: ModelConfig, *, n_classes: int = 5) -> BaseEstimator:
    """
    Multiclass ANN for compound choice.

    Notes:
    - Expects y to be integer encoded 0..n_classes-1 (sparse categorical).
    - In VSE, they train on filtered pit-stop rows (your --stage2 path does this).
    """
    if not _HAS_KERAS:
        raise ImportError("SciKeras/TensorFlow not installed. pip install scikeras tensorflow")
    if n_classes < 2:
        raise ValueError("n_classes must be >= 2")

    def model_fn(meta):
        return _build_keras_vse_compound_mlp(meta["n_features_in_"], n_classes=n_classes)

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    return KerasClassifier(
        model=model_fn,
        epochs=200,            # paper trains/tunes; early stopping will halt earlier
        batch_size=32,         # matches VSE Table 14
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1,
        random_state=cfg.random_state,
    )

# -------------------------
# Generic ANN baselines (binary + multiclass)
# -------------------------

def _build_keras_mlp_binary(input_dim: int) -> "keras.Model":
    reg = keras.regularizers.l2(5e-4)
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu", kernel_regularizer=reg),
        layers.Dense(64, activation="relu", kernel_regularizer=reg),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


def make_ann_binary(cfg: ModelConfig) -> BaseEstimator:
    if not _HAS_KERAS:
        raise ImportError("SciKeras/TensorFlow not installed. pip install scikeras tensorflow(-macos)")

    def model_fn(meta):
        return _build_keras_mlp_binary(meta["n_features_in_"])

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    return KerasClassifier(
        model=model_fn,
        epochs=40,
        batch_size=256,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1,
        random_state=cfg.random_state,
    )


def _build_keras_mlp_multiclass(input_dim: int, n_classes: int) -> "keras.Model":
    reg = keras.regularizers.l2(5e-4)
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu", kernel_regularizer=reg),
        layers.Dense(64, activation="relu", kernel_regularizer=reg),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    return model


def make_ann_multiclass(cfg: ModelConfig, *, n_classes: int) -> BaseEstimator:
    if not _HAS_KERAS:
        raise ImportError("SciKeras/TensorFlow not installed. pip install scikeras tensorflow(-macos)")
    if n_classes < 2:
        raise ValueError("n_classes must be >= 2")

    def model_fn(meta):
        return _build_keras_mlp_multiclass(meta["n_features_in_"], n_classes=n_classes)

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    return KerasClassifier(
        model=model_fn,
        epochs=40,
        batch_size=256,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1,
        random_state=cfg.random_state,
    )


# -------------------------
# ANN + LSTM hybrid (VSE-style)
# -------------------------

def _build_keras_vse_ffnn(input_dim: int) -> "keras.Model":
    reg = keras.regularizers.l2(5e-4)  # 0.0005

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu", kernel_regularizer=reg),
        layers.Dense(64, activation="relu", kernel_regularizer=reg),
        layers.Dense(64, activation="relu", kernel_regularizer=reg),
        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=keras.optimizers.Nadam(),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


def make_ann_binary_vse_ffnn(cfg: ModelConfig) -> BaseEstimator:
    """
    VSE pit-stop FFNN.
    IMPORTANT: pass class weights at fit time:
      pipe.fit(X, y, model__class_weight={0: 1.0, 1: 5.0})
    """
    if not _HAS_KERAS:
        raise ImportError("SciKeras/TensorFlow not installed. pip install scikeras tensorflow")

    def model_fn(meta):
        return _build_keras_vse_ffnn(meta["n_features_in_"])

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    return KerasClassifier(
        model=model_fn,
        epochs=200,              # early stopping will stop sooner
        batch_size=256,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1,
        random_state=cfg.random_state,
    )


def _build_keras_lstm_head(seq_len: int) -> "keras.Model":
    """
    Consumes a sequence of FFNN outputs: shape (seq_len, 1).
    VSE says: "add one LSTM unit after original output" -> LSTM(units=1).
    """
    x_in = layers.Input(shape=(seq_len, 1))
    x = layers.LSTM(1)(x_in)
    y = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(x_in, y)
    model.compile(
        optimizer=keras.optimizers.Nadam(),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


def make_lstm_binary_head_vse(cfg: ModelConfig, *, seq_len: int = 4) -> BaseEstimator:
    if not _HAS_KERAS:
        raise ImportError("SciKeras/TensorFlow not installed. pip install scikeras tensorflow")

    def model_fn(meta):
        return _build_keras_lstm_head(seq_len=seq_len)

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    return KerasClassifier(
        model=model_fn,
        epochs=200,
        batch_size=256,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1,
        random_state=cfg.random_state,
    )


# -------------------------
# Registries (explicit)
# -------------------------

def get_binary_base_learners(cfg: ModelConfig) -> Dict[str, BaseEstimator]:
    learners: Dict[str, BaseEstimator] = {
        "svm": make_svm_binary(cfg),
        "rf": make_rf_binary(cfg),
    }
    if _HAS_XGB:
        learners["xgb"] = make_xgb_binary(cfg)
    if _HAS_KERAS:
        learners["ann"] = make_ann_binary(cfg)
    return learners

def get_multiclass_base_learners(cfg: ModelConfig, *, n_classes: int) -> Dict[str, BaseEstimator]:
    learners: Dict[str, BaseEstimator] = {
        "svm": make_svm_multiclass(cfg),
        "rf": make_rf_multiclass(cfg),
    }
    if _HAS_XGB:
        learners["xgb"] = make_xgb_multiclass(cfg, n_classes=n_classes)
    if _HAS_KERAS:
        learners["ann"] = make_ann_multiclass(cfg, n_classes=n_classes)
        learners["vse_compound_ann"] = make_vse_compound_ann(cfg, n_classes=n_classes)

    return learners




#-------------
#temporal convolutional network
#-------------
# src/models.py (add near your other Keras builders)

import numpy as np

def _tcn_residual_block(x, filters: int, kernel_size: int, dilation: int, dropout: float):
    """
    Basic TCN residual block:
    (Conv -> LN -> ReLU -> Dropout) x2 + residual
    """
    shortcut = x

    x = layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout)(x)

    # Match channels if needed
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding="same")(shortcut)

    x = layers.Add()([shortcut, x])
    return x


def _build_keras_tcn_binary(seq_len: int, n_features: int) -> "keras.Model":
    """
    TCN for pit/no-pit.
    Input: (seq_len, n_features)
    Output: p(pit=1 at last timestep)
    """
    x_in = layers.Input(shape=(seq_len, n_features))

    x = layers.Conv1D(64, 3, padding="causal", dilation_rate=1)(x_in)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)

    # Dilation stack: receptive field grows quickly
    for d in [1, 2, 4, 8]:
        x = _tcn_residual_block(x, filters=64, kernel_size=3, dilation=d, dropout=0.15)

    # Pool over time (or use last timestep)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    y_out = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(x_in, y_out)

    # Focal loss helps minority class learning without exploding class weights
    loss = keras.losses.BinaryFocalCrossentropy(gamma=2.0, alpha=0.75)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss,
        metrics=[
            keras.metrics.AUC(curve="PR", name="pr_auc"),
            keras.metrics.AUC(curve="ROC", name="roc_auc"),
        ],
    )
    return model


def make_tcn_binary(cfg: ModelConfig, *, seq_len: int = 8) -> BaseEstimator:
    """
    Binary TCN base learner.
    Expects X shaped (n_samples, seq_len, n_features) and y in {0,1}.
    """
    if not _HAS_KERAS:
        raise ImportError("SciKeras/TensorFlow not installed. pip install scikeras tensorflow")

    def model_fn(meta):
        # meta["X_shape_"] should be (n, seq_len, n_features)
        _, sl, nf = meta["X_shape_"]
        return _build_keras_tcn_binary(seq_len=int(sl), n_features=int(nf))

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_pr_auc",
        mode="max",
        patience=6,
        restore_best_weights=True,
    )

    return KerasClassifier(
        model=model_fn,
        epochs=80,
        batch_size=512,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1,
        random_state=cfg.random_state,
    )
