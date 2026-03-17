from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
import numpy as np
from sklearn.base import TransformerMixin

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


# -------------------------
# Config / helpers
# -------------------------

@dataclass(frozen=True)
class ModelConfig:
    random_state: int = 42
    n_jobs: int = -1
    # NOTE: Your tuned RF runs had class_weight=null.
    # Keep this False to match tuning; set True if you intentionally want balanced weighting.
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
# RF (binary + multiclass) — UPDATED with tuned params
# -------------------------

def make_rf_binary(cfg: ModelConfig) -> BaseEstimator:
    # Optuna best_params (stage1_binary_rf):
    # n_estimators=1832, max_depth=18, min_samples_leaf=18,
    # min_samples_split=13, max_features=0.5447035601509961,
    # bootstrap=True, class_weight=None
    return RandomForestClassifier(
        n_estimators=1832,
        max_depth=18,
        min_samples_leaf=18,
        min_samples_split=13,
        max_features=0.5447035601509961,
        bootstrap=True,
        class_weight="balanced" if cfg.use_class_weight else None,
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_state,
    )


def make_rf_multiclass(cfg: ModelConfig) -> BaseEstimator:
    # Optuna best_params (stage2_multiclass_rf):
    # n_estimators=482, max_depth=None, min_samples_leaf=2,
    # min_samples_split=41, max_features=0.13109109859596693,
    # bootstrap=True, class_weight=None
    return RandomForestClassifier(
        n_estimators=482,
        max_depth=None,
        min_samples_leaf=2,
        min_samples_split=41,
        max_features=0.13109109859596693,
        bootstrap=True,
        class_weight="balanced" if cfg.use_class_weight else None,
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_state,
    )


# -------------------------
# XGBoost (binary + multiclass) — UPDATED with tuned params
# -------------------------

def make_xgb_binary(cfg: ModelConfig) -> BaseEstimator:
    if not _HAS_XGB:
        raise ImportError("XGBoost not installed. pip install xgboost")

    # Optuna best_params (stage1_binary_xgb):
    # n_estimators=345, max_depth=8, learning_rate=0.12106896936002161,
    # subsample=0.6849356442713105, colsample_bytree=0.5090949803242604,
    # min_child_weight=2, reg_lambda=0.026892128247368887,
    # reg_alpha=2.6237821581611893, gamma=2.1597250932105787,
    # scale_pos_weight=26.646294743608884
    return XGBClassifier(
        n_estimators=345,
        max_depth=8,
        learning_rate=0.12106896936002161,
        subsample=0.6849356442713105,
        colsample_bytree=0.5090949803242604,
        min_child_weight=2,
        reg_lambda=0.026892128247368887,
        reg_alpha=2.6237821581611893,
        gamma=2.1597250932105787,
        scale_pos_weight=26.646294743608884,
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

    # Optuna best_params (stage2_multiclass_xgb):
    # n_estimators=4404, max_depth=7, learning_rate=0.20634912494467203,
    # subsample=0.9897570538738893, colsample_bytree=0.5151275191150386,
    # min_child_weight=2, reg_lambda=0.02591056758604487,
    # reg_alpha=2.2733958364357734, gamma=0.4938415319283539
    return XGBClassifier(
        n_estimators=4404,
        max_depth=7,
        learning_rate=0.20634912494467203,
        subsample=0.9897570538738893,
        colsample_bytree=0.5151275191150386,
        min_child_weight=2,
        reg_lambda=0.02591056758604487,
        reg_alpha=2.2733958364357734,
        gamma=0.4938415319283539,
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
        epochs=200,
        batch_size=32,
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
        learners["tcn"] = make_tcn_binary(cfg)  # expose seq model if you want it in ensembles
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


# -------------------------
# Temporal Convolutional Network (binary) — UPDATED with tuned params support
# -------------------------

def _tcn_residual_block(x, filters: int, kernel_size: int, dilation: int, dropout: float):
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


def _build_keras_tcn_binary(
    seq_len: int,
    n_features: int,
    *,
    filters: int = 64,
    kernel_size: int = 3,
    dropout: float = 0.10879485872574356,
    learning_rate: float = 0.00011662890273931399,
    pooling: str = "gap",
    use_focal: bool = True,
    focal_gamma: float = 1.0,
    focal_alpha: float = 0.5,
) -> "keras.Model":
    x_in = layers.Input(shape=(seq_len, n_features))

    x = layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=1)(x_in)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)

    for d in [1, 2, 4, 8]:
        x = _tcn_residual_block(x, filters=filters, kernel_size=kernel_size, dilation=d, dropout=dropout)

    if pooling == "gap":
        x = layers.GlobalAveragePooling1D()(x)
    elif pooling == "last":
        x = layers.Lambda(lambda t: t[:, -1, :])(x)
    else:
        raise ValueError(f"Unsupported pooling='{pooling}' (use 'gap' or 'last')")

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    y_out = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(x_in, y_out)

    if use_focal:
        # BinaryFocalCrossentropy exists in TF >= 2.12-ish. If your TF lacks it, switch use_focal=False.
        loss = keras.losses.BinaryFocalCrossentropy(gamma=focal_gamma, alpha=focal_alpha)
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


def make_tcn_binary(cfg: ModelConfig, *, seq_len: int = 12) -> BaseEstimator:
    """
    Binary TCN base learner.
    NOTE: seq_len must match however you're building (n_samples, seq_len, n_features).
    """
    if not _HAS_KERAS:
        raise ImportError("SciKeras/TensorFlow not installed. pip install scikeras tensorflow")

    def model_fn(meta):
        _, sl, nf = meta["X_shape_"]
        keras.utils.set_random_seed(cfg.random_state)
        return _build_keras_tcn_binary(
            seq_len=int(sl),
            n_features=int(nf),
            filters=64,
            kernel_size=3,
            dropout=0.10879485872574356,
            learning_rate=0.00011662890273931399,
            pooling="gap",
            use_focal=True,
            focal_gamma=1.0,
            focal_alpha=0.5,
        )

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


# -------------------------
# (Optional) Other seq models you already had (GRU/LSTM/TCN+GRU)
# -------------------------

def _compile_seq_binary(model: "keras.Model", *, learning_rate: float = 1e-3) -> "keras.Model":
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(curve="PR", name="pr_auc"),
            keras.metrics.AUC(curve="ROC", name="roc_auc"),
        ],
    )
    return model


def _build_keras_gru_binary(seq_len: int, n_features: int) -> "keras.Model":
    x_in = layers.Input(shape=(seq_len, n_features))
    x = layers.GRU(
        64, return_sequences=True,
        dropout=0.2, recurrent_dropout=0.0,
        kernel_regularizer=keras.regularizers.l2(5e-4),
    )(x_in)
    x = layers.GRU(
        32, return_sequences=True,
        dropout=0.2, recurrent_dropout=0.0,
        kernel_regularizer=keras.regularizers.l2(5e-4),
    )(x)
    x = layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(5e-4))(x)
    x = layers.Dropout(0.2)(x)
    y_out = layers.Dense(1, activation="sigmoid")(x)
    return _compile_seq_binary(keras.Model(x_in, y_out))


def _build_keras_lstm_binary(seq_len: int, n_features: int) -> "keras.Model":
    x_in = layers.Input(shape=(seq_len, n_features))
    x = layers.LSTM(
        64, return_sequences=True,
        dropout=0.2, recurrent_dropout=0.0,
        kernel_regularizer=keras.regularizers.l2(5e-4),
    )(x_in)
    x = layers.LSTM(
        32, return_sequences=True,
        dropout=0.2, recurrent_dropout=0.0,
        kernel_regularizer=keras.regularizers.l2(5e-4),
    )(x)
    x = layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(5e-4))(x)
    x = layers.Dropout(0.2)(x)
    y_out = layers.Dense(1, activation="sigmoid")(x)
    return _compile_seq_binary(keras.Model(x_in, y_out))


def _build_keras_tcn_gru_binary(seq_len: int, n_features: int) -> "keras.Model":
    x_in = layers.Input(shape=(seq_len, n_features))
    x = layers.Conv1D(64, 3, padding="causal", dilation_rate=1)(x_in)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)
    for d in [1, 2, 4, 8]:
        x = _tcn_residual_block(x, filters=64, kernel_size=3, dilation=d, dropout=0.15)
    x = layers.GRU(
        32, return_sequences=False,
        dropout=0.2, recurrent_dropout=0.0,
        kernel_regularizer=keras.regularizers.l2(5e-4),
    )(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(5e-4))(x)
    x = layers.Dropout(0.2)(x)
    y_out = layers.Dense(1, activation="sigmoid")(x)
    return _compile_seq_binary(keras.Model(x_in, y_out))


def _make_seq_classifier(cfg: ModelConfig, build_fn) -> BaseEstimator:
    if not _HAS_KERAS:
        raise ImportError("SciKeras/TensorFlow not installed. pip install scikeras tensorflow")

    def model_fn(meta):
        _, sl, nf = meta["X_shape_"]
        keras.utils.set_random_seed(cfg.random_state)
        return build_fn(int(sl), int(nf))

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


def make_gru_binary(cfg: ModelConfig) -> BaseEstimator:
    return _make_seq_classifier(cfg, _build_keras_gru_binary)


def make_lstm_binary(cfg: ModelConfig) -> BaseEstimator:
    return _make_seq_classifier(cfg, _build_keras_lstm_binary)


def make_tcn_gru_binary(cfg: ModelConfig) -> BaseEstimator:
    return _make_seq_classifier(cfg, _build_keras_tcn_gru_binary)

# ============================================================
# Multiclass sequence models (TCN / GRU / LSTM / TCN+GRU)
# Drop-in block for src/models/models.py
# Paste BELOW your existing binary seq models.
# ============================================================

def _compile_seq_multiclass(model: "keras.Model", *, learning_rate: float = 1e-3) -> "keras.Model":
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    return model


def _build_keras_gru_multiclass(seq_len: int, n_features: int, n_classes: int) -> "keras.Model":
    x_in = layers.Input(shape=(seq_len, n_features))
    x = layers.GRU(
        64, return_sequences=True,
        dropout=0.2, recurrent_dropout=0.0,
        kernel_regularizer=keras.regularizers.l2(5e-4),
    )(x_in)
    x = layers.GRU(
        32, return_sequences=True,
        dropout=0.2, recurrent_dropout=0.0,
        kernel_regularizer=keras.regularizers.l2(5e-4),
    )(x)
    x = layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(5e-4))(x)
    x = layers.Dropout(0.2)(x)
    y_out = layers.Dense(n_classes, activation="softmax")(x)
    return _compile_seq_multiclass(keras.Model(x_in, y_out), learning_rate=1e-3)


def _build_keras_lstm_multiclass(seq_len: int, n_features: int, n_classes: int) -> "keras.Model":
    x_in = layers.Input(shape=(seq_len, n_features))
    x = layers.LSTM(
        64, return_sequences=True,
        dropout=0.2, recurrent_dropout=0.0,
        kernel_regularizer=keras.regularizers.l2(5e-4),
    )(x_in)
    x = layers.LSTM(
        32, return_sequences=True,
        dropout=0.2, recurrent_dropout=0.0,
        kernel_regularizer=keras.regularizers.l2(5e-4),
    )(x)
    x = layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(5e-4))(x)
    x = layers.Dropout(0.2)(x)
    y_out = layers.Dense(n_classes, activation="softmax")(x)
    return _compile_seq_multiclass(keras.Model(x_in, y_out), learning_rate=1e-3)


def _build_keras_tcn_gru_multiclass(seq_len: int, n_features: int, n_classes: int) -> "keras.Model":
    # Uses your existing _tcn_residual_block()
    x_in = layers.Input(shape=(seq_len, n_features))
    x = layers.Conv1D(64, 3, padding="causal", dilation_rate=1)(x_in)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)
    for d in [1, 2, 4, 8]:
        x = _tcn_residual_block(x, filters=64, kernel_size=3, dilation=d, dropout=0.15)

    x = layers.GRU(
        32, return_sequences=False,
        dropout=0.2, recurrent_dropout=0.0,
        kernel_regularizer=keras.regularizers.l2(5e-4),
    )(x)

    x = layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(5e-4))(x)
    x = layers.Dropout(0.2)(x)
    y_out = layers.Dense(n_classes, activation="softmax")(x)
    return _compile_seq_multiclass(keras.Model(x_in, y_out), learning_rate=1e-3)


def _build_keras_tcn_multiclass(
    seq_len: int,
    n_features: int,
    n_classes: int,
    *,
    filters: int = 64,
    kernel_size: int = 3,
    dropout: float = 0.12,
    learning_rate: float = 1e-4,
    pooling: str = "gap",
) -> "keras.Model":
    # Uses your existing _tcn_residual_block()
    x_in = layers.Input(shape=(seq_len, n_features))

    x = layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=1)(x_in)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)

    for d in [1, 2, 4, 8]:
        x = _tcn_residual_block(x, filters=filters, kernel_size=kernel_size, dilation=d, dropout=dropout)

    if pooling == "gap":
        x = layers.GlobalAveragePooling1D()(x)
    elif pooling == "last":
        x = layers.Lambda(lambda t: t[:, -1, :])(x)
    else:
        raise ValueError(f"Unsupported pooling='{pooling}' (use 'gap' or 'last')")

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    y_out = layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model(x_in, y_out)
    return _compile_seq_multiclass(model, learning_rate=learning_rate)


def _make_seq_classifier_multiclass(cfg: ModelConfig, build_fn, *, n_classes: int) -> BaseEstimator:
    """
    SciKeras wrapper for multiclass sequence models.
    build_fn signature: (seq_len, n_features, n_classes) -> keras.Model
    """
    if not _HAS_KERAS:
        raise ImportError("SciKeras/TensorFlow not installed. pip install scikeras tensorflow")

    def model_fn(meta):
        # meta["X_shape_"] = (n_samples, seq_len, n_features)
        _, sl, nf = meta["X_shape_"]
        keras.utils.set_random_seed(cfg.random_state)
        return build_fn(int(sl), int(nf), int(n_classes))

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_acc",
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


def make_gru_multiclass(cfg: ModelConfig, *, n_classes: int) -> BaseEstimator:
    return _make_seq_classifier_multiclass(cfg, _build_keras_gru_multiclass, n_classes=n_classes)


def make_lstm_multiclass(cfg: ModelConfig, *, n_classes: int) -> BaseEstimator:
    return _make_seq_classifier_multiclass(cfg, _build_keras_lstm_multiclass, n_classes=n_classes)


def make_tcn_gru_multiclass(cfg: ModelConfig, *, n_classes: int) -> BaseEstimator:
    return _make_seq_classifier_multiclass(cfg, _build_keras_tcn_gru_multiclass, n_classes=n_classes)


def make_tcn_multiclass(cfg: ModelConfig, *, n_classes: int) -> BaseEstimator:
    """
    Multiclass TCN base learner.
    NOTE: seq_len is inferred from X_shape_ by SciKeras (same as your binary TCN).
    """
    if not _HAS_KERAS:
        raise ImportError("SciKeras/TensorFlow not installed. pip install scikeras tensorflow")

    def model_fn(meta):
        _, sl, nf = meta["X_shape_"]
        keras.utils.set_random_seed(cfg.random_state)
        return _build_keras_tcn_multiclass(
            seq_len=int(sl),
            n_features=int(nf),
            n_classes=int(n_classes),
            filters=64,
            kernel_size=3,
            dropout=0.12,
            learning_rate=1e-4,
            pooling="gap",
        )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_acc",
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