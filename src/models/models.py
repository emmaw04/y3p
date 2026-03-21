from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from xgboost import XGBClassifier
from scikeras.wrappers import KerasClassifier
from tensorflow import keras
from tensorflow.keras import layers


# shared bits

@dataclass(frozen=True)
class ModelConfig:
    """small config object used across the model factories"""

    random_state: int = 42
    n_jobs: int = -1
    use_class_weight: bool = False


def build_model_pipeline(preprocessor, estimator: BaseEstimator) -> Pipeline:
    """wraps preprocessing and the estimator together so each fold fits both from scratch"""

    return Pipeline([
        ("preprocess", preprocessor),
        ("model", estimator),
    ])


def _make_keras_classifier(
    cfg: ModelConfig,
    model_fn: Callable,
    *,
    epochs: int,
    batch_size: int,
    monitor: str,
    mode: str,
    patience: int,
) -> BaseEstimator:
    """shared scikeras wrapper so the keras factories stay short and readable"""

    early_stop = keras.callbacks.EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=patience,
        restore_best_weights=True,
    )

    return KerasClassifier(
        model=model_fn,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1,
        random_state=cfg.random_state,
    )


def _compile_binary_seq_model(
    model: "keras.Model",
    *,
    learning_rate: float = 1e-3,
    use_focal: bool = False,
    focal_gamma: float = 1.0,
    focal_alpha: float = 0.5,
) -> "keras.Model":
    """compiles a binary sequence model with the metrics used in stage 1"""

    if use_focal:
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


def _compile_multiclass_seq_model(
    model: "keras.Model",
    *,
    learning_rate: float = 1e-3,
) -> "keras.Model":
    """compiles a multiclass sequence model with the metric used in stage 2"""

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    return model


def _make_binary_seq_classifier(cfg: ModelConfig, build_fn: Callable) -> BaseEstimator:
    """shared wrapper for the stage 1 sequence models"""

    def model_fn(meta):
        _, seq_len, n_features = meta["X_shape_"]
        keras.utils.set_random_seed(cfg.random_state)
        return build_fn(int(seq_len), int(n_features))

    return _make_keras_classifier(
        cfg,
        model_fn,
        epochs=80,
        batch_size=512,
        monitor="val_pr_auc",
        mode="max",
        patience=6,
    )


def _make_multiclass_seq_classifier(
    cfg: ModelConfig,
    build_fn: Callable,
    *,
    n_classes: int,
) -> BaseEstimator:
    """shared wrapper for the stage 2 sequence models"""

    def model_fn(meta):
        _, seq_len, n_features = meta["X_shape_"]
        keras.utils.set_random_seed(cfg.random_state)
        return build_fn(int(seq_len), int(n_features), int(n_classes))

    return _make_keras_classifier(
        cfg,
        model_fn,
        epochs=80,
        batch_size=512,
        monitor="val_acc",
        mode="max",
        patience=6,
    )


def _tcn_residual_block(x, *, filters: int, kernel_size: int, dilation: int, dropout: float):
    """basic residual block used by the tcn models"""

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

# stage 1 tabular

def make_stage1_rf(cfg: ModelConfig) -> BaseEstimator:
    """returns the tuned random forest for stage 1

    this is the main tree ensemble for lap level pit stop prediction
    """

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


def make_stage1_xgb(cfg: ModelConfig) -> BaseEstimator:
    """returns the tuned xgboost model for stage 1

    this is the boosted tree baseline for binary pit stop prediction
    """

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


def make_stage1_svm(cfg: ModelConfig) -> BaseEstimator:
    """returns the stage 1 svm

    this is the tabular margin based classifier with probability output enabled
    """

    return SVC(
        C=2.0,
        kernel="rbf",
        gamma="scale",
        probability=True,
        class_weight="balanced" if cfg.use_class_weight else None,
        random_state=cfg.random_state,
    )


def get_stage1_tabular_models(cfg: ModelConfig) -> Dict[str, BaseEstimator]:
    """registry for the stage 1 tabular models"""

    return {
        "rf": make_stage1_rf(cfg),
        "xgb": make_stage1_xgb(cfg),
        "svm": make_stage1_svm(cfg),
    }

# stage 1 sequential

def _build_tcn_binary(
    seq_len: int,
    n_features: int,
    filters: int = 64,
    kernel_size: int = 3,
    dropout: float = 0.10879485872574356,
    pooling: str = "gap",
    learning_rate: float = 0.00011662890273931399,
    focal_gamma: float = 1.0,
    focal_alpha: float = 0.5,
) -> "keras.Model":
    """builds the binary tcn

    this is a causal conv front end followed by residual tcn blocks
    then global pooling and a small dense head
    """

    x_in = layers.Input(shape=(seq_len, n_features))

    x = layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=1)(x_in)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)

    for d in [1, 2, 4, 8]:
        x = _tcn_residual_block(
            x,
            filters=filters,
            kernel_size=kernel_size,
            dilation=d,
            dropout=dropout,
        )

    if pooling == "gap":
        x = layers.GlobalAveragePooling1D()(x)
    elif pooling == "last":
        x = layers.Lambda(lambda z: z[:, -1, :])(x)
        
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    y_out = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(x_in, y_out)
    return _compile_binary_seq_model(
        model,
        learning_rate=learning_rate,
        use_focal=True,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
    )


def _build_tcn_gru_binary(
    seq_len: int,
    n_features: int,
    filters: int = 64,
    kernel_size: int = 3,
    dropout: float = 0.15,
    rnn_units: int = 32,
    rnn_dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> "keras.Model":
    """builds the binary tcn gru model

    this starts with tcn style conv blocks then hands the sequence to a gru head
    """

    x_in = layers.Input(shape=(seq_len, n_features))

    x = layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=1)(x_in)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)

    for d in [1, 2, 4, 8]:
        x = _tcn_residual_block(
            x,
            filters=filters,
            kernel_size=kernel_size,
            dilation=d,
            dropout=dropout,
        )

    x = layers.GRU(
        rnn_units,
        return_sequences=False,
        dropout=rnn_dropout,
        recurrent_dropout=0.0,
        kernel_regularizer=keras.regularizers.l2(5e-4),
    )(x)

    x = layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(5e-4))(x)
    x = layers.Dropout(0.2)(x)
    y_out = layers.Dense(1, activation="sigmoid")(x)

    return _compile_binary_seq_model(keras.Model(x_in, y_out), learning_rate=learning_rate)


def _build_lstm_binary(
    seq_len: int,
    n_features: int,
    rnn_units: int = 64,
    rnn_dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> "keras.Model":
    """builds the binary lstm

    this is a stacked lstm with pooling and a small dense output head
    """

    x_in = layers.Input(shape=(seq_len, n_features))

    x = layers.LSTM(
        rnn_units,
        return_sequences=True,
        dropout=rnn_dropout,
        recurrent_dropout=0.0,
        kernel_regularizer=keras.regularizers.l2(5e-4),
    )(x_in)
    x = layers.LSTM(
        rnn_units // 2,
        return_sequences=True,
        dropout=rnn_dropout,
        recurrent_dropout=0.0,
        kernel_regularizer=keras.regularizers.l2(5e-4),
    )(x)

    x = layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(5e-4))(x)
    x = layers.Dropout(0.2)(x)
    y_out = layers.Dense(1, activation="sigmoid")(x)

    return _compile_binary_seq_model(keras.Model(x_in, y_out), learning_rate=learning_rate)


def _build_gru_binary(
    seq_len: int,
    n_features: int,
    rnn_units: int = 64,
    rnn_dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> "keras.Model":
    """builds the binary gru

    this is the same general idea as the lstm model but using gru layers
    """

    x_in = layers.Input(shape=(seq_len, n_features))

    x = layers.GRU(
        rnn_units,
        return_sequences=True,
        dropout=rnn_dropout,
        recurrent_dropout=0.0,
        kernel_regularizer=keras.regularizers.l2(5e-4),
    )(x_in)
    x = layers.GRU(
        rnn_units // 2,
        return_sequences=True,
        dropout=rnn_dropout,
        recurrent_dropout=0.0,
        kernel_regularizer=keras.regularizers.l2(5e-4),
    )(x)

    x = layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(5e-4))(x)
    x = layers.Dropout(0.2)(x)
    y_out = layers.Dense(1, activation="sigmoid")(x)

    return _compile_binary_seq_model(keras.Model(x_in, y_out), learning_rate=learning_rate)


def make_tcn_binary(cfg: ModelConfig, *, seq_len: int = 12) -> BaseEstimator:
    """returns the stage 1 tcn classifier

    the seq_len argument is kept for compatibility even though the fitted shape is read at runtime
    """

    return _make_binary_seq_classifier(cfg, _build_tcn_binary)


def make_tcn_gru_binary(cfg: ModelConfig) -> BaseEstimator:
    """returns the stage 1 tcn gru classifier"""

    return _make_binary_seq_classifier(cfg, _build_tcn_gru_binary)


def make_lstm_binary(cfg: ModelConfig) -> BaseEstimator:
    """returns the stage 1 lstm classifier"""

    return _make_binary_seq_classifier(cfg, _build_lstm_binary)


def make_gru_binary(cfg: ModelConfig) -> BaseEstimator:
    """returns the stage 1 gru classifier"""

    return _make_binary_seq_classifier(cfg, _build_gru_binary)


def get_stage1_sequential_models(cfg: ModelConfig) -> Dict[str, BaseEstimator]:
    """registry for the stage 1 sequence models"""

    return {
        "tcn": make_tcn_binary(cfg),
        "tcn_gru": make_tcn_gru_binary(cfg),
        "lstm": make_lstm_binary(cfg),
        "gru": make_gru_binary(cfg),
        "hybrid_vse": make_hybrid_vse_binary(cfg),
    }


# stage 1 four lap hybrid

def _build_vse_hybrid_binary(
    seq_len: int,
    n_features: int,
    rnn_units: int = 32,
    rnn_dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> "keras.Model":
    """
    builds the four lap hybrid model as taken from Heilmeier et al.'s proposed VSE model
    uses a time distributed feed forward network to extract a probability per lap,
    which then feeds directly into an lstm head.
    """
    x_in = layers.Input(shape=(seq_len, n_features))
    reg = keras.regularizers.l2(5e-4)

    # ffnn applied to each lap in the sequence independently
    x = layers.TimeDistributed(layers.Dense(64, activation="relu", kernel_regularizer=reg))(x_in)
    x = layers.TimeDistributed(layers.Dense(64, activation="relu", kernel_regularizer=reg))(x)
    x = layers.TimeDistributed(layers.Dense(1, activation="sigmoid"))(x)

    # lstm head processes the sequence of single-lap probabilities
    x = layers.LSTM(rnn_units, return_sequences=False, dropout=rnn_dropout, recurrent_dropout=0.0)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    y_out = layers.Dense(1, activation="sigmoid")(x)

    return _compile_binary_seq_model(keras.Model(x_in, y_out), learning_rate=learning_rate)

def make_hybrid_vse_binary(cfg: ModelConfig) -> BaseEstimator:
    """returns the single end-to-end hybrid sequence model for stage 1"""
    return _make_binary_seq_classifier(cfg, _build_vse_hybrid_binary)


# stage 2 tabular

def make_stage2_ffnn(cfg: ModelConfig, *, n_classes: int) -> BaseEstimator:
    """returns the stage 2 feed forward network

    this is the main dense multiclass baseline for compound prediction
    """

    def build_ffnn(input_dim: int) -> "keras.Model":
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

    def model_fn(meta):
        keras.utils.set_random_seed(cfg.random_state)
        return build_ffnn(meta["n_features_in_"])

    return _make_keras_classifier(
        cfg,
        model_fn,
        epochs=40,
        batch_size=256,
        monitor="val_loss",
        mode="min",
        patience=5,
    )


def make_stage2_rf(cfg: ModelConfig) -> BaseEstimator:
    """returns the tuned random forest for stage 2

    this is the multiclass tree ensemble for tyre compound prediction
    """

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


def make_stage2_svm(cfg: ModelConfig) -> BaseEstimator:
    """returns the stage 2 svm

    this is the multiclass tabular svm with probability output
    """

    return SVC(
        C=2.0,
        kernel="rbf",
        gamma="scale",
        probability=True,
        decision_function_shape="ovr",
        class_weight="balanced" if cfg.use_class_weight else None,
        random_state=cfg.random_state,
    )


def make_stage2_xgb(cfg: ModelConfig, *, n_classes: int) -> BaseEstimator:
    """returns the tuned xgboost model for stage 2"""

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


def get_stage2_tabular_models(cfg: ModelConfig, *, n_classes: int) -> Dict[str, BaseEstimator]:
    """registry for the stage 2 tabular models"""

    return {
        "ann": make_stage2_ffnn(cfg, n_classes=n_classes),
        "rf": make_stage2_rf(cfg),
        "svm": make_stage2_svm(cfg),
        "xgb": make_stage2_xgb(cfg, n_classes=n_classes),
    }


# stage 2 sequential

def _build_tcn_multiclass(
    seq_len: int,
    n_features: int,
    n_classes: int,
    filters: int = 64,
    kernel_size: int = 3,
    dropout: float = 0.12,
    pooling: str = "gap",
    learning_rate: float = 1e-4,
) -> "keras.Model":
    """builds the multiclass tcn for stage 2"""

    x_in = layers.Input(shape=(seq_len, n_features))

    x = layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=1)(x_in)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)

    for d in [1, 2, 4, 8]:
        x = _tcn_residual_block(
            x,
            filters=filters,
            kernel_size=kernel_size,
            dilation=d,
            dropout=dropout,
        )

    if pooling == "gap":
        x = layers.GlobalAveragePooling1D()(x)
    elif pooling == "last":
        x = layers.Lambda(lambda z: z[:, -1, :])(x)
        
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    y_out = layers.Dense(n_classes, activation="softmax")(x)

    return _compile_multiclass_seq_model(keras.Model(x_in, y_out), learning_rate=learning_rate)


def _build_tcn_gru_multiclass(
    seq_len: int,
    n_features: int,
    n_classes: int,
    filters: int = 64,
    kernel_size: int = 3,
    dropout: float = 0.15,
    rnn_units: int = 32,
    rnn_dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> "keras.Model":
    """builds the multiclass tcn gru model for stage 2"""

    x_in = layers.Input(shape=(seq_len, n_features))

    x = layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=1)(x_in)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)

    for d in [1, 2, 4, 8]:
        x = _tcn_residual_block(
            x,
            filters=filters,
            kernel_size=kernel_size,
            dilation=d,
            dropout=dropout,
        )

    x = layers.GRU(
        rnn_units,
        return_sequences=False,
        dropout=rnn_dropout,
        recurrent_dropout=0.0,
        kernel_regularizer=keras.regularizers.l2(5e-4),
    )(x)

    x = layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(5e-4))(x)
    x = layers.Dropout(0.2)(x)
    y_out = layers.Dense(n_classes, activation="softmax")(x)

    return _compile_multiclass_seq_model(keras.Model(x_in, y_out), learning_rate=learning_rate)


def _build_lstm_multiclass(
    seq_len: int,
    n_features: int,
    n_classes: int,
    rnn_units: int = 64,
    rnn_dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> "keras.Model":
    """builds the multiclass lstm for stage 2"""

    x_in = layers.Input(shape=(seq_len, n_features))

    x = layers.LSTM(
        rnn_units,
        return_sequences=True,
        dropout=rnn_dropout,
        recurrent_dropout=0.0,
        kernel_regularizer=keras.regularizers.l2(5e-4),
    )(x_in)
    x = layers.LSTM(
        rnn_units // 2,
        return_sequences=True,
        dropout=rnn_dropout,
        recurrent_dropout=0.0,
        kernel_regularizer=keras.regularizers.l2(5e-4),
    )(x)

    x = layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(5e-4))(x)
    x = layers.Dropout(0.2)(x)
    y_out = layers.Dense(n_classes, activation="softmax")(x)

    return _compile_multiclass_seq_model(keras.Model(x_in, y_out), learning_rate=learning_rate)


def _build_gru_multiclass(
    seq_len: int,
    n_features: int,
    n_classes: int,
    rnn_units: int = 64,
    rnn_dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> "keras.Model":
    """builds the multiclass gru for stage 2"""

    x_in = layers.Input(shape=(seq_len, n_features))

    x = layers.GRU(
        rnn_units,
        return_sequences=True,
        dropout=rnn_dropout,
        recurrent_dropout=0.0,
        kernel_regularizer=keras.regularizers.l2(5e-4),
    )(x_in)
    x = layers.GRU(
        rnn_units // 2,
        return_sequences=True,
        dropout=rnn_dropout,
        recurrent_dropout=0.0,
        kernel_regularizer=keras.regularizers.l2(5e-4),
    )(x)

    x = layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(5e-4))(x)
    x = layers.Dropout(0.2)(x)
    y_out = layers.Dense(n_classes, activation="softmax")(x)

    return _compile_multiclass_seq_model(keras.Model(x_in, y_out), learning_rate=learning_rate)


def make_tcn_multiclass(cfg: ModelConfig, *, n_classes: int) -> BaseEstimator:
    """returns the stage 2 tcn classifier"""

    return _make_multiclass_seq_classifier(cfg, _build_tcn_multiclass, n_classes=n_classes)


def make_tcn_gru_multiclass(cfg: ModelConfig, *, n_classes: int) -> BaseEstimator:
    """returns the stage 2 tcn gru classifier"""

    return _make_multiclass_seq_classifier(cfg, _build_tcn_gru_multiclass, n_classes=n_classes)


def make_lstm_multiclass(cfg: ModelConfig, *, n_classes: int) -> BaseEstimator:
    """returns the stage 2 lstm classifier"""

    return _make_multiclass_seq_classifier(cfg, _build_lstm_multiclass, n_classes=n_classes)


def make_gru_multiclass(cfg: ModelConfig, *, n_classes: int) -> BaseEstimator:
    """returns the stage 2 gru classifier"""

    return _make_multiclass_seq_classifier(cfg, _build_gru_multiclass, n_classes=n_classes)


def get_stage2_sequential_models(cfg: ModelConfig, *, n_classes: int) -> Dict[str, BaseEstimator]:
    """registry for the stage 2 sequence models"""

    return {
        "tcn": make_tcn_multiclass(cfg, n_classes=n_classes),
        "tcn_gru": make_tcn_gru_multiclass(cfg, n_classes=n_classes),
        "lstm": make_lstm_multiclass(cfg, n_classes=n_classes),
        "gru": make_gru_multiclass(cfg, n_classes=n_classes),
    }


#binary meta learners

def make_meta_binary_mlp(cfg: ModelConfig) -> BaseEstimator:
    """returns the binary mlp meta learner used on stacked stage 1 outputs"""

    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=200,
        early_stopping=True,
        random_state=cfg.random_state,
    )


def make_meta_binary_lr(cfg: ModelConfig) -> BaseEstimator:
    """returns the binary logistic regression meta learner"""

    return LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=4000,
        class_weight="balanced" if cfg.use_class_weight else None,
        random_state=cfg.random_state,
    )


def make_meta_binary_xgb(cfg: ModelConfig) -> BaseEstimator:
    """returns the binary xgboost meta learner"""

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

#multiclass meta learners

def make_meta_multiclass_mlp(cfg: ModelConfig) -> BaseEstimator:
    """returns the multiclass mlp meta learner used on stacked stage 2 outputs"""

    return MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=250,
        early_stopping=True,
        random_state=cfg.random_state,
    )


def make_meta_multiclass_lr(cfg: ModelConfig) -> BaseEstimator:
    """returns the multiclass logistic regression meta learner"""

    return LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=6000,
        multi_class="multinomial",
        class_weight="balanced" if cfg.use_class_weight else None,
        random_state=cfg.random_state,
    )


def make_meta_multiclass_xgb(cfg: ModelConfig, *, n_classes: int) -> BaseEstimator:
    """returns the multiclass xgboost meta learner"""

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


def get_meta_binary_learners(cfg: ModelConfig) -> Dict[str, BaseEstimator]:
    """registry for the binary meta learners"""

    return {
        "mlp": make_meta_binary_mlp(cfg),
        "lr": make_meta_binary_lr(cfg),
        "xgb": make_meta_binary_xgb(cfg),
    }


def get_meta_multiclass_learners(cfg: ModelConfig, *, n_classes: int) -> Dict[str, BaseEstimator]:
    """registry for the multiclass meta learners"""

    return {
        "mlp": make_meta_multiclass_mlp(cfg),
        "lr": make_meta_multiclass_lr(cfg),
        "xgb": make_meta_multiclass_xgb(cfg, n_classes=n_classes),
    }


# backward compatible names for the rest of the codebase

def make_rf_binary(cfg: ModelConfig) -> BaseEstimator:
    """kept so the existing evaluation code still works"""

    return make_stage1_rf(cfg)


def make_xgb_binary(cfg: ModelConfig) -> BaseEstimator:
    """kept so the existing evaluation code still works"""

    return make_stage1_xgb(cfg)


def make_svm_binary(cfg: ModelConfig, *, kernel: str = "rbf") -> BaseEstimator:
    """kept so the existing evaluation code still works"""

    if kernel != "rbf":
        raise ValueError("only the rbf svm is kept in the submission version")
    return make_stage1_svm(cfg)


def make_rf_multiclass(cfg: ModelConfig) -> BaseEstimator:
    """kept so the existing evaluation code still works"""

    return make_stage2_rf(cfg)


def make_xgb_multiclass(cfg: ModelConfig, *, n_classes: int) -> BaseEstimator:
    """kept so the existing evaluation code still works"""

    return make_stage2_xgb(cfg, n_classes=n_classes)


def make_svm_multiclass(cfg: ModelConfig, *, kernel: str = "rbf") -> BaseEstimator:
    """kept so the existing evaluation code still works"""

    if kernel != "rbf":
        raise ValueError("only the rbf svm is kept in the submission version")
    return make_stage2_svm(cfg)


def make_ann_multiclass(cfg: ModelConfig, *, n_classes: int) -> BaseEstimator:
    """kept so the existing evaluation code still works"""

    return make_stage2_ffnn(cfg, n_classes=n_classes)


def get_binary_base_learners(cfg: ModelConfig) -> Dict[str, BaseEstimator]:
    """kept so the existing evaluation code still works"""

    return get_stage1_tabular_models(cfg)


def get_multiclass_base_learners(cfg: ModelConfig, *, n_classes: int) -> Dict[str, BaseEstimator]:
    """kept so the existing evaluation code still works"""

    return get_stage2_tabular_models(cfg, n_classes=n_classes)
