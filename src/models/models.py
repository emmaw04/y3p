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

# shared helpers
@dataclass(frozen=True)
class ModelConfig:
    """small config object used across the model factories
    ensures seed number is consistent across classes
    """
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


# stage 1 tabular factories

def make_stage1_rf(cfg: ModelConfig) -> BaseEstimator:
    """returns the tuned random forest for stage 1
    """

    return RandomForestClassifier(
        n_estimators=841,
        max_depth=14,
        min_samples_leaf=5,
        min_samples_split=8,
        max_features=0.3,
        bootstrap=True,
        class_weight=None,
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_state,
    )


def make_stage1_xgb(cfg: ModelConfig) -> BaseEstimator:
    """returns the tuned xgboost model for stage 1
    """

    return XGBClassifier(
        n_estimators=469,
        max_depth=6,
        learning_rate=0.10934656142438419,
        subsample=0.8406507528138978,
        colsample_bytree=0.7318670875985085,
        min_child_weight=2,
        reg_lambda=0.2707713602235789,
        reg_alpha=0.00888399485712417,
        gamma=0.8808306415506197,
        max_bin=256,
        scale_pos_weight=26.646294743608884,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_state,
    )


def make_stage1_svm(cfg: ModelConfig) -> BaseEstimator:
    """returns the stage 1 svm
    """

    return SVC(
        C=0.1506884593067545,
        kernel="rbf",
        gamma=0.04233624238517596,
        probability=True,
        class_weight="balanced",
        random_state=cfg.random_state,
    )

def make_stage1_ann(cfg: ModelConfig) -> BaseEstimator:
    """returns the tuned stage 1 artificial neural network"""

    def build_ann(input_dim: int) -> "keras.Model":
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(32, activation="relu", kernel_regularizer=keras.regularizers.l2(0.0003129707937151823)),
            layers.Dropout(0.3482830988348138),
            layers.Dense(1, activation="sigmoid"),
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0007127590653708736),
            loss="binary_crossentropy",
            metrics=[
                keras.metrics.AUC(curve="PR", name="pr_auc"),
                keras.metrics.AUC(curve="ROC", name="roc_auc"),
            ],
        )
        return model

    def model_fn(meta):
        keras.utils.set_random_seed(cfg.random_state)
        return build_ann(meta["n_features_in_"])

    return _make_keras_classifier(
        cfg,
        model_fn,
        epochs=80,
        batch_size=64,
        monitor="val_pr_auc",
        mode="max",
        patience=10,
    )

# stage 1 sequential factories

def make_tcn_binary(cfg: ModelConfig) -> BaseEstimator:
    """Stage 1 TCN classifier."""

    def model_fn(meta):
        _, seq_len, n_features = meta["X_shape_"]
        keras.utils.set_random_seed(cfg.random_state)

        x_in = layers.Input(shape=(int(seq_len), int(n_features)))

        x = layers.Conv1D(64, 4, padding="causal", dilation_rate=1)(x_in)
        x = layers.LayerNormalization()(x)
        x = layers.Activation("relu")(x)

        for d in [1, 2, 4, 8]:
            x = _tcn_residual_block(
                x,
                filters=64,
                kernel_size=4,
                dilation=d,
                dropout=0.15117968234330592,
            )

        x = layers.Lambda(lambda z: z[:, -1, :])(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        y_out = layers.Dense(1, activation="sigmoid")(x)

        return _compile_binary_seq_model(
            keras.Model(x_in, y_out),
            learning_rate=0.0009484229044417891,
            use_focal=True,
            focal_gamma=1.0,
            focal_alpha=0.75,
        )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_pr_auc",
        mode="max",
        patience=10,
        restore_best_weights=True,
    )

    return KerasClassifier(
        model=model_fn,
        epochs=80,
        batch_size=128,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1,
        random_state=cfg.random_state,
    )


def make_tcn_gru_binary(cfg: ModelConfig) -> BaseEstimator:
    """Stage 1 TCN-GRU classifier."""

    def model_fn(meta):
        _, seq_len, n_features = meta["X_shape_"]
        keras.utils.set_random_seed(cfg.random_state)

        x_in = layers.Input(shape=(int(seq_len), int(n_features)))

        x = layers.Conv1D(64, 2, padding="causal", dilation_rate=1)(x_in)
        x = layers.LayerNormalization()(x)
        x = layers.Activation("relu")(x)

        for d in [1, 2, 4, 8]:
            x = _tcn_residual_block(
                x,
                filters=64,
                kernel_size=2,
                dilation=d,
                dropout=0.0444341795866854,
            )

        x = layers.GRU(
            16,
            return_sequences=False,
            dropout=0.2741761381775658,
            recurrent_dropout=0.0,
            kernel_regularizer=keras.regularizers.l2(5e-4),
        )(x)

        x = layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(5e-4))(x)
        x = layers.Dropout(0.2)(x)
        y_out = layers.Dense(1, activation="sigmoid")(x)

        return _compile_binary_seq_model(
            keras.Model(x_in, y_out),
            learning_rate=0.0014409354395782717,
        )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_pr_auc",
        mode="max",
        patience=10,
        restore_best_weights=True,
    )

    return KerasClassifier(
        model=model_fn,
        epochs=80,
        batch_size=128,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1,
        random_state=cfg.random_state,
    )


def make_lstm_binary(cfg: ModelConfig) -> BaseEstimator:
    """Stage 1 LSTM classifier."""

    def model_fn(meta):
        _, seq_len, n_features = meta["X_shape_"]
        keras.utils.set_random_seed(cfg.random_state)

        x_in = layers.Input(shape=(int(seq_len), int(n_features)))

        x = layers.LSTM(
            64,
            return_sequences=True,
            dropout=0.03852451305608232,
            recurrent_dropout=0.0,
            kernel_regularizer=keras.regularizers.l2(5e-4),
        )(x_in)
        x = layers.LSTM(
            32,
            return_sequences=True,
            dropout=0.03852451305608232,
            recurrent_dropout=0.0,
            kernel_regularizer=keras.regularizers.l2(5e-4),
        )(x)

        x = layers.LayerNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(5e-4))(x)
        x = layers.Dropout(0.2)(x)
        y_out = layers.Dense(1, activation="sigmoid")(x)

        return _compile_binary_seq_model(
            keras.Model(x_in, y_out),
            learning_rate=0.0007454170873871039,
        )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_pr_auc",
        mode="max",
        patience=10,
        restore_best_weights=True,
    )

    return KerasClassifier(
        model=model_fn,
        epochs=80,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1,
        random_state=cfg.random_state,
    )


def make_gru_binary(cfg: ModelConfig) -> BaseEstimator:
    """Stage 1 GRU classifier."""

    def model_fn(meta):
        _, seq_len, n_features = meta["X_shape_"]
        keras.utils.set_random_seed(cfg.random_state)

        x_in = layers.Input(shape=(int(seq_len), int(n_features)))

        x = layers.GRU(
            64,
            return_sequences=True,
            dropout=0.010142639375770673,
            recurrent_dropout=0.0,
            kernel_regularizer=keras.regularizers.l2(5e-4),
        )(x_in)
        x = layers.GRU(
            32,
            return_sequences=True,
            dropout=0.010142639375770673,
            recurrent_dropout=0.0,
            kernel_regularizer=keras.regularizers.l2(5e-4),
        )(x)

        x = layers.LayerNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(5e-4))(x)
        x = layers.Dropout(0.2)(x)
        y_out = layers.Dense(1, activation="sigmoid")(x)

        return _compile_binary_seq_model(
            keras.Model(x_in, y_out),
            learning_rate=0.0009731654271003453,
        )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_pr_auc",
        mode="max",
        patience=10,
        restore_best_weights=True,
    )

    return KerasClassifier(
        model=model_fn,
        epochs=80,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1,
        random_state=cfg.random_state,
    )


def make_hybrid_vse_binary(cfg: ModelConfig) -> BaseEstimator:
    """Stage 1 hybrid VSE classifier."""

    def model_fn(meta):
        _, seq_len, n_features = meta["X_shape_"]
        keras.utils.set_random_seed(cfg.random_state)

        x_in = layers.Input(shape=(int(seq_len), int(n_features)))
        reg = keras.regularizers.l2(5e-4)

        x = layers.TimeDistributed(
            layers.Dense(64, activation="relu", kernel_regularizer=reg)
        )(x_in)
        x = layers.TimeDistributed(
            layers.Dense(64, activation="relu", kernel_regularizer=reg)
        )(x)
        x = layers.TimeDistributed(
            layers.Dense(1, activation="sigmoid")
        )(x)

        x = layers.LSTM(
            32,
            return_sequences=False,
            dropout=0.2,
            recurrent_dropout=0.0,
        )(x)
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        y_out = layers.Dense(1, activation="sigmoid")(x)

        return _compile_binary_seq_model(
            keras.Model(x_in, y_out),
            learning_rate=1e-3,
        )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_pr_auc",
        mode="max",
        patience=10,
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

# stage 1 registries

def get_stage1_tabular_models(cfg: ModelConfig) -> Dict[str, BaseEstimator]:
    """registry for the stage 1 tabular models"""
    return {
        "rf": make_stage1_rf(cfg),
        "xgb": make_stage1_xgb(cfg),
        "svm": make_stage1_svm(cfg),
        "ann": make_stage1_ann(cfg),
    }

def get_stage1_sequential_models(cfg: ModelConfig) -> Dict[str, BaseEstimator]:
    """registry for the stage 1 sequence models"""
    return {
        "tcn": make_tcn_binary(cfg),
        "tcn_gru": make_tcn_gru_binary(cfg),
        "lstm": make_lstm_binary(cfg),
        "gru": make_gru_binary(cfg),
        "hybrid_vse": make_hybrid_vse_binary(cfg),
    }

# stage 2 tabular factories
def make_stage2_ffnn(cfg: ModelConfig, *, n_classes: int) -> BaseEstimator:
    """the stage 2 feed forward network
    """

    def build_ffnn(input_dim: int) -> "keras.Model":
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(32, activation="relu", kernel_regularizer=keras.regularizers.l2(1.161257222624932e-05)),
            layers.Dropout(0.2696839799381978),
            layers.Dense(n_classes, activation="softmax"),
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0014153610406428582),
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
        batch_size=64,
        monitor="val_loss",
        mode="min",
        patience=10,
    )


def make_stage2_rf(cfg: ModelConfig) -> BaseEstimator:
    """the tuned random forest for stage 2
    """

    return RandomForestClassifier(
        n_estimators=892,
        max_depth=14,
        min_samples_leaf=5,
        min_samples_split=20,
        max_features="sqrt",
        bootstrap=True,
        class_weight=None,
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_state,
    )


def make_stage2_svm(cfg: ModelConfig) -> BaseEstimator:
    """stage 2 svm
    """
    return SVC(
        C=27.461363284183133,
        kernel="rbf",
        gamma=0.04106194081414191,
        probability=True,
        decision_function_shape="ovr",
        class_weight=None,
        random_state=cfg.random_state,
    )


def make_stage2_xgb(cfg: ModelConfig, *, n_classes: int) -> BaseEstimator:
    """xgboost model for stage 2"""
 
    return XGBClassifier(
        n_estimators=444,
        max_depth=4,
        learning_rate=0.03314120416924632,
        subsample=0.9352813595213753,
        colsample_bytree=0.6034143183279435,
        min_child_weight=1,
        reg_lambda=0.08174791935277638,
        reg_alpha=0.007071279795461194,
        gamma=0.6179551616665386,
        max_bin=256,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_state,
    )

# stage 2 sequential model factories

def make_tcn_multiclass(cfg: ModelConfig, *, n_classes: int) -> BaseEstimator:
    """Stage 2 TCN classifier."""

    def model_fn(meta):
        _, seq_len, n_features = meta["X_shape_"]
        keras.utils.set_random_seed(cfg.random_state)

        x_in = layers.Input(shape=(int(seq_len), int(n_features)))

        x = layers.Conv1D(32, 3, padding="causal", dilation_rate=1)(x_in)
        x = layers.LayerNormalization()(x)
        x = layers.Activation("relu")(x)

        for d in [1, 2, 4, 8]:
            x = _tcn_residual_block(
                x,
                filters=32,
                kernel_size=3,
                dilation=d,
                dropout=0.3189536660208673,
            )

        x = layers.Lambda(lambda z: z[:, -1, :])(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        y_out = layers.Dense(n_classes, activation="softmax")(x)

        return _compile_multiclass_seq_model(
            keras.Model(x_in, y_out),
            learning_rate=0.0012208142144753873,
        )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_acc",
        mode="max",
        patience=10,
        restore_best_weights=True,
    )

    return KerasClassifier(
        model=model_fn,
        epochs=80,
        batch_size=128,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1,
        random_state=cfg.random_state,
    )


def make_tcn_gru_multiclass(cfg: ModelConfig, *, n_classes: int) -> BaseEstimator:
    """Stage 2 TCN-GRU classifier."""

    def model_fn(meta):
        _, seq_len, n_features = meta["X_shape_"]
        keras.utils.set_random_seed(cfg.random_state)

        x_in = layers.Input(shape=(int(seq_len), int(n_features)))

        x = layers.Conv1D(32, 3, padding="causal", dilation_rate=1)(x_in)
        x = layers.LayerNormalization()(x)
        x = layers.Activation("relu")(x)

        for d in [1, 2, 4, 8]:
            x = _tcn_residual_block(
                x,
                filters=32,
                kernel_size=3,
                dilation=d,
                dropout=0.1421548856945066,
            )

        x = layers.GRU(
            64,
            return_sequences=False,
            dropout=0.0010077077435941516,
            recurrent_dropout=0.0,
            kernel_regularizer=keras.regularizers.l2(5e-4),
        )(x)

        x = layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(5e-4))(x)
        x = layers.Dropout(0.2)(x)
        y_out = layers.Dense(n_classes, activation="softmax")(x)

        return _compile_multiclass_seq_model(
            keras.Model(x_in, y_out),
            learning_rate=0.0006671159654368591,
        )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_acc",
        mode="max",
        patience=10,
        restore_best_weights=True,
    )

    return KerasClassifier(
        model=model_fn,
        epochs=80,
        batch_size=128,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1,
        random_state=cfg.random_state,
    )


def make_lstm_multiclass(cfg: ModelConfig, *, n_classes: int) -> BaseEstimator:
    """Stage 2 LSTM classifier."""

    def model_fn(meta):
        _, seq_len, n_features = meta["X_shape_"]
        keras.utils.set_random_seed(cfg.random_state)

        x_in = layers.Input(shape=(int(seq_len), int(n_features)))

        x = layers.LSTM(
            32,
            return_sequences=True,
            dropout=0.25791781440280437,
            recurrent_dropout=0.0,
            kernel_regularizer=keras.regularizers.l2(5e-4),
        )(x_in)
        x = layers.LSTM(
            16,
            return_sequences=True,
            dropout=0.25791781440280437,
            recurrent_dropout=0.0,
            kernel_regularizer=keras.regularizers.l2(5e-4),
        )(x)

        x = layers.LayerNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(5e-4))(x)
        x = layers.Dropout(0.2)(x)
        y_out = layers.Dense(n_classes, activation="softmax")(x)

        return _compile_multiclass_seq_model(
            keras.Model(x_in, y_out),
            learning_rate=0.0008962278983461516,
        )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_acc",
        mode="max",
        patience=10,
        restore_best_weights=True,
    )

    return KerasClassifier(
        model=model_fn,
        epochs=80,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1,
        random_state=cfg.random_state,
    )


def make_gru_multiclass(cfg: ModelConfig, *, n_classes: int) -> BaseEstimator:
    """Stage 2 GRU classifier."""

    def model_fn(meta):
        _, seq_len, n_features = meta["X_shape_"]
        keras.utils.set_random_seed(cfg.random_state)

        x_in = layers.Input(shape=(int(seq_len), int(n_features)))

        x = layers.GRU(
            32,
            return_sequences=True,
            dropout=0.16754040659127542,
            recurrent_dropout=0.0,
            kernel_regularizer=keras.regularizers.l2(5e-4),
        )(x_in)
        x = layers.GRU(
            16,
            return_sequences=True,
            dropout=0.16754040659127542,
            recurrent_dropout=0.0,
            kernel_regularizer=keras.regularizers.l2(5e-4),
        )(x)

        x = layers.LayerNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(5e-4))(x)
        x = layers.Dropout(0.2)(x)
        y_out = layers.Dense(n_classes, activation="softmax")(x)

        return _compile_multiclass_seq_model(
            keras.Model(x_in, y_out),
            learning_rate=0.0008375958618273803,
        )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_acc",
        mode="max",
        patience=10,
        restore_best_weights=True,
    )

    return KerasClassifier(
        model=model_fn,
        epochs=80,
        batch_size=128,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1,
        random_state=cfg.random_state,
    )

# stage 2 registries

def get_stage2_tabular_models(cfg: ModelConfig, *, n_classes: int) -> Dict[str, BaseEstimator]:
    """tabular models registry"""
    return {
        "ann": make_stage2_ffnn(cfg, n_classes=n_classes),
        "rf": make_stage2_rf(cfg),
        "svm": make_stage2_svm(cfg),
        "xgb": make_stage2_xgb(cfg, n_classes=n_classes),
    }


def get_stage2_sequential_models(cfg: ModelConfig, *, n_classes: int) -> Dict[str, BaseEstimator]:
    """sequential models registry"""
    return {
        "tcn": make_tcn_multiclass(cfg, n_classes=n_classes),
        "tcn_gru": make_tcn_gru_multiclass(cfg, n_classes=n_classes),
        "lstm": make_lstm_multiclass(cfg, n_classes=n_classes),
        "gru": make_gru_multiclass(cfg, n_classes=n_classes),
    }


# meta learner factories

def make_meta_binary_mlp(cfg: ModelConfig) -> BaseEstimator:
    """binary mlp meta learner"""

    return MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        alpha=0.0014750765684701768,
        learning_rate_init=0.013743140045520868,
        batch_size=128,
        max_iter=200,
        early_stopping=True,
        random_state=cfg.random_state,
    )


def make_meta_binary_lr(cfg: ModelConfig) -> BaseEstimator:
    """binary logistic regression meta learner"""

    return LogisticRegression(
        penalty="l2",
        C=0.16876312778393562,
        solver="lbfgs",
        max_iter=4000,
        class_weight=None,
        random_state=cfg.random_state,
    )


def make_meta_binary_xgb(cfg: ModelConfig) -> BaseEstimator:
    """binary xgboost meta learner"""

    return XGBClassifier(
        n_estimators=650,
        max_depth=3,
        learning_rate=0.0351660559697577,
        subsample=0.7033869080464379,
        colsample_bytree=0.962851669188502,
        reg_alpha=1.1258660482262575e-06,
        reg_lambda=13.994968791682169,
        min_child_weight=8,
        gamma=3.2257301877317572,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_state,
    )

def make_meta_multiclass_mlp(cfg: ModelConfig) -> BaseEstimator:
    """multiclass mlp meta learner"""
    return MLPClassifier(
        hidden_layer_sizes=(32,),
        activation="relu",
        alpha=0.0005053944133370265,
        learning_rate_init=0.00033361078666799233,
        batch_size=32,
        max_iter=250,
        early_stopping=True,
        random_state=cfg.random_state,
    )

def make_meta_multiclass_lr(cfg: ModelConfig) -> BaseEstimator:
    """multiclass logistic regression meta learner"""
    return LogisticRegression(
        penalty="l2",
        C=0.08933948320060756,
        solver="lbfgs",
        max_iter=6000,
        multi_class="multinomial",
        class_weight=None,
        random_state=cfg.random_state,
    )

def make_meta_multiclass_xgb(cfg: ModelConfig, *, n_classes: int) -> BaseEstimator:
    """multiclass xgboost meta learner"""
    return XGBClassifier(
        n_estimators=338,
        max_depth=3,
        learning_rate=0.05720671296025396,
        subsample=0.962183523136069,
        colsample_bytree=0.7031882165929242,
        reg_alpha=1.2022222899804231,
        reg_lambda=0.015654414442086057,
        min_child_weight=1,
        gamma=4.655392345755547,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_state,
    )


#meta learner registries
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