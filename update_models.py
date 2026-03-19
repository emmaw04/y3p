import re

with open("src/models/models.py", "r") as f:
    code = f.read()

old_vse = '''# stage 1 four lap hybrid

def _build_vse_ffnn_binary(input_dim: int) -> "keras.Model":
    """builds the small feed forward network used before the lstm head

    this gives a probability per lap which then becomes the short input sequence
    """

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


def make_ann_binary_vse_ffnn(cfg: ModelConfig) -> BaseEstimator:
    """returns the stage 1 feed forward model used in the four lap hybrid setup"""

    def model_fn(meta):
        keras.utils.set_random_seed(cfg.random_state)
        return _build_vse_ffnn_binary(meta["n_features_in_"])

    return _make_keras_classifier(
        cfg,
        model_fn,
        epochs=40,
        batch_size=256,
        monitor="val_loss",
        mode="min",
        patience=5,
    )


def _build_vse_lstm_binary_head(seq_len: int) -> "keras.Model":
    """builds the small lstm head for the four lap probability sequences"""

    x_in = layers.Input(shape=(seq_len, 1))
    x = layers.LSTM(32, return_sequences=False, dropout=0.2, recurrent_dropout=0.0)(x_in)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    y_out = layers.Dense(1, activation="sigmoid")(x)

    return _compile_binary_seq_model(keras.Model(x_in, y_out))


def make_lstm_binary_head_vse(cfg: ModelConfig, *, seq_len: int = 4) -> BaseEstimator:
    """returns the lstm head used after the four lap feed forward probabilities"""

    def model_fn(meta):
        keras.utils.set_random_seed(cfg.random_state)
        _, inferred_seq_len, _ = meta["X_shape_"]
        return _build_vse_lstm_binary_head(int(inferred_seq_len))

    return _make_keras_classifier(
        cfg,
        model_fn,
        epochs=60,
        batch_size=256,
        monitor="val_pr_auc",
        mode="max",
        patience=6,
    )'''

new_vse = '''# stage 1 four lap hybrid

def _build_vse_hybrid_binary(seq_len: int, n_features: int) -> "keras.Model":
    """
    builds the four lap hybrid model in a single end-to-end architecture.
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
    x = layers.LSTM(32, return_sequences=False, dropout=0.2, recurrent_dropout=0.0)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    y_out = layers.Dense(1, activation="sigmoid")(x)

    return _compile_binary_seq_model(keras.Model(x_in, y_out))

def make_hybrid_vse_binary(cfg: ModelConfig) -> BaseEstimator:
    """returns the single end-to-end hybrid sequence model for stage 1"""
    return _make_binary_seq_classifier(cfg, _build_vse_hybrid_binary)'''

new_code = code.replace(old_vse, new_vse)

with open("src/models/models.py", "w") as f:
    f.write(new_code)
