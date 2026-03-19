import re

with open('src/models/models.py', 'r') as f:
    content = f.read()

# 1. _build_tcn_binary
content = re.sub(
    r'def _build_tcn_binary\(seq_len: int, n_features: int\) -> "keras\.Model":\s*"""builds the binary tcn',
    r'''def _build_tcn_binary(seq_len: int, n_features: int, filters: int = 64, kernel_size: int = 3, dropout: float = 0.10879485872574356, pooling: str = "gap", learning_rate: float = 0.00011662890273931399, focal_gamma: float = 1.0, focal_alpha: float = 0.5) -> "keras.Model":
    """builds the binary tcn''',
    content
)
content = content.replace('layers.Conv1D(64, 3', 'layers.Conv1D(filters, kernel_size')
content = content.replace('filters=64,\n            kernel_size=3,\n            dilation=d,\n            dropout=0.10879485872574356,', 'filters=filters,\n            kernel_size=kernel_size,\n            dilation=d,\n            dropout=dropout,')
content = content.replace('''    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)''', '''    if pooling == "gap":
        x = layers.GlobalAveragePooling1D()(x)
    elif pooling == "last":
        x = layers.Lambda(lambda z: z[:, -1, :])(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)''')
content = content.replace('''    return _compile_binary_seq_model(
        model,
        learning_rate=0.00011662890273931399,
        use_focal=True,
    )''', '''    return _compile_binary_seq_model(
        model,
        learning_rate=learning_rate,
        use_focal=True,
    )''')
# Wait, I need to pass focal parameters into _compile_binary_seq_model as well, since the focal loss is created there.
# Let's fix _compile_binary_seq_model first
