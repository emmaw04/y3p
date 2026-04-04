from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import OrdinalEncoder

@dataclass(frozen=True)
class PreprocessConfig:
    scale_numeric: bool = True
    numeric_impute_strategy: str = "median"
    categorical_impute_strategy: str = "most_frequent"
    onehot_drop: Optional[str] = None
    sparse_onehot: bool = True

def build_preprocessor(
    num_cols: Sequence[str],
    cat_cols: Sequence[str],
    cfg: PreprocessConfig = PreprocessConfig(),
) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=cfg.numeric_impute_strategy)),
        ("scaler", StandardScaler()) if cfg.scale_numeric else ("passthrough", "passthrough"),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=cfg.categorical_impute_strategy)),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            drop=cfg.onehot_drop,
            sparse_output=cfg.sparse_onehot,
        )),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, list(num_cols)),
            ("cat", cat_pipe, list(cat_cols)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

def make_preprocessor_for_model(model_name: str, num_cols: Sequence[str], cat_cols: Sequence[str]) -> ColumnTransformer:
    model = model_name.lower()

    # ANN/sequence models need dense, scaled numeric
    if model in {"ann", "hybrid_vse"}:
        return make_preprocessor_for_keras(num_cols, cat_cols)

    if model in {"tcn", "gru", "lstm", "tcn_gru"}: # TCN needs one-hot
        return build_preprocessor(num_cols, cat_cols, PreprocessConfig(scale_numeric=True, sparse_onehot=False))

    if model in {"svm"}:
        return build_preprocessor(num_cols, cat_cols, PreprocessConfig(scale_numeric=True, sparse_onehot=True))

    if model in {"rf", "xgb"}:
        return build_preprocessor(num_cols, cat_cols, PreprocessConfig(scale_numeric=False, sparse_onehot=True))

    return build_preprocessor(num_cols, cat_cols)

def make_preprocessor_for_keras(num_cols, cat_cols) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Ordinal encode categoricals -> small dense matrix
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ord", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, list(num_cols)),
            ("cat", cat_pipe, list(cat_cols)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )