from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import OrdinalEncoder


@dataclass
class PreprocessConfig:
    # default settings for most models
    scale_numeric: bool = True
    numeric_impute_strategy: str = "median"
    categorical_impute_strategy: str = "most_frequent"  # current compound missing values get filled with the most common class
    onehot_drop: Optional[str] = None
    sparse_onehot: bool = True


def build_preprocessor(
    num_cols: Sequence[str],
    cat_cols: Sequence[str],
    cfg: PreprocessConfig = PreprocessConfig(),
) -> ColumnTransformer:
    # numeric pipeline fills missing values, then scales if needed
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=cfg.numeric_impute_strategy)),
        ("scaler", StandardScaler()) if cfg.scale_numeric else ("passthrough", "passthrough"),
    ])

    # categorical pipeline fills missing values, then one hot encodes
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=cfg.categorical_impute_strategy)),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            drop=cfg.onehot_drop,
            sparse_output=cfg.sparse_onehot,
        )),
    ])

    # apply numeric and categorical pipelines to their own columns
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

    # keras style models want dense inputs
    if model in {"ann", "hybrid_vse"}:
        return make_preprocessor_for_keras(num_cols, cat_cols)

    # sequence models also want dense output, with scaled numeric features
    if model in {"tcn", "gru", "lstm", "tcn_gru"}:
        return build_preprocessor(num_cols, cat_cols, PreprocessConfig(scale_numeric=True, sparse_onehot=False))

    # svm benefits from scaling, and sparse one hot is fine
    if model in {"svm"}:
        return build_preprocessor(num_cols, cat_cols, PreprocessConfig(scale_numeric=True, sparse_onehot=True))

    # tree models do not need scaling
    if model in {"rf", "xgb"}:
        return build_preprocessor(num_cols, cat_cols, PreprocessConfig(scale_numeric=False, sparse_onehot=True))

    #default
    return build_preprocessor(num_cols, cat_cols)


def make_preprocessor_for_keras(num_cols, cat_cols) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # ordinal encoding keeps the matrix small and dense for keras
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