import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss

from imblearn.over_sampling import SMOTENC

from xgboost import XGBClassifier
from typing import Optional

# -----------------------
# Config
# -----------------------
DATA_PATH = "data/processed/output2.csv"
OUTDIR = "data/processed/SMOTE"
N_SPLITS = 5
SEED = 42

# Up to 5 datasets / settings (WET multiplier relative to original count in *each training fold*)
# Convention: 500% oversampling => add 5x synthetic => final count = 6x original
WET_MULTIPLIERS = [
    ("wet_x1",   1.0),  # no WET oversampling (still can oversample other minorities)
    ("wet_100",  2.0),  # +100% => 2x
    ("wet_200",  3.0),  # +200% => 3x
    ("wet_300",  4.0),  # +300% => 4x
    ("wet_500",  6.0),  # +500% => 6x (absolute max)
]

# Whether to bring MEDIUM/SOFT/INTERMEDIATE up to the majority count (HARD) in each fold
BALANCE_NON_WET_TO_MAJORITY = True

# Save resampled *full-dataset* versions (in preprocessed feature space) for each setting
SAVE_FULL_RESAMPLED_DATASETS = True


# -----------------------
# Load
# -----------------------
df = pd.read_csv(DATA_PATH, na_values=[""])
target = "y_compound"
group = "race_id"

X_all = df.drop(columns=[target])
y_all = df[target].astype(str)
groups_all = df[group].values

# IMPORTANT: don't use race_id as a feature
X_feat_all = X_all.drop(columns=[group])

# Define categorical vs numeric
cat_cols = ["current_compound", "race_track", "fulfilled_second_compound", "rained_yet", "is_raining"]
num_cols = [c for c in X_feat_all.columns if c not in cat_cols]

# Preprocess: numeric impute+scale; categorical impute+ordinal encode (needed for SMOTENC)
preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        ]), cat_cols),
    ],
    remainder="drop",
)

# After preprocess, columns are [num..., cat...]
cat_indices = list(range(len(num_cols), len(num_cols) + len(cat_cols)))

# Label-encode target for consistent ordering + XGB compatibility
le = LabelEncoder()
y_enc_all = le.fit_transform(y_all)
class_names = list(le.classes_)

if "WET" not in class_names:
    raise ValueError(f"'WET' not found in y_compound classes: {class_names}")
wet_label = int(le.transform(["WET"])[0])

# Model (you can swap this out for rf/svm, etc.)
def make_model(num_classes: int, seed: int = 42):
    return XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=num_classes,
        random_state=seed,
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
    )


# -----------------------
# Helper: build per-fold sampling_strategy dict
# -----------------------
def build_sampling_strategy(y_train_enc: np.ndarray, wet_multiplier: float) -> dict:
    """
    Returns a dict {class_int: target_count} for SMOTENC.
    - If BALANCE_NON_WET_TO_MAJORITY: oversample all non-majority, non-WET classes to majority count.
    - WET is capped by wet_multiplier relative to its fold count (max 6x for 500%).
    """
    counts = pd.Series(y_train_enc).value_counts().to_dict()
    # majority class in this fold
    maj_label = max(counts, key=lambda k: counts[k])
    maj_count = counts[maj_label]

    strategy = {}

    # Balance non-WET classes up to majority (optional)
    if BALANCE_NON_WET_TO_MAJORITY:
        for cls, n in counts.items():
            if cls == maj_label:
                continue
            if cls == wet_label:
                continue
            if n < maj_count:
                strategy[int(cls)] = int(maj_count)

    # WET capped to wet_multiplier * wet_count (but never above majority)
    wet_count = counts.get(wet_label, 0)
    if wet_count >= 2:
        wet_target = int(min(round(wet_count * wet_multiplier), maj_count))
        if wet_target > wet_count:
            strategy[wet_label] = wet_target

    # If a class is absent in this fold, it won't be in counts; SMOTE can't create it from nothing.
    return strategy


def safe_k_neighbors(y_train_enc: np.ndarray, strategy: dict, k_max: int = 5) -> Optional[int]:
    """
    SMOTE/SMOTENC requires at least (k_neighbors + 1) samples in *each* class being oversampled.
    Returns a safe k_neighbors or None if oversampling isn't feasible.
    """
    if not strategy:
        return None
    counts = pd.Series(y_train_enc).value_counts()
    min_n = min(int(counts[c]) for c in strategy.keys() if c in counts.index)
    if min_n < 2:
        return None
    return int(max(1, min(k_max, min_n - 1)))


# -----------------------
# CV Sweep
# -----------------------
os.makedirs(OUTDIR, exist_ok=True)

cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
num_classes = len(class_names)

results = {}

for setting_name, wet_mult in WET_MULTIPLIERS:
    fold_metrics = []

    for fold, (tr, va) in enumerate(cv.split(X_feat_all, y_enc_all, groups=groups_all), 1):
        X_tr_raw = X_feat_all.iloc[tr]
        y_tr = y_enc_all[tr]
        X_va_raw = X_feat_all.iloc[va]
        y_va = y_enc_all[va]

        # Fit preprocessing on train fold only
        X_tr = preprocess.fit_transform(X_tr_raw)
        X_va = preprocess.transform(X_va_raw)

        # Build sampling strategy for this fold
        strategy = build_sampling_strategy(y_tr, wet_multiplier=wet_mult)
        k = safe_k_neighbors(y_tr, strategy, k_max=5)

        # Apply SMOTENC on train fold only (if feasible)
        if strategy and k is not None:
            sampler = SMOTENC(
                categorical_features=cat_indices,
                sampling_strategy=strategy,
                k_neighbors=k,
                random_state=SEED,
            )
            X_tr_res, y_tr_res = sampler.fit_resample(X_tr, y_tr)
        else:
            X_tr_res, y_tr_res = X_tr, y_tr

        model = make_model(num_classes=num_classes, seed=SEED)
        model.fit(X_tr_res, y_tr_res)

        proba = model.predict_proba(X_va)
        pred = np.argmax(proba, axis=1)

        fold_metrics.append({
            "accuracy": float(accuracy_score(y_va, pred)),
            "precision_macro": float(precision_score(y_va, pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_va, pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_va, pred, average="macro", zero_division=0)),
            "logloss": float(log_loss(y_va, proba, labels=list(range(num_classes)))),
        })

    # Aggregate
    dfm = pd.DataFrame(fold_metrics)
    summary = {f"{c}_mean": float(dfm[c].mean()) for c in dfm.columns}
    summary.update({f"{c}_std": float(dfm[c].std(ddof=1)) for c in dfm.columns})
    summary["wet_multiplier"] = wet_mult
    summary["setting"] = setting_name

    results[setting_name] = summary
    print(f"\n=== {setting_name} (WET x{wet_mult:g}) ===")
    print(json.dumps(summary, indent=2))

# Save sweep summary
with open(os.path.join(OUTDIR, "sweep_summary.json"), "w") as f:
    json.dump({"classes": class_names, "results": results}, f, indent=2)

"""
helpers: scales back to original values
"""
def inverse_to_original_units(X_res: np.ndarray, preprocess: ColumnTransformer,
                              num_cols: list, cat_cols: list) -> pd.DataFrame:
    """
    Convert SMOTE output (in preprocessed space) back to:
      - numeric in original units (inverse StandardScaler)
      - categoricals back to original string categories (inverse OrdinalEncoder)

    Note: this cannot restore missing values (imputation already happened).
    """
    n_num = len(num_cols)
    n_cat = len(cat_cols)

    X_num = X_res[:, :n_num]
    X_cat = X_res[:, n_num:n_num + n_cat]

    # Grab fitted steps
    num_pipe = preprocess.named_transformers_["num"]
    cat_pipe = preprocess.named_transformers_["cat"]

    scaler = num_pipe.named_steps["scaler"]
    enc = cat_pipe.named_steps["enc"]

    # Inverse scale numerics back to original units
    X_num_inv = scaler.inverse_transform(X_num)

    # SMOTENC should keep categoricals “categorical”, but float dtype can happen; round safely
    X_cat_rounded = np.rint(X_cat).astype(int)

    # Inverse encode categoricals back to original labels
    # OrdinalEncoder expects 2D float array
    X_cat_inv = enc.inverse_transform(X_cat_rounded.astype(float))

    df_inv = pd.DataFrame(X_num_inv, columns=num_cols)
    for j, col in enumerate(cat_cols):
        df_inv[col] = X_cat_inv[:, j]

    return df_inv

# -----------------------
# Optional: generate & save 5 resampled datasets from the FULL dataset
# (preprocessed feature space; useful for training DL models directly)
# -----------------------
if SAVE_FULL_RESAMPLED_DATASETS:
    # Fit preprocess on full data
    X_full = preprocess.fit_transform(X_feat_all)
    y_full = y_enc_all

    feature_names = preprocess.get_feature_names_out()

    for setting_name, wet_mult in WET_MULTIPLIERS:
        strategy = build_sampling_strategy(y_full, wet_multiplier=wet_mult)
        k = safe_k_neighbors(y_full, strategy, k_max=5)

        if strategy and k is not None:
            sampler = SMOTENC(
                categorical_features=cat_indices,
                sampling_strategy=strategy,
                k_neighbors=k,
                random_state=SEED,
            )
            X_res, y_res = sampler.fit_resample(X_full, y_full)
        else:
            X_res, y_res = X_full, y_full

        out_csv = os.path.join(OUTDIR, f"stage2_{setting_name}_preprocessed.csv")
        out_npz = os.path.join(OUTDIR, f"stage2_{setting_name}_preprocessed.npz")

        df_out = pd.DataFrame(X_res, columns=feature_names)
        df_out["y_compound_int"] = y_res
        df_out["y_compound"] = le.inverse_transform(y_res)
        df_out.to_csv(out_csv, index=False)

        # NEW: inverse-transform numerics + inverse-transform categoricals
        df_human = inverse_to_original_units(X_res, preprocess, num_cols, cat_cols)
        df_human["y_compound_int"] = y_res
        df_human["y_compound"] = le.inverse_transform(y_res)

        out_csv_human = os.path.join(OUTDIR, f"stage2_{setting_name}_original_units.csv")
        df_human.to_csv(out_csv_human, index=False)
        np.savez_compressed(out_npz, X=X_res, y=y_res, feature_names=feature_names, classes=np.array(class_names, dtype=object))

    with open(os.path.join(OUTDIR, "label_mapping.json"), "w") as f:
        json.dump({"classes_in_order": class_names}, f, indent=2)

    print(f"\nSaved resampled datasets + summary to: {OUTDIR}")
