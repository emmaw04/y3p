"""
generates smote upsampled datasets for stage 2 compound prediction.
it takes the original dataset and creates 6 new datasets with different multipliers for the wet compound class.
other minority classes are balanced to match the majority class count.
outputs the final datasets in original units so they can be dropped into the pipeline seamlessly.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

# config
DATA_PATH = "data/processed/dataset2.csv"
OUTDIR = "data/processed/SMOTE"
SEED = 42

WET_MULTIPLIERS = [
    ("wet_x1", 1.0),
    ("wet_100", 2.0),
    ("wet_200", 3.0),
    ("wet_300", 4.0),
    ("wet_400", 5.0),
    ("wet_500", 6.0),
]

ID_COLS = ["race_id", "driver_id", "lapno"]


def inverse_to_original_units(
    x_res: np.ndarray,
    preprocessor: ColumnTransformer,
    num_cols: list[str],
    cat_cols: list[str],
) -> pd.DataFrame:
    """
    reverses the standard scaling and ordinal encoding so the smote outputs
    look like the original dataset again.
    """
    n_num = len(num_cols)
    n_cat = len(cat_cols)

    x_num = x_res[:, :n_num]
    x_cat = x_res[:, n_num:n_num + n_cat]

    scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]
    enc = preprocessor.named_transformers_["cat"].named_steps["enc"]

    x_num_inv = scaler.inverse_transform(x_num)

    # ensure categoricals map back nicely
    x_cat_rounded = np.rint(x_cat).astype(int)
    x_cat_inv = enc.inverse_transform(x_cat_rounded.astype(float))

    df_inv = pd.DataFrame(x_num_inv, columns=num_cols)
    for j, col in enumerate(cat_cols):
        df_inv[col] = x_cat_inv[:, j]

    return df_inv


def build_sampling_strategy(y_enc: np.ndarray, wet_mult: float, wet_label: int) -> dict[int, int]:
    """
    figures out exactly how many samples each class needs.
    brings all minority classes up to the majority count, except for wet which is capped by the multiplier.
    """
    counts = pd.Series(y_enc).value_counts().to_dict()
    maj_label = max(counts, key=counts.get)
    maj_count = counts[maj_label]

    strategy = {}

    for cls, n in counts.items():
        if cls == maj_label:
            continue

        if cls == wet_label:
            wet_count = counts.get(wet_label, 0)
            if wet_count >= 2:
                wet_target = int(min(round(wet_count * wet_mult), maj_count))
                if wet_target > wet_count:
                    strategy[wet_label] = wet_target
        else:
            if n < maj_count:
                strategy[int(cls)] = int(maj_count)

    return strategy


def get_safe_k_neighbors(y_enc: np.ndarray, strategy: dict[int, int], k_max: int = 5) -> Optional[int]:
    """smote needs enough neighbors to work with. this ensures we dont crash if a class is too rare."""
    if not strategy:
        return None

    counts = pd.Series(y_enc).value_counts()
    min_n = min(int(counts[c]) for c in strategy.keys() if c in counts.index)

    if min_n < 2:
        return None

    return int(max(1, min(k_max, min_n - 1)))


def build_resampled_metadata(
    meta_df: pd.DataFrame,
    y_orig: pd.Series,
    y_res_enc: np.ndarray,
    le: LabelEncoder,
    seed: int,
) -> pd.DataFrame:
    """
    keeps the original metadata rows for original samples and assigns metadata
    to synthetic rows by sampling from real rows of the same target class.
    allows us to run evaluate_stack.py
    """
    rng = np.random.default_rng(seed)

    n_orig = len(meta_df)
    n_res = len(y_res_enc)

    if n_res == n_orig:
        return meta_df.reset_index(drop=True).copy()

    synth_n = n_res - n_orig
    synth_labels = le.inverse_transform(y_res_enc[n_orig:])

    meta_orig = meta_df.reset_index(drop=True).copy()
    y_orig = y_orig.reset_index(drop=True).astype(str)

    synth_meta = pd.DataFrame(index=range(synth_n), columns=meta_orig.columns)

    for cls_name in np.unique(synth_labels):
        cls_mask = synth_labels == cls_name
        n_cls = int(np.sum(cls_mask))

        pool = meta_orig.loc[y_orig == cls_name].reset_index(drop=True)
        if pool.empty:
            raise ValueError(f"no metadata rows available for class {cls_name}")

        chosen_idx = rng.integers(0, len(pool), size=n_cls)
        synth_meta.loc[cls_mask, :] = pool.iloc[chosen_idx].to_numpy()

    out = pd.concat([meta_orig, synth_meta], ignore_index=True)

    # restore integer-like id columns if present
    for col in ["race_id", "driver_id", "lapno"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round().astype("Int64")

    return out


def generate_smote_datasets():
    Path(OUTDIR).mkdir(parents=True, exist_ok=True)

    print("loading dataset...")
    df = pd.read_csv(DATA_PATH, na_values=[""])

    target = "y_compound"
    original_cols = list(df.columns)

    id_cols = [c for c in ID_COLS if c in df.columns]
    meta_df = df[id_cols].copy()

    # remove ids from the feature matrix so smote does not interpolate them
    x_all = df.drop(columns=id_cols + [target], errors="ignore")
    y_all = df[target].astype(str)

    cat_cols = [
        "pit_stops_so_far",
        "current_compound",
        "race_track",
        "fulfilled_second_compound",
        "is_wet_race",
        "is_raining",
        "gap_behind_s_missing",
        "tyre_age_diff_to_ahead_missing",
        "rejoin_gap_ahead_est_s_missing",
        "rejoin_gap_behind_est_s_missing",
    ]

    num_cols = [
        "race_progress",
        "minutes_rain",
        "gap_behind_s",
        "n_cars_within_5s_ahead",
        "tyre_age_diff_to_ahead",
        "rejoin_gap_ahead_est_s",
        "rejoin_gap_behind_est_s",
    ]

    # only keep columns that actually exist in the dataframe to prevent crashes
    cat_cols = [c for c in cat_cols if c in x_all.columns]
    num_cols = [c for c in num_cols if c in x_all.columns]

    preprocessor = ColumnTransformer(
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

    cat_indices = list(range(len(num_cols), len(num_cols) + len(cat_cols)))

    le = LabelEncoder()
    y_enc_all = le.fit_transform(y_all)
    class_names = list(le.classes_)

    if "WET" not in class_names:
        raise ValueError("wet compound missing from labels. smote generator expects wet.")

    wet_label = int(le.transform(["WET"])[0])

    print("fitting preprocessor on full dataset...")
    x_full = preprocessor.fit_transform(x_all)

    for setting_name, wet_mult in WET_MULTIPLIERS:
        print(f"generating dataset for {setting_name} (wet multiplier x{wet_mult})")

        strategy = build_sampling_strategy(y_enc_all, wet_mult, wet_label)
        k = get_safe_k_neighbors(y_enc_all, strategy)

        if strategy and k is not None:
            sampler = SMOTENC(
                categorical_features=cat_indices,
                sampling_strategy=strategy,
                k_neighbors=k,
                random_state=SEED,
            )
            x_res, y_res = sampler.fit_resample(x_full, y_enc_all)
        else:
            x_res, y_res = x_full, y_enc_all

        print("  reversing transformations to save in original units...")
        df_features = inverse_to_original_units(x_res, preprocessor, num_cols, cat_cols)
        df_features[target] = le.inverse_transform(y_res)

        print("  rebuilding metadata columns...")
        df_meta = build_resampled_metadata(meta_df, y_all, y_res, le, seed=SEED)

        df_out = pd.concat([df_meta.reset_index(drop=True), df_features.reset_index(drop=True)], axis=1)

        # restore original column order exactly
        missing_cols = [c for c in original_cols if c not in df_out.columns]
        if missing_cols:
            raise ValueError(f"output is missing expected columns: {missing_cols}")

        df_out = df_out[original_cols]

        out_csv = os.path.join(OUTDIR, f"{setting_name}_dataset.csv")
        df_out.to_csv(out_csv, index=False)
        print(f"  saved to {out_csv}")

    print("all datasets generated successfully.")


if __name__ == "__main__":
    generate_smote_datasets()