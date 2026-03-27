from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
import pandas as pd


# -----------------------------
# Constants
# -----------------------------

COMPOUND_CLASSES: Tuple[str, ...] = ("HARD", "MEDIUM", "SOFT", "INTERMEDIATE", "WET")
COMPOUND_TO_INT: Dict[str, int] = {c: i for i, c in enumerate(COMPOUND_CLASSES)}
INT_TO_COMPOUND: Dict[int, str] = {i: c for c, i in COMPOUND_TO_INT.items()}

# holdout race IDs (for case studies, 64 and 139 are removed because the races are unsuitable for training)
HOLDOUT_RACE_IDS: Tuple[int, ...] = (53, 73, 24, 75, 2, 64, 139)
#HOLDOUT_RACE_IDS: Tuple[int, ...] = (53, 73, 24, 100, 2, 64, 139)

# hard-coded categorical features for this project
CATEGORICAL_FEATURES: set[str] = {
    "fcy_status",
    "track_category",
    "position",
    "pit_stops_so_far",
    "race_track",
    "current_compound",
    "tyre_change_pursuer",
    "fulfilled_second_compound",
    "close_ahead",
    "is_wet_race",
    "is_raining",
    "gap_behind_s_missing",
    "tyre_age_diff_to_ahead_missing",
    "rejoin_gap_ahead_est_s_missing",
    "rejoin_gap_behind_est_s_missing",
    "tyre_age_missing",
    #"pit_stops_left",
}

def _exclude_holdout_races(
    df: pd.DataFrame,
    race_ids: Sequence[int] = HOLDOUT_RACE_IDS,
) -> pd.DataFrame:
    if "race_id" not in df.columns:
        return df
    return df.loc[~df["race_id"].isin(set(map(int, race_ids)))].copy()


def _read_any(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported file type: {path.suffix}. Use .csv or .parquet")


def _normalize_compound_series(s: pd.Series, *, allow_null: bool) -> pd.Series:
    s = s.replace({"": np.nan, "None": np.nan, "nan": np.nan})
    s = s.apply(lambda x: x.strip().upper() if isinstance(x, str) else x)

    if not allow_null and s.isna().any():
        raise ValueError("Compound column has missing values")

    non_null = s.dropna().unique().tolist()
    unknown = [c for c in non_null if c not in COMPOUND_TO_INT]
    if unknown:
        raise ValueError(f"Unexpected compound values: {unknown}")

    return s


# -----------------------------
# Sequence builders
# -----------------------------

def build_feature_sequences(
    keys: pd.DataFrame,
    X_all: np.ndarray,
    y_all: pd.Series,
    *,
    seq_len: int = 8,
    pad_left: bool = False,
    add_timestep_mask: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = {"race_id", "driver_id", "lapno"}
    if not required.issubset(keys.columns):
        raise ValueError(f"keys must contain columns {sorted(required)}")

    n = len(keys)
    if X_all.shape[0] != n or len(y_all) != n:
        raise ValueError("keys, X_all, and y_all must have the same number of rows")

    if not isinstance(y_all, pd.Series):
        y_all = pd.Series(y_all)

    df_idx = keys[["race_id", "driver_id", "lapno"]].copy()
    df_idx["_row"] = np.arange(n, dtype=int)
    df_idx = df_idx.sort_values(["race_id", "driver_id", "lapno"], kind="mergesort")

    d = X_all.shape[1]
    d_out = d + 1 if add_timestep_mask else d

    X_seq_list = []
    y_seq_list = []
    idx_last_list = []
    seq_idx_list = []
    eff_len_list = []

    for (_, _), g in df_idx.groupby(["race_id", "driver_id"], sort=False):
        g_rows = g["_row"].to_numpy(dtype=int)

        if len(g_rows) == 0:
            continue

        if (not pad_left) and (len(g_rows) < seq_len):
            continue

        start_end = 0 if pad_left else (seq_len - 1)

        for end in range(start_end, len(g_rows)):
            if pad_left:
                real = g_rows[max(0, end - seq_len + 1): end + 1]
                n_pad = seq_len - len(real)
                window = np.concatenate([np.full(n_pad, -1, dtype=int), real])
            else:
                window = g_rows[end - seq_len + 1: end + 1]

            real_mask = window != -1
            eff_len = int(np.sum(real_mask))

            Xw = np.zeros((seq_len, d), dtype=np.float32)
            if eff_len > 0:
                Xw[real_mask] = X_all[window[real_mask]].astype(np.float32, copy=False)

            if add_timestep_mask:
                mask = real_mask.astype(np.float32).reshape(seq_len, 1)
                Xw = np.concatenate([Xw, mask], axis=1)

            X_seq_list.append(Xw)
            y_seq_list.append(int(y_all.iloc[g_rows[end]]))
            idx_last_list.append(int(g_rows[end]))
            seq_idx_list.append(window)
            eff_len_list.append(eff_len)

    if not X_seq_list:
        return (
            np.zeros((0, seq_len, d_out), dtype=np.float32),
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=int),
            np.zeros((0, seq_len), dtype=int),
            np.zeros((0,), dtype=int),
        )

    return (
        np.asarray(X_seq_list, dtype=np.float32),
        np.asarray(y_seq_list, dtype=int),
        np.asarray(idx_last_list, dtype=int),
        np.asarray(seq_idx_list, dtype=int),
        np.asarray(eff_len_list, dtype=int),
    )


def build_prob_sequences_4lap(
    keys_df: pd.DataFrame,
    proba_pos: np.ndarray,
    y: pd.Series,
    *,
    seq_len: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = {"race_id", "driver_id", "lapno"}
    if not required.issubset(keys_df.columns):
        raise ValueError("keys_df must have columns: race_id, driver_id, lapno")

    keys_df = keys_df.reset_index(drop=True)
    proba_pos = np.asarray(proba_pos).reshape(-1)

    if len(proba_pos) != len(keys_df):
        raise ValueError("proba_pos must be same length as keys_df")
    if len(y) != len(keys_df):
        raise ValueError("y must be same length as keys_df")

    order = keys_df.sort_values(["race_id", "driver_id", "lapno"], kind="mergesort").index.to_numpy()

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    idx_list: List[int] = []
    seq_idx_list: List[np.ndarray] = []

    tmp = keys_df.loc[order, ["race_id", "driver_id", "lapno"]].copy()
    tmp["_idx"] = order

    for (_, _), g in tmp.groupby(["race_id", "driver_id"], sort=False):
        idxs = g["_idx"].to_numpy(dtype=int)
        if len(idxs) < seq_len:
            continue

        for j in range(seq_len - 1, len(idxs)):
            last_idx = int(idxs[j])
            seq_idxs = idxs[j - seq_len + 1 : j + 1].astype(int)

            X_list.append(proba_pos[seq_idxs].reshape(seq_len, 1))
            y_list.append(int(y.iloc[last_idx]))
            idx_list.append(last_idx)
            seq_idx_list.append(seq_idxs)

    if not X_list:
        return (
            np.zeros((0, seq_len, 1), dtype=np.float32),
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=int),
            np.zeros((0, seq_len), dtype=int),
        )

    return (
        np.stack(X_list, axis=0).astype(np.float32),
        np.asarray(y_list, dtype=int),
        np.asarray(idx_list, dtype=int),
        np.stack(seq_idx_list, axis=0).astype(int),
    )


# -----------------------------
# Loading
# -----------------------------

def load_stage1_dataset(
    path: Union[str, Path],
    *,
    exclude_holdouts: bool = True,
) -> pd.DataFrame:
    df = _read_any(path)

    for col in ["race_id", "driver_id", "lapno", "y_pit"]:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)

    if exclude_holdouts:
        df = _exclude_holdout_races(df)

    df["current_compound"] = _normalize_compound_series(df["current_compound"], allow_null=True)

    if "race_track" in df.columns:
        df["race_track"] = df["race_track"].replace({"": np.nan, "None": np.nan, "nan": np.nan})
        df["race_track"] = df["race_track"].apply(lambda x: x.strip() if isinstance(x, str) else x)

    if "fcy_status" in df.columns:
        df["fcy_status"] = df["fcy_status"].replace({"": np.nan, "None": np.nan, "nan": np.nan})
        df["fcy_status"] = df["fcy_status"].apply(lambda x: x.strip() if isinstance(x, str) else x)
        df["fcy_status"] = df["fcy_status"].astype("category")

    numeric_cols = [
        "race_progress",
        "position",
        "pit_stops_so_far",
        "track_category",
        "lap_time",
        "interval",
        "tyre_age",
        "gap_behind_s",
        "n_cars_within_5s_ahead",
        "tyre_age_diff_to_ahead",
        "rejoin_gap_ahead_est_s",
        "rejoin_gap_behind_est_s",
        "minutes_rain",
        "close_ahead",
        "is_wet_race",
        "is_raining",
        "fulfilled_second_compound",
        "gap_behind_s_missing",
        "tyre_age_diff_to_ahead_missing",
        "rejoin_gap_ahead_est_s_missing",
        "rejoin_gap_behind_est_s_missing",
        "tyre_age_missing",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_stage2_dataset(
    path: Union[str, Path],
    *,
    strict: bool = True,
    exclude_holdouts: bool = True,
) -> pd.DataFrame:
    df = _read_any(path)

    df["race_id"] = pd.to_numeric(df["race_id"], errors="raise").astype(int)

    if exclude_holdouts:
        df = _exclude_holdout_races(df)

    df["y_compound"] = _normalize_compound_series(df["y_compound"], allow_null=True)

    if strict and df["y_compound"].isna().any():
        n_missing = int(df["y_compound"].isna().sum())
        df = df.loc[~df["y_compound"].isna()].copy()
        warnings.warn(
            f"load_stage2_dataset dropped {n_missing} rows with missing y_compound.",
            stacklevel=2,
        )

    df["current_compound"] = _normalize_compound_series(df["current_compound"], allow_null=True)

    if "race_track" in df.columns:
        df["race_track"] = df["race_track"].replace({"": np.nan, "None": np.nan, "nan": np.nan})
        df["race_track"] = df["race_track"].apply(lambda x: x.strip() if isinstance(x, str) else x)

    numeric_cols = [
        "race_progress",
        "pit_stops_so_far",
        "minutes_rain",
        "is_wet_race",
        "is_raining",
        "fulfilled_second_compound",
        "gap_behind_s",
        "n_cars_within_5s_ahead",
        "tyre_age_diff_to_ahead",
        "rejoin_gap_ahead_est_s",
        "rejoin_gap_behind_est_s",
        "gap_behind_s_missing",
        "tyre_age_diff_to_ahead_missing",
        "rejoin_gap_ahead_est_s_missing",
        "rejoin_gap_behind_est_s_missing",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def add_pit_next_compound_labels_from_stage1(
    df: pd.DataFrame,
    *,
    pit_col: str = "y_pit",
    compound_col: str = "current_compound",
    out_col: str = "y_compound",
) -> pd.DataFrame:
    df2 = df.copy()
    df2 = df2.sort_values(["race_id", "driver_id", "lapno"], kind="mergesort").reset_index(drop=True)

    next_comp = df2.groupby(["race_id", "driver_id"], sort=False)[compound_col].shift(-1)
    df2[out_col] = np.where(df2[pit_col].astype(int).to_numpy() == 1, next_comp, np.nan)
    df2[out_col] = _normalize_compound_series(df2[out_col], allow_null=True)

    return df2


def encode_y_compound(
    df: pd.DataFrame,
    col: str = "y_compound",
    out_col: str = "y_compound_encoded",
) -> pd.DataFrame:
    df2 = df.copy()
    df2[col] = _normalize_compound_series(df2[col], allow_null=True)
    df2[out_col] = df2[col].map(COMPOUND_TO_INT)
    return df2


def load_stage2_seq_from_stage1(path: Union[str, Path]) -> pd.DataFrame:
    df = load_stage1_dataset(path)
    df = add_pit_next_compound_labels_from_stage1(
        df,
        pit_col="y_pit",
        compound_col="current_compound",
        out_col="y_compound",
    )
    df = encode_y_compound(df, col="y_compound", out_col="y_compound_encoded")
    return df


# -----------------------------
# Feature extraction
# -----------------------------

def make_xy(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    drop_cols = list(drop_cols) if drop_cols is not None else []
    to_drop = set(drop_cols + [target_col])

    X = df.drop(columns=[c for c in to_drop if c in df.columns]).copy()
    y = df[target_col].copy()
    return X, y


def get_stage1_xy(
    df1: pd.DataFrame,
    *,
    drop_cols: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    default_drop = ["y_pit", "race_id", "driver_id"]
    if drop_cols is None:
        drop_cols = default_drop
    else:
        drop_cols = list(set(drop_cols).union(default_drop))

    return make_xy(df1, target_col="y_pit", drop_cols=drop_cols)


def get_stage2_xy(
    df2: pd.DataFrame,
    *,
    drop_cols: Optional[Sequence[str]] = None,
    encoded_target_col: str = "y_compound_encoded",
) -> Tuple[pd.DataFrame, pd.Series]:
    df2 = encode_y_compound(df2, col="y_compound", out_col=encoded_target_col)
    df2 = df2.loc[~df2[encoded_target_col].isna()].copy()

    default_drop = ["y_compound", encoded_target_col, "race_id"]
    if drop_cols is None:
        drop_cols = default_drop
    else:
        drop_cols = list(set(drop_cols).union(default_drop))

    X, y = make_xy(df2, target_col=encoded_target_col, drop_cols=drop_cols)
    return X, y.astype(int)


def infer_feature_types(
    X_df: pd.DataFrame,
    *,
    force_categorical: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[str]]:
    cat_set = set(CATEGORICAL_FEATURES)
    if force_categorical is not None:
        cat_set.update(force_categorical)

    num_cols: List[str] = []
    cat_cols: List[str] = []

    for col in X_df.columns:
        if col in cat_set:
            cat_cols.append(col)
        else:
            num_cols.append(col)

    return num_cols, cat_cols


# -----------------------------
# Race-wise folds
# -----------------------------

@dataclass(frozen=True)
class FoldBundle:
    folds: List[Tuple[np.ndarray, np.ndarray]]
    fold_race_ids: List[List[int]]


def _group_label_counts(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((n_classes,), dtype=float)
    for v in y:
        if np.isnan(v):
            continue
        out[int(v)] += 1.0
    return out


def make_race_group_folds(
    df: pd.DataFrame,
    *,
    group_col: str = "race_id",
    target_col: str,
    n_splits: int = 5,
    seed: int = 42,
    balance_labels: bool = True,
) -> FoldBundle:
    """
    aggregates laps by the race they belong to
    """
    y_raw = df[target_col].to_numpy()

    n_classes = None
    if balance_labels:
        uniq = pd.Series(y_raw).dropna().unique().tolist()
        if uniq:
            uniq_int = sorted({int(u) for u in uniq})
            n_classes = max(uniq_int) + 1 if uniq_int else None

    groups = df.groupby(group_col, sort=False).indices
    race_ids = list(groups.keys())

    rng = np.random.default_rng(seed)
    rng.shuffle(race_ids)

    race_meta = []
    for rid in race_ids:
        idx = groups[rid]
        size = len(idx)
        counts = _group_label_counts(y_raw[idx], n_classes) if (balance_labels and n_classes is not None) else None
        race_meta.append((rid, size, counts))

    race_meta.sort(key=lambda t: t[1], reverse=True)

    fold_sizes = np.zeros((n_splits,), dtype=float)
    fold_counts = np.zeros((n_splits, n_classes), dtype=float) if (balance_labels and n_classes is not None) else None
    fold_races: List[List[int]] = [[] for _ in range(n_splits)]

    if balance_labels and n_classes is not None:
        global_counts = np.zeros((n_classes,), dtype=float)
        for _, _, c in race_meta:
            global_counts += c
        global_props = global_counts / max(global_counts.sum(), 1.0)

    for rid, size, counts in race_meta:
        best_fold = None
        best_score = None

        for f in range(n_splits):
            new_size = fold_sizes[f] + size
            score = new_size

            if balance_labels and n_classes is not None and counts is not None:
                new_counts = fold_counts[f] + counts
                new_props = new_counts / max(new_counts.sum(), 1.0)
                label_score = np.sum(np.abs(new_props - global_props))
                score = new_size + (label_score * (size * 0.05))

            if best_score is None or score < best_score or (score == best_score and fold_sizes[f] < fold_sizes[best_fold]):
                best_score = score
                best_fold = f

        fold_races[best_fold].append(int(rid))
        fold_sizes[best_fold] += size
        if fold_counts is not None and counts is not None:
            fold_counts[best_fold] += counts

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    all_idx = np.arange(len(df))

    for f in range(n_splits):
        valid_rids = set(fold_races[f])
        valid_mask = df[group_col].isin(valid_rids).to_numpy()
        valid_idx = all_idx[valid_mask]
        train_idx = all_idx[~valid_mask]
        folds.append((train_idx, valid_idx))

    return FoldBundle(folds=folds, fold_race_ids=fold_races)