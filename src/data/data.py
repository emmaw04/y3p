# src/data.py
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

import warnings

# encoding labels for compounds
COMPOUND_CLASSES: Tuple[str, ...] = ("HARD", "MEDIUM", "SOFT", "INTERMEDIATE", "WET")
COMPOUND_TO_INT: Dict[str, int] = {c: i for i, c in enumerate(COMPOUND_CLASSES)}
INT_TO_COMPOUND: Dict[int, str] = {i: c for c, i in COMPOUND_TO_INT.items()}

# holdout race IDs (for case studies)
HOLDOUT_RACE_IDS: Tuple[int, ...] = (53, 73, 24, 75, 2)

def _exclude_holdout_races(df: pd.DataFrame, race_ids: Sequence[int] = HOLDOUT_RACE_IDS) -> pd.DataFrame:
    """remove rows belonging to holdout races"""
    if "race_id" not in df.columns:
        return df
    race_ids_set = set(int(r) for r in race_ids)
    return df.loc[~df["race_id"].isin(race_ids_set)].copy()

# -----------------------------
# Schema for each dataset
# -----------------------------

REQUIRED_STAGE1_COLUMNS: Tuple[str, ...] = (
    "race_id",
    "driver_id",
    "lapno",
    "race_progress",
    "position",
    "fcy_status",
    "pit_stops_so_far",
    "tyre_change_pursuer",
    "track_category",
    "y_pit",
    "current_compound",
    "lap_time",
    "interval",
    "tyre_age",
    "close_ahead",
    "race_track",
    "rained_yet",
    "is_raining",
    "minutes_rain",
    "fulfilled_second_compound",
)

REQUIRED_STAGE2_COLUMNS: Tuple[str, ...] = (
    "race_id",
    "race_progress",
    "pit_stops_so_far",
    "current_compound",
    "race_track",
    "fulfilled_second_compound",
    "rained_yet",
    "is_raining",
    "minutes_rain",
    "y_compound",
)


# -----------------------------
# Loading + validation
# -----------------------------
def _read_any(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported file type: {path.suffix}. Use .csv or .parquet")


def _normalize_compound_series(s: pd.Series, *, allow_null: bool) -> pd.Series:
    s = s.replace({"": np.nan, "None": np.nan, "nan": np.nan})
    s = s.apply(lambda x: x.strip().upper() if isinstance(x, str) else x)

    if not allow_null and s.isna().any():
        bad_n = int(s.isna().sum())
        raise ValueError(f"Compound column has {bad_n} missing values; expected none.")

    non_null = s.dropna().unique().tolist()
    unknown = [c for c in non_null if c not in COMPOUND_TO_INT]
    if unknown:
        raise ValueError(
            f"Compound column has unexpected values: {unknown}. "
            f"Expected subset of {list(COMPOUND_TO_INT.keys())} (or null if allowed)."
        )
    return s

def build_feature_sequences(
    keys: "pd.DataFrame",
    X_all: np.ndarray,
    y_all: "pd.Series",
    *,
    seq_len: int = 8,
    pad_left: bool = False,
    add_timestep_mask: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build fixed-length sequences for each (race_id, driver_id) time series.

    Args:
        keys: DataFrame with columns ["race_id","driver_id","lapno"] aligned with rows of X_all/y_all
        X_all: (N, D) numpy array of features for all rows (already preprocessed)
        y_all: length-N binary labels (0/1), ideally a pd.Series aligned with keys
        seq_len: timesteps per sequence (e.g., 8)
        pad_left: if True, create sequences for early laps by left-padding with -1 indices
        add_timestep_mask: if True, append a per-timestep feature (1 if real timestep else 0)

    Returns:
        X_seq: (M, seq_len, D or D+1)
        y_seq: (M,) label from the last real timestep
        idx_last: (M,) original row index of last timestep
        seq_idx: (M, seq_len) original row indices (padded timesteps = -1)
        eff_len: (M,) number of real timesteps in the window (1..seq_len)
    """
    required = {"race_id", "driver_id", "lapno"}
    if not required.issubset(keys.columns):
        raise ValueError(f"keys must contain columns {sorted(required)}")

    N = len(keys)
    if X_all.shape[0] != N or len(y_all) != N:
        raise ValueError("keys, X_all, and y_all must have the same number of rows")

    # Ensure y_all is indexable by iloc
    if not isinstance(y_all, pd.Series):
        y_all = pd.Series(y_all)

    df_idx = keys[["race_id", "driver_id", "lapno"]].copy()
    df_idx["_row"] = np.arange(N, dtype=int)
    df_idx = df_idx.sort_values(["race_id", "driver_id", "lapno"], kind="mergesort")

    D = X_all.shape[1]
    D_out = D + 1 if add_timestep_mask else D

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

            real_mask = (window != -1)
            eff_len = int(np.sum(real_mask))

            # padded feature tensor
            Xw = np.zeros((seq_len, D), dtype=np.float32)
            if eff_len > 0:
                Xw[real_mask] = X_all[window[real_mask]].astype(np.float32, copy=False)

            if add_timestep_mask:
                m = real_mask.astype(np.float32).reshape(seq_len, 1)
                Xw = np.concatenate([Xw, m], axis=1)  # (seq_len, D+1)

            X_seq_list.append(Xw)
            y_seq_list.append(int(y_all.iloc[g_rows[end]]))
            idx_last_list.append(int(g_rows[end]))
            seq_idx_list.append(window)
            eff_len_list.append(eff_len)

    if len(X_seq_list) == 0:
        return (
            np.zeros((0, seq_len, D_out), dtype=np.float32),
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=int),
            np.zeros((0, seq_len), dtype=int),
            np.zeros((0,), dtype=int),
        )

    X_seq = np.asarray(X_seq_list, dtype=np.float32)
    y_seq = np.asarray(y_seq_list, dtype=int)
    idx_last = np.asarray(idx_last_list, dtype=int)
    seq_idx = np.asarray(seq_idx_list, dtype=int)
    eff_len = np.asarray(eff_len_list, dtype=int)

    return X_seq, y_seq, idx_last, seq_idx, eff_len

def _coerce_boolish_to_01(series: pd.Series, colname: str, *, allow_null: bool = True) -> pd.Series:
    s = series.replace({"": np.nan, "None": np.nan, "nan": np.nan})

    true_vals = {"true", "1", "yes", "y", "t"}
    false_vals = {"false", "0", "no", "n", "f"}

    def _map(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, (bool, np.bool_)):
            return int(v)
        if isinstance(v, (int, np.integer, float, np.floating)) and not pd.isna(v):
            if v in (0, 1):
                return int(v)
        if isinstance(v, str):
            vv = v.strip().lower()
            if vv in true_vals:
                return 1
            if vv in false_vals:
                return 0
        return "__INVALID__"

    out = s.map(_map)

    invalid = out.eq("__INVALID__")
    if invalid.any():
        bad = sorted(series[invalid].dropna().astype(str).unique().tolist())
        raise ValueError(f"{colname} contains invalid boolean-like values: {bad}")

    out = out.astype("float")
    if not allow_null and out.isna().any():
        raise ValueError(f"{colname} has missing values; expected none.")

    return out


def load_stage1_dataset(
    path: Union[str, Path],
    *,
    exclude_holdouts: bool = True,
) -> pd.DataFrame:
    """
    Load and validate the Stage 1 lap-level dataset.

    Expected target:
        y_pit in {0, 1}

    By default, holdout races are excluded so this loader can be used directly
    for training and cross-validation without leaking final test races.
    """
    df = _read_any(path)

    missing = [c for c in REQUIRED_STAGE1_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Stage 1 dataset missing required columns: {missing}")

    # Core identifiers
    for col in ["race_id", "driver_id", "lapno"]:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)

    if exclude_holdouts:
        df = _exclude_holdout_races(df)

    # Target
    if df["y_pit"].isna().any():
        n_missing = int(df["y_pit"].isna().sum())
        raise ValueError(f"y_pit contains {n_missing} missing values; expected none.")

    df["y_pit"] = pd.to_numeric(df["y_pit"], errors="raise").astype(int)
    y_vals = set(df["y_pit"].unique().tolist())
    if not y_vals.issubset({0, 1}):
        raise ValueError(f"y_pit must be binary 0/1. Found values: {sorted(y_vals)}")

    # Canonicalise categorical/string fields
    df["current_compound"] = _normalize_compound_series(df["current_compound"], allow_null=True)

    df["race_track"] = (
        df["race_track"]
        .replace({"": np.nan, "None": np.nan, "nan": np.nan})
        .apply(lambda x: x.strip() if isinstance(x, str) else x)
    )

    df["fcy_status"] = (
        df["fcy_status"]
        .replace({"": np.nan, "None": np.nan, "nan": np.nan})
        .apply(lambda x: x.strip() if isinstance(x, str) else x)
        .astype("category")
    )

    # Boolean / binary fields
    for col in ["close_ahead", "rained_yet", "is_raining", "tyre_change_pursuer"]:
        df[col] = _coerce_boolish_to_01(df[col], col, allow_null=True)

    # fulfilled_second_compound may be binary or categorical
    try:
        df["fulfilled_second_compound"] = _coerce_boolish_to_01(
            df["fulfilled_second_compound"],
            "fulfilled_second_compound",
            allow_null=True,
        )
    except ValueError:
        df["fulfilled_second_compound"] = (
            df["fulfilled_second_compound"]
            .replace({"": np.nan, "None": np.nan, "nan": np.nan})
            .apply(lambda x: x.strip().upper() if isinstance(x, str) else x)
            .astype("category")
        )

    # Numeric fields
    numeric_cols = [
        "race_progress",
        "position",
        "pit_stops_so_far",
        "track_category",
        "lap_time",
        "interval",
        "tyre_age",
        "minutes_rain",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_stage2_dataset(
    path: Union[str, Path],
    *,
    strict: bool = True,
    exclude_holdouts: bool = True,
) -> pd.DataFrame:
    """
    Load and validate the Stage 2 pit-stop dataset.

    Expected target:
        y_compound in {"HARD", "MEDIUM", "SOFT", "INTERMEDIATE", "WET"}

    If strict=True, rows with missing y_compound are dropped after normalisation.
    By default, holdout races are excluded so this loader can be used directly
    for training and cross-validation.
    """
    df = _read_any(path)

    missing = [c for c in REQUIRED_STAGE2_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Stage 2 dataset missing required columns: {missing}")

    df["race_id"] = pd.to_numeric(df["race_id"], errors="raise").astype(int)

    if exclude_holdouts:
        df = _exclude_holdout_races(df)

    # Target
    df["y_compound"] = _normalize_compound_series(df["y_compound"], allow_null=True)

    if strict:
        n_missing = int(df["y_compound"].isna().sum())
        if n_missing > 0:
            df = df.loc[~df["y_compound"].isna()].copy()
            warnings.warn(
                f"load_stage2_dataset dropped {n_missing} rows with missing y_compound.",
                stacklevel=2,
            )

    non_null = df["y_compound"].dropna().unique().tolist()
    unknown = [c for c in non_null if c not in COMPOUND_TO_INT]
    if unknown:
        raise ValueError(
            f"y_compound contains unexpected values: {unknown}. "
            f"Expected subset of {list(COMPOUND_TO_INT.keys())}."
        )

    # Canonicalise categorical/string fields
    df["current_compound"] = _normalize_compound_series(df["current_compound"], allow_null=True)

    df["race_track"] = (
        df["race_track"]
        .replace({"": np.nan, "None": np.nan, "nan": np.nan})
        .apply(lambda x: x.strip() if isinstance(x, str) else x)
    )

    # Boolean / binary fields
    for col in ["rained_yet", "is_raining"]:
        df[col] = _coerce_boolish_to_01(df[col], col, allow_null=True)

    # fulfilled_second_compound may be binary or categorical
    try:
        df["fulfilled_second_compound"] = _coerce_boolish_to_01(
            df["fulfilled_second_compound"],
            "fulfilled_second_compound",
            allow_null=True,
        )
    except ValueError:
        df["fulfilled_second_compound"] = (
            df["fulfilled_second_compound"]
            .replace({"": np.nan, "None": np.nan, "nan": np.nan})
            .apply(lambda x: x.strip().upper() if isinstance(x, str) else x)
            .astype("category")
        )

    # Numeric fields
    numeric_cols = [
        "race_progress",
        "pit_stops_so_far",
        "minutes_rain",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def load_stage2_seq_from_stage1(path: Union[str, Path]) -> pd.DataFrame:
    """
    Use stage1 (all laps) but add y_compound labels for pit laps only.
    Keeps ALL laps so sequences have context; labels are NaN except pit laps.
    """
    df = load_stage1_dataset(path)  # includes holdout exclusion already
    df = add_pit_next_compound_labels_from_stage1(df, pit_col="y_pit", compound_col="current_compound", out_col="y_compound")
    df = encode_y_compound(df, col="y_compound", out_col="y_compound_encoded")  # NaN where y_compound is NaN
    return df

# -----------------------------
# Row IDs
# -----------------------------
def add_row_id(df: pd.DataFrame, id_col: str = "row_id") -> pd.DataFrame:
    """
    Add a stable row_id.

    Stage 1: (race_id, driver_id, lapno) is usually unique.
    Stage 2: no driver_id/lapno, so we fall back to (race_id + original index) hash.
    """
    if id_col in df.columns:
        return df

    df2 = df.copy()

    if {"race_id", "driver_id", "lapno"}.issubset(df2.columns):
        keys = (
            df2["race_id"].astype(str) + "_" +
            df2["driver_id"].astype(str) + "_" +
            df2["lapno"].astype(str)
        )
        if keys.is_unique:
            df2[id_col] = keys
            return df2

        # fallback: hash key + index
        df2[id_col] = [
            hashlib.sha1(f"{r}|{d}|{l}|{i}".encode("utf-8")).hexdigest()
            for i, (r, d, l) in enumerate(zip(df2["race_id"], df2["driver_id"], df2["lapno"]))
        ]
        return df2

    # Stage 2 or anything else: hash race_id + index
    if "race_id" in df2.columns:
        df2[id_col] = [
            hashlib.sha1(f"{int(r)}|{i}".encode("utf-8")).hexdigest()
            for i, r in enumerate(df2["race_id"].tolist())
        ]
        return df2

    df2[id_col] = [hashlib.sha1(str(i).encode("utf-8")).hexdigest() for i in df2.index]
    return df2


# -----------------------------
# Feature/target extraction
# -----------------------------
def make_xy(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not in df columns")

    drop_cols = list(drop_cols) if drop_cols is not None else []
    to_drop = set(drop_cols + [target_col])

    X = df.drop(columns=[c for c in to_drop if c in df.columns]).copy()
    y = df[target_col].copy()
    return X, y


def encode_y_compound(df: pd.DataFrame, col: str = "y_compound", out_col: str = "y_compound_encoded") -> pd.DataFrame:
    df2 = df.copy()
    if col not in df2.columns:
        raise ValueError(f"'{col}' not in df")

    df2[col] = _normalize_compound_series(df2[col], allow_null=True)
    df2[out_col] = df2[col].map(COMPOUND_TO_INT)
    return df2

def add_pit_next_compound_labels_from_stage1(
    df: pd.DataFrame,
    *,
    pit_col: str = "y_pit",
    compound_col: str = "current_compound",
    out_col: str = "y_compound",
) -> pd.DataFrame:
    """
    From lap-by-lap stage1 df, create y_compound ONLY on pit laps:
      y_compound[t] = current_compound[t+1] for same (race_id, driver_id)
    Non-pit laps -> NaN.
    """
    df2 = df.copy()

    # Ensure sorted within each driver-race
    df2 = df2.sort_values(["race_id", "driver_id", "lapno"], kind="mergesort").reset_index(drop=True)

    # Next lap compound within each time series
    next_comp = df2.groupby(["race_id", "driver_id"], sort=False)[compound_col].shift(-1)

    # Label only where pit happened
    df2[out_col] = np.where(df2[pit_col].astype(int).to_numpy() == 1, next_comp, np.nan)

    # Normalise to canonical strings (HARD/MEDIUM/...)
    df2[out_col] = _normalize_compound_series(df2[out_col], allow_null=True)

    return df2

# -----------------------------
# Feature type inference
# -----------------------------
def infer_feature_types(
    X_df: pd.DataFrame,
    *,
    force_categorical: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[str]]:
    force = set(force_categorical or [])
    force.update({
        # stage 1 common cats
        "fcy_status",
        "track_category",
        "position",
        "pit_stops_so_far",
        "race_track",
        "current_compound",
        "tyre_change_pursuer",
        "fulfilled_second_compound",
    })

    num_cols: List[str] = []
    cat_cols: List[str] = []

    for col in X_df.columns:
        if col in force:
            cat_cols.append(col)
            continue

        dtype = X_df[col].dtype
        if dtype == bool or str(dtype) == "boolean":
            cat_cols.append(col)
            continue
        if str(dtype) == "category" or dtype == object:
            cat_cols.append(col)
            continue
        if pd.api.types.is_numeric_dtype(dtype):
            num_cols.append(col)
            continue

        cat_cols.append(col)

    return num_cols, cat_cols


# -----------------------------
# Grouped (race-wise) folds (no leakage across a race)
# -----------------------------
@dataclass(frozen=True)
class FoldBundle:
    folds: List[Tuple[np.ndarray, np.ndarray]]  # (train_idx, valid_idx)
    fold_race_ids: List[List[int]]              # which race_ids in each valid fold


def _group_label_counts(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Return counts vector length n_classes."""
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
    Create folds where validation sets contain whole races (race_id groups).
    Tries to balance:
      - total number of rows per fold (always)
      - label distribution per fold (if balance_labels=True)

    Works for:
      - binary y_pit (0/1)
      - multiclass y_compound_encoded (0..K-1)
    """
    if group_col not in df.columns:
        raise ValueError(f"'{group_col}' not in df")
    if target_col not in df.columns:
        raise ValueError(f"'{target_col}' not in df")

    # Prepare target values for label balancing
    y_raw = df[target_col].to_numpy()

    # Determine n_classes if we want to balance labels
    n_classes = None
    if balance_labels:
        uniq = pd.Series(y_raw).dropna().unique().tolist()
        # If binary or multiclass integers
        # (If target is y_pit, should be {0,1})
        if len(uniq) > 0:
            # handle float ints too
            uniq_int = sorted(list({int(u) for u in uniq}))
            n_classes = (max(uniq_int) + 1) if uniq_int else None

    # Group indices by race
    groups = df.groupby(group_col, sort=False).indices  # race_id -> np.array(indices)
    race_ids = list(groups.keys())

    rng = np.random.default_rng(seed)
    rng.shuffle(race_ids)

    # For each race, compute size and label counts (optional)
    race_meta = []
    for rid in race_ids:
        idx = groups[rid]
        size = len(idx)

        if balance_labels and n_classes is not None:
            y_g = y_raw[idx]
            counts = _group_label_counts(y_g, n_classes)
        else:
            counts = None

        race_meta.append((rid, size, counts))

    # Sort by size desc so greedy packing is stable
    race_meta.sort(key=lambda t: t[1], reverse=True)

    fold_sizes = np.zeros((n_splits,), dtype=float)
    fold_counts = np.zeros((n_splits, n_classes), dtype=float) if (balance_labels and n_classes is not None) else None
    fold_races: List[List[int]] = [[] for _ in range(n_splits)]

    # Global targets for label balancing
    if balance_labels and n_classes is not None:
        global_counts = np.zeros((n_classes,), dtype=float)
        for _, _, c in race_meta:
            global_counts += c
        # Avoid divide by zero
        global_props = global_counts / max(global_counts.sum(), 1.0)

    # Greedy assignment of races to folds
    for rid, size, counts in race_meta:
        best_fold = None
        best_score = None

        for f in range(n_splits):
            # size balance term
            new_size = fold_sizes[f] + size
            size_score = new_size  # minimize max sizes, but we compare relative below

            score = size_score

            if balance_labels and n_classes is not None and counts is not None:
                # label balance term: how far would this fold's label proportions move from global?
                new_counts = fold_counts[f] + counts
                new_props = new_counts / max(new_counts.sum(), 1.0)
                label_score = np.sum(np.abs(new_props - global_props))
                # weight label balance gently relative to size
                score = (size_score * 1.0) + (label_score * (size * 0.05))

            # choose fold with minimal score, tie-break by smallest current size
            if best_score is None or score < best_score or (score == best_score and fold_sizes[f] < fold_sizes[best_fold]):
                best_score = score
                best_fold = f

        fold_races[best_fold].append(int(rid))
        fold_sizes[best_fold] += size
        if fold_counts is not None and counts is not None:
            fold_counts[best_fold] += counts

    # Build (train_idx, valid_idx) arrays
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    all_idx = np.arange(len(df))

    for f in range(n_splits):
        valid_rids = set(fold_races[f])
        valid_mask = df[group_col].isin(valid_rids).to_numpy()
        valid_idx = all_idx[valid_mask]
        train_idx = all_idx[~valid_mask]
        folds.append((train_idx, valid_idx))

    return FoldBundle(folds=folds, fold_race_ids=fold_races)

def build_prob_sequences_4lap(
    keys_df: pd.DataFrame,
    proba_pos: np.ndarray,
    y: pd.Series,
    *,
    seq_len: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build fixed-length sequences of *probabilities* (e.g. FFNN outputs) per (race_id, driver_id).

    Returns:
        X_seq:   (n_seq, seq_len, 1)   probability sequences
        y_seq:   (n_seq,)              label at final timestep (0/1)
        idx_last:(n_seq,)              original row index of final timestep
        seq_idx: (n_seq, seq_len)      original row indices used in each sequence
    """
    keys_df = keys_df.reset_index(drop=True)
    required = {"race_id", "driver_id", "lapno"}
    if not required.issubset(keys_df.columns):
        raise ValueError("keys_df must have columns: race_id, driver_id, lapno")

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
        # return empty but well-formed arrays (nicer than throwing in CV)
        return (
            np.zeros((0, seq_len, 1), dtype=np.float32),
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=int),
            np.zeros((0, seq_len), dtype=int),
        )

    X_seq = np.stack(X_list, axis=0).astype(np.float32)
    y_seq = np.asarray(y_list, dtype=int)
    idx_last = np.asarray(idx_list, dtype=int)
    seq_idx = np.stack(seq_idx_list, axis=0).astype(int)

    return X_seq, y_seq, idx_last, seq_idx


def fold_iter(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    folds: List[Tuple[np.ndarray, np.ndarray]],
):
    if isinstance(y, pd.Series):
        y_arr = y.to_numpy()
    else:
        y_arr = np.asarray(y)

    for k, (tr_idx, va_idx) in enumerate(folds):
        yield (
            k,
            X.iloc[tr_idx],
            y_arr[tr_idx],
            X.iloc[va_idx],
            y_arr[va_idx],
            tr_idx,
            va_idx,
        )


# -----------------------------
# Convenience helpers for your two stages
# -----------------------------
def get_stage1_xy(
    df1: pd.DataFrame,
    *,
    drop_cols: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Stage 1: predict y_pit.
    Default: drop identifiers to reduce memorisation, but keep race_id in df for fold-making.
    """
    df1 = add_row_id(df1)

    default_drop = ["y_pit", "race_id", "driver_id"]  # <-- drop race/driver from features
    if drop_cols is None:
        drop_cols = default_drop
    else:
        drop_cols = list(set(drop_cols).union(default_drop))

    X, y = make_xy(df1, target_col="y_pit", drop_cols=drop_cols)
    return X, y


def get_stage2_xy(
    df2: pd.DataFrame,
    *,
    drop_cols: Optional[Sequence[str]] = None,
    encoded_target_col: str = "y_compound_encoded",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Stage 2: predict y_compound (encoded).
    """
    df2 = add_row_id(df2)
    df2 = encode_y_compound(df2, col="y_compound", out_col=encoded_target_col)

    if df2[encoded_target_col].isna().any():
        # stage2 dataset should basically never have missing y_compound
        df2 = df2[~df2[encoded_target_col].isna()].copy()

    default_drop = ["y_compound", encoded_target_col, "race_id"]  # drop race_id from features
    if drop_cols is None:
        drop_cols = default_drop
    else:
        drop_cols = list(set(drop_cols).union(default_drop))

    X, y = make_xy(df2, target_col=encoded_target_col, drop_cols=drop_cols)
    y = y.astype(int)
    return X, y
