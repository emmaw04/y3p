from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import warnings
import numpy as np
import pandas as pd


# constants for tyre compounds
COMPOUND_CLASSES: Tuple[str, ...] = ("HARD", "MEDIUM", "SOFT", "INTERMEDIATE", "WET")
COMPOUND_TO_INT: Dict[str, int] = {c: i for i, c in enumerate(COMPOUND_CLASSES)}
INT_TO_COMPOUND: Dict[int, str] = {i: c for c, i in COMPOUND_TO_INT.items()}

# holdout race ids for our case studies
# we skip sixty four and one hundred thirty nine since those races are too weird for training
HOLDOUT_RACE_IDS: Tuple[int, ...] = (53, 73, 24, 75, 2, 64, 139)

# hardcoded categorical features we care about for this project
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
}

def _exclude_holdout_races(
    df: pd.DataFrame,
    race_ids: Sequence[int] = HOLDOUT_RACE_IDS,
) -> pd.DataFrame:
    """
    drops any rows belonging to the holdout races
    """
    if "race_id" not in df.columns:
        return df
    return df.loc[~df["race_id"].isin(set(map(int, race_ids)))].copy()

def _read_any(path: Union[str, Path]) -> pd.DataFrame:
    """
    super simple helper to load a dataframe from csv
    """
    path = Path(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    
    raise ValueError(f"Unsupported file type: {path.suffix} use .csv")


def _normalize_compound_series(s: pd.Series, *, allow_null: bool) -> pd.Series:
    """
    cleans up the tyre compound text making sure everything is uppercase and valid
    it throws an error if it sees something weird or if there are blanks when there shouldnt be
    """
    s = s.replace({"": np.nan}) #replaces all empty/not defined tyres with nan
    s = s.apply(lambda x: x.strip().upper() if isinstance(x, str) else x) #triple checks everything is valid

    if not allow_null and s.isna().any():
        raise ValueError("Compound column has missing values")

    #check there aren't any incorrect compounds
    non_null = s.dropna().unique().tolist()
    unknown = [c for c in non_null if c not in COMPOUND_TO_INT]
    if unknown:
        raise ValueError(f"Unexpected compound values: {unknown}")

    return s

# sequence builders for the recurrent models
def build_feature_sequences(
    keys: pd.DataFrame,
    X_all: np.ndarray,
    y_all: pd.Series,
    *,
    seq_len: int = 8,
    pad_left: bool = False,
    add_timestep_mask: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    turns tabular lap data into sequences for our sequential models, the model making a decision at lap 8 will be fed a sequence of lap data from the 8 previous available laps (so laps 1-8 if they are all present)
    it groups by driver and race so we dont accidentally mix different races together
    """
    #ensure we have the key columns which let us uniquely identify an entry in the dataset
    required = {"race_id", "driver_id", "lapno"}
    if not required.issubset(keys.columns):
        raise ValueError(f"keys must contain columns {sorted(required)}")

    #double check inputs have the same number of rows
    n = len(keys)
    if X_all.shape[0] != n or len(y_all) != n:
        raise ValueError("keys, X_all, and y_all must have the same number of rows")

    if not isinstance(y_all, pd.Series):
        y_all = pd.Series(y_all)

    # keep track of the original row index so we know each rows original location in X_all and y_all
    df_idx = keys[["race_id", "driver_id", "lapno"]].copy()
    df_idx["_row"] = np.arange(n, dtype=int)
    df_idx = df_idx.sort_values(["race_id", "driver_id", "lapno"], kind="mergesort") #sort by race, then driver, then lap

    d = X_all.shape[1]
    d_out = d + 1 if add_timestep_mask else d #add an extra dimension to store the mask that tells us how much of the sequence is real or padding

    X_seq_list = [] #the sequence
    y_seq_list = [] #the label for the sequence
    idx_last_list = [] #the original row index for the last lap
    seq_idx_list = [] #the original row indices for all laps in the sequence
    eff_len_list = [] #how many real laps are in the sequence

    #split the data up by driver per race
    for (_, _), g in df_idx.groupby(["race_id", "driver_id"], sort=False):
        g_rows = g["_row"].to_numpy(dtype=int) #get original row indices

        if len(g_rows) == 0:
            continue

        if (not pad_left) and (len(g_rows) < seq_len): #if we choose not to pad sequences (we do choose to pad sequences in the final model) we skip groups shorter than the given sequence length
            continue

        start_end = 0 if pad_left else (seq_len - 1) #decides what lap the sequences can start from

        for end in range(start_end, len(g_rows)): #slide a window through the drivers laps to get all possible sequences for a specific driver at a specific race
            if pad_left:
                real = g_rows[max(0, end - seq_len + 1): end + 1]
                n_pad = seq_len - len(real)
                # use negative one for padding when there aren't a sequence of 8 laps
                window = np.concatenate([np.full(n_pad, -1, dtype=int), real])
            else:
                window = g_rows[end - seq_len + 1: end + 1]

            real_mask = window != -1 #figure out which laps in the sequence are real and which are padding
            eff_len = int(np.sum(real_mask))

            #create the feature matrix
            Xw = np.zeros((seq_len, d), dtype=np.float32)
            if eff_len > 0:
                Xw[real_mask] = X_all[window[real_mask]].astype(np.float32, copy=False)

            if add_timestep_mask: #add a feature which explicitly tells the model what input is real and what is padding
                mask = real_mask.astype(np.float32).reshape(seq_len, 1)
                Xw = np.concatenate([Xw, mask], axis=1)

            X_seq_list.append(Xw)
            y_seq_list.append(int(y_all.iloc[g_rows[end]])) #attach the label for the final lap in the sequence
            idx_last_list.append(int(g_rows[end]))
            seq_idx_list.append(window)
            eff_len_list.append(eff_len)

    if not X_seq_list:
        # return empty arrays with the right shapes if we found nothing
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
    """
    takes the probability predictions and builds a sequence out of them
    used for implementing the VSE model that Heilmeier proposed (not included in the final model)
    """
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

# functions for loading datasets
def load_stage1_dataset(path: Union[str, Path],*,exclude_holdouts: bool = True,) -> pd.DataFrame:
    """
    loads the data we need for the first stage which predicts if a pit stop happens
    cleans up the text columns and makes sure the numeric ones are actually numbers
    """
    df = _read_any(path) #loads the csv file to a pandas dataframe

    for col in ["race_id", "driver_id", "lapno", "y_pit"]: #forces key columns to be integrers
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)

    if exclude_holdouts: #get rid of races that we exclude because they aren't good training examples of if we want to use them in evaluation
        df = _exclude_holdout_races(df)

    df["current_compound"] = _normalize_compound_series(df["current_compound"], allow_null=True) #normalises tyres

    #normalise race track
    if "race_track" in df.columns:
        df["race_track"] = df["race_track"].replace({"": np.nan, "None": np.nan, "nan": np.nan})
        df["race_track"] = df["race_track"].apply(lambda x: x.strip() if isinstance(x, str) else x)

    #normalise fcy
    if "fcy_status" in df.columns:
        df["fcy_status"] = df["fcy_status"].replace({"": np.nan, "None": np.nan, "nan": np.nan})
        df["fcy_status"] = df["fcy_status"].apply(lambda x: x.strip() if isinstance(x, str) else x)
        df["fcy_status"] = df["fcy_status"].astype("category")

    #convert the relevant feature columns to numeric
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
    """
    grabs the data for the second stage which predicts the next tyre compound
    it throws away rows where we dont know the next compound if strict is true
    """
    df = _read_any(path) #read csv file

    for col in ["race_id", "driver_id", "lapno"]: #forces key columns to be integrers
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)

    if exclude_holdouts: #remove races for evaluation
        df = _exclude_holdout_races(df)

    #normalising tyres
    df["y_compound"] = _normalize_compound_series(df["y_compound"], allow_null=True)
    # drop the mystery compounds if we are being strict about it
    if strict and df["y_compound"].isna().any():
        n_missing = int(df["y_compound"].isna().sum())
        df = df.loc[~df["y_compound"].isna()].copy()
        warnings.warn(
            f"load stage two dataset dropped {n_missing} rows with missing y compound.",
            stacklevel=2,
        )
    df["current_compound"] = _normalize_compound_series(df["current_compound"], allow_null=True)

    #clean race track attribute
    if "race_track" in df.columns:
        df["race_track"] = df["race_track"].replace({"": np.nan, "None": np.nan, "nan": np.nan})
        df["race_track"] = df["race_track"].apply(lambda x: x.strip() if isinstance(x, str) else x)

    #ensure selected columns are numeric
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

def encode_y_compound(
    df: pd.DataFrame,
    col: str = "y_compound",
    out_col: str = "y_compound_encoded",
) -> pd.DataFrame:
    """
    changes the text based compound names into integers that the models can understand, using the COMPOUND_TO_INT dictionary
    """
    df2 = df.copy()
    df2[col] = _normalize_compound_series(df2[col], allow_null=True)
    df2[out_col] = df2[col].map(COMPOUND_TO_INT)
    return df2

# splitting features from targets
def make_xy(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    pulls the target column and gets rid of any columns we dont want
    """
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
    """
    grabs the features and target for predicting if someone pits
    throws away stuff like race id, driver ids, and lapno since they arent features and are just used to uniquely identify entries in the dataset
    """
    default_drop = ["y_pit", "race_id", "driver_id", "lapno"]
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
    """
    grabs the features and target for predicting which compound is selected
    also drops rows where y_compound isnt defined
    """
    #make sure compound labels are encoded
    df2 = encode_y_compound(df2, col="y_compound", out_col=encoded_target_col)
    df2 = df2.loc[~df2[encoded_target_col].isna()].copy() #drop rows where the encoded target is missing

    #drop the label column and the columns used to uniquely identify each data entry
    default_drop = ["y_compound", encoded_target_col, "race_id", "driver_id", "lapno"]
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
    """
    determines which columns are treated as numeric and which are treated as categorical 
    """
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

def build_feature_sequences_from_reference(
    reference_keys: pd.DataFrame,
    X_reference: np.ndarray,
    target_keys: pd.DataFrame,
    y_target: pd.Series,
    *,
    seq_len: int = 8,
    pad_left: bool = True,
    add_timestep_mask: bool = False,
    strict_match: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    builds sequences for stage 2 sequential models. gets the laps from dataset 1 (Restricted to the attributes dataset 2 has) and the label from dataset 2
    reference_keys and X_reference come from dataset 1
    target_keys and y_target come from the pit-event dataset dataset 2

    each target row is identified by (race_id, driver_id, lapno)
    we find the matching lap in dataset 1, then take the seq_len most recent available laps for that same driver in that same race, ending on the matched lap in dataset 2

    importantly, this function only uses whatever columns are already present in X_reference/dataset 2, we don't use attributes that are in dataset 1 but not dataset 2 (e.g., lap time, not relevant)

    returns:
    X_seq: shape (n_targets_found, seq_len, d or d+1)
    y_seq: labels aligned to each built sequence
    idx_target:row indices into the target table
    seq_idx_ref:row indices into the reference table for each timestep
    eff_len:number of real laps before left-padding
    """
    required = {"race_id", "driver_id", "lapno"}

    if not required.issubset(reference_keys.columns):
        raise ValueError(f"reference_keys must contain columns {sorted(required)}")
    if not required.issubset(target_keys.columns):
        raise ValueError(f"target_keys must contain columns {sorted(required)}")

    n_ref = len(reference_keys)
    n_tgt = len(target_keys)

    if X_reference.shape[0] != n_ref:
        raise ValueError("reference_keys and X_reference must have the same number of rows")
    if len(y_target) != n_tgt:
        raise ValueError("target_keys and y_target must have the same number of rows")

    if not isinstance(y_target, pd.Series):
        y_target = pd.Series(y_target)

    # sort the full lap-level reference rows by race / driver / lap
    ref_idx = reference_keys[["race_id", "driver_id", "lapno"]].copy()
    ref_idx["_ref_row"] = np.arange(n_ref, dtype=int)
    ref_idx = ref_idx.sort_values(["race_id", "driver_id", "lapno"], kind="mergesort")

    # build fast lookup by (race_id, driver_id)
    ref_groups: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
    for (race_id, driver_id), g in ref_idx.groupby(["race_id", "driver_id"], sort=False):
        lapnos = g["lapno"].to_numpy(dtype=int)
        ref_rows = g["_ref_row"].to_numpy(dtype=int)
        ref_groups[(int(race_id), int(driver_id))] = (lapnos, ref_rows)

    # keep target row ids in original target-table order
    tgt_idx = target_keys[["race_id", "driver_id", "lapno"]].copy()
    tgt_idx["_target_row"] = np.arange(n_tgt, dtype=int)
    tgt_idx = tgt_idx.sort_values(["race_id", "driver_id", "lapno"], kind="mergesort")

    d = X_reference.shape[1]
    d_out = d + 1 if add_timestep_mask else d

    X_seq_list = []
    y_seq_list = []
    idx_target_list = []
    seq_idx_ref_list = []
    eff_len_list = []

    missing_targets: List[Tuple[int, int, int, int]] = []

    for (race_id, driver_id), g in tgt_idx.groupby(["race_id", "driver_id"], sort=False):
        key = (int(race_id), int(driver_id))

        if key not in ref_groups:
            for _, row in g.iterrows():
                missing_targets.append((
                    int(row["_target_row"]),
                    int(row["race_id"]),
                    int(row["driver_id"]),
                    int(row["lapno"]),
                ))
            continue

        ref_lapnos, ref_rows = ref_groups[key]

        for _, row in g.iterrows():
            target_row = int(row["_target_row"])
            lapno = int(row["lapno"])

            # locate the target lap inside the full lap-level reference history
            pos = np.searchsorted(ref_lapnos, lapno)

            if pos >= len(ref_lapnos) or ref_lapnos[pos] != lapno:
                missing_targets.append((
                    target_row,
                    int(row["race_id"]),
                    int(row["driver_id"]),
                    int(row["lapno"]),
                ))
                continue

            if pad_left:
                real = ref_rows[max(0, pos - seq_len + 1): pos + 1]
                n_pad = seq_len - len(real)
                window = np.concatenate([np.full(n_pad, -1, dtype=int), real])
            else:
                if pos + 1 < seq_len:
                    continue
                window = ref_rows[pos - seq_len + 1: pos + 1]

            real_mask = window != -1
            eff_len = int(np.sum(real_mask))

            Xw = np.zeros((seq_len, d), dtype=np.float32)
            if eff_len > 0:
                Xw[real_mask] = X_reference[window[real_mask]].astype(np.float32, copy=False)

            if add_timestep_mask:
                mask = real_mask.astype(np.float32).reshape(seq_len, 1)
                Xw = np.concatenate([Xw, mask], axis=1)

            X_seq_list.append(Xw)
            y_seq_list.append(int(y_target.iloc[target_row]))
            idx_target_list.append(target_row)
            seq_idx_ref_list.append(window)
            eff_len_list.append(eff_len)

    if missing_targets and strict_match:
        preview = missing_targets[:5]
        raise ValueError(
            "some stage 2 target laps were not found in the full lap-level reference data. "
            f"first few missing targets: {preview}"
        )

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
        np.asarray(idx_target_list, dtype=int),
        np.asarray(seq_idx_ref_list, dtype=int),
        np.asarray(eff_len_list, dtype=int),
    )

#splitting data into training and validation (randomly shuffling races instead of laps to avoid temporal leakage)
@dataclass(frozen=True)
class FoldBundle:
    """
    holds the split data ready for cross validation
    """
    folds: List[Tuple[np.ndarray, np.ndarray]] 
    fold_race_ids: List[List[int]] # stores a list of folds where each fold is a list of race ids belonging to it

def make_race_group_folds(
    df: pd.DataFrame,
    *,
    group_col: str = "race_id",
    n_splits: int = 5,
    seed: int = 42,
) -> FoldBundle:
    """
    keeps all rows from the same race in the same fold
    and tries to keep folds similar in total size
    """
    # group rows by race
    groups = df.groupby(group_col, sort=False).indices
    race_ids = list(groups.keys())

    # shuffle races for reproducibility
    rng = np.random.default_rng(seed)
    rng.shuffle(race_ids)

    # collect race sizes
    race_meta = []
    for rid in race_ids:
        idx = groups[rid]
        size = len(idx)
        race_meta.append((rid, size))

    #order races to keep races with more laps first
    race_meta.sort(key=lambda t: t[1], reverse=True)

    # track current fold sizes and assigned races
    fold_sizes = np.zeros(n_splits, dtype=float)
    fold_races: List[List[int]] = [[] for _ in range(n_splits)]

    # greedily assign each race to the currently smallest fold
    for rid, size in race_meta:
        best_fold = int(np.argmin(fold_sizes))
        fold_races[best_fold].append(int(rid))
        fold_sizes[best_fold] += size

    # build train/validation index arrays
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    all_idx = np.arange(len(df))

    for f in range(n_splits):
        valid_rids = set(fold_races[f])
        valid_mask = df[group_col].isin(valid_rids).to_numpy()
        valid_idx = all_idx[valid_mask]
        train_idx = all_idx[~valid_mask]
        folds.append((train_idx, valid_idx))

    return FoldBundle(folds=folds, fold_race_ids=fold_races)