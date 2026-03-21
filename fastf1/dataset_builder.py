#!/usr/bin/env python3
"""

takes the clean and less clean sqlite databases and transforms them into
csv datasets ready for model training

dataset 1 comes from the clean db and prepares features for tire strategy
dataset 2 comes from the less clean db and focuses on pit stop events

usage
  python dataset_builder.py --db-clean path/to/clean.sqlite --db-less-clean path/to/less_clean.sqlite
"""

import argparse
import math
import re
import sqlite3
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# suppress future warnings for cleaner output
pd.set_option('future.no_silent_downcasting', True)


# -----------------------------
# configuration and constants
# -----------------------------

# set this to true to include the pit_stops_left feature, useful for certain structural analyses.
INCLUDE_PIT_STOPS_LEFT = True

# estimated time lost in pit lane for different tracks in seconds
# taken from historical averages
PIT_LOSS_BY_TRACK_S: dict[str, float] = {
    "70th Anniversary Grand Prix": 28.378,
    "Abu Dhabi Grand Prix": 21.7175,
    "Australian Grand Prix": 18.0205,
    "Austrian Grand Prix": 21.6775,
    "Azerbaijan Grand Prix": 20.6375,
    "Bahrain Grand Prix": 24.8,
    "Belgian Grand Prix": 23.076,
    "Brazilian Grand Prix": 23.113,
    "British Grand Prix": 28.914,
    "Canadian Grand Prix": 23.711,
    "Chinese Grand Prix": 22.777,
    "Dutch Grand Prix": 19.643,
    "Eifel Grand Prix": 22.61,
    "Emilia Romagna Grand Prix": 30.289,
    "French Grand Prix": 30.36,
    "German Grand Prix": 20.563,
    "Hungarian Grand Prix": 21.814,
    "Italian Grand Prix": 24.445,
    "Japanese Grand Prix": 23.396,
    "Las Vegas Grand Prix": 21.5585,
    "Mexican Grand Prix": 22.3685,
    "Mexico City Grand Prix": 22.492,
    "Miami Grand Prix": 22.1465,
    "Monaco Grand Prix": 24.2765,
    "Portuguese Grand Prix": 26.1275,
    "Qatar Grand Prix": 28.317,
    "Russian Grand Prix": 29.933,
    "Sakhir Grand Prix": 24.2485,
    "Saudi Arabian Grand Prix": 20.9425,
    "Singapore Grand Prix": 29.518,
    "Spanish Grand Prix": 22.2165,
    "Styrian Grand Prix": 21.627,
    "São Paulo Grand Prix": 23.602,
    "Turkish Grand Prix": 23.5235,
    "United States Grand Prix": 24.023,
}

# fallback value if track is not in the list
DEFAULT_PIT_LOSS_S = 23.113

# mapping for compound hardness to an absolute scale where 1 is hardest and 7 is softest
# standardizes across 2018's A-compounds and the current C-compounds
ABSOLUTE_HARDNESS_MAP = {
    # 2018 names to absolute
    "A1": 1, # superhard
    "A2": 2, # hard
    "A3": 3, # medium
    "A4": 4, # soft
    "A5": 5, # supersoft
    "A6": 6, # ultrasoft
    "A7": 7, # hypersoft
    # c names to absolute (c1 is roughly a2, etc)
    "C1": 2,
    "C2": 3,
    "C3": 4,
    "C4": 6,
    "C5": 7,
}


# -----------------------------
# helper functions
# -----------------------------

def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    """checks if a specific table exists in the sqlite db"""
    q = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
    return conn.execute(q, (name,)).fetchone() is not None


def pick_race_lap_count(races: pd.DataFrame) -> pd.Series:
    """determines the total laps for the race using planned if available otherwise actual"""
    if "nolapsplanned" in races.columns:
        base = races["nolapsplanned"].copy()
    else:
        base = pd.Series([np.nan] * len(races), index=races.index)

    if "nolaps" in races.columns:
        base = base.fillna(races["nolaps"])
    return base


def normalize_compound(x: Optional[str]) -> Optional[str]:
    """standardizes compound names to a common format"""
    if x is None:
        return None
    s = str(x).strip().upper()
    if s in {"SOFT", "S"}:
        return "SOFT"
    if s in {"MEDIUM", "M"}:
        return "MEDIUM"
    if s in {"HARD", "H"}:
        return "HARD"
    if s in {"INTERMEDIATE", "INTER", "IN", "I"}:
        return "INTERMEDIATE"
    if s in {"WET", "FULLWET", "FW"}:
        return "WET"
    return None


def estimate_pit_loss_seconds(race_track: pd.Series, fcy_status: pd.Series) -> pd.Series:
    """calculates estimated time lost in pits adjusting for safety car status"""
    track = race_track.astype(str).str.strip()
    base = track.map(PIT_LOSS_BY_TRACK_S).fillna(DEFAULT_PIT_LOSS_S).astype(float)

    f = pd.to_numeric(fcy_status, errors="coerce").fillna(0).astype(int)
    # reduce loss under safety car conditions
    mult = np.where((f == 1) | (f == 2), 0.65, np.where((f == 3) | (f == 4), 0.45, 1.0))
    return base * mult


def extract_hardest_slick_from_availablecompounds(s: Optional[str]) -> Optional[int]:
    """
    parses the available compounds string to find the hardest slick tire on an absolute scale
    1 is hardest (superhard) 7 is softest (hypersoft/c5)
    """
    if s is None:
        return None

    txt = str(s).upper()

    # extract all standard designations
    matches = re.findall(r"\b([AC][1-7])\b", txt)
    if not matches:
        return None

    # map to absolute hardness
    abs_hardness = [ABSOLUTE_HARDNESS_MAP.get(m) for m in matches]
    abs_hardness = [h for h in abs_hardness if h is not None]
    
    if not abs_hardness:
        return None

    # smaller number is harder
    return min(abs_hardness)


def track_category_from_hardest(hardest_abs: Optional[int]) -> Optional[int]:
    """
    categorizes track degradation based on the hardest compound brought to the track
    1 = high degradation (brings the hardest tires, eg. abs hardness 1 or 2 / c1)
    2 = medium degradation (brings medium tires, eg. abs hardness 3 / c2)
    3 = low degradation (brings soft tires, eg. abs hardness >= 4 / c3+)
    """
    if hardest_abs is None or (isinstance(hardest_abs, float) and math.isnan(hardest_abs)):
        return None
    
    if hardest_abs <= 2:
        return 1 # high degradation
    if hardest_abs == 3:
        return 2 # medium degradation
    return 3 # low degradation


# -----------------------------
# feature engineering logic
# -----------------------------

def _rejoin_gaps_one_lap(grp: pd.DataFrame) -> pd.DataFrame:
    """estimates gaps to cars ahead and behind after a hypothetical pit stop"""
    # operate on a copy to avoid side effects
    g = grp.copy()
    times = pd.to_numeric(g["racetime_sofar"], errors="coerce").to_numpy()
    ids = g["driver_id"].to_numpy()
    pit_loss = pd.to_numeric(g["pit_loss_est_s"], errors="coerce").to_numpy()

    t_proj = times + pit_loss

    # sort by current race time to find insertion points
    order = np.argsort(times)
    times_sorted = times[order]
    ids_sorted = ids[order]

    ahead_gap = np.full(len(g), np.nan, dtype=float)
    behind_gap = np.full(len(g), np.nan, dtype=float)

    for i in range(len(g)):
        # find where the car would rejoin in the sorted race times
        ins = np.searchsorted(times_sorted, t_proj[i], side="left")

        prev_idx = ins - 1
        next_idx = ins

        # skip self in comparisons (a driver doesn't race against their own ghost)
        while prev_idx >= 0 and ids_sorted[prev_idx] == ids[i]:
            prev_idx -= 1
        while next_idx < len(times_sorted) and ids_sorted[next_idx] == ids[i]:
            next_idx += 1

        if prev_idx >= 0:
            ahead_gap[i] = t_proj[i] - times_sorted[prev_idx]
        if next_idx < len(times_sorted):
            behind_gap[i] = times_sorted[next_idx] - t_proj[i]

    # return only the new columns to facilitate clean merging
    return pd.DataFrame({
        "rejoin_gap_ahead_est_s": ahead_gap,
        "rejoin_gap_behind_est_s": behind_gap
    }, index=g.index)


def add_pit_stops_left(df: pd.DataFrame) -> pd.DataFrame:
    """calculates how many pit stops are remaining for the driver in the race"""
    df = df.sort_values(["race_id", "driver_id", "lapno"]).copy()

    if "y_pit" not in df.columns:
        df["y_pit"] = 0
    df["y_pit"] = pd.to_numeric(df["y_pit"], errors="coerce").fillna(0).astype(int)

    # cheat by looking ahead at the total pits in the race
    total_pits = df.groupby(["race_id", "driver_id"])["y_pit"].transform("sum").astype(int)

    if "pit_stops_so_far" not in df.columns:
        df["pit_stops_so_far"] = (
            df.groupby(["race_id", "driver_id"])["y_pit"]
            .transform(lambda s: s.cumsum().shift(1).fillna(0).astype(int))
        )

    df["pit_stops_left"] = (total_pits - df["pit_stops_so_far"]).astype(int).clip(lower=0)
    return df


def add_undercut_features(df: pd.DataFrame) -> pd.DataFrame:
    """adds features related to track position and gaps relevant for undercutting"""
    df = df.copy()
    
    # check for minimum required columns
    need = {"race_id", "driver_id", "lapno", "position", "racetime_sofar"}
    if not need.issubset(df.columns):
        for c in ["gap_behind_s", "n_cars_within_5s_ahead", "tyre_age_diff_to_ahead",
                  "rejoin_gap_ahead_est_s", "rejoin_gap_behind_est_s"]:
            df[c] = np.nan
        return df

    df["position"] = pd.to_numeric(df["position"], errors="coerce")
    df["racetime_sofar"] = pd.to_numeric(df["racetime_sofar"], errors="coerce")
    df["tyre_age"] = pd.to_numeric(df.get("tyre_age", np.nan), errors="coerce")

    df = df.sort_values(["race_id", "lapno", "position", "driver_id"]).copy()
    g = df.groupby(["race_id", "lapno"], sort=False)

    # calculate gap to the car directly behind
    df["behind_racetime"] = g["racetime_sofar"].shift(-1)
    df["gap_behind_s"] = df["behind_racetime"] - df["racetime_sofar"]

    # count cars directly ahead within striking distance
    def _count_ahead_within_5s(grp: pd.DataFrame) -> pd.Series:
        g2 = grp.sort_values(["position", "driver_id"]).copy()
        t = pd.to_numeric(g2["racetime_sofar"], errors="coerce").to_numpy()

        out = np.full(len(g2), np.nan, dtype=float)
        for i in range(len(g2)):
            if np.isnan(t[i]):
                continue
            j0 = np.searchsorted(t, t[i] - 5.0, side="left")
            out[i] = max(0, i - j0)
        return pd.Series(out, index=g2.index)

    df["n_cars_within_5s_ahead"] = (
        df.groupby(["race_id", "lapno"], sort=False, group_keys=False)
        .apply(_count_ahead_within_5s, include_groups=False)
        .fillna(0)
        .astype(int)
    )

    # tyre age delta to the car ahead
    df["ahead1_tyre_age"] = g["tyre_age"].shift(1)
    df["tyre_age_diff_to_ahead"] = df["tyre_age"] - df["ahead1_tyre_age"]

    # estimate gaps after a potential pit stop
    if "race_track" in df.columns and "fcy_status" in df.columns:
        df["pit_loss_est_s"] = estimate_pit_loss_seconds(df["race_track"], df["fcy_status"])
    else:
        df["pit_loss_est_s"] = DEFAULT_PIT_LOSS_S

    # apply calculation per lap-group and join results back
    rejoin_cols = df.groupby(["race_id", "lapno"], sort=False, group_keys=False).apply(_rejoin_gaps_one_lap, include_groups=False)
    df = pd.concat([df, rejoin_cols], axis=1)

    # clean up temporary columns
    df = df.drop(
        columns=["behind_racetime", "ahead1_tyre_age", "pit_loss_est_s"],
        errors="ignore",
    )
    return df


def add_weather_flags(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """processes weather data to add flags for current rain and accumulated rain"""
    df = df.copy()
    df["is_raining"] = 0
    df["minutes_rain"] = np.nan

    if weather is None or weather.empty or "race_id" not in df.columns:
        return df

    if "racetime_sofar" in df.columns:
        t_s = pd.to_numeric(df["racetime_sofar"], errors="coerce")
    elif "racetime" in df.columns:
        t_s = pd.to_numeric(df["racetime"], errors="coerce")
    else:
        return df

    df["_time_ms"] = (t_s * 1000.0).round().astype("Int64")

    weather = weather.copy()
    weather["time_ms"] = pd.to_numeric(weather["time_ms"], errors="coerce")
    weather["rainfall"] = pd.to_numeric(weather["rainfall"], errors="coerce").fillna(0).astype(int)
    weather = weather.dropna(subset=["race_id", "time_ms"]).copy()

    out = []
    for race_id, df_r in df.groupby("race_id", sort=False):
        df_r = df_r.copy()
        w = weather[weather["race_id"] == race_id].sort_values("time_ms").copy()

        if w.empty:
            out.append(df_r)
            continue

        # identify rain streaks for duration calculation
        is_rain = w["rainfall"].eq(1)
        change = is_rain.ne(is_rain.shift(1, fill_value=False))
        w["streak_id"] = change.cumsum()
        w["rain_streak_start_ms"] = np.where(
            is_rain,
            w.groupby("streak_id")["time_ms"].transform("min"),
            np.nan
        )

        df_r["_time_ms_num"] = pd.to_numeric(df_r["_time_ms"], errors="coerce")
        has_t = df_r["_time_ms_num"].notna()
        df_t = df_r.loc[has_t].sort_values("_time_ms_num").copy()
        df_nt = df_r.loc[~has_t].copy()

        # perform asof merge to map weather states to laps
        df_t["_time_ms_i64"] = df_t["_time_ms_num"].astype(np.int64)
        w = w.dropna(subset=["time_ms"]).copy()
        w["time_ms_i64"] = w["time_ms"].astype(np.int64)
        w = w.sort_values("time_ms_i64")

        merged = pd.merge_asof(
            df_t,
            w[["time_ms_i64", "rainfall", "rain_streak_start_ms"]],
            left_on="_time_ms_i64",
            right_on="time_ms_i64",
            direction="backward",
            allow_exact_matches=True,
        )
        merged = merged.drop(columns=["time_ms_i64", "_time_ms_i64"], errors="ignore")

        merged["is_raining"] = merged["rainfall"].fillna(0).astype(int)

        dur_ms = merged["_time_ms_num"] - pd.to_numeric(merged["rain_streak_start_ms"], errors="coerce")
        merged["minutes_rain"] = np.where(
            merged["is_raining"].eq(1) & dur_ms.notna(),
            dur_ms / 60000.0,
            np.nan
        )

        merged = merged.drop(columns=["time_ms", "rainfall", "rain_streak_start_ms"], errors="ignore")

        out.append(pd.concat([merged, df_nt], ignore_index=True) if not df_nt.empty else merged)

    df2 = pd.concat(out, ignore_index=True)
    df2 = df2.drop(columns=["_time_ms_num"], errors="ignore")
    return df2


def add_is_wet_race(df: pd.DataFrame) -> pd.DataFrame:
    """
    flags if the driver has used intermediate or wet tires at any point
    up to and including the current lap in the race.
    """
    df = df.sort_values(["race_id", "driver_id", "lapno"]).copy()
    
    if "compound" in df.columns:
        comp = df["compound"].apply(normalize_compound)
    elif "current_compound" in df.columns:
        comp = df["current_compound"].apply(normalize_compound)
    else:
        df["is_wet_race"] = 0
        return df

    is_wet_tyre = comp.isin({"INTERMEDIATE", "WET"}).astype(int)
    
    df["is_wet_race"] = (
        is_wet_tyre.groupby([df["race_id"], df["driver_id"]])
        .cumsum()
        .clip(upper=1)
        .astype(int)
    )
    return df


def add_pit_stops_so_far(df: pd.DataFrame) -> pd.DataFrame:
    """calculates cumulative pit stops taken strictly prior to the current lap"""
    df = df.sort_values(["race_id", "driver_id", "lapno"]).copy()
    
    if "y_pit" not in df.columns:
        df["pit_stops_so_far"] = 0
        return df

    df["pit_stops_so_far"] = (
        df.groupby(["race_id", "driver_id"])["y_pit"]
          .transform(lambda s: s.cumsum().shift(1).fillna(0).astype(int))
    )
    return df


def add_tyre_change_pursuer(df: pd.DataFrame) -> pd.DataFrame:
    """checks if the car immediately behind changed tyres in the previous lap"""
    df = df.sort_values(["race_id", "driver_id", "lapno"]).copy()

    if "compound" not in df.columns:
        df["tyre_change_pursuer"] = False
        return df

    df["compound_norm"] = df["compound"].apply(normalize_compound)
    
    # check if compound changed since previous lap
    def did_change(s):
        shifted = s.shift(1)
        return s.notna() & shifted.notna() & (s != shifted)
        
    df["did_change_since_prev"] = df.groupby(["race_id", "driver_id"])["compound_norm"].transform(did_change)

    pursuer_lookup = df[["race_id", "lapno", "position", "did_change_since_prev"]].copy()
    pursuer_lookup = pursuer_lookup.rename(columns={"did_change_since_prev": "pursuer_changed_prev"})
    df["pursuer_position"] = df["position"] + 1

    df = df.merge(
        pursuer_lookup,
        left_on=["race_id", "lapno", "pursuer_position"],
        right_on=["race_id", "lapno", "position"],
        how="left",
        suffixes=("", "_pursuer_row")
    )

    df["tyre_change_pursuer"] = df["pursuer_changed_prev"].fillna(False).astype(bool)

    df = df.drop(
        columns=[c for c in ["compound_norm", "did_change_since_prev", "pursuer_position",
                             "pursuer_changed_prev", "position_pursuer_row"] if c in df.columns],
        errors="ignore"
    )
    return df


def add_close_ahead(df: pd.DataFrame, threshold_s: float = 1.5) -> pd.DataFrame:
    """flags if the driver was closely followed by another car in the previous lap"""
    df = df.copy()

    if not {"race_id", "driver_id", "lapno", "position", "racetime"}.issubset(df.columns):
        df["close_ahead"] = 0
        return df

    base = df[["race_id", "driver_id", "lapno", "position", "racetime"]].copy()
    base["racetime"] = pd.to_numeric(base["racetime"], errors="coerce")
    base["position"] = pd.to_numeric(base["position"], errors="coerce")

    pursuer = base[["race_id", "lapno", "position", "racetime"]].rename(
        columns={"position": "pursuer_position", "racetime": "pursuer_racetime"}
    )
    base["pursuer_position"] = base["position"] + 1

    merged = base.merge(
        pursuer,
        on=["race_id", "lapno", "pursuer_position"],
        how="left"
    )

    merged["ahead_by_pursuer_s_prev"] = merged["pursuer_racetime"] - merged["racetime"]

    merged["lapno_next"] = merged["lapno"] + 1
    attach = merged[["race_id", "driver_id", "lapno_next", "ahead_by_pursuer_s_prev"]].rename(
        columns={"lapno_next": "lapno"}
    )

    df = df.merge(attach, on=["race_id", "driver_id", "lapno"], how="left")

    df["close_ahead"] = (
        df["ahead_by_pursuer_s_prev"].notna()
        & (pd.to_numeric(df["ahead_by_pursuer_s_prev"], errors="coerce") <= threshold_s)
    ).astype(int)

    df = df.drop(columns=["ahead_by_pursuer_s_prev"], errors="ignore")
    return df


def add_fcy_status_table6(df: pd.DataFrame, fcy: pd.DataFrame) -> pd.DataFrame:
    """maps full course yellow phases to a categorical status for each lap"""
    df = df.copy()
    df["fcy_status"] = 0

    if "racetime" not in df.columns or df["racetime"].isna().all():
        return df
    if "type" not in fcy.columns:
        return df

    fcy = fcy.copy()
    fcy["type_norm"] = fcy["type"].astype(str).str.upper()
    fcy = fcy[fcy["type_norm"].isin({"VSC", "SC"})].copy()
    if fcy.empty:
        return df

    df["racetime"] = pd.to_numeric(df["racetime"], errors="coerce")
    fcy["startracetime"] = pd.to_numeric(fcy["startracetime"], errors="coerce")
    fcy["endracetime"] = pd.to_numeric(fcy["endracetime"], errors="coerce")

    out_frames = []
    for race_id, df_r in df.groupby("race_id", sort=False):
        df_r = df_r.copy()
        fcy_r = fcy[fcy["race_id"] == race_id].copy()
        
        # logic to determine status based on timing overlap
        if fcy_r.empty:
            df_r["phase_type_end"] = None
            out_frames.append(df_r)
            continue

        fcy_r = fcy_r[fcy_r["startracetime"].notna()].sort_values("startracetime")
        has_time = df_r["racetime"].notna()
        df_r_time = df_r.loc[has_time].sort_values("racetime").copy()
        df_r_notime = df_r.loc[~has_time].copy()

        if df_r_time.empty:
            df_r["phase_type_end"] = None
            out_frames.append(df_r)
            continue

        merged_time = pd.merge_asof(
            df_r_time,
            fcy_r[["startracetime", "endracetime", "type_norm"]],
            left_on="racetime",
            right_on="startracetime",
            direction="backward",
            allow_exact_matches=True,
        )

        active = merged_time["endracetime"].notna() & (merged_time["racetime"] <= merged_time["endracetime"])
        merged_time["phase_type_end"] = np.where(active, merged_time["type_norm"], None)

        df_r_notime["phase_type_end"] = None

        merged_race = pd.concat(
            [
                merged_time.drop(columns=["startracetime", "endracetime", "type_norm"], errors="ignore"),
                df_r_notime,
            ],
            ignore_index=True,
        )
        out_frames.append(merged_race)

    df2 = pd.concat(out_frames, ignore_index=True)
    df2 = df2.sort_values(["race_id", "driver_id", "lapno"]).copy()
    prev_phase = df2.groupby(["race_id", "driver_id"])["phase_type_end"].shift(1)

    is_vsc = df2["phase_type_end"].eq("VSC")
    is_sc = df2["phase_type_end"].eq("SC")

    # encode states 1=vsc_start 2=vsc_cont 3=sc_start 4=sc_cont
    first_vsc = is_vsc & (prev_phase.ne("VSC"))
    further_vsc = is_vsc & (prev_phase.eq("VSC"))
    first_sc = is_sc & (prev_phase.ne("SC"))
    further_sc = is_sc & (prev_phase.eq("SC"))

    df2["fcy_status"] = 0
    df2.loc[first_vsc, "fcy_status"] = 1
    df2.loc[further_vsc, "fcy_status"] = 2
    df2.loc[first_sc, "fcy_status"] = 3
    df2.loc[further_sc, "fcy_status"] = 4

    df2 = df2.drop(columns=["phase_type_end"], errors="ignore")
    return df2


def add_fulfilled_second_compound(df: pd.DataFrame) -> pd.DataFrame:
    """checks if the regulation to use two distinct compounds has been met"""
    df = df.sort_values(["race_id", "driver_id", "lapno"]).copy()

    if "compound" in df.columns:
        comp = df["compound"].apply(normalize_compound)
    elif "current_compound" in df.columns:
        comp = df["current_compound"].apply(normalize_compound)
    else:
        comp = pd.Series([None] * len(df), index=df.index)

    missing = comp.isna()
    tmp = comp.fillna("__MISSING__")

    first_occ = tmp.groupby([df["race_id"], df["driver_id"]]).transform(lambda s: ~s.duplicated())
    first_occ = first_occ & (~missing)

    cum_distinct = first_occ.groupby([df["race_id"], df["driver_id"]]).cumsum().astype(int)
    
    # rule exemption if intermediate or wet tyres have been used
    is_wet_race = pd.to_numeric(df.get("is_wet_race", 0), errors="coerce").fillna(0).astype(int)

    df["fulfilled_second_compound"] = ((is_wet_race == 1) | (cum_distinct >= 2)).astype(int)
    return df


def load_core_tables(conn: sqlite3.Connection) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """retrieves raw tables from the database for processing"""
    required = ["laps", "races", "fcyphases"]
    missing = [t for t in required if not table_exists(conn, t)]
    if missing:
        raise RuntimeError(f"missing required tables {missing}")

    # load minimal columns for laps
    laps_cols = [
        "race_id", "driver_id", "lapno", "position",
        "laptime", "racetime", "gap", "interval",
        "compound", "tireage", "nextcompound",
        "pitintime", "pitstopduration", "pit_in_elapsed"
    ]
    laps_info = pd.read_sql_query("PRAGMA table_info(laps);", conn)
    existing_laps_cols = set(laps_info["name"].tolist())
    laps_cols = [c for c in laps_cols if c in existing_laps_cols]
    laps = pd.read_sql_query(f"SELECT {', '.join(laps_cols)} FROM laps;", conn)

    # load races
    races_cols = ["id", "location", "nolapsplanned", "nolaps", "availablecompounds", "availablecompounds_c"]
    races_info = pd.read_sql_query("PRAGMA table_info(races);", conn)
    existing_races_cols = set(races_info["name"].tolist())
    races_cols = [c for c in races_cols if c in existing_races_cols]
    races = pd.read_sql_query(f"SELECT {', '.join(races_cols)} FROM races;", conn).rename(columns={"id": "race_id"})

    # load fcy phases
    fcy_cols = ["race_id", "startracetime", "endracetime", "startlap", "endlap", "type"]
    fcy_info = pd.read_sql_query("PRAGMA table_info(fcyphases);", conn)
    existing_fcy_cols = set(fcy_info["name"].tolist())
    fcy_cols = [c for c in fcy_cols if c in existing_fcy_cols]
    fcy = pd.read_sql_query(f"SELECT {', '.join(fcy_cols)} FROM fcyphases;", conn)

    # load weather
    weather = pd.DataFrame(columns=["race_id", "time_ms", "rainfall"])
    if table_exists(conn, "sessions") and table_exists(conn, "weather_samples"):
        sess = pd.read_sql_query("SELECT id AS session_id, race_id, session_code FROM sessions;", conn)
        sess = sess[sess["session_code"].astype(str).str.upper() == "R"][["session_id", "race_id"]]
        ws = pd.read_sql_query("SELECT session_id, time_ms, rainfall FROM weather_samples;", conn)
        weather = ws.merge(sess, on="session_id", how="inner")[["race_id", "time_ms", "rainfall"]].copy()

    return laps, races, fcy, weather


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """generates target variables for training"""
    df = df.copy()

    pit_by_time = df["pitintime"].notna() if "pitintime" in df.columns else pd.Series(False, index=df.index)
    df["y_pit"] = pit_by_time.astype(int)

    if "nextcompound" in df.columns:
        df["y_compound"] = df["nextcompound"].apply(normalize_compound)
        df.loc[df["y_pit"] != 1, "y_compound"] = None
    else:
        df["y_compound"] = None

    return df


def add_race_progress_and_track_category(laps: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """adds derived features for race completion and track characteristics"""
    df = laps.merge(races, on="race_id", how="left")

    total_laps = pick_race_lap_count(df)
    df["race_progress"] = df["lapno"] / total_laps.replace({0: np.nan})
    df["race_progress"] = df["race_progress"].clip(lower=0.0, upper=1.0)

    if "availablecompounds_c" in df.columns:
        ac = df["availablecompounds_c"].where(df["availablecompounds_c"].notna(), df.get("availablecompounds"))
    else:
        ac = df.get("availablecompounds")

    hardest_abs = ac.apply(extract_hardest_slick_from_availablecompounds)
    df["track_category"] = hardest_abs.apply(track_category_from_hardest)

    if "location" in df.columns:
        df["race_track"] = df["location"].astype(str)
    else:
        df["race_track"] = None

    return df


def add_extra_output_features(df: pd.DataFrame) -> pd.DataFrame:
    """includes useful columns from raw data to the processing dataframe"""
    df = df.copy()

    if "compound" in df.columns:
        df["current_compound"] = df["compound"].apply(normalize_compound)
    else:
        df["current_compound"] = None

    lap_time_base = pd.to_numeric(df.get("laptime", np.nan), errors="coerce")
    
    # if pit_in_elapsed exists, use it for pit laps to avoid inflated lap times
    if "pit_in_elapsed" in df.columns:
        pit_in_time = pd.to_numeric(df["pit_in_elapsed"], errors="coerce")
        df["lap_time"] = np.where(df.get("y_pit", 0) == 1, pit_in_time.fillna(lap_time_base), lap_time_base)
    else:
        df["lap_time"] = lap_time_base

    df["racetime_sofar"] = pd.to_numeric(df.get("racetime", np.nan), errors="coerce")
    df["gap_to_leader"] = pd.to_numeric(df.get("gap", np.nan), errors="coerce")
    df["interval"] = pd.to_numeric(df.get("interval", np.nan), errors="coerce")
    df["tyre_age"] = pd.to_numeric(df.get("tireage", np.nan), errors="coerce")

    return df


def finalize_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """
    handles missing values in a principled way for the final dataset.
    semantically fills certain columns and adds missingness flags for others.
    leaves main numeric columns as nan for downstream statistical imputation.
    """
    df = df.copy()

    # a. direct semantic fills inside dataset_builder
    if "is_raining" in df.columns:
        df["is_raining"] = df["is_raining"].fillna(0)
    if "minutes_rain" in df.columns and "is_raining" in df.columns:
        # only fill minutes_rain with 0 where it is not raining
        mask_not_raining = df["is_raining"] == 0
        df.loc[mask_not_raining, "minutes_rain"] = df.loc[mask_not_raining, "minutes_rain"].fillna(0.0)
    if "tyre_change_pursuer" in df.columns:
        df["tyre_change_pursuer"] = df["tyre_change_pursuer"].fillna(False)
    if "close_ahead" in df.columns:
        df["close_ahead"] = df["close_ahead"].fillna(0)
    if "fcy_status" in df.columns:
        df["fcy_status"] = df["fcy_status"].fillna(0)

    # b. explicit missingness flags for structurally undefined features
    features_to_flag = [
        "gap_behind_s",
        "tyre_age_diff_to_ahead",
        "rejoin_gap_ahead_est_s",
        "rejoin_gap_behind_est_s",
        "tyre_age"
    ]

    for feat in features_to_flag:
        if feat in df.columns:
            df[f"{feat}_missing"] = df[feat].isna().astype(int)

    return df


def build_dataset(db_path: str) -> pd.DataFrame:
    """pipeline that coordinates data loading and feature engineering"""
    conn = sqlite3.connect(db_path)
    try:
        laps, races, fcy, weather = load_core_tables(conn)
    finally:
        conn.close()

    laps = laps.dropna(subset=["race_id", "driver_id", "lapno", "position"]).copy()
    for c in ["race_id", "driver_id", "lapno", "position"]:
        laps[c] = laps[c].astype(int)

    df = add_labels(laps)
    df = add_extra_output_features(df)
    df = add_race_progress_and_track_category(df, races)
    df = add_weather_flags(df, weather)
    df = add_is_wet_race(df)
    df = add_pit_stops_so_far(df)
    
    if INCLUDE_PIT_STOPS_LEFT:
        df = add_pit_stops_left(df)
        
    df = add_tyre_change_pursuer(df)
    df = add_fcy_status_table6(df, fcy)
    df = add_undercut_features(df)
    df = add_close_ahead(df, threshold_s=1.5)
    df = add_fulfilled_second_compound(df)
    
    # perform principled missing value handling
    df = finalize_missingness(df)

    # enforce numeric bounds for safety
    if "race_progress" in df.columns:
        df["race_progress"] = df["race_progress"].astype(float).clip(0.0, 1.0)
    if "position" in df.columns:
        df["position"] = df["position"].astype(int).clip(0, 22)
    if "fcy_status" in df.columns:
        df["fcy_status"] = df["fcy_status"].astype(int).clip(0, 4)

    return df


def make_dataset_1(df_full: pd.DataFrame) -> pd.DataFrame:
    """filters columns for the main tire strategy dataset"""
    keep_cols = [
        "race_id", "driver_id", "lapno",
        "race_progress", "position", "fcy_status",
        "pit_stops_so_far", 
        "tyre_change_pursuer", "track_category",
        "y_pit",
        "current_compound", "lap_time", "interval", "tyre_age",
        "gap_behind_s", "n_cars_within_5s_ahead",
        "tyre_age_diff_to_ahead",
        "rejoin_gap_ahead_est_s", "rejoin_gap_behind_est_s",
        "close_ahead", "race_track",
        "is_wet_race", "is_raining", "minutes_rain",
        "fulfilled_second_compound",
        # missingness flags
        "gap_behind_s_missing", "tyre_age_diff_to_ahead_missing",
        "rejoin_gap_ahead_est_s_missing", "rejoin_gap_behind_est_s_missing",
        "tyre_age_missing"
    ]
    
    if INCLUDE_PIT_STOPS_LEFT:
        keep_cols.append("pit_stops_left")
        
    return df_full[[c for c in keep_cols if c in df_full.columns]].copy()


def make_dataset_2(df_full: pd.DataFrame) -> pd.DataFrame:
    """filters columns for the pit event dataset"""
    df_pit = df_full[df_full.get("y_pit", 0).astype(int) == 1].copy()

    keep_cols = [
        "race_id", "driver_id", "lapno", "race_progress",
        "pit_stops_so_far", 
        "current_compound", "race_track",
        "fulfilled_second_compound",
        "is_wet_race", "is_raining", "minutes_rain",
        "gap_behind_s", "n_cars_within_5s_ahead",
        "tyre_age_diff_to_ahead",
        "rejoin_gap_ahead_est_s", "rejoin_gap_behind_est_s",
        "y_compound",
        # missingness flags
        "gap_behind_s_missing", "tyre_age_diff_to_ahead_missing",
        "rejoin_gap_ahead_est_s_missing", "rejoin_gap_behind_est_s_missing"
    ]
    
    if INCLUDE_PIT_STOPS_LEFT:
        keep_cols.append("pit_stops_left")
        
    return df_pit[[c for c in keep_cols if c in df_pit.columns]].copy()


# -----------------------------
# main entry point
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-clean", required=True, help="path to clean sqlite db")
    ap.add_argument("--db-less-clean", required=True, help="path to less clean sqlite db")
    args = ap.parse_args()

    # process dataset 1
    out1_path = Path(args.db_clean).with_name(Path(args.db_clean).stem + "_dataset.csv")
    df_clean_full = build_dataset(args.db_clean)
    ds1 = make_dataset_1(df_clean_full)
    ds1.to_csv(out1_path, index=False)
    print(f"dataset 1 wrote {len(ds1)} rows to {out1_path}")

    # process dataset 2
    out2_path = Path(args.db_less_clean).with_name(Path(args.db_less_clean).stem + "_dataset.csv")
    df_less_full = build_dataset(args.db_less_clean)
    ds2 = make_dataset_2(df_less_full)
    ds2.to_csv(out2_path, index=False)
    print(f"dataset 2 wrote {len(ds2)} rows to {out2_path}")


if __name__ == "__main__":
    main()
