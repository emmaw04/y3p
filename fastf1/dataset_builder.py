#!/usr/bin/env python3
"""
Takes the clean and less clean sqlite databases and transforms them into
csv datasets ready for model training.

Dataset 1 comes from the clean db and prepares features for tire strategy.
Dataset 2 comes from the less clean db and focuses on pit stop events.
"""

import argparse
import logging
import math
import re
import sqlite3
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

pd.set_option('future.no_silent_downcasting', True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Set to True to include the pit stops left feature for structural analyses
INCLUDE_PIT_STOPS_LEFT = False

# Estimated net time loss from making a pit stop in seconds
NET_PIT_LOSS_BY_TRACK_S: dict[str, float] = {
    "70th Anniversary Grand Prix": 9.01,
    "Abu Dhabi Grand Prix": 11.216,
    "Australian Grand Prix": 5.783,
    "Austrian Grand Prix": 9.668,
    "Azerbaijan Grand Prix": 6.65,
    "Bahrain Grand Prix": 9.629,
    "Belgian Grand Prix": 6.754,
    "Brazilian Grand Prix": 10.345,
    "British Grand Prix": 9.602,
    "Canadian Grand Prix": 8.075,
    "Chinese Grand Prix": 9.13,
    "Dutch Grand Prix": 8.043,
    "Eifel Grand Prix": 8.759,
    "Emilia Romagna Grand Prix": 13.08,
    "French Grand Prix": 9.982,
    "German Grand Prix": 8.53,
    "Hungarian Grand Prix": 7.905,
    "Italian Grand Prix": 9.408,
    "Japanese Grand Prix": 8.353,
    "Las Vegas Grand Prix": 7.284,
    "Mexican Grand Prix": 8.249,
    "Mexico City Grand Prix": 8.405,
    "Miami Grand Prix": 6.367,
    "Monaco Grand Prix": 10.712,
    "Portuguese Grand Prix": 11.787,
    "Qatar Grand Prix": 11.32,
    "Russian Grand Prix": 9.231,
    "Sakhir Grand Prix": 11.915,
    "Saudi Arabian Grand Prix": 7.927,
    "Singapore Grand Prix": 12.235,
    "Spanish Grand Prix": 9.141,
    "Styrian Grand Prix": 9.707,
    "São Paulo Grand Prix": 10.487,
    "Tuscan Grand Prix": 8.522,
    "United States Grand Prix": 9.958,
}

DEFAULT_NET_PIT_LOSS_S = 9.141

# Mapping for compound hardness to an absolute scale where 1 is hardest and 7 is softest
ABSOLUTE_HARDNESS_MAP = {
    "A1": 1,
    "A2": 2,
    "A3": 3,
    "A4": 4,
    "A5": 5,
    "A6": 6,
    "A7": 7,
    "C1": 2,
    "C2": 3,
    "C3": 4,
    "C4": 6,
    "C5": 7,
}


def pick_race_lap_count(races: pd.DataFrame) -> pd.Series:
    """Determines the total laps for the race using planned if available otherwise actual."""
    if "nolapsplanned" in races.columns:
        base = races["nolapsplanned"].copy()
    else:
        base = pd.Series([np.nan] * len(races), index=races.index)

    if "nolaps" in races.columns:
        base = base.fillna(races["nolaps"])
    return base


def normalize_compound(x: Optional[str]) -> Optional[str]:
    """Standardizes compound names to a common format."""
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
    """Calculates estimated net time loss from pitting adjusted for safety car status."""
    track = race_track.astype(str).str.strip()
    base = track.map(NET_PIT_LOSS_BY_TRACK_S).fillna(DEFAULT_NET_PIT_LOSS_S).astype(float)

    f = pd.to_numeric(fcy_status, errors="coerce").fillna(0).astype(int)

    # Reduce net loss under VSC/SC conditions based on simulator assumptions
    mult = np.where(
        (f == 1) | (f == 2),
        0.71,
        np.where((f == 3) | (f == 4), 0.63, 1.0)
    )
    return base * mult


def extract_hardest_slick_from_availablecompounds(s: Optional[str]) -> Optional[int]:
    """
    Parses the available compounds string to find the hardest slick tire on an absolute scale.
    1 is hardest (superhard) and 7 is softest (hypersoft or C5).
    """
    if s is None:
        return None

    txt = str(s).upper()
    matches = re.findall(r"\b([AC][1-7])\b", txt)
    if not matches:
        return None

    abs_hardness = [ABSOLUTE_HARDNESS_MAP.get(m) for m in matches]
    abs_hardness = [h for h in abs_hardness if h is not None]
    
    if not abs_hardness:
        return None

    return min(abs_hardness)


def track_category_from_hardest(hardest_abs: Optional[int]) -> Optional[int]:
    """
    Categorizes track degradation based on the hardest compound brought to the track.
    1: high degradation (hardest tires brought)
    2: medium degradation (medium tires brought)
    3: low degradation (soft tires brought)
    """
    if hardest_abs is None or (isinstance(hardest_abs, float) and math.isnan(hardest_abs)):
        return None
    
    if hardest_abs <= 2:
        return 1 
    if hardest_abs == 3:
        return 2 
    return 3 


def _rejoin_gaps_one_lap(grp: pd.DataFrame) -> pd.DataFrame:
    """Estimates gaps to cars ahead and behind after a hypothetical pit stop."""
    g = grp.copy()
    times = pd.to_numeric(g["racetime_sofar"], errors="coerce").to_numpy()
    ids = g["driver_id"].to_numpy()
    pit_loss = pd.to_numeric(g["pit_loss_est_s"], errors="coerce").to_numpy()

    t_proj = times + pit_loss

    order = np.argsort(times)
    times_sorted = times[order]
    ids_sorted = ids[order]

    ahead_gap = np.full(len(g), np.nan, dtype=float)
    behind_gap = np.full(len(g), np.nan, dtype=float)

    for i in range(len(g)):
        ins = np.searchsorted(times_sorted, t_proj[i], side="left")

        prev_idx = ins - 1
        next_idx = ins

        # Skip self comparisons (driver does not race against their own ghost)
        while prev_idx >= 0 and ids_sorted[prev_idx] == ids[i]:
            prev_idx -= 1
        while next_idx < len(times_sorted) and ids_sorted[next_idx] == ids[i]:
            next_idx += 1

        if prev_idx >= 0:
            ahead_gap[i] = t_proj[i] - times_sorted[prev_idx]
        if next_idx < len(times_sorted):
            behind_gap[i] = times_sorted[next_idx] - t_proj[i]

    return pd.DataFrame({
        "rejoin_gap_ahead_est_s": ahead_gap,
        "rejoin_gap_behind_est_s": behind_gap
    }, index=g.index)


def add_pit_stops_left(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates how many pit stops are remaining for the driver in the race."""
    df = df.sort_values(["race_id", "driver_id", "lapno"]).copy()

    total_pits = df.groupby(["race_id", "driver_id"])["y_pit"].transform("sum").astype(int)

    if "pit_stops_so_far" not in df.columns:
        df["pit_stops_so_far"] = (
            df.groupby(["race_id", "driver_id"])["y_pit"]
            .transform(lambda s: s.cumsum().shift(1).fillna(0).astype(int))
        )

    df["pit_stops_left"] = (total_pits - df["pit_stops_so_far"]).astype(int).clip(lower=0)
    return df


def add_undercut_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds features related to track position and gaps relevant for undercutting."""
    df = df.copy()
    
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

    df["behind_racetime"] = g["racetime_sofar"].shift(-1)
    df["gap_behind_s"] = df["behind_racetime"] - df["racetime_sofar"]

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

    df["ahead1_tyre_age"] = g["tyre_age"].shift(1)
    df["tyre_age_diff_to_ahead"] = df["tyre_age"] - df["ahead1_tyre_age"]

    if "race_track" in df.columns and "fcy_status" in df.columns:
        df["pit_loss_est_s"] = estimate_pit_loss_seconds(df["race_track"], df["fcy_status"])
    else:
        df["pit_loss_est_s"] = DEFAULT_NET_PIT_LOSS_S

    rejoin_cols = (
        df.groupby(["race_id", "lapno"], sort=False, group_keys=False)
        .apply(_rejoin_gaps_one_lap, include_groups=False)
    )
    df = df.join(rejoin_cols)

    df = df.drop(
        columns=["behind_racetime", "ahead1_tyre_age", "pit_loss_est_s"],
        errors="ignore",
    )
    return df


def add_weather_flags(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Processes weather data to add flags for current rain and accumulated rain."""
    df = df.copy()
    df["is_raining"] = 0
    df["minutes_rain"] = np.nan

    if weather is None or weather.empty:
        return df

    df["_time_ms"] = pd.to_numeric(df["decision_time_ms"], errors="coerce").round().astype("Int64")

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
    df2 = df2.drop(columns=["_time_ms_num", "_time_ms"], errors="ignore")
    return df2


def add_is_wet_race(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flags if the driver has used intermediate or wet tires at any point
    up to and including the current lap in the race.
    """
    df = df.sort_values(["race_id", "driver_id", "lapno"]).copy()
    
    comp = df["current_compound"]
    is_wet_tyre = comp.isin({"INTERMEDIATE", "WET"}).astype(int)
    
    df["is_wet_race"] = (
        is_wet_tyre.groupby([df["race_id"], df["driver_id"]])
        .cumsum()
        .clip(upper=1)
        .astype(int)
    )
    return df


def add_pit_stops_so_far(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates cumulative pit stops taken strictly prior to the current lap."""
    df = df.sort_values(["race_id", "driver_id", "lapno"]).copy()
    
    df["pit_stops_so_far"] = (
        df.groupby(["race_id", "driver_id"])["y_pit"]
          .transform(lambda s: s.cumsum().shift(1).fillna(0).astype(int))
    )
    return df


def add_tyre_change_pursuer(df: pd.DataFrame) -> pd.DataFrame:
    """Checks if the car immediately behind changed tyres in the previous lap."""
    df = df.sort_values(["race_id", "driver_id", "lapno"]).copy()

    def did_change(s):
        shifted = s.shift(1)
        return s.notna() & shifted.notna() & (s != shifted)
        
    df["did_change_since_prev"] = df.groupby(["race_id", "driver_id"])["current_compound"].transform(did_change)

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
        columns=["did_change_since_prev", "pursuer_position", "pursuer_changed_prev", "position_pursuer_row"],
        errors="ignore"
    )
    return df


def add_close_ahead(df: pd.DataFrame, threshold_s: float = 1.5) -> pd.DataFrame:
    """Flags if the driver was closely followed by another car in the previous lap."""
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
    """Maps full course yellow phases to a categorical status for each lap."""
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
    prev_phase = df2.groupby(["race_id", "driver_id"], sort=False)["phase_type_end"].shift(1)

    is_vsc = df2["phase_type_end"].eq("VSC")
    is_sc = df2["phase_type_end"].eq("SC")

    # Encode states: 1 (VSC start), 2 (VSC cont), 3 (SC start), 4 (SC cont)
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
    """Checks if the regulation to use two distinct compounds has been met."""
    df = df.sort_values(["race_id", "driver_id", "lapno"]).copy()

    comp = df["current_compound"]
    missing = comp.isna()
    tmp = comp.fillna("__MISSING__")

    first_occ = tmp.groupby([df["race_id"], df["driver_id"]]).transform(lambda s: ~s.duplicated())
    first_occ = first_occ & (~missing)

    cum_distinct = first_occ.groupby([df["race_id"], df["driver_id"]]).cumsum().astype(int)
    
    is_wet_race = pd.to_numeric(df["is_wet_race"], errors="coerce").fillna(0).astype(int)

    df["fulfilled_second_compound"] = ((is_wet_race == 1) | (cum_distinct >= 2)).astype(int)
    return df


def load_core_tables(conn: sqlite3.Connection) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Retrieves raw tables from the database for processing using fixed schema queries."""
    laps_cols = [
        "race_id", "driver_id", "lapno", "position",
        "laptime", "racetime", "gap", "interval",
        "compound", "tireage", "nextcompound",
        "pitintime", "pitstopduration", "pit_in_elapsed",
        "pitintimenum", "pitouttimenum",
        "sector1session_ms", "sector2session_ms", "sector3session_ms"
    ]
    laps = pd.read_sql_query(f"SELECT {', '.join(laps_cols)} FROM laps;", conn)

    races_cols = ["id", "location", "nolapsplanned", "nolaps", "availablecompounds", "availablecompounds_c"]
    races = pd.read_sql_query(f"SELECT {', '.join(races_cols)} FROM races;", conn).rename(columns={"id": "race_id"})

    fcy_cols = ["race_id", "startracetime", "endracetime", "startlap", "endlap", "type"]
    fcy = pd.read_sql_query(f"SELECT {', '.join(fcy_cols)} FROM fcyphases;", conn)

    weather_query = """
        SELECT ws.race_id, ws.time_ms, ws.rainfall 
        FROM weather_samples ws
        INNER JOIN sessions s ON ws.session_id = s.id
        WHERE UPPER(s.session_code) = 'R';
    """
    weather = pd.read_sql_query(weather_query, conn)

    return laps, races, fcy, weather


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Generates target variables for training."""
    df = df.copy()

    df["y_pit"] = df["pitintime"].notna().astype(int)

    df["y_compound"] = df["nextcompound"].apply(normalize_compound)
    df.loc[df["y_pit"] != 1, "y_compound"] = None

    return df


def add_race_progress_and_track_category(laps: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """Adds derived features for race completion and track characteristics."""
    df = laps.merge(races, on="race_id", how="left")

    total_laps = pick_race_lap_count(df)
    df["race_progress"] = df["lapno"] / total_laps.replace({0: np.nan})
    df["race_progress"] = df["race_progress"].clip(lower=0.0, upper=1.0)

    ac = df["availablecompounds_c"].fillna(df["availablecompounds"])
    hardest_abs = ac.apply(extract_hardest_slick_from_availablecompounds)
    df["track_category"] = hardest_abs.apply(track_category_from_hardest)

    df["race_track"] = df["location"].astype(str)

    return df


def add_extra_output_features(df: pd.DataFrame) -> pd.DataFrame:
    """Includes useful columns from raw data to the processing dataframe."""
    df = df.copy()

    df["current_compound"] = df["compound"].apply(normalize_compound)

    lap_time_base = pd.to_numeric(df["laptime"], errors="coerce")
    pit_in_time = pd.to_numeric(df["pit_in_elapsed"], errors="coerce")
    df["lap_time"] = np.where(df["y_pit"] == 1, pit_in_time.fillna(lap_time_base), lap_time_base)

    df["racetime_sofar"] = pd.to_numeric(df["racetime"], errors="coerce")
    df["gap_to_leader"] = pd.to_numeric(df["gap"], errors="coerce")
    df["interval"] = pd.to_numeric(df["interval"], errors="coerce")
    df["tyre_age"] = pd.to_numeric(df["tireage"], errors="coerce")

    for c in ["pitintimenum", "pitouttimenum", "sector1session_ms", "sector2session_ms", "sector3session_ms"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    session_time_cols = ["pitintimenum", "sector3session_ms", "sector2session_ms", "sector1session_ms"]
    df["decision_time_ms"] = df[session_time_cols].bfill(axis=1).iloc[:, 0]

    return df


def finalize_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values in a principled way for the final dataset.
    Semantically fills certain columns and adds missingness flags for others.
    Leaves main numeric columns as NaN for downstream statistical imputation.
    """
    df = df.copy()

    if "is_raining" in df.columns:
        df["is_raining"] = df["is_raining"].fillna(0)
    if "minutes_rain" in df.columns and "is_raining" in df.columns:
        mask_not_raining = df["is_raining"] == 0
        df.loc[mask_not_raining, "minutes_rain"] = df.loc[mask_not_raining, "minutes_rain"].fillna(0.0)
    if "tyre_change_pursuer" in df.columns:
        df["tyre_change_pursuer"] = df["tyre_change_pursuer"].fillna(False)
    if "close_ahead" in df.columns:
        df["close_ahead"] = df["close_ahead"].fillna(0)
    if "fcy_status" in df.columns:
        df["fcy_status"] = df["fcy_status"].fillna(0)

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
    """Pipeline that coordinates data loading and feature engineering."""
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
    
    df = finalize_missingness(df)

    if "race_progress" in df.columns:
        df["race_progress"] = df["race_progress"].astype(float).clip(0.0, 1.0)
    if "position" in df.columns:
        df["position"] = df["position"].astype(int).clip(0, 22)
    if "fcy_status" in df.columns:
        df["fcy_status"] = df["fcy_status"].astype(int).clip(0, 4)

    return df


def make_dataset_1(df_full: pd.DataFrame) -> pd.DataFrame:
    """Filters columns for the main tire strategy dataset."""
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
        "gap_behind_s_missing", "tyre_age_diff_to_ahead_missing",
        "rejoin_gap_ahead_est_s_missing", "rejoin_gap_behind_est_s_missing",
        "tyre_age_missing"
    ]
    
    if INCLUDE_PIT_STOPS_LEFT:
        keep_cols.append("pit_stops_left")
        
    return df_full[[c for c in keep_cols if c in df_full.columns]].copy()


def make_dataset_2(df_full: pd.DataFrame) -> pd.DataFrame:
    """Filters columns for the pit event dataset."""
    df_pit = df_full[df_full["y_pit"] == 1].copy()

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
        "gap_behind_s_missing", "tyre_age_diff_to_ahead_missing",
        "rejoin_gap_ahead_est_s_missing", "rejoin_gap_behind_est_s_missing"
    ]
    
    if INCLUDE_PIT_STOPS_LEFT:
        keep_cols.append("pit_stops_left")
        
    return df_pit[[c for c in keep_cols if c in df_pit.columns]].copy()

def main() -> None:
    db_clean_path = "../data/raw/f1_database__clean.sqlite"
    db_less_clean_path = "../data/raw/f1_database__less_clean.sqlite"

    out1_path = Path("../data/processed/dataset1.csv")
    out2_path = Path("../data/processed/dataset2.csv")

    out1_path.parent.mkdir(parents=True, exist_ok=True)

    df_clean_full = build_dataset(db_clean_path)
    ds1 = make_dataset_1(df_clean_full)
    ds1.to_csv(out1_path, index=False)
    logger.info(f"Dataset 1 (tire strategy) wrote {len(ds1)} rows to {out1_path}")

    df_less_full = build_dataset(db_less_clean_path)
    ds2 = make_dataset_2(df_less_full)
    ds2.to_csv(out2_path, index=False)
    logger.info(f"Dataset 2 (pit events) wrote {len(ds2)} rows to {out2_path}")

if __name__ == "__main__":
    main()
