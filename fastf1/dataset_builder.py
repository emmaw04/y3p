#!/usr/bin/env python3
"""
Dataset builder: SQLite -> pandas (TWO datasets)

Produces two datasets from two different cleaned DB variants:

Dataset 1 (from *clean.sqlite*):
  - Same as your current dataset, BUT:
      * drops y_compound
      * adds fulfilled_second_compound
  - Keeps y_pit.

Dataset 2 (from *less_clean.sqlite*):
  - ONLY rows where y_pit == 1
  - Columns ONLY (no y_pit):
    race_id
      Race_progress
      Pit_stops_so_far
      Current_compound
      Race_track
      Fulfilled_second_compound
      Rained_yet
      Is_raining
      Minutes_rain
      y_compound

fulfilled_second_compound at lap t is True iff:
  - rained_yet == 1 at lap t, OR
  - the driver has used >= 2 DISTINCT tyre compounds in that race up to and including lap t
    (computed from laps.compound, cumulative up to lap t).

Usage:
  python dataset_builder.py \
    --db-clean /path/to/yourdb__clean.sqlite \
    --db-less-clean /path/to/yourdb__less_clean.sqlite \
    --out1 /path/to/dataset_clean.parquet \
    --out2 /path/to/dataset_pit_events.parquet

Output format inferred from extension (.parquet or .csv), or override via --format1/--format2.
"""

from __future__ import annotations

import argparse
import math
import re
import sqlite3
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Utilities
# -----------------------------
def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    q = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
    return conn.execute(q, (name,)).fetchone() is not None


def pick_race_lap_count(races: pd.DataFrame) -> pd.Series:
    """Prefer nolapsplanned, fallback to nolaps."""
    if "nolapsplanned" in races.columns:
        base = races["nolapsplanned"].copy()
    else:
        base = pd.Series([np.nan] * len(races), index=races.index)

    if "nolaps" in races.columns:
        base = base.fillna(races["nolaps"])
    return base

def normalize_compound(x: Optional[str]) -> Optional[str]:
    """Map raw strings to {SOFT, MEDIUM, HARD, INTERMEDIATE, WET}. Return None if unknown."""
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
    if s in {"", "NAN", "NONE", "UNKNOWN"}:
        return None
    return None

# undercut / clean air features

# stores pit loss per track
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

DEFAULT_PIT_LOSS_S = 23.113 #median of dictionary, added just in case


def estimate_pit_loss_seconds(race_track: pd.Series, fcy_status: pd.Series) -> pd.Series:
    track = race_track.astype(str).str.strip()
    base = track.map(PIT_LOSS_BY_TRACK_S).fillna(DEFAULT_PIT_LOSS_S).astype(float)

    f = pd.to_numeric(fcy_status, errors="coerce").fillna(0).astype(int)
    mult = np.where((f == 1) | (f == 2), 0.65, np.where((f == 3) | (f == 4), 0.45, 1.0))
    return base * mult


def _rejoin_gaps_one_lap(grp: pd.DataFrame) -> pd.DataFrame:
    """
    For one (race_id, lapno) snapshot:
      t_proj = racetime_sofar + pit_loss_est_s

    Race-time ordering: smaller racetime => ahead.
    So:
      rejoin_gap_ahead = t_proj - max(time < t_proj)   (excluding self)
      rejoin_gap_behind = min(time > t_proj) - t_proj  (excluding self)
    """
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
        # insertion point in sorted list
        ins = np.searchsorted(times_sorted, t_proj[i], side="left")

        prev_idx = ins - 1
        next_idx = ins

        # exclude self (otherwise prev could be self => ahead_gap ~ pit_loss, wrong)
        while prev_idx >= 0 and ids_sorted[prev_idx] == ids[i]:
            prev_idx -= 1
        while next_idx < len(times_sorted) and ids_sorted[next_idx] == ids[i]:
            next_idx += 1

        if prev_idx >= 0:
            ahead_gap[i] = t_proj[i] - times_sorted[prev_idx]
        if next_idx < len(times_sorted):
            behind_gap[i] = times_sorted[next_idx] - t_proj[i]

    g["rejoin_gap_ahead_est_s"] = ahead_gap
    g["rejoin_gap_behind_est_s"] = behind_gap
    return g

def add_pit_stops_left(df: pd.DataFrame, clip_to_0_3: bool = False) -> pd.DataFrame:
    """
    pit_stops_left at lap t:
      = total number of pit events the driver makes in that race
        minus pit_stops_so_far(t) (which counts pit events up to lap t-1)

    So if total pits = 3:
      - before 1st pit: 3
      - on 1st pit lap: 3
      - after 1st pit: 2
      - on 2nd pit lap: 2
      - after 2nd pit: 1
      - after 3rd pit: 0
    """
    df = df.sort_values(["race_id", "driver_id", "lapno"]).copy()

    if "y_pit" not in df.columns:
        df["y_pit"] = 0
    df["y_pit"] = pd.to_numeric(df["y_pit"], errors="coerce").fillna(0).astype(int)

    # total number of pit stops actually taken in the full race (uses future info)
    total_pits = df.groupby(["race_id", "driver_id"])["y_pit"].transform("sum").astype(int)

    # make sure pit_stops_so_far exists (your pipeline already adds it)
    if "pit_stops_so_far" not in df.columns:
        df["pit_stops_so_far"] = (
            df.groupby(["race_id", "driver_id"])["y_pit"]
              .cumsum()
              .shift(1)
              .fillna(0)
              .astype(int)
        )

    df["pit_stops_left"] = (total_pits - df["pit_stops_so_far"]).astype(int)

    # safety
    df["pit_stops_left"] = df["pit_stops_left"].clip(lower=0)

    if clip_to_0_3:
        df["pit_stops_left"] = df["pit_stops_left"].clip(0, 3).astype(int)

    return df

def add_undercut_features(df: pd.DataFrame, lag_one_lap: bool = True) -> pd.DataFrame:
    """
    Adds (optionally lagged by 1 lap within each (race,driver)):
      - gap_behind_s
      - n_cars_within_5s_ahead   (count among up to 3 cars directly ahead)
      - tyre_age_diff_to_ahead
      - rejoin_gap_ahead_est_s
      - rejoin_gap_behind_est_s
    """
    df = df.copy()

    need = {"race_id", "driver_id", "lapno", "position", "racetime_sofar"}
    if not need.issubset(df.columns):
        # create empty cols if insufficient inputs
        for c in ["gap_behind_s", "n_cars_within_5s_ahead", "tyre_age_diff_to_ahead",
                  "rejoin_gap_ahead_est_s", "rejoin_gap_behind_est_s"]:
            df[c] = np.nan
        return df

    # Ensure numeric
    df["position"] = pd.to_numeric(df["position"], errors="coerce")
    df["racetime_sofar"] = pd.to_numeric(df["racetime_sofar"], errors="coerce")
    df["tyre_age"] = pd.to_numeric(df.get("tyre_age", np.nan), errors="coerce")

    # Per-lap snapshot ordering by position
    df = df.sort_values(["race_id", "lapno", "position", "driver_id"]).copy()
    g = df.groupby(["race_id", "lapno"], sort=False)

    # gap behind: (car behind by position) racetime - self racetime
    df["behind_racetime"] = g["racetime_sofar"].shift(-1)
    df["gap_behind_s"] = df["behind_racetime"] - df["racetime_sofar"]

    # up to 3 cars ahead by position: shift(1..3)
    def _count_ahead_within_5s(grp: pd.DataFrame) -> pd.Series:
        # grp is one (race_id, lapno), already small (~20)
        g2 = grp.sort_values(["position", "driver_id"]).copy()
        t = pd.to_numeric(g2["racetime_sofar"], errors="coerce").to_numpy()

        out = np.full(len(g2), np.nan, dtype=float)
        for i in range(len(g2)):
            if np.isnan(t[i]):
                continue
            # cars ahead are indices < i (smaller racetime normally)
            # count those with t[i] - t[j] <= 5  => t[j] >= t[i] - 5
            # find earliest index j0 among ahead cars that satisfies this
            j0 = np.searchsorted(t, t[i] - 5.0, side="left")
            out[i] = max(0, i - j0)  # number of ahead cars within 5s
        return pd.Series(out, index=g2.index)

    df["n_cars_within_5s_ahead"] = (
        df.groupby(["race_id", "lapno"], sort=False, group_keys=False)
        .apply(_count_ahead_within_5s)
        .fillna(0)
        .astype(int)
    )

    # tyre age diff to car ahead (position-1)
    df["ahead1_tyre_age"] = g["tyre_age"].shift(1)
    df["tyre_age_diff_to_ahead"] = df["tyre_age"] - df["ahead1_tyre_age"]

    # --- Rejoin gaps (needs race_track + fcy_status for pit loss estimate) ---
    if "race_track" in df.columns and "fcy_status" in df.columns:
        df["pit_loss_est_s"] = estimate_pit_loss_seconds(df["race_track"], df["fcy_status"])
    else:
        df["pit_loss_est_s"] = DEFAULT_PIT_LOSS_S

    # Compute rejoin gaps per (race_id, lapno)
    df = (
        df.groupby(["race_id", "lapno"], sort=False, group_keys=False)
          .apply(_rejoin_gaps_one_lap)
    )

    # Optional: lag all these features by one lap to match your "close_ahead uses t-1" convention
    if lag_one_lap:
        df = df.sort_values(["race_id", "driver_id", "lapno"]).copy()
        lag_cols = [
            "gap_behind_s",
            "n_cars_within_5s_ahead",
            "tyre_age_diff_to_ahead",
            "rejoin_gap_ahead_est_s",
            "rejoin_gap_behind_est_s",
        ]
        for c in lag_cols:
            df[c] = df.groupby(["race_id", "driver_id"], sort=False)[c].shift(1)

    # cleanup intermediates
    df = df.drop(
        columns=[
            "behind_racetime",
            "ahead1_racetime", "ahead2_racetime", "ahead3_racetime",
            "ahead1_tyre_age",
            "pit_loss_est_s",
        ],
        errors="ignore",
    )
    return df


# Invert your mapping: A -> C
A_TO_C = {
    "A2": 1,  # C1
    "A3": 2,  # C2
    "A4": 3,  # C3
    "A6": 4,  # C4
    "A7": 5,  # C5
}


def extract_hardest_slick_from_availablecompounds(s: Optional[str]) -> Optional[int]:
    """
    Supports both formats:
      - "C1,C2,C3"
      - "A3,A4,A6,I,W" with mapping A2->C1, A3->C2, A4->C3, A6->C4, A7->C5
    Returns: hardest slick as C-number in {1..5} (smallest is hardest), or None if not found.
    """
    if s is None:
        return None

    txt = str(s).upper()

    # 1) Try C-notation directly
    c_matches = re.findall(r"\bC([1-5])\b", txt)
    if c_matches:
        return min(int(m) for m in c_matches)

    # 2) Fall back to A-notation (ignore I/W)
    a_matches = re.findall(r"\bA(2|3|4|6|7)\b", txt)
    if not a_matches:
        return None

    c_nums = [A_TO_C.get(f"A{m}") for m in a_matches]
    c_nums = [c for c in c_nums if c is not None]
    return min(c_nums) if c_nums else None


def track_category_from_hardest(hardest_c: Optional[int]) -> Optional[int]:
    """
    Binning:
      1 = low degradation    (hardest = C3/C4/C5)
      2 = medium             (hardest = C2)
      3 = high degradation   (hardest = C1)
    """
    if hardest_c is None or (isinstance(hardest_c, float) and math.isnan(hardest_c)):
        return None
    if hardest_c == 1:
        return 3
    if hardest_c == 2:
        return 2
    return 1


# -----------------------------
# Core table loader
# -----------------------------
def load_core_tables(conn: sqlite3.Connection) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load laps, races, fcyphases + weather timeline (mapped to race_id) needed for features/labels.
    """
    required = ["laps", "races", "fcyphases"]
    missing = [t for t in required if not table_exists(conn, t)]
    if missing:
        raise RuntimeError(f"Missing required tables: {missing}")

    # --- Laps ---
    laps_cols = [
        "race_id", "driver_id", "lapno", "position",
        "laptime",
        "racetime",
        "gap", "interval",
        "compound",
        "tireage",
        "nextcompound",
        "pitintime", "pitstopduration",
    ]
    laps_info = pd.read_sql_query("PRAGMA table_info(laps);", conn)
    existing_laps_cols = set(laps_info["name"].tolist())
    laps_cols = [c for c in laps_cols if c in existing_laps_cols]

    laps = pd.read_sql_query(f"SELECT {', '.join(laps_cols)} FROM laps;", conn)

    # --- Races ---
    races_cols = ["id", "location", "nolapsplanned", "nolaps", "availablecompounds", "availablecompounds_c"]
    races_info = pd.read_sql_query("PRAGMA table_info(races);", conn)
    existing_races_cols = set(races_info["name"].tolist())
    races_cols = [c for c in races_cols if c in existing_races_cols]

    races = pd.read_sql_query(f"SELECT {', '.join(races_cols)} FROM races;", conn).rename(columns={"id": "race_id"})

    # --- FCY phases ---
    fcy_cols = ["race_id", "startracetime", "endracetime", "startlap", "endlap", "type"]
    fcy_info = pd.read_sql_query("PRAGMA table_info(fcyphases);", conn)
    existing_fcy_cols = set(fcy_info["name"].tolist())
    fcy_cols = [c for c in fcy_cols if c in existing_fcy_cols]

    fcy = pd.read_sql_query(f"SELECT {', '.join(fcy_cols)} FROM fcyphases;", conn)

    # --- Weather: map weather_samples -> race_id via sessions (Race session only) ---
    weather = pd.DataFrame(columns=["race_id", "time_ms", "rainfall"])

    if table_exists(conn, "sessions") and table_exists(conn, "weather_samples"):
        sess = pd.read_sql_query("SELECT id AS session_id, race_id, session_code FROM sessions;", conn)
        sess = sess[sess["session_code"].astype(str).str.upper() == "R"][["session_id", "race_id"]]

        ws = pd.read_sql_query("SELECT session_id, time_ms, rainfall FROM weather_samples;", conn)
        weather = ws.merge(sess, on="session_id", how="inner")[["race_id", "time_ms", "rainfall"]].copy()

    return laps, races, fcy, weather


# -----------------------------
# Labels + features
# -----------------------------
def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add:
      y_pit: 1 if pit this lap, else 0
      y_compound: normalized nextcompound for pit laps; None otherwise
    """
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
    df = laps.merge(races, on="race_id", how="left")

    total_laps = pick_race_lap_count(df)
    df["race_progress"] = df["lapno"] / total_laps.replace({0: np.nan})
    df["race_progress"] = df["race_progress"].clip(lower=0.0, upper=1.0)

    if "availablecompounds_c" in df.columns:
        ac = df["availablecompounds_c"].where(df["availablecompounds_c"].notna(), df.get("availablecompounds"))
    else:
        ac = df.get("availablecompounds")

    hardest = ac.apply(extract_hardest_slick_from_availablecompounds)
    df["track_category"] = hardest.apply(track_category_from_hardest)

    if "location" in df.columns:
        df["race_track"] = df["location"].astype(str)
    else:
        df["race_track"] = None

    return df


def add_extra_output_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - current_compound  <- laps.compound (normalized)
      - lap_time          <- laps.laptime
      - racetime_sofar    <- laps.racetime
      - gap_to_leader     <- laps.gap
      - interval          <- laps.interval
      - tyre_age          <- laps.tireage
    """
    df = df.copy()

    if "compound" in df.columns:
        df["current_compound"] = df["compound"].apply(normalize_compound)
    else:
        df["current_compound"] = None

    df["lap_time"] = pd.to_numeric(df["laptime"], errors="coerce") if "laptime" in df.columns else np.nan
    df["racetime_sofar"] = pd.to_numeric(df["racetime"], errors="coerce") if "racetime" in df.columns else np.nan

    df["gap_to_leader"] = pd.to_numeric(df["gap"], errors="coerce") if "gap" in df.columns else np.nan
    df["interval"] = pd.to_numeric(df["interval"], errors="coerce") if "interval" in df.columns else np.nan

    df["tyre_age"] = pd.to_numeric(df["tireage"], errors="coerce") if "tireage" in df.columns else np.nan

    return df


def add_weather_flags(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - rained_yet: 1 if ANY weather sample up to current time had rainfall=1
      - is_raining: 1 if latest sample at/before current time has rainfall=1
      - minutes_rain: only when is_raining=1; duration of consecutive rain streak up to now
    """
    df = df.copy()
    df["rained_yet"] = 0
    df["is_raining"] = 0
    df["minutes_rain"] = np.nan

    if weather is None or weather.empty:
        return df
    if "race_id" not in df.columns:
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

        w["rained_yet_cum"] = w["rainfall"].cummax()

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
        w["time_ms_i64"] = pd.to_numeric(w["time_ms"], errors="coerce")
        w = w.dropna(subset=["time_ms_i64"]).copy()
        w["time_ms_i64"] = w["time_ms_i64"].astype(np.int64)
        w = w.sort_values("time_ms_i64")

        merged = pd.merge_asof(
            df_t,
            w[["time_ms_i64", "rainfall", "rained_yet_cum", "rain_streak_start_ms"]],
            left_on="_time_ms_i64",
            right_on="time_ms_i64",
            direction="backward",
            allow_exact_matches=True,
        )
        merged = merged.drop(columns=["time_ms_i64", "_time_ms_i64"], errors="ignore")

        merged["is_raining"] = merged["rainfall"].fillna(0).astype(int)
        merged["rained_yet"] = merged["rained_yet_cum"].fillna(0).astype(int)

        dur_ms = merged["_time_ms_num"] - pd.to_numeric(merged["rain_streak_start_ms"], errors="coerce")
        merged["minutes_rain"] = np.where(
            merged["is_raining"].eq(1) & dur_ms.notna(),
            dur_ms / 60000.0,
            np.nan
        )

        merged = merged.drop(columns=["time_ms", "rainfall", "rained_yet_cum", "rain_streak_start_ms"], errors="ignore")

        out.append(pd.concat([merged, df_nt], ignore_index=True) if not df_nt.empty else merged)

    df2 = pd.concat(out, ignore_index=True)
    df2 = df2.drop(columns=["_time_ms_num"], errors="ignore")
    return df2


def add_pit_stops_so_far(df: pd.DataFrame, clip_to_0_3: bool = False) -> pd.DataFrame:
    """
    pit_stops_so_far at lap t = count of pit events up to lap t-1 (online-computable)
    """
    df = df.sort_values(["race_id", "driver_id", "lapno"]).copy()
    df["pit_stops_so_far"] = (
        df.groupby(["race_id", "driver_id"])["y_pit"]
          .cumsum()
          .shift(1)
          .fillna(0)
          .astype(int)
    )
    if clip_to_0_3:
        df["pit_stops_so_far"] = df["pit_stops_so_far"].clip(lower=0, upper=3).astype(int)
    return df


def add_tyre_change_pursuer(df: pd.DataFrame) -> pd.DataFrame:
    """
    tyre_change_pursuer: whether the pursuer (position+1) changed tyres in the previous lap.
    Returns 0/1 instead of False/True.
    """
    df = df.sort_values(["race_id", "driver_id", "lapno"]).copy()

    if "compound" not in df.columns:
        df["tyre_change_pursuer"] = 0
        return df

    df["compound_norm"] = df["compound"].apply(normalize_compound)
    prev_comp = df.groupby(["race_id", "driver_id"])["compound_norm"].shift(1)
    df["did_change_since_prev"] = (
        df["compound_norm"].notna()
        & prev_comp.notna()
        & (df["compound_norm"] != prev_comp)
    )

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

    df["tyre_change_pursuer"] = df["pursuer_changed_prev"].fillna(False).astype(int)

    df = df.drop(
        columns=[c for c in [
            "compound_norm", "did_change_since_prev", "pursuer_position",
            "pursuer_changed_prev", "position_pursuer_row"
        ] if c in df.columns],
        errors="ignore"
    )
    return df


def add_close_ahead(df: pd.DataFrame, threshold_s: float = 1.5) -> pd.DataFrame:
    """
    close_ahead at lap t:
      True if, at the end of lap t-1, the driver was <= threshold_s ahead of their pursuer
      (the car directly behind them at lap t-1).
    """
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
    """
    Table 6 FCY status (active at END of lap):
      0: No FCY active
      1: First lap of VSC active
      2: Further laps of VSC active
      3: First lap of SC active
      4: Further laps of SC active
    """
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

    if not {"startracetime", "endracetime", "race_id"}.issubset(fcy.columns):
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
        if fcy_r.empty:
            df_r["phase_type_end"] = None
            out_frames.append(df_r)
            continue

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
    """
    fulfilled_second_compound at lap t:
      True iff rained_yet(t)==1 OR cumulative distinct compounds used by the driver in the race
      up to and including lap t is >= 2.

    Distinct compounds are computed from laps.compound (normalized to 5-class),
    with missing/unknown compounds ignored in the distinct count.
    """
    df = df.sort_values(["race_id", "driver_id", "lapno"]).copy()

    # Need compound for the distinct-count part
    if "compound" in df.columns:
        comp = df["compound"].apply(normalize_compound)
    elif "current_compound" in df.columns:
        comp = df["current_compound"].apply(normalize_compound)
    else:
        comp = pd.Series([None] * len(df), index=df.index)

    missing = comp.isna()
    tmp = comp.fillna("__MISSING__")

    # First occurrence of each compound within a (race,driver) sequence
    first_occ = tmp.groupby([df["race_id"], df["driver_id"]]).transform(lambda s: ~s.duplicated())
    # Ignore missing compound in count
    first_occ = first_occ & (~missing)

    # Cumulative distinct count
    cum_distinct = first_occ.groupby([df["race_id"], df["driver_id"]]).cumsum().astype(int)

    rained_yet = pd.to_numeric(df.get("rained_yet", 0), errors="coerce").fillna(0).astype(int)

    df["fulfilled_second_compound"] = ((rained_yet == 1) | (cum_distinct >= 2)).astype(int)
    return df


# -----------------------------
# Dataset build
# -----------------------------
def build_dataset(db_path: str) -> pd.DataFrame:
    """
    Builds the full per-(race, driver, lap) dataframe with all intermediate columns
    (we subset into Dataset 1 / Dataset 2 afterwards).
    """
    conn = sqlite3.connect(db_path)
    try:
        laps, races, fcy, weather = load_core_tables(conn)
    finally:
        conn.close()

    # Basic hygiene: drop rows missing core keys
    laps = laps.dropna(subset=["race_id", "driver_id", "lapno", "position"]).copy()
    laps["race_id"] = laps["race_id"].astype(int)
    laps["driver_id"] = laps["driver_id"].astype(int)
    laps["lapno"] = laps["lapno"].astype(int)
    laps["position"] = laps["position"].astype(int)

    # Labels
    df = add_labels(laps)

    # Extra outputs
    df = add_extra_output_features(df)

    # Help features
    df = add_race_progress_and_track_category(df, races)

    # Weather flags
    df = add_weather_flags(df, weather)

    # pit stops so far
    df = add_pit_stops_so_far(df, clip_to_0_3=False)

    # NEW: pit stops left
    df = add_pit_stops_left(df, clip_to_0_3=False)

    # pursuer tyre change
    df = add_tyre_change_pursuer(df)

    # FCY
    df = add_fcy_status_table6(df, fcy)

    # NEW: undercut / clear-air features
    df = add_undercut_features(df, lag_one_lap=False)

    # close_ahead
    df = add_close_ahead(df, threshold_s=1.5)

    # NEW: fulfilled_second_compound
    df = add_fulfilled_second_compound(df)

    # Enforce ranges (optional)
    if "race_progress" in df.columns:
        df["race_progress"] = df["race_progress"].astype(float).clip(0.0, 1.0)
    if "position" in df.columns:
        df["position"] = df["position"].astype(int).clip(0, 22)
    if "fcy_status" in df.columns:
        df["fcy_status"] = df["fcy_status"].astype(int).clip(0, 4)

    return df


def make_dataset_1(df_full: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "race_id", "driver_id", "lapno",

        "race_progress", "position", "fcy_status",
        "pit_stops_so_far", 
        "pit_stops_left",
        "tyre_change_pursuer", "track_category",

        "y_pit",

        "current_compound",
        "lap_time",
        "interval",
        "tyre_age",

        # NEW undercut / clear-air features
        "gap_behind_s",
        "n_cars_within_5s_ahead",
        "tyre_age_diff_to_ahead",
        "rejoin_gap_ahead_est_s",
        "rejoin_gap_behind_est_s",

        "close_ahead",
        "race_track",
        "rained_yet",
        "is_raining",
        "minutes_rain",

        "fulfilled_second_compound",
    ]
    return df_full[[c for c in keep_cols if c in df_full.columns]].copy()


def make_dataset_2(df_full: pd.DataFrame) -> pd.DataFrame:
    df_pit = df_full[df_full.get("y_pit", 0).astype(int) == 1].copy()

    keep_cols = [
        "race_id",
        "race_progress",
        "pit_stops_so_far",
        "pit_stops_left",
        "current_compound",
        "race_track",
        "fulfilled_second_compound",
        "rained_yet",
        "is_raining",
        "minutes_rain",

        # optional adds:
        "gap_behind_s",
        "n_cars_within_5s_ahead",
        "tyre_age_diff_to_ahead",
        "rejoin_gap_ahead_est_s",
        "rejoin_gap_behind_est_s",

        "y_compound",
    ]
    return df_pit[[c for c in keep_cols if c in df_pit.columns]].copy()



# -----------------------------
# Output helpers
# -----------------------------
def infer_format(path: str, override: Optional[str]) -> str:
    if override is not None:
        return override
    return "parquet" if path.lower().endswith(".parquet") else "csv"


def write_df(df: pd.DataFrame, out_path: str, fmt: str) -> None:
    if fmt == "parquet":
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)


# -----------------------------
# CLI
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-clean", required=True, help="Path to CLEAN sqlite DB (e.g., *_clean.sqlite)")
    ap.add_argument("--db-less-clean", required=True, help="Path to LESS-CLEAN sqlite DB (e.g., *_less_clean.sqlite)")

    ap.add_argument("--out1", required=True, help="Output path for Dataset 1 (.parquet or .csv)")
    ap.add_argument("--out2", required=True, help="Output path for Dataset 2 (.parquet or .csv)")

    ap.add_argument("--format1", choices=["parquet", "csv"], default=None, help="Optional override for Dataset 1 format")
    ap.add_argument("--format2", choices=["parquet", "csv"], default=None, help="Optional override for Dataset 2 format")

    args = ap.parse_args()

    # ---- Dataset 1 from clean DB ----
    df_clean_full = build_dataset(args.db_clean)
    ds1 = make_dataset_1(df_clean_full)

    fmt1 = infer_format(args.out1, args.format1)
    write_df(ds1, args.out1, fmt1)
    print(f"[OK] Dataset 1 (clean DB) wrote {len(ds1):,} rows -> {args.out1}")
    print(ds1.head(10).to_string(index=False))

    # ---- Dataset 2 from less-clean DB ----
    df_less_full = build_dataset(args.db_less_clean)
    ds2 = make_dataset_2(df_less_full)

    fmt2 = infer_format(args.out2, args.format2)
    write_df(ds2, args.out2, fmt2)
    print(f"[OK] Dataset 2 (less-clean DB, pit events only) wrote {len(ds2):,} rows -> {args.out2}")
    print(ds2.head(10).to_string(index=False))


if __name__ == "__main__":
    main()