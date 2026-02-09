#!/usr/bin/env python3
"""
Dataset builder for the **MINIMAL TWO-TIER** reduced schema (no weather tables).

Takes TWO SQLite DBs produced by db_clean_minimal_two_tiers.py and writes TWO datasets:

Dataset 1 (from *clean.sqlite*):
  - Per-(race, driver, lap) rows
  - Keeps y_pit
  - Drops y_compound
  - Adds fulfilled_second_compound
  - Weather outputs are NULL/NaN (rained_yet, is_raining, minutes_rain), because weather tables are absent.

Dataset 2 (from *less_clean.sqlite*):
  - Only pit-event rows (y_pit == 1)
  - Does NOT include y_pit column
  - Includes y_compound
  - Includes race_id for race-wise splitting.

fulfilled_second_compound at lap t is True iff:
  - rained_yet == 1 at lap t, OR
  - the driver has used >= 2 DISTINCT tyre compounds in that race up to and including lap t.
Since weather is unavailable here, rained_yet is set to NaN and treated as 0 for the fulfilled_second_compound computation.

Usage:
  python dataset_builder_ergast.py \
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
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd


# -----------------------------
# Utilities
# -----------------------------
def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    q = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
    return conn.execute(q, (name,)).fetchone() is not None


def pick_race_lap_count(df: pd.DataFrame) -> pd.Series:
    """Prefer nolapsplanned, fallback to nolaps."""
    base = pd.Series([np.nan] * len(df), index=df.index)
    if "nolapsplanned" in df.columns:
        base = df["nolapsplanned"].copy()
    if "nolaps" in df.columns:
        base = base.fillna(df["nolaps"])
    return base



def normalize_compound_5class(x: Optional[str]) -> Optional[str]:
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
    if s in {"WET", "FULLWET", "FW", "W"}:
        return "WET"
    if s in {"", "NAN", "NONE", "UNKNOWN"}:
        return None
    return None


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
    Returns: hardest slick as C-number in {1..5} (smaller is harder), or None if not found.
    """
    if s is None:
        return None

    txt = str(s).upper()

    # 1) C-notation directly
    c_matches = re.findall(r"\bC([1-5])\b", txt)
    if c_matches:
        return min(int(m) for m in c_matches)

    # 2) A-notation (ignore I/W)
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


def _parse_alloc_tokens(s: Optional[str]) -> List[str]:
    """
    Parse races.availablecompounds or races.availablecompounds_c like:
      'A2,A3,A4,I,W' or 'C1,C2,C3,I,W' or with spaces/brackets/quotes.
    Returns tokens uppercased, stripped, keeping order.
    """
    if s is None:
        return []
    txt = str(s).strip().upper()
    # Split on commas, strip brackets/quotes
    parts = [p.strip().strip("[](){}\"'") for p in txt.split(",")]
    return [p for p in parts if p]


def build_relative_compound_map(availablecompounds_any: Optional[str]) -> Dict[str, Optional[str]]:
    """
    Build a mapping from absolute codes -> relative 5-class labels for ONE race.

    Supports weekend allocations given as A-codes (A2/A3/A4/A6/A7) or C-codes (C1..C5),
    plus I/W for intermediates/wets.

    Rules:
      - I -> INTERMEDIATE, W -> WET
      - For slicks, collect A* or C* tokens present (exclude I/W).
        Sort by hardness ascending (hardest first).
        If 3 slicks: HARD, MEDIUM, SOFT assigned hardest..softest.
        If 2 slicks: HARD, MEDIUM.
        If 1 slick: HARD.
    """
    toks = _parse_alloc_tokens(availablecompounds_any)

    # collect slick tokens (A or C)
    a_slicks = []
    c_slicks = []
    for t in toks:
        if re.fullmatch(r"A(2|3|4|6|7)", t):
            a_slicks.append(t)
        elif re.fullmatch(r"C[1-5]", t):
            c_slicks.append(t)

    # unify as "hardness rank": smaller means harder (C1 hardest)
    slicks_ranked: List[Tuple[int, str]] = []

    for a in set(a_slicks):
        c = A_TO_C.get(a)
        if c is not None:
            slicks_ranked.append((c, a))  # rank by mapped C-number

    for c in set(c_slicks):
        slicks_ranked.append((int(c[1:]), c))  # rank by C-number directly

    slicks_ranked.sort(key=lambda x: x[0])  # hardest first

    m: Dict[str, Optional[str]] = {"I": "INTERMEDIATE", "INTER": "INTERMEDIATE", "IN": "INTERMEDIATE", "W": "WET"}

    slick_tokens_sorted = [tok for _, tok in slicks_ranked]

    if len(slick_tokens_sorted) >= 3:
        m[slick_tokens_sorted[0]] = "HARD"
        m[slick_tokens_sorted[1]] = "MEDIUM"
        m[slick_tokens_sorted[2]] = "SOFT"
    elif len(slick_tokens_sorted) == 2:
        m[slick_tokens_sorted[0]] = "HARD"
        m[slick_tokens_sorted[1]] = "MEDIUM"
    elif len(slick_tokens_sorted) == 1:
        m[slick_tokens_sorted[0]] = "HARD"

    return m


def normalize_compound_relative(code: Optional[str], rel_map: Dict[str, Optional[str]]) -> Optional[str]:
    """
    Map a lap compound code using the race-specific rel_map.
    Accepts already-normalised strings too (SOFT/MEDIUM/...).
    """
    if code is None:
        return None
    s = str(code).strip().upper()

    # already 5-class
    if s in {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}:
        return s

    # common shorthands -> map keys
    if s in {"INTER", "IN"}:
        s = "I"
    if s in {"FULLWET", "FW"}:
        s = "W"

    # allow "C3" / "A4" / "I" / "W"
    return rel_map.get(s, None)


# -----------------------------
# Load tables
# -----------------------------
def load_core_tables(conn: sqlite3.Connection) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = ["laps", "races", "fcyphases"]
    missing = [t for t in required if not table_exists(conn, t)]
    if missing:
        raise RuntimeError(f"Missing required tables: {missing}")

    # --- Laps ---
    laps_cols = [
        "race_id", "driver_id", "lapno", "position",
        "laptime", "racetime", "gap", "interval",
        "compound", "tireage",
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

    return laps, races, fcy


# -----------------------------
# Labels + derived columns
# -----------------------------
def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add:
      y_pit: 1 if pit this lap, else 0
      y_compound: relative-normalised nextcompound for pit laps; None otherwise
    """
    df = df.copy()

    if "pitintime" in df.columns:
        pit = df["pitintime"].notna() & (~df["pitintime"].astype(str).str.strip().isin(["", "NaT", "nan", "None"]))
    else:
        pit = pd.Series(False, index=df.index)
    df["y_pit"] = pit.astype(int)

    # Prefer already-built relative nextcompound if present
    if "nextcompound_rel" in df.columns:
        df["y_compound"] = df["nextcompound_rel"]
    elif "nextcompound" in df.columns:
        df["y_compound"] = df["nextcompound"].apply(normalize_compound_5class)
    else:
        df["y_compound"] = None

    df.loc[df["y_pit"] != 1, "y_compound"] = None
    return df


def add_extra_output_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - current_compound  <- compound_rel (preferred) else None
      - lap_time          <- laps.laptime
      - interval          <- laps.interval
      - tyre_age          <- laps.tireage
      - weather outputs set to NaN (no weather tables)
    """
    df = df.copy()

    if "compound_rel" in df.columns:
        df["current_compound"] = df["compound_rel"]
    elif "compound" in df.columns:
        df["current_compound"] = df["compound"].apply(normalize_compound_5class)
    else:
        df["current_compound"] = None

    df["lap_time"] = pd.to_numeric(df["laptime"], errors="coerce") if "laptime" in df.columns else np.nan
    df["interval"] = pd.to_numeric(df["interval"], errors="coerce") if "interval" in df.columns else np.nan
    df["tyre_age"] = pd.to_numeric(df["tireage"], errors="coerce") if "tireage" in df.columns else np.nan

    # Weather unavailable in minimal schema
    df["rained_yet"] = np.nan
    df["is_raining"] = np.nan
    df["minutes_rain"] = np.nan

    return df


def add_race_progress_track_category_and_relative_compounds(laps: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """
    Merges races, computes:
      - race_progress
      - race_track
      - track_category (from hardest slick in availablecompounds_c else availablecompounds)
      - compound_rel / nextcompound_rel (race-wise mapping from weekend allocation)
    """
    df = laps.merge(races, on="race_id", how="left")

    # race_progress
    total_laps = pick_race_lap_count(df).replace({0: np.nan})
    df["race_progress"] = (pd.to_numeric(df["lapno"], errors="coerce") / total_laps).astype(float)
    df["race_progress"] = df["race_progress"].clip(lower=0.0, upper=1.0)

    # race_track
    df["race_track"] = df["location"].astype(str) if "location" in df.columns else None

    # track_category (prefer availablecompounds_c)
    alloc_for_track = None
    if "availablecompounds_c" in df.columns:
        alloc_for_track = df["availablecompounds_c"].where(df["availablecompounds_c"].notna(), df.get("availablecompounds"))
    else:
        alloc_for_track = df.get("availablecompounds")

    if alloc_for_track is not None:
        hardest = alloc_for_track.apply(extract_hardest_slick_from_availablecompounds)
        df["track_category"] = hardest.apply(track_category_from_hardest)
    else:
        df["track_category"] = None

    # relative compound mapping per race (prefer availablecompounds_c else availablecompounds)
    df["compound_rel"] = None
    df["nextcompound_rel"] = None

    alloc_col = "availablecompounds_c" if "availablecompounds_c" in df.columns else "availablecompounds"
    if alloc_col in df.columns:
        for race_id, idx in df.groupby("race_id").groups.items():
            alloc = df.loc[idx, alloc_col].iloc[0]
            rel_map = build_relative_compound_map(alloc)

            if "compound" in df.columns:
                df.loc[idx, "compound_rel"] = df.loc[idx, "compound"].apply(
                    lambda x: normalize_compound_relative(x, rel_map)
                )
            if "nextcompound" in df.columns:
                df.loc[idx, "nextcompound_rel"] = df.loc[idx, "nextcompound"].apply(
                    lambda x: normalize_compound_relative(x, rel_map)
                )

    return df


def add_pit_stops_so_far(df: pd.DataFrame, clip_to_0_3: bool = False) -> pd.DataFrame:
    """
    pit_stops_so_far at lap t = count of pit events up to lap t-1
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
    Uses compound_rel if present, else falls back to compound normalised to 5-class.
    """
    df = df.sort_values(["race_id", "driver_id", "lapno"]).copy()

    comp_col = "compound_rel" if "compound_rel" in df.columns else None
    if comp_col is None or df[comp_col].isna().all():
        if "compound" in df.columns:
            df["_comp_tmp"] = df["compound"].apply(normalize_compound_5class)
            comp_col = "_comp_tmp"
        else:
            df["tyre_change_pursuer"] = False
            return df

    prev_comp = df.groupby(["race_id", "driver_id"])[comp_col].shift(1)
    df["did_change_since_prev"] = df[comp_col].notna() & prev_comp.notna() & (df[comp_col] != prev_comp)

    lookup = df[["race_id", "lapno", "position", "did_change_since_prev"]].rename(
        columns={"did_change_since_prev": "pursuer_changed_prev"}
    )
    df["pursuer_position"] = pd.to_numeric(df["position"], errors="coerce") + 1

    df = df.merge(
        lookup,
        left_on=["race_id", "lapno", "pursuer_position"],
        right_on=["race_id", "lapno", "position"],
        how="left",
        suffixes=("", "_pursuer_row"),
    )

    df["tyre_change_pursuer"] = df["pursuer_changed_prev"].fillna(False).astype(bool)

    df = df.drop(
        columns=[c for c in ["_comp_tmp", "did_change_since_prev", "pursuer_position", "pursuer_changed_prev", "position_pursuer_row"] if c in df.columns],
        errors="ignore",
    )
    return df


def add_close_ahead(df: pd.DataFrame, threshold_s: float = 1.5) -> pd.DataFrame:
    """
    close_ahead at lap t:
      True if, at end of lap t-1, the driver was <= threshold_s ahead of their pursuer (position+1).
    Uses racetime difference at lap t-1:
        delta = racetime(pursuer) - racetime(driver)
    """
    df = df.copy()

    needed = {"race_id", "driver_id", "lapno", "position", "racetime"}
    if not needed.issubset(df.columns):
        df["close_ahead"] = 0
        return df

    base = df[["race_id", "driver_id", "lapno", "position", "racetime"]].copy()
    base["racetime"] = pd.to_numeric(base["racetime"], errors="coerce")
    base["position"] = pd.to_numeric(base["position"], errors="coerce")

    pursuer = base[["race_id", "lapno", "position", "racetime"]].rename(
        columns={"position": "pursuer_position", "racetime": "pursuer_racetime"}
    )
    base["pursuer_position"] = base["position"] + 1

    merged = base.merge(pursuer, on=["race_id", "lapno", "pursuer_position"], how="left")
    merged["ahead_by_pursuer_s_prev"] = merged["pursuer_racetime"] - merged["racetime"]

    merged["lapno_next"] = merged["lapno"] + 1
    attach = merged[["race_id", "driver_id", "lapno_next", "ahead_by_pursuer_s_prev"]].rename(columns={"lapno_next": "lapno"})

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
    if not {"race_id", "startracetime", "endracetime"}.issubset(fcy.columns):
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
        if fcy_r.empty:
            df_r["phase_type_end"] = None
            out_frames.append(df_r)
            continue

        has_time = df_r["racetime"].notna()
        df_time = df_r.loc[has_time].sort_values("racetime").copy()
        df_notime = df_r.loc[~has_time].copy()

        if df_time.empty:
            df_r["phase_type_end"] = None
            out_frames.append(df_r)
            continue

        merged_time = pd.merge_asof(
            df_time,
            fcy_r[["startracetime", "endracetime", "type_norm"]],
            left_on="racetime",
            right_on="startracetime",
            direction="backward",
            allow_exact_matches=True,
        )

        active = merged_time["endracetime"].notna() & (merged_time["racetime"] <= merged_time["endracetime"])
        merged_time["phase_type_end"] = np.where(active, merged_time["type_norm"], None)

        df_notime["phase_type_end"] = None

        merged_r = pd.concat(
            [merged_time.drop(columns=["startracetime", "endracetime", "type_norm"], errors="ignore"), df_notime],
            ignore_index=True,
        )
        out_frames.append(merged_r)

    df2 = pd.concat(out_frames, ignore_index=True)
    df2 = df2.sort_values(["race_id", "driver_id", "lapno"]).copy()

    prev_phase = df2.groupby(["race_id", "driver_id"])["phase_type_end"].shift(1)

    is_vsc = df2["phase_type_end"].eq("VSC")
    is_sc = df2["phase_type_end"].eq("SC")

    df2.loc[is_vsc & prev_phase.ne("VSC"), "fcy_status"] = 1
    df2.loc[is_vsc & prev_phase.eq("VSC"), "fcy_status"] = 2
    df2.loc[is_sc & prev_phase.ne("SC"), "fcy_status"] = 3
    df2.loc[is_sc & prev_phase.eq("SC"), "fcy_status"] = 4

    df2 = df2.drop(columns=["phase_type_end"], errors="ignore")
    return df2


def add_fulfilled_second_compound(df: pd.DataFrame) -> pd.DataFrame:
    """
    fulfilled_second_compound at lap t:
      True iff rained_yet(t)==1 OR cumulative distinct compounds used by the driver in the race
      up to and including lap t is >= 2.

    In this MINIMAL schema, rained_yet is NaN -> treated as 0 for the boolean.
    Distinct compounds are computed from current_compound (preferred) / compound_rel,
    ignoring missing values.
    """
    df = df.sort_values(["race_id", "driver_id", "lapno"]).copy()

    if "current_compound" in df.columns:
        comp = df["current_compound"].apply(normalize_compound_5class)
    elif "compound_rel" in df.columns:
        comp = df["compound_rel"].apply(normalize_compound_5class)
    elif "compound" in df.columns:
        comp = df["compound"].apply(normalize_compound_5class)
    else:
        comp = pd.Series([None] * len(df), index=df.index)

    missing = comp.isna()
    tmp = comp.fillna("__MISSING__")

    # First occurrence of each compound within a (race,driver)
    first_occ = tmp.groupby([df["race_id"], df["driver_id"]]).transform(lambda s: ~s.duplicated())
    first_occ = first_occ & (~missing)

    # Cumulative distinct count
    cum_distinct = first_occ.groupby([df["race_id"], df["driver_id"]]).cumsum().astype(int)

    # Weather unavailable -> treat as 0
    rained_yet = pd.to_numeric(df.get("rained_yet", 0), errors="coerce").fillna(0).astype(int)

    df["fulfilled_second_compound"] = ((rained_yet == 1) | (cum_distinct >= 2)).astype(int)
    return df


# -----------------------------
# Build full DF from a DB
# -----------------------------
def build_full_df(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        laps, races, fcy = load_core_tables(conn)
    finally:
        conn.close()

    # Drop rows missing core keys
    laps = laps.dropna(subset=["race_id", "driver_id", "lapno", "position"]).copy()
    laps["race_id"] = pd.to_numeric(laps["race_id"], errors="coerce").astype("Int64")
    laps["driver_id"] = pd.to_numeric(laps["driver_id"], errors="coerce").astype("Int64")
    laps["lapno"] = pd.to_numeric(laps["lapno"], errors="coerce").astype("Int64")
    laps["position"] = pd.to_numeric(laps["position"], errors="coerce").astype("Int64")
    laps = laps.dropna(subset=["race_id", "driver_id", "lapno", "position"]).copy()

    # cast to int now that NaNs removed
    laps["race_id"] = laps["race_id"].astype(int)
    laps["driver_id"] = laps["driver_id"].astype(int)
    laps["lapno"] = laps["lapno"].astype(int)
    laps["position"] = laps["position"].astype(int)

    # merge races & compute race_progress/track_category/compound_rel mappings
    df = add_race_progress_track_category_and_relative_compounds(laps, races)

    # labels (uses nextcompound_rel if present)
    df = add_labels(df)

    # outputs (sets weather to NaN)
    df = add_extra_output_features(df)

    # remaining derived features
    df = add_pit_stops_so_far(df, clip_to_0_3=False)
    df = add_tyre_change_pursuer(df)
    df = add_fcy_status_table6(df, fcy)
    df = add_close_ahead(df, threshold_s=1.5)
    df = add_fulfilled_second_compound(df)

    # basic ranges / dtypes
    if "race_progress" in df.columns:
        df["race_progress"] = pd.to_numeric(df["race_progress"], errors="coerce").clip(0.0, 1.0)
    if "position" in df.columns:
        df["position"] = pd.to_numeric(df["position"], errors="coerce").fillna(-1).astype(int).clip(0, 22)
    if "fcy_status" in df.columns:
        df["fcy_status"] = pd.to_numeric(df["fcy_status"], errors="coerce").fillna(0).astype(int).clip(0, 4)

    return df


# -----------------------------
# Dataset 1 / Dataset 2 projections
# -----------------------------
def make_dataset_1(df_full: pd.DataFrame) -> pd.DataFrame:
    """
    Dataset 1 (clean DB):
      - Same “core” structure as the two-dataset builder:
      - WITHOUT y_compound
      - WITH y_pit
      - WITH fulfilled_second_compound
      - Weather fields are present but NaN
    """
    keep_cols = [
        "race_id", "driver_id", "lapno",

        # baseline features
        "race_progress", "position", "fcy_status",
        "pit_stops_so_far", "tyre_change_pursuer", "track_category",

        # label
        "y_pit",

        # extra outputs
        "current_compound",
        "lap_time",
        "interval",
        "tyre_age",
        "close_ahead",
        "race_track",
        "rained_yet",
        "is_raining",
        "minutes_rain",

        # new
        "fulfilled_second_compound",
    ]
    out = df_full[[c for c in keep_cols if c in df_full.columns]].copy()
    return out


def make_dataset_2(df_full: pd.DataFrame) -> pd.DataFrame:
    """
    Dataset 2 (less-clean DB):
      - Only pit-stop rows (y_pit == 1)
      - Does NOT include y_pit
      - Includes y_compound
      - Includes race_id for group splits
      - Same feature set as your two-dataset builder (weather NaN)
    """
    df_pit = df_full[df_full.get("y_pit", 0).astype(int) == 1].copy()

    keep_cols = [
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
    ]
    out = df_pit[[c for c in keep_cols if c in df_pit.columns]].copy()
    return out


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
    df_clean_full = build_full_df(args.db_clean)
    ds1 = make_dataset_1(df_clean_full)

    fmt1 = infer_format(args.out1, args.format1)
    write_df(ds1, args.out1, fmt1)
    print(f"[OK] Dataset 1 (clean DB) wrote {len(ds1):,} rows -> {args.out1}")
    print(ds1.head(10).to_string(index=False))

    # ---- Dataset 2 from less-clean DB ----
    df_less_full = build_full_df(args.db_less_clean)
    ds2 = make_dataset_2(df_less_full)

    fmt2 = infer_format(args.out2, args.format2)
    write_df(ds2, args.out2, fmt2)
    print(f"[OK] Dataset 2 (less-clean DB, pit events only) wrote {len(ds2):,} rows -> {args.out2}")
    print(ds2.head(10).to_string(index=False))


if __name__ == "__main__":
    main()