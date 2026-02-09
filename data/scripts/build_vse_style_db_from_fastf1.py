import argparse
import hashlib
import json
import logging
import os
import platform
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import fastf1
import pandas as pd
from thefuzz import process
import numpy as np

# --- Constants and Configuration ---

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

VSE_SCHEMA = {
    "provenance": """
        CREATE TABLE IF NOT EXISTS provenance (
            provenance_id INTEGER PRIMARY KEY,
            race_id INTEGER,
            source_year INTEGER,
            source_location TEXT,
            source_session TEXT,
            fastf1_version TEXT,
            python_version TEXT,
            ingestion_code_commit_hash TEXT,
            ingestion_timestamp_utc TEXT,
            source_fastf1_cache_enabled BOOLEAN,
            notes TEXT,
            FOREIGN KEY (race_id) REFERENCES races(id)
        )
    """,
    "job_report": """
        CREATE TABLE IF NOT EXISTS job_report (
            report_id INTEGER PRIMARY KEY,
            run_timestamp_utc TEXT,
            total_races_processed INTEGER,
            races_succeeded INTEGER,
            races_failed INTEGER,
            report_summary_json TEXT
        )
    """,
    "data_hashes": """
        CREATE TABLE IF NOT EXISTS data_hashes (
            hash_id INTEGER PRIMARY KEY,
            race_id INTEGER,
            table_name TEXT,
            data_hash TEXT,
            UNIQUE(race_id, table_name),
            FOREIGN KEY (race_id) REFERENCES races(id)
        )
    """,
    "drivers": """
        CREATE TABLE IF NOT EXISTS drivers (
            id INTEGER PRIMARY KEY,
            carno INTEGER,
            initials TEXT UNIQUE,
            name TEXT
        )
    """,
    "races": """
        CREATE TABLE IF NOT EXISTS races (
            id INTEGER PRIMARY KEY,
            date TEXT,
            season INTEGER,
            location TEXT,
            availablecompounds TEXT,
            comment TEXT,
            nolaps INTEGER,
            nolapsplanned INTEGER,
            tracklength REAL,
            ingestion_status TEXT DEFAULT 'pending',
            UNIQUE(season, location)
        )
    """,
    "starterfields": """
        CREATE TABLE IF NOT EXISTS starterfields (
            race_id INTEGER,
            driver_id INTEGER,
            team TEXT,
            teamcolor TEXT,
            enginemanufacturer TEXT,
            gridposition INTEGER,
            status TEXT,
            resultposition INTEGER,
            completedlaps INTEGER,
            speedtrap REAL,
            PRIMARY KEY (race_id, driver_id),
            FOREIGN KEY (race_id) REFERENCES races(id),
            FOREIGN KEY (driver_id) REFERENCES drivers(id)
        )
    """,
    "laps": """
        CREATE TABLE IF NOT EXISTS laps (
            race_id INTEGER,
            lapno INTEGER,
            position INTEGER,
            driver_id INTEGER,
            laptime REAL,
            racetime REAL,
            gap REAL,
            interval REAL,
            compound TEXT,
            tireage INTEGER,
            pitintime TEXT,
            pitstopduration REAL,
            nextcompound TEXT,
            startlapprog_vsc REAL,
            endlapprog_vsc REAL,
            age_vsc REAL,
            startlapprog_sc REAL,
            endlapprog_sc REAL,
            age_sc REAL,
            PRIMARY KEY (race_id, lapno, driver_id),
            FOREIGN KEY (race_id) REFERENCES races(id),
            FOREIGN KEY (driver_id) REFERENCES drivers(id)
        )
    """,
    "fcyphases": """
        CREATE TABLE IF NOT EXISTS fcyphases (
            id INTEGER PRIMARY KEY,
            race_id INTEGER,
            startracetime REAL,
            endracetime REAL,
            startraceprog REAL,
            endraceprog REAL,
            startlap INTEGER,
            endlap INTEGER,
            type TEXT,
            FOREIGN KEY (race_id) REFERENCES races(id)
        )
    """,
    "qualifyings": """
        CREATE TABLE IF NOT EXISTS qualifyings (
            race_id INTEGER,
            position INTEGER,
            driver_id INTEGER,
            q1laptime REAL,
            q2laptime REAL,
            q3laptime REAL,
            speedtrap REAL,
            PRIMARY KEY (race_id, position),
            FOREIGN KEY (race_id) REFERENCES races(id),
            FOREIGN KEY (driver_id) REFERENCES drivers(id)
        )
    """,
    "retirements": """
        CREATE TABLE IF NOT EXISTS retirements (
            season INTEGER,
            driver_id INTEGER,
            accidents INTEGER,
            failures INTEGER,
            PRIMARY KEY (season, driver_id),
            FOREIGN KEY (driver_id) REFERENCES drivers(id)
        )
    """,
}

# Absolute Compound Allocation Table
COMPOUND_ALLOCATIONS = {
    2018: {
        "Australian Grand Prix": "A4,A5,A6", "Bahrain Grand Prix": "A3,A4,A5", "Chinese Grand Prix": "A3,A4,A6",
        "Azerbaijan Grand Prix": "A4,A5,A6", "Spanish Grand Prix": "A3,A4,A5", "Monaco Grand Prix": "A5,A6,A7",
        "Canadian Grand Prix": "A5,A6,A7", "French Grand Prix": "A4,A5,A6", "Austrian Grand Prix": "A4,A5,A6",
        "British Grand Prix": "A2,A3,A4", "German Grand Prix": "A3,A4,A6", "Hungarian Grand Prix": "A3,A4,A6",
        "Belgian Grand Prix": "A3,A4,A5", "Italian Grand Prix": "A3,A4,A5", "Singapore Grand Prix": "A4,A6,A7",
        "Russian Grand Prix": "A4,A6,A7", "Japanese Grand Prix": "A3,A4,A5", "United States Grand Prix": "A4,A5,A6",
        "Mexican Grand Prix": "A5,A6,A7", "Brazilian Grand Prix": "A3,A4,A5", "Abu Dhabi Grand Prix": "A5,A6,A7"
    },
    2019: {
        "Australian Grand Prix": "C2,C3,C4", "Bahrain Grand Prix": "C1,C2,C3", "Chinese Grand Prix": "C2,C3,C4",
        "Azerbaijan Grand Prix": "C2,C3,C4", "Spanish Grand Prix": "C1,C2,C3", "Monaco Grand Prix": "C3,C4,C5",
        "Canadian Grand Prix": "C3,C4,C5", "French Grand Prix": "C2,C3,C4", "Austrian Grand Prix": "C2,C3,C4",
        "British Grand Prix": "C1,C2,C3", "German Grand Prix": "C2,C3,C4", "Hungarian Grand Prix": "C2,C3,C4",
        "Belgian Grand Prix": "C1,C2,C3", "Italian Grand Prix": "C2,C3,C4", "Singapore Grand Prix": "C3,C4,C5",
        "Russian Grand Prix": "C2,C3,C4", "Japanese Grand Prix": "C1,C2,C3", "Mexican Grand Prix": "C2,C3,C4",
        "United States Grand Prix": "C2,C3,C4", "Brazilian Grand Prix": "C1,C2,C3", "Abu Dhabi Grand Prix": "C3,C4,C5"
    },
    2020: {
        "Austrian Grand Prix": "C2,C3,C4", "Styrian Grand Prix": "C2,C3,C4", "Hungarian Grand Prix": "C2,C3,C4",
        "British Grand Prix": "C1,C2,C3", "70th Anniversary Grand Prix": "C2,C3,C4", "Spanish Grand Prix": "C1,C2,C3",
        "Belgian Grand Prix": "C2,C3,C4", "Italian Grand Prix": "C2,C3,C4", "Tuscan Grand Prix": "C1,C2,C3",
        "Russian Grand Prix": "C3,C4,C5", "Eifel Grand Prix": "C2,C3,C4", "Portuguese Grand Prix": "C1,C2,C3",
        "Emilia Romagna Grand Prix": "C2,C3,C4", "Turkish Grand Prix": "C1,C2,C3", "Bahrain Grand Prix": "C2,C3,C4",
        "Sakhir Grand Prix": "C2,C3,C4", "Abu Dhabi Grand Prix": "C3,C4,C5"
    },
    2021: {
        "Bahrain Grand Prix": "C2,C3,C4", "Emilia Romagna Grand Prix": "C2,C3,C4", "Portuguese Grand Prix": "C1,C2,C3",
        "Spanish Grand Prix": "C1,C2,C3", "Monaco Grand Prix": "C3,C4,C5", "Azerbaijan Grand Prix": "C3,C4,C5",
        "French Grand Prix": "C2,C3,C4", "Styrian Grand Prix": "C2,C3,C4", "Austrian Grand Prix": "C3,C4,C5",
        "British Grand Prix": "C1,C2,C3", "Hungarian Grand Prix": "C2,C3,C4", "Belgian Grand Prix": "C2,C3,C4",
        "Dutch Grand Prix": "C1,C2,C3", "Italian Grand Prix": "C2,C3,C4", "Russian Grand Prix": "C3,C4,C5",
        "Turkish Grand Prix": "C2,C3,C4", "United States Grand Prix": "C2,C3,C4", "Mexico City Grand Prix": "C2,C3,C4",
        "São Paulo Grand Prix": "C2,C3,C4", "Qatar Grand Prix": "C1,C2,C3", "Saudi Arabian Grand Prix": "C2,C3,C4",
        "Abu Dhabi Grand Prix": "C3,C4,C5"
    },
    2022: {
        "Bahrain Grand Prix": "C1,C2,C3", "Saudi Arabian Grand Prix": "C2,C3,C4", "Australian Grand Prix": "C2,C3,C5",
        "Emilia Romagna Grand Prix": "C2,C3,C4", "Miami Grand Prix": "C2,C3,C4", "Spanish Grand Prix": "C1,C2,C3",
        "Monaco Grand Prix": "C3,C4,C5", "Azerbaijan Grand Prix": "C3,C4,C5", "Canadian Grand Prix": "C3,C4,C5",
        "British Grand Prix": "C1,C2,C3", "Austrian Grand Prix": "C3,C4,C5", "French Grand Prix": "C2,C3,C4",
        "Hungarian Grand Prix": "C2,C3,C4", "Belgian Grand Prix": "C2,C3,C4", "Dutch Grand Prix": "C1,C2,C3",
        "Italian Grand Prix": "C2,C3,C4", "Singapore Grand Prix": "C3,C4,C5", "Japanese Grand Prix": "C1,C2,C3",
        "United States Grand Prix": "C2,C3,C4", "Mexico City Grand Prix": "C2,C3,C4", "São Paulo Grand Prix": "C2,C3,C4",
        "Abu Dhabi Grand Prix": "C3,C4,C5"
    },
    2023: {
        "Bahrain Grand Prix": "C1,C2,C3", "Saudi Arabian Grand Prix": "C2,C3,C4", "Australian Grand Prix": "C2,C3,C4",
        "Azerbaijan Grand Prix": "C3,C4,C5", "Miami Grand Prix": "C2,C3,C4", "Monaco Grand Prix": "C3,C4,C5",
        "Spanish Grand Prix": "C1,C2,C3", "Canadian Grand Prix": "C3,C4,C5", "Austrian Grand Prix": "C3,C4,C5",
        "British Grand Prix": "C1,C2,C3", "Hungarian Grand Prix": "C3,C4,C5", "Belgian Grand Prix": "C2,C3,C4",
        "Dutch Grand Prix": "C1,C2,C3", "Italian Grand Prix": "C3,C4,C5", "Singapore Grand Prix": "C3,C4,C5",
        "Japanese Grand Prix": "C1,C2,C3", "Qatar Grand Prix": "C1,C2,C3", "United States Grand Prix": "C2,C3,C4",
        "Mexico City Grand Prix": "C3,C4,C5", "São Paulo Grand Prix": "C2,C3,C4", "Las Vegas Grand Prix": "C3,C4,C5",
        "Abu Dhabi Grand Prix": "C3,C4,C5"
    },
    2024: {
        "Bahrain Grand Prix": "C1,C2,C3", "Saudi Arabian Grand Prix": "C2,C3,C4", "Australian Grand Prix": "C3,C4,C5",
        "Japanese Grand Prix": "C1,C2,C3", "Chinese Grand Prix": "C2,C3,C4", "Miami Grand Prix": "C2,C3,C4",
        "Emilia Romagna Grand Prix": "C3,C4,C5", "Monaco Grand Prix": "C3,C4,C5", "Canadian Grand Prix": "C3,C4,C5",
        "Spanish Grand Prix": "C1,C2,C3", "Austrian Grand Prix": "C3,C4,C5", "British Grand Prix": "C1,C2,C3",
        "Hungarian Grand Prix": "C3,C4,C5", "Belgian Grand Prix": "C1,C3,C4", "Dutch Grand Prix": "C1,C2,C3",
        "Italian Grand Prix": "C3,C4,C5", "Azerbaijan Grand Prix": "C3,C4,C5", "Singapore Grand Prix": "C3,C4,C5",
        "United States Grand Prix": "C2,C3,C4", "Mexico City Grand Prix": "C2,C4,C5", "São Paulo Grand Prix": "C3,C4,C5",
        "Las Vegas Grand Prix": "C3,C4,C5", "Qatar Grand Prix": "C1,C2,C3", "Abu Dhabi Grand Prix": "C3,C4,C5"
    },
    2025: {
        "Australian Grand Prix": "C3,C4,C5", "Chinese Grand Prix": "C2,C3,C4", "Japanese Grand Prix": "C1,C2,C3",
        "Bahrain Grand Prix": "C1,C2,C3", "Saudi Arabian Grand Prix": "C3,C4,C5", "Miami Grand Prix": "C3,C4,C5",
        "Emilia Romagna Grand Prix": "C4,C5,C6", "Monaco Grand Prix": "C4,C5,C6", "Spanish Grand Prix": "C1,C2,C3",
        "Canadian Grand Prix": "C4,C5,C6", "Austrian Grand Prix": "C3,C4,C5", "British Grand Prix": "C2,C3,C4",
        "Belgian Grand Prix": "C1,C3,C4", "Hungarian Grand Prix": "C3,C4,C5", "Dutch Grand Prix": "C2,C3,C4",
        "Italian Grand Prix": "C3,C4,C5", "Azerbaijan Grand Prix": "C4,C5,C6", "Singapore Grand Prix": "C3,C4,C5",
        "United States Grand Prix": "C1,C3,C4", "Mexico City Grand Prix": "C2,C4,C5", "São Paulo Grand Prix": "C2,C3,C4",
        "Las Vegas Grand Prix": "C3,C4,C5", "Qatar Grand Prix": "C1,C2,C3", "Abu Dhabi Grand Prix": "C3,C4,C5"
    }
}

def compute_gap_interval_by_lap_completion(laps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes gap/interval for each lapnumber based on lap completion order.

    Definitions (per lapnumber):
      - gap_calc:    end_time(driver) - end_time(first finisher on that lapnumber)
      - interval_calc: 0 for first finisher; otherwise end_time(driver) - end_time(previous finisher)

    Uses absolute lap end time:
      - Prefer 'time' if present,
      - else uses lapstarttime + laptime if both present.
    """
    df = laps_df.copy()

    # Ensure numeric
    df["lapnumber"] = pd.to_numeric(df.get("lapnumber"), errors="coerce")
    df["laptime"] = pd.to_numeric(df.get("laptime"), errors="coerce")
    df["time"] = pd.to_numeric(df.get("time"), errors="coerce") if "time" in df.columns else np.nan
    df["lapstarttime"] = pd.to_numeric(df.get("lapstarttime"), errors="coerce") if "lapstarttime" in df.columns else np.nan

    # Build an absolute lap-end time in seconds
    end_time = df["time"]
    if end_time.isna().all():
        # fallback: end = lapstarttime + laptime
        if "lapstarttime" in df.columns and "laptime" in df.columns:
            end_time = df["lapstarttime"] + df["laptime"]

    df["_end_time_s"] = end_time

    # Only compute where we have essentials
    tmp = df[["driver", "lapnumber", "_end_time_s"]].dropna(subset=["driver", "lapnumber", "_end_time_s"]).copy()
    tmp["lapnumber"] = tmp["lapnumber"].astype(int)

    # Order cars by who finished that lap sooner
    tmp = tmp.sort_values(["lapnumber", "_end_time_s", "driver"])

    # gap to "lap leader" (first finisher of that lapnumber)
    first_time = tmp.groupby("lapnumber")["_end_time_s"].transform("min")
    tmp["gap_calc"] = tmp["_end_time_s"] - first_time

    # interval to previous finisher on that lapnumber
    prev_time = tmp.groupby("lapnumber")["_end_time_s"].shift(1)
    tmp["interval_calc"] = tmp["_end_time_s"] - prev_time

    # First finisher gets interval 0
    is_first = tmp.groupby("lapnumber").cumcount() == 0
    tmp.loc[is_first, "interval_calc"] = 0.0

    # Merge back
    df = df.merge(
        tmp[["driver", "lapnumber", "gap_calc", "interval_calc"]],
        on=["driver", "lapnumber"],
        how="left"
    )

    df.drop(columns=["_end_time_s"], inplace=True, errors="ignore")
    return df


# --- Missing Data Tracking ---

class MissingDataTracker:
    def __init__(self):
        self.issues = []

    def add(self, season, location, session, filename, reason, path=None, severity="WARNING"):
        self.issues.append({
            "season": season,
            "grand_prix": location,
            "session": session,
            "filename": filename,
            "reason": reason,
            "path": str(path) if path is not None else "",
            "severity": severity
        })

    def log_summary(self):
        if not self.issues:
            logging.info("Missing-data report: no missing/failed inputs detected.")
            return

        issues_sorted = sorted(
            self.issues,
            key=lambda x: (
                int(x["season"]) if str(x["season"]).isdigit() else 9999,
                x["grand_prix"],
                x["session"],
                x["filename"],
                x["reason"]
            )
        )

        logging.warning("Missing-data report: inputs were missing OR data could not be inserted. Details below:")
        for it in issues_sorted:
            msg = (
                f"[{it['severity']}] {it['season']} | {it['grand_prix']} | {it['session']} | "
                f"{it['filename']} -> {it['reason']}"
            )
            if it["path"]:
                msg += f" | path={it['path']}"
            logging.warning(msg)

        logging.warning(f"Missing-data report: total issues = {len(issues_sorted)}")

    def write_csv(self, out_path):
        if not self.issues:
            df = pd.DataFrame(columns=["season", "grand_prix", "session", "filename", "reason", "path", "severity"])
        else:
            df = pd.DataFrame(self.issues).sort_values(
                by=["season", "grand_prix", "session", "filename", "reason"],
                kind="stable"
            )
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        logging.info(f"Missing-data CSV written to: {out_path}")


# --- Database Functions ---

def create_database(db_path):
    if os.path.exists(db_path):
        os.remove(db_path)
        logging.info(f"Removed existing database at {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for table_name, create_sql in VSE_SCHEMA.items():
        cursor.execute(create_sql)
        logging.info(f"Table '{table_name}' created successfully.")

    conn.commit()
    logging.info("Database schema created successfully.")
    return conn


# --- Helpers ---

import re
import pandas as pd

ABS_2018_TO_A = {
    "SUPERHARD": "A1",
    "HARD": "A2",
    "MEDIUM": "A3",
    "SOFT": "A4",
    "SUPERSOFT": "A5",
    "ULTRASOFT": "A6",
    "HYPERSOFT": "A7",
}

def _clean_compound_str(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip().upper()
    if s in {"", "NAN", "NONE", "NULL"}:
        return None
    return s

def parse_allocated_slicks(availablecompounds: str):
    """
    From races.availablecompounds like "A4,A5,A6,I,W" or "C2,C3,C4,I,W"
    return the 3 slick codes (A* or C*).
    """
    if not availablecompounds:
        return []
    txt = str(availablecompounds).upper()

    # FIX: include A1 and A5 (A1..A7 minus any missing years)
    a = re.findall(r"\bA(1|2|3|4|5|6|7)\b", txt)
    c = re.findall(r"\bC([1-6])\b", txt)  # allow C6 just in case (your 2025 table includes C6)

    if a:
        return [f"A{x}" for x in a]
    if c:
        return [f"C{x}" for x in c]
    return []

def build_relative_map_from_alloc(slicks):
    """
    slicks is 3 codes in either A-range or C-range.
    Returns dict like {"A4":"HARD","A5":"MEDIUM","A6":"SOFT"} (relative labels).
    """
    if len(slicks) != 3:
        return {}

    if slicks[0].startswith("A"):
        # A2 hardest ... A7 softest
        hardness_key = lambda s: int(s[1:])  # smaller = harder
        ordered = sorted(slicks, key=hardness_key)  # hardest -> softest
    elif slicks[0].startswith("C"):
        # C1 hardest ... C5 softest
        hardness_key = lambda s: int(s[1:])  # smaller = harder
        ordered = sorted(slicks, key=hardness_key)  # hardest -> softest
    else:
        return {}

    # hardest -> HARD, middle -> MEDIUM, softest -> SOFT
    return {ordered[0]: "HARD", ordered[1]: "MEDIUM", ordered[2]: "SOFT"}

def normalise_to_relative(compound_value, rel_map):
    """
    Map laps.parquet 'compound' to {SOFT, MEDIUM, HARD, INTERMEDIATE, WET} for this race.

    Works for:
      - 2018 absolute names: SUPERHARD/HARD/MEDIUM/SOFT/SUPERSOFT/ULTRASOFT/HYPERSOFT
      - 2019+: relative names already: SOFT/MEDIUM/HARD
      - A*/C* codes if they ever appear
    """
    s = _clean_compound_str(compound_value)
    if s is None:
        return None

    # Wet tyres (keep)
    if s in {"INTERMEDIATE", "INTER", "IN", "I"}:
        return "INTERMEDIATE"
    if s in {"WET", "FULLWET", "FW"}:
        return "WET"

    # Detect whether this race uses 2018-style A allocations or modern C allocations
    rel_keys = set(rel_map.keys()) if isinstance(rel_map, dict) else set()
    is_a_based = any(k.startswith("A") for k in rel_keys)
    is_c_based = any(k.startswith("C") for k in rel_keys)

    # If it's already relative (modern era), keep it
    # IMPORTANT: don't do this for A-based (2018) because "HARD/MEDIUM/SOFT" are absolute tyre names then.
    if (is_c_based or not rel_keys) and s in {"SOFT", "MEDIUM", "HARD"}:
        return s

    # If it’s an A-code or C-code directly
    if re.fullmatch(r"A(1|2|3|4|5|6|7)", s) or re.fullmatch(r"C[1-6]", s):
        return rel_map.get(s)

    # 2018-style named slicks -> convert to A-code then map to relative
    if s in ABS_2018_TO_A:
        return rel_map.get(ABS_2018_TO_A[s])

    return None


def get_compound_allocation(season, gp_name):
    def normalize(name):
        return name.lower().replace("grand prix", "").replace("gp", "").strip()

    normalized_gp_name = normalize(gp_name)
    season_allocations = COMPOUND_ALLOCATIONS.get(season)
    if not season_allocations:
        logging.warning(f"No compound allocations found for season {season}.")
        return None

    for name, compounds in season_allocations.items():
        if normalize(name) == normalized_gp_name:
            return compounds.replace(" ", "")

    choices = list(season_allocations.keys())
    best_match = process.extractOne(gp_name, choices)
    if best_match and best_match[1] > 80:
        logging.warning(
            f"No exact match for '{gp_name}' in season {season}. "
            f"Using fuzzy match '{best_match[0]}' with score {best_match[1]}."
        )
        return season_allocations[best_match[0]].replace(" ", "")

    logging.error(f"Could not find compound allocation for '{gp_name}' in season {season}.")
    return None


def _safe_read_parquet(path, tracker, season, location, session, filename_label):
    try:
        df = pd.read_parquet(path)
        df.columns = [x.lower() for x in df.columns]
        return df
    except Exception as e:
        tracker.add(season, location, session, filename_label, f"failed to read parquet: {type(e).__name__}: {e}", path=path, severity="ERROR")
        return None


def _safe_timedelta_to_seconds(x):
    if pd.isna(x):
        return None
    try:
        return pd.to_timedelta(x).total_seconds()
    except Exception:
        return None


def _clean_and_dedupe_positions(df, tracker, season, location, session, source_path):
    """
    Ensures df has unique integer positions, logs duplicates, drops invalid rows.
    Returns cleaned df or None.
    """
    if "position" not in df.columns:
        tracker.add(season, location, session, "results.parquet", "missing 'position' column", path=source_path, severity="ERROR")
        return None

    out = df[pd.notna(df["position"])].copy()
    out["position"] = pd.to_numeric(out["position"], errors="coerce")
    out = out[pd.notna(out["position"])].copy()
    out["position"] = out["position"].astype(int)

    dup = out[out.duplicated(subset=["position"], keep=False)]
    if not dup.empty:
        logging.warning(
            f"{season} {location} {session}: duplicate positions found in results.parquet. "
            f"Dropping duplicates (keeping first)."
        )

    out = out.sort_values("position").drop_duplicates(subset=["position"], keep="first")
    return out


def _extract_fcy_phases(track_status_df):
    if track_status_df is None or track_status_df.empty:
        return []
    if "time" not in track_status_df.columns or "status" not in track_status_df.columns:
        return []

    df = track_status_df.copy().sort_values("time")
    df["time"] = df["time"].apply(_safe_timedelta_to_seconds)

    sc_codes = {"4"}
    vsc_codes = {"6", "7"}

    def classify(s):
        s = str(s)
        if s in sc_codes:
            return "SC"
        if s in vsc_codes:
            return "VSC"
        return None

    df["fcy_type"] = df["status"].apply(classify)

    phases = []
    current_type = None
    start_time = None

    for _, row in df.iterrows():
        t = row["time"]
        ft = row["fcy_type"]

        if current_type is None:
            if ft is not None:
                current_type = ft
                start_time = t
            continue

        if ft != current_type:
            phases.append({"start": start_time, "end": t, "type": current_type})
            current_type = None
            start_time = None
            if ft is not None:
                current_type = ft
                start_time = t

    if current_type is not None and start_time is not None:
        phases.append({"start": start_time, "end": None, "type": current_type})

    cleaned = []
    for p in phases:
        if p["start"] is None:
            continue
        if p["end"] is not None and p["end"] <= p["start"]:
            continue
        cleaned.append(p)
    return cleaned


def _phase_times_to_laps(phases, leader_laps_df):
    if not phases or leader_laps_df is None or leader_laps_df.empty:
        return phases

    ll = leader_laps_df.sort_values("lapnumber")
    lap_numbers = ll["lapnumber"].tolist()
    lap_racetimes = ll["racetime"].tolist()

    def time_to_lap(t):
        if t is None:
            return None
        for ln, rt in zip(lap_numbers, lap_racetimes):
            if rt is not None and rt >= t:
                return int(ln)
        return int(lap_numbers[-1]) if lap_numbers else None

    for p in phases:
        p["startlap"] = time_to_lap(p.get("start"))
        p["endlap"] = time_to_lap(p.get("end")) if p.get("end") is not None else None
    return phases


# --- Main Processing ---

def process_sessions(input_dir, conn, tracker, missing_report_path=None):
    logging.info(f"Starting to process sessions in '{input_dir}'...")
    cursor = conn.cursor()

    driver_cache = {}   # abbreviation -> driver_id
    race_cache = {}     # (season, location) -> race_id
    next_driver_id = 1
    next_race_id = 1
    next_fcyphase_id = 1

    # --- Pass 1: Discover races + populate races + drivers ---
    logging.info("--- Pass 1: Discovering events and populating 'races' and 'drivers' tables ---")
    season_dirs = sorted([p for p in Path(input_dir).iterdir() if p.is_dir() and p.name.isdigit()])
    if not season_dirs:
        logging.warning(f"No season directories found under input dir: {input_dir}")

    for season_dir in season_dirs:
        season = int(season_dir.name)
        gp_dirs = sorted([p for p in season_dir.iterdir() if p.is_dir()])

        for gp_dir in gp_dirs:
            location = gp_dir.name
            logging.info(f"--- Discovering event: {season} {location} ---")

            race_session_dir = gp_dir / "Race"
            if not race_session_dir.is_dir():
                tracker.add(season, location, "Race", "(session folder)", "missing Race session folder", path=race_session_dir)
                continue

            results_path = race_session_dir / "results.parquet"
            if not results_path.exists():
                tracker.add(season, location, "Race", "results.parquet", "missing results.parquet (event skipped)", path=results_path)
                continue

            results_df = _safe_read_parquet(results_path, tracker, season, location, "Race", "results.parquet")
            if results_df is None:
                continue

            # --- races row ---
            if (season, location) not in race_cache:
                race_date = f"{season}-01-01"
                laps_path = race_session_dir / "laps.parquet"
                if laps_path.exists():
                    laps_df_for_date = _safe_read_parquet(laps_path, tracker, season, location, "Race", "laps.parquet")
                    if laps_df_for_date is not None:
                        if (
                            "lapstartdate" in laps_df_for_date.columns
                            and not laps_df_for_date.empty
                            and pd.notna(laps_df_for_date["lapstartdate"].iloc[0])
                        ):
                            race_date = pd.to_datetime(laps_df_for_date["lapstartdate"].iloc[0]).strftime("%Y-%m-%d")
                else:
                    tracker.add(season, location, "Race", "laps.parquet", "missing laps.parquet (race date fallback used)", path=laps_path)

                dry_compounds = get_compound_allocation(season, location)
                available_compounds = f"{dry_compounds},I,W" if dry_compounds else "A1,A2,A3,I,W"

                nolaps = 0
                if "position" in results_df.columns and "laps" in results_df.columns:
                    winner = results_df[results_df["position"] == 1]
                    nolaps = int(winner["laps"].iloc[0]) if not winner.empty else 0
                else:
                    tracker.add(season, location, "Race", "results.parquet", "missing 'position'/'laps' columns; nolaps set to 0", path=results_path)

                cursor.execute(
                    """
                    INSERT INTO races (id, date, season, location, availablecompounds, comment, nolaps, nolapsplanned, tracklength)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (next_race_id, race_date, season, location, available_compounds, None, nolaps, nolaps, None),
                )
                race_cache[(season, location)] = next_race_id
                next_race_id += 1

            # --- drivers rows (from race results + qualifying results if present) ---
            all_results_dfs = [results_df]

            quali_session_dir = gp_dir / "Qualifying"
            if quali_session_dir.is_dir():
                quali_results_path = quali_session_dir / "results.parquet"
                if quali_results_path.exists():
                    quali_df = _safe_read_parquet(quali_results_path, tracker, season, location, "Qualifying", "results.parquet")
                    if quali_df is not None:
                        all_results_dfs.append(quali_df)
                else:
                    tracker.add(season, location, "Qualifying", "results.parquet", "missing qualifying results.parquet (drivers may still exist from Race)", path=quali_results_path)
            else:
                tracker.add(season, location, "Qualifying", "(session folder)", "missing Qualifying session folder (drivers may still exist from Race)", path=quali_session_dir)

            for df in all_results_dfs:
                for _, row in df.iterrows():
                    driver_abbr = row.get("abbreviation")
                    if not driver_abbr:
                        continue
                    if driver_abbr in driver_cache:
                        continue

                    driver_name = row.get("broadcastname", row.get("fullname", driver_abbr))
                    carno = row.get("drivernumber", 0)
                    try:
                        carno = int(carno) if pd.notna(carno) else 0
                    except Exception:
                        carno = 0

                    cursor.execute(
                        "INSERT INTO drivers (id, carno, initials, name) VALUES (?, ?, ?, ?)",
                        (next_driver_id, carno, driver_abbr, driver_name),
                    )
                    driver_cache[driver_abbr] = next_driver_id
                    next_driver_id += 1

    conn.commit()
    logging.info(f"Discovered and inserted {len(race_cache)} races and {len(driver_cache)} drivers.")

    if len(driver_cache) == 0:
        tracker.add("ALL", "ALL", "ALL", "drivers", "driver_cache is empty after Pass 1; laps insertion will be 0", severity="ERROR")

    # --- Pass 2: starterfields, qualifyings, retirements ---
    logging.info("--- Pass 2: Populating 'starterfields', 'qualifyings', and collecting retirement data ---")
    retirements_data = {}

    def to_seconds(time_val):
        if pd.isna(time_val):
            return None
        if isinstance(time_val, (int, float)):
            return float(time_val)
        try:
            return pd.to_timedelta(time_val).total_seconds()
        except Exception:
            return None

    for season_dir in season_dirs:
        season = int(season_dir.name)
        gp_dirs = sorted([p for p in season_dir.iterdir() if p.is_dir()])

        for gp_dir in gp_dirs:
            location = gp_dir.name
            race_id = race_cache.get((season, location))
            if not race_id:
                continue

            # starterfields from Race results
            race_session_dir = gp_dir / "Race"
            results_path = race_session_dir / "results.parquet"
            if results_path.exists():
                results_df = _safe_read_parquet(results_path, tracker, season, location, "Race", "results.parquet")
                if results_df is not None:
                    for _, row in results_df.iterrows():
                        abbr = row.get("abbreviation")
                        driver_id = driver_cache.get(abbr)
                        if not driver_id:
                            continue

                        cursor.execute(
                            """
                            INSERT INTO starterfields (race_id, driver_id, team, teamcolor, enginemanufacturer,
                                                     gridposition, status, resultposition, completedlaps, speedtrap)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                race_id, driver_id, row.get("teamname"), row.get("teamcolor"), None,
                                row.get("gridposition"), row.get("status"), row.get("position"),
                                row.get("laps"), None
                            )
                        )

                        status = str(row.get("status", "")).lower()
                        if "finished" not in status and "+" not in status:
                            key = (season, driver_id)
                            retirements_data.setdefault(key, {"accidents": 0, "failures": 0})
                            if any(kw in status for kw in ["accident", "collision", "crash", "spun", "damage", "contact"]):
                                retirements_data[key]["accidents"] += 1
                            else:
                                retirements_data[key]["failures"] += 1
            else:
                tracker.add(season, location, "Race", "results.parquet", "missing results.parquet (starterfields cannot be populated)", path=results_path)

            # qualifyings from Qualifying results (DEDUPED)
            quali_session_dir = gp_dir / "Qualifying"
            q_results_path = quali_session_dir / "results.parquet"
            if q_results_path.exists():
                q_df = _safe_read_parquet(q_results_path, tracker, season, location, "Qualifying", "results.parquet")
                if q_df is not None:
                    q_df = _clean_and_dedupe_positions(q_df, tracker, season, location, "Qualifying", q_results_path)
                    if q_df is not None:
                        for _, row in q_df.iterrows():
                            abbr = row.get("abbreviation")
                            driver_id = driver_cache.get(abbr)
                            if not driver_id:
                                continue
                            cursor.execute(
                                """
                                INSERT INTO qualifyings (race_id, position, driver_id, q1laptime, q2laptime, q3laptime, speedtrap)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    race_id,
                                    int(row.get("position")),
                                    driver_id,
                                    to_seconds(row.get("q1")),
                                    to_seconds(row.get("q2")),
                                    to_seconds(row.get("q3")),
                                    None
                                )
                            )
            else:
                tracker.add(season, location, "Qualifying", "results.parquet", "missing results.parquet (qualifyings cannot be populated)", path=q_results_path)

    conn.commit()
    logging.info("Populated starterfields, qualifyings, and collected retirement data.")

    # --- Pass 3: laps + fcyphases + speedtraps ---
    logging.info("--- Pass 3: Populating Laps, FCY, and Speedtrap data ---")

    for season_dir in season_dirs:
        season = int(season_dir.name)
        gp_dirs = sorted([p for p in season_dir.iterdir() if p.is_dir()])

        for gp_dir in gp_dirs:
            location = gp_dir.name
            race_id = race_cache.get((season, location))
            if not race_id:
                continue

            race_session_dir = gp_dir / "Race"
            if not race_session_dir.is_dir():
                tracker.add(season, location, "Race", "(session folder)", "missing Race session folder (laps/FCY cannot be populated)", path=race_session_dir)
                continue

            laps_path = race_session_dir / "laps.parquet"
            track_status_path = race_session_dir / "track_status.parquet"

            if not laps_path.exists():
                tracker.add(season, location, "Race", "laps.parquet", "missing laps.parquet (laps table cannot be populated)", path=laps_path)
                continue

            laps_df = _safe_read_parquet(laps_path, tracker, season, location, "Race", "laps.parquet")
            if laps_df is None:
                continue
            # --- Normalize FastF1 column names to lowercase to match your pipeline ---
            rename_map = {
                "Driver": "driver",
                "LapNumber": "lapnumber",
                "Position": "position",
                "LapTime": "laptime",
                "PitInTime": "pitintime",
                "PitOutTime": "pitouttime",
                "Compound": "compound",
                "TyreLife": "tyrelife",
                "TrackStatus": "trackstatus",
                "Deleted": "deleted",
                "DeletedReason": "deletedreason",
                "IsPersonalBest": "ispersonalbest",
                "IsAccurate": "isaccurate",
                "Sector1Time": "sector1time",
                "Sector2Time": "sector2time",
                "Sector3Time": "sector3time",
                "Sector1SessionTime": "sector1sessiontime",
                "Sector2SessionTime": "sector2sessiontime",
                "Sector3SessionTime": "sector3sessiontime",
                "SpeedI1": "speedi1",
                "SpeedI2": "speedi2",
                "SpeedFL": "speedfl",
                "SpeedST": "speedst",
                "LapStartTime": "lapstarttime",
                "Time": "time",
            }
            laps_df = laps_df.rename(columns={k: v for k, v in rename_map.items() if k in laps_df.columns})

            # Get this race's available compounds from DB (so it's the single source of truth)
            cursor.execute("SELECT availablecompounds FROM races WHERE id = ?", (race_id,))
            row = cursor.fetchone()
            availablecompounds = row[0] if row else None

            slicks = parse_allocated_slicks(availablecompounds)          # e.g. ["A4","A5","A6"]
            rel_map = build_relative_map_from_alloc(slicks)              # e.g. {"A4":"HARD","A5":"MEDIUM","A6":"SOFT"}

            # Normalise compound to relative labels
            laps_df["compound_rel"] = laps_df["compound"].apply(lambda x: normalise_to_relative(x, rel_map))

            for col in ["laptime", "pitintime", "pitouttime", "time"]:
                if col in laps_df.columns:
                    laps_df[col] = laps_df[col].apply(_safe_timedelta_to_seconds)


            required_cols = {"driver", "lapnumber", "position", "laptime"}
            missing_cols = required_cols - set(laps_df.columns)
            if missing_cols:
                tracker.add(
                    season, location, "Race", "laps.parquet",
                    f"missing required columns: {sorted(list(missing_cols))} (laps table cannot be populated)",
                    path=laps_path, severity="ERROR"
                )
                continue

            laps_df = laps_df.sort_values(by=["driver", "lapnumber"])
            laps_df["racetime"] = laps_df.groupby("driver")["laptime"].cumsum()
            # Ensure sort order is correct
            laps_df = laps_df.sort_values(by=["driver", "lapnumber"])
            # Next lap's compound for the same driver (next completed lap)
            laps_df["nextcompound_calc"] = laps_df.groupby("driver")["compound_rel"].shift(-1)
            # Only keep nextcompound for pit-entry laps
            laps_df["nextcompound_calc"] = laps_df["nextcompound_calc"].where(laps_df["pitintime"].notna(), None)
            # after you've converted pitintime to seconds and sorted
                        # ------------------------------------------------------------
            # Compute GAP + INTERVAL using leader-lap snapshots (robust to lapping/retirements)
            if "time" not in laps_df.columns:
                tracker.add(season, location, "Race", "laps.parquet",
                            "missing 'time' column (cannot compute robust gap/interval snapshots)",
                            path=laps_path, severity="ERROR")
                continue

            laps_df = compute_gap_interval_by_lap_completion(laps_df)

            # Optional: sanity check (should basically never be negative now)
            neg_int = laps_df["interval_calc"].notna() & (laps_df["interval_calc"] < 0)
            neg_gap = laps_df["gap_calc"].notna() & (laps_df["gap_calc"] < 0)
            if neg_int.any() or neg_gap.any():
                logging.warning(f"{season} {location} Race: unexpected negative gap/interval after snapshot calc.")


            # FCY
            if track_status_path.exists():
                ts_df = _safe_read_parquet(track_status_path, tracker, season, location, "Race", "track_status.parquet")
                if ts_df is not None:
                    phases = _extract_fcy_phases(ts_df)
                    leader_df = laps_df[laps_df["position"] == 1][["lapnumber", "racetime"]].dropna()
                    phases = _phase_times_to_laps(phases, leader_df)

                    for phase in phases:
                        cursor.execute(
                            """
                            INSERT INTO fcyphases (id, race_id, startracetime, endracetime, startlap, endlap, type)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                next_fcyphase_id, race_id,
                                phase.get("start"), phase.get("end"),
                                phase.get("startlap"), phase.get("endlap"),
                                phase.get("type"),
                            )
                        )
                        next_fcyphase_id += 1
            else:
                tracker.add(season, location, "Race", "track_status.parquet", "missing track_status.parquet (fcyphases cannot be populated)", path=track_status_path)

            # Laps insert
            inserted = 0
            missing_driver_codes = set()

            for _, lap in laps_df.iterrows():
                drv = lap.get("driver")
                driver_id = driver_cache.get(drv)
                if not driver_id:
                    missing_driver_codes.add(str(drv))
                    continue

                lapno = lap.get("lapnumber")
                pos = lap.get("position")
                rt = lap.get("racetime")

                gap = lap.get("gap_calc")
                interval = lap.get("interval_calc")

                cursor.execute(
                        """
                        INSERT INTO laps (race_id, lapno, position, driver_id, laptime, racetime, gap, interval,
                                        compound, tireage, pitintime, pitstopduration, nextcompound,
                                        startlapprog_vsc, endlapprog_vsc, age_vsc,
                                        startlapprog_sc, endlapprog_sc, age_sc)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            race_id, lapno, pos, driver_id, lap.get("laptime"),
                            rt, gap, interval, lap.get("compound_rel"), lap.get("tyrelife"),
                            str(lap.get("pitintime")) if pd.notna(lap.get("pitintime")) else None,
                            None,
                            lap.get("nextcompound_calc"),  # <--- HERE
                            None, None, None,
                            None, None, None,
                        )
                    )
                inserted += 1

            if inserted == 0:
                sample = ", ".join(list(sorted(missing_driver_codes))[:10])
                tracker.add(
                    season, location, "Race", "laps.parquet",
                    f"laps.parquet was present but 0 laps inserted (no driver_id matches). "
                    f"Sample lap driver codes not found in driver_cache: {sample}",
                    path=laps_path, severity="ERROR"
                )

            # Speedtrap update (if available)
            if "speedst" in laps_df.columns:
                speedtrap_df = laps_df.groupby("driver")["speedst"].max().reset_index()
                for _, row in speedtrap_df.iterrows():
                    driver_id = driver_cache.get(row["driver"])
                    if driver_id and pd.notna(row["speedst"]):
                        cursor.execute(
                            """
                            UPDATE starterfields SET speedtrap = ?
                            WHERE race_id = ? AND driver_id = ? AND speedtrap IS NULL
                            """,
                            (row["speedst"], race_id, driver_id),
                        )
            else:
                tracker.add(season, location, "Race", "laps.parquet", "column 'speedst' not present (speedtrap cannot be derived)", path=laps_path)

    conn.commit()
    logging.info("Populated laps, fcyphases, and updated speedtraps.")

    # --- Pass 4: retirements + validation ---
    logging.info("--- Pass 4: Finalizing and Validating ---")

    for (season, driver_id), counts in retirements_data.items():
        if counts["accidents"] > 0 or counts["failures"] > 0:
            cursor.execute(
                """
                INSERT INTO retirements (season, driver_id, accidents, failures)
                VALUES (?, ?, ?, ?)
                """,
                (season, driver_id, counts["accidents"], counts["failures"]),
            )

    conn.commit()

    # Simple validation counts
    logging.info("--- Running Validation Checks ---")
    for table in VSE_SCHEMA.keys():
        count = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        logging.info(f"Table '{table}': {count} rows inserted.")

    tracker.log_summary()
    if missing_report_path:
        tracker.write_csv(missing_report_path)


def main():
    parser = argparse.ArgumentParser(description="Build a VSE-style SQLite database from FastF1 exports.")
    parser.add_argument("--input", required=True, help="Directory containing FastF1 session data.")
    parser.add_argument("--output", required=True, help="Path to the output SQLite database file.")
    parser.add_argument("--missing-report", default=None, help="CSV path for missing/failed inputs and logical insert failures.")
    args = parser.parse_args()

    missing_report_path = args.missing_report
    if missing_report_path is None:
        out = Path(args.output)
        missing_report_path = str(out.with_suffix("")) + "_missing_report.csv"

    tracker = MissingDataTracker()
    conn = create_database(args.output)
    if not conn:
        return

    try:
        process_sessions(args.input, conn, tracker, missing_report_path=missing_report_path)
        conn.commit()
        logging.info("All data has been committed to the database.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        conn.rollback()
    finally:
        conn.close()
        logging.info("Database connection closed.")


if __name__ == "__main__":
    main()