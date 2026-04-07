"""
build_f1_db.py

database builder. reads parquet files storing fastf1 api data, and writes an sqlite database in four steps
pass 1 discover events, populate races and drivers
pass 2 populate starterfields, qualifyings, collect retirement data
pass 3 for every race insert session row, weather samples, track status, events, race control messages, laps, fcy phases, update driver car numbers
pass 4 insert retirements, run validation
"""
import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# logging setup just to see what is going on while the database is being built
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# full database schema
# stores ad a dictionary of table names and their schema
FULL_SCHEMA: Dict[str, str] = {
    "drivers": """
        CREATE TABLE IF NOT EXISTS drivers (
            id      INTEGER PRIMARY KEY,
            carno   INTEGER,
            initials TEXT UNIQUE,
            name    TEXT
        )
    """,
    "races": """
        CREATE TABLE IF NOT EXISTS races (
            id                   INTEGER PRIMARY KEY,
            date                 TEXT,
            season               INTEGER,
            location             TEXT,
            availablecompounds   TEXT,
            availablecompounds_c TEXT,
            comment              TEXT,
            nolaps               INTEGER,
            nolapsplanned        INTEGER,
            tracklength          REAL,
            wet_race             INTEGER DEFAULT 0,
            ingestion_status     TEXT DEFAULT 'pending',
            UNIQUE(season, location)
        )
    """,
    "sessions": """
        CREATE TABLE IF NOT EXISTS sessions (
            id           INTEGER PRIMARY KEY,
            race_id      INTEGER NOT NULL,
            session_code TEXT NOT NULL,
            session_name TEXT,
            date_utc     TEXT,
            timezone     TEXT,
            is_official  INTEGER DEFAULT 1,
            UNIQUE(race_id, session_code),
            FOREIGN KEY (race_id) REFERENCES races(id)
        )
    """,
    "starterfields": """
        CREATE TABLE IF NOT EXISTS starterfields (
            race_id             INTEGER,
            driver_id           INTEGER,
            team                TEXT,
            teamcolor           TEXT,
            enginemanufacturer  TEXT,
            gridposition        INTEGER,
            status              TEXT,
            resultposition      INTEGER,
            completedlaps       INTEGER,
            speedtrap           REAL,
            PRIMARY KEY (race_id, driver_id),
            FOREIGN KEY (race_id)   REFERENCES races(id),
            FOREIGN KEY (driver_id) REFERENCES drivers(id)
        )
    """,
    "laps": """
        CREATE TABLE IF NOT EXISTS laps (
            race_id            INTEGER,
            lapno              INTEGER,
            position           INTEGER,
            driver_id          INTEGER,
            -- core timing
            laptime            REAL,
            racetime           REAL,
            gap                REAL,
            interval           REAL,
            -- tyre
            compound           TEXT,
            tireage            INTEGER,
            nextcompound       TEXT,
            -- pit timing text is original timedelta string while s is seconds and num is ms
            pitintime          TEXT,
            pitouttime         TEXT,
            pitintime_s        REAL,
            pitouttime_s       REAL,
            pitintimenum       INTEGER,
            pitouttimenum      INTEGER,
            pitstopduration    REAL,
            pit_in_elapsed     REAL,
            -- sector times seconds
            sector1time        REAL,
            sector2time        REAL,
            sector3time        REAL,
            -- sector session times ms since session start
            sector1session_ms  INTEGER,
            sector2session_ms  INTEGER,
            sector3session_ms  INTEGER,
            -- speed traps kph
            speed_i1_kph       REAL,
            speed_i2_kph       REAL,
            speed_fl_kph       REAL,
            speed_st_kph       REAL,
            -- flags
            track_status_code  TEXT,
            is_personal_best   INTEGER,
            is_accurate        INTEGER,
            is_deleted         INTEGER,
            deleted_reason     TEXT,
            -- fcy lap progress fields populated downstream if needed
            startlapprog_vsc   REAL,
            endlapprog_vsc     REAL,
            age_vsc            REAL,
            startlapprog_sc    REAL,
            endlapprog_sc      REAL,
            age_sc             REAL,
            PRIMARY KEY (race_id, lapno, position),
            FOREIGN KEY (race_id)   REFERENCES races(id),
            FOREIGN KEY (driver_id) REFERENCES drivers(id)
        )
    """,
    "fcyphases": """
        CREATE TABLE IF NOT EXISTS fcyphases (
            id             INTEGER PRIMARY KEY,
            race_id        INTEGER,
            startracetime  REAL,
            endracetime    REAL,
            startraceprog  REAL,
            endraceprog    REAL,
            startlap       INTEGER,
            endlap         INTEGER,
            type           TEXT,
            FOREIGN KEY (race_id) REFERENCES races(id)
        )
    """,
    "qualifyings": """
        CREATE TABLE IF NOT EXISTS qualifyings (
            race_id    INTEGER,
            position   INTEGER,
            driver_id  INTEGER,
            q1laptime  REAL,
            q2laptime  REAL,
            q3laptime  REAL,
            speedtrap  REAL,
            PRIMARY KEY (race_id, position),
            FOREIGN KEY (race_id)   REFERENCES races(id),
            FOREIGN KEY (driver_id) REFERENCES drivers(id)
        )
    """,
    "retirements": """
        CREATE TABLE IF NOT EXISTS retirements (
            season    INTEGER,
            driver_id INTEGER,
            accidents INTEGER,
            failures  INTEGER,
            PRIMARY KEY (season, driver_id),
            FOREIGN KEY (driver_id) REFERENCES drivers(id)
        )
    """,
    "weather_samples": """
        CREATE TABLE IF NOT EXISTS weather_samples (
            session_id    INTEGER NOT NULL,
            time_ms       INTEGER NOT NULL,
            date_utc      TEXT,
            air_temp_c    REAL,
            track_temp_c  REAL,
            humidity_pct  REAL,
            pressure_hpa  REAL,
            wind_speed_ms REAL,
            wind_dir_deg  REAL,
            rainfall      INTEGER,
            PRIMARY KEY (session_id, time_ms),
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        ) WITHOUT ROWID
    """,
    "track_status_events": """
        CREATE TABLE IF NOT EXISTS track_status_events (
            id          INTEGER PRIMARY KEY,
            session_id  INTEGER NOT NULL,
            time_ms     INTEGER,
            date_utc    TEXT,
            status_code TEXT NOT NULL,
            message     TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """,
    "race_control_messages": """
        CREATE TABLE IF NOT EXISTS race_control_messages (
            id         INTEGER PRIMARY KEY,
            session_id INTEGER NOT NULL,
            utc        TEXT,
            time_ms    INTEGER,
            category   TEXT,
            message    TEXT,
            status     TEXT,
            flag       TEXT,
            scope      TEXT,
            sector     INTEGER,
            lap        INTEGER,
            driver_no  INTEGER,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """,
}

#commands to create indexes to later speed up queries
SCHEMA_INDICES: List[str] = [
    "CREATE INDEX IF NOT EXISTS idx_sessions_race ON sessions(race_id)",
    "CREATE INDEX IF NOT EXISTS idx_laps_race_driver ON laps(race_id, driver_id)",
    "CREATE INDEX IF NOT EXISTS idx_laps_race_drv_lap ON laps(race_id, driver_id, lapno)",
    "CREATE INDEX IF NOT EXISTS idx_rcm_session_utc ON race_control_messages(session_id, utc)",
    "CREATE INDEX IF NOT EXISTS idx_rcm_session_lap ON race_control_messages(session_id, lap)",
    "CREATE INDEX IF NOT EXISTS idx_rcm_driver_no ON race_control_messages(session_id, driver_no)",
]

# manually collected compound allocations for each weekend to be added to the database
COMPOUND_ALLOCATIONS: Dict[int, Dict[str, str]] = {
    2018: { #2018 season didnt use the C1-C5 range, they used hypersoft, ultrasoft, supersoft, soft, medium, hard, superhard, so I use the mapping hypersoft = A1, ultrasoft = A2, etc
        "Australian Grand Prix": "A4,A5,A6", "Bahrain Grand Prix": "A3,A4,A5",
        "Chinese Grand Prix": "A3,A4,A6", "Azerbaijan Grand Prix": "A4,A5,A6",
        "Spanish Grand Prix": "A3,A4,A5", "Monaco Grand Prix": "A5,A6,A7",
        "Canadian Grand Prix": "A5,A6,A7", "French Grand Prix": "A4,A5,A6",
        "Austrian Grand Prix": "A4,A5,A6", "British Grand Prix": "A2,A3,A4",
        "German Grand Prix": "A3,A4,A6", "Hungarian Grand Prix": "A3,A4,A6",
        "Belgian Grand Prix": "A3,A4,A5", "Italian Grand Prix": "A3,A4,A5",
        "Singapore Grand Prix": "A4,A6,A7", "Russian Grand Prix": "A4,A6,A7",
        "Japanese Grand Prix": "A3,A4,A5", "United States Grand Prix": "A4,A5,A6",
        "Mexican Grand Prix": "A5,A6,A7", "Brazilian Grand Prix": "A3,A4,A5",
        "Abu Dhabi Grand Prix": "A5,A6,A7",
    },
    2019: {
        "Australian Grand Prix": "C2,C3,C4", "Bahrain Grand Prix": "C1,C2,C3",
        "Chinese Grand Prix": "C2,C3,C4", "Azerbaijan Grand Prix": "C2,C3,C4",
        "Spanish Grand Prix": "C1,C2,C3", "Monaco Grand Prix": "C3,C4,C5",
        "Canadian Grand Prix": "C3,C4,C5", "French Grand Prix": "C2,C3,C4",
        "Austrian Grand Prix": "C2,C3,C4", "British Grand Prix": "C1,C2,C3",
        "German Grand Prix": "C2,C3,C4", "Hungarian Grand Prix": "C2,C3,C4",
        "Belgian Grand Prix": "C1,C2,C3", "Italian Grand Prix": "C2,C3,C4",
        "Singapore Grand Prix": "C3,C4,C5", "Russian Grand Prix": "C2,C3,C4",
        "Japanese Grand Prix": "C1,C2,C3", "Mexican Grand Prix": "C2,C3,C4",
        "United States Grand Prix": "C2,C3,C4", "Brazilian Grand Prix": "C1,C2,C3",
        "Abu Dhabi Grand Prix": "C3,C4,C5",
    },
    2020: {
        "Austrian Grand Prix": "C2,C3,C4", "Styrian Grand Prix": "C2,C3,C4",
        "Hungarian Grand Prix": "C2,C3,C4", "British Grand Prix": "C1,C2,C3",
        "70th Anniversary Grand Prix": "C2,C3,C4", "Spanish Grand Prix": "C1,C2,C3",
        "Belgian Grand Prix": "C2,C3,C4", "Italian Grand Prix": "C2,C3,C4",
        "Tuscan Grand Prix": "C1,C2,C3", "Russian Grand Prix": "C3,C4,C5",
        "Eifel Grand Prix": "C2,C3,C4", "Portuguese Grand Prix": "C1,C2,C3",
        "Emilia Romagna Grand Prix": "C2,C3,C4", "Turkish Grand Prix": "C1,C2,C3",
        "Bahrain Grand Prix": "C2,C3,C4", "Sakhir Grand Prix": "C2,C3,C4",
        "Abu Dhabi Grand Prix": "C3,C4,C5",
    },
    2021: {
        "Bahrain Grand Prix": "C2,C3,C4", "Emilia Romagna Grand Prix": "C2,C3,C4",
        "Portuguese Grand Prix": "C1,C2,C3", "Spanish Grand Prix": "C1,C2,C3",
        "Monaco Grand Prix": "C3,C4,C5", "Azerbaijan Grand Prix": "C3,C4,C5",
        "French Grand Prix": "C2,C3,C4", "Styrian Grand Prix": "C2,C3,C4",
        "Austrian Grand Prix": "C3,C4,C5", "British Grand Prix": "C1,C2,C3",
        "Hungarian Grand Prix": "C2,C3,C4", "Belgian Grand Prix": "C2,C3,C4",
        "Dutch Grand Prix": "C1,C2,C3", "Italian Grand Prix": "C2,C3,C4",
        "Russian Grand Prix": "C3,C4,C5", "Turkish Grand Prix": "C2,C3,C4",
        "United States Grand Prix": "C2,C3,C4", "Mexico City Grand Prix": "C2,C3,C4",
        "São Paulo Grand Prix": "C2,C3,C4", "Qatar Grand Prix": "C1,C2,C3",
        "Saudi Arabian Grand Prix": "C2,C3,C4", "Abu Dhabi Grand Prix": "C3,C4,C5",
    },
    2022: {
        "Bahrain Grand Prix": "C1,C2,C3", "Saudi Arabian Grand Prix": "C2,C3,C4",
        "Australian Grand Prix": "C2,C3,C5", "Emilia Romagna Grand Prix": "C2,C3,C4",
        "Miami Grand Prix": "C2,C3,C4", "Spanish Grand Prix": "C1,C2,C3",
        "Monaco Grand Prix": "C3,C4,C5", "Azerbaijan Grand Prix": "C3,C4,C5",
        "Canadian Grand Prix": "C3,C4,C5", "British Grand Prix": "C1,C2,C3",
        "Austrian Grand Prix": "C3,C4,C5", "French Grand Prix": "C2,C3,C4",
        "Hungarian Grand Prix": "C2,C3,C4", "Belgian Grand Prix": "C2,C3,C4",
        "Dutch Grand Prix": "C1,C2,C3", "Italian Grand Prix": "C2,C3,C4",
        "Singapore Grand Prix": "C3,C4,C5", "Japanese Grand Prix": "C1,C2,C3",
        "United States Grand Prix": "C2,C3,C4", "Mexico City Grand Prix": "C2,C3,C4",
        "São Paulo Grand Prix": "C2,C3,C4", "Abu Dhabi Grand Prix": "C3,C4,C5",
    },
    2023: {
        "Bahrain Grand Prix": "C1,C2,C3", "Saudi Arabian Grand Prix": "C2,C3,C4",
        "Australian Grand Prix": "C2,C3,C4", "Azerbaijan Grand Prix": "C3,C4,C5",
        "Miami Grand Prix": "C2,C3,C4", "Monaco Grand Prix": "C3,C4,C5",
        "Spanish Grand Prix": "C1,C2,C3", "Canadian Grand Prix": "C3,C4,C5",
        "Austrian Grand Prix": "C3,C4,C5", "British Grand Prix": "C1,C2,C3",
        "Hungarian Grand Prix": "C3,C4,C5", "Belgian Grand Prix": "C2,C3,C4",
        "Dutch Grand Prix": "C1,C2,C3", "Italian Grand Prix": "C3,C4,C5",
        "Singapore Grand Prix": "C3,C4,C5", "Japanese Grand Prix": "C1,C2,C3",
        "Qatar Grand Prix": "C1,C2,C3", "United States Grand Prix": "C2,C3,C4",
        "Mexico City Grand Prix": "C3,C4,C5", "São Paulo Grand Prix": "C2,C3,C4",
        "Las Vegas Grand Prix": "C3,C4,C5", "Abu Dhabi Grand Prix": "C3,C4,C5",
    },
    2024: {
        "Bahrain Grand Prix": "C1,C2,C3", "Saudi Arabian Grand Prix": "C2,C3,C4",
        "Australian Grand Prix": "C3,C4,C5", "Japanese Grand Prix": "C1,C2,C3",
        "Chinese Grand Prix": "C2,C3,C4", "Miami Grand Prix": "C2,C3,C4",
        "Emilia Romagna Grand Prix": "C3,C4,C5", "Monaco Grand Prix": "C3,C4,C5",
        "Canadian Grand Prix": "C3,C4,C5", "Spanish Grand Prix": "C1,C2,C3",
        "Austrian Grand Prix": "C3,C4,C5", "British Grand Prix": "C1,C2,C3",
        "Hungarian Grand Prix": "C3,C4,C5", "Belgian Grand Prix": "C1,C3,C4",
        "Dutch Grand Prix": "C1,C2,C3", "Italian Grand Prix": "C3,C4,C5",
        "Azerbaijan Grand Prix": "C3,C4,C5", "Singapore Grand Prix": "C3,C4,C5",
        "United States Grand Prix": "C2,C3,C4", "Mexico City Grand Prix": "C2,C4,C5",
        "São Paulo Grand Prix": "C3,C4,C5", "Las Vegas Grand Prix": "C3,C4,C5",
        "Qatar Grand Prix": "C1,C2,C3", "Abu Dhabi Grand Prix": "C3,C4,C5",
    },
    2025: {
        "Australian Grand Prix": "C3,C4,C5", "Chinese Grand Prix": "C2,C3,C4",
        "Japanese Grand Prix": "C1,C2,C3", "Bahrain Grand Prix": "C1,C2,C3",
        "Saudi Arabian Grand Prix": "C3,C4,C5", "Miami Grand Prix": "C3,C4,C5",
        "Emilia Romagna Grand Prix": "C4,C5,C6", "Monaco Grand Prix": "C4,C5,C6",
        "Spanish Grand Prix": "C1,C2,C3", "Canadian Grand Prix": "C4,C5,C6",
        "Austrian Grand Prix": "C3,C4,C5", "British Grand Prix": "C2,C3,C4",
        "Belgian Grand Prix": "C1,C3,C4", "Hungarian Grand Prix": "C3,C4,C5",
        "Dutch Grand Prix": "C2,C3,C4", "Italian Grand Prix": "C3,C4,C5",
        "Azerbaijan Grand Prix": "C4,C5,C6", "Singapore Grand Prix": "C3,C4,C5",
        "United States Grand Prix": "C1,C3,C4", "Mexico City Grand Prix": "C2,C4,C5",
        "São Paulo Grand Prix": "C2,C3,C4", "Las Vegas Grand Prix": "C3,C4,C5",
        "Qatar Grand Prix": "C1,C2,C3", "Abu Dhabi Grand Prix": "C3,C4,C5",
    },
}

# mapping 2018 absolute tyre compound names to simplified a codes consistent with heilmeier's database
_ABS_2018_TO_A: Dict[str, str] = {
    "SUPERHARD": "A1", "HARD": "A2", "MEDIUM": "A3",
    "SOFT": "A4", "SUPERSOFT": "A5", "ULTRASOFT": "A6", "HYPERSOFT": "A7",
}

# general purpose helpers

def td_to_s(val: Any) -> Optional[float]:
    """convert a timedelta like value to seconds float"""
    # handle nulls and nans
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    # try to just get total seconds
    try:
        return float(val.total_seconds())
    except Exception:
        # fallback to pandas to timedelta
        try:
            return float(pd.to_timedelta(val).total_seconds())
        except Exception:
            return None

def time_to_ms(val: Any) -> Optional[int]:
    """convert a timedelta like or numeric value to integer milliseconds"""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except Exception:
        pass
    # check for total seconds
    if hasattr(val, "total_seconds"):
        try:
            return int(val.total_seconds() * 1000)
        except Exception:
            return None
    # try parsing it as a float
    try:
        x = float(val)
        return int(x * 1000)
    except Exception:
        return None

def _safe_read_parquet(path: Path) -> Optional[pd.DataFrame]:
    """safely read a parquet file and normalise column names to lowercase"""
    try:
        df = pd.read_parquet(path)
        df.columns = [str(c).lower() for c in df.columns]
        return df
    except Exception:
        return None

def _norm(n):
    """
    normalises grand prix name for get_compound_a
    """
    # remove grand prix so it is easier to match
    return n.lower().replace("grand prix", "").replace("gp", "").strip()

# compound helpers
def get_compound_allocation(season: int, gp_name: str) -> Optional[str]:
    """retrieve the tire compound allocation for a specific race event"""

    # get the whole season of allocs
    allocs = COMPOUND_ALLOCATIONS.get(season)
    # normalise the target string by removing grand prix so its easier to match
    norm = _norm(gp_name)
    # search for the compounds used for that race
    for name, compounds in allocs.items():
        if _norm(name) == norm:
            return compounds.replace(" ", "")
    return None

def _parse_allocated_slicks(availablecompounds: str) -> List[str]:
    """parse the available compound string into a list of individual codes"""
    # handle empty strings
    if not availablecompounds:
        return []
    # uppercase everything just in case
    txt = str(availablecompounds).upper()
    # regex for old a codes (A1-A7 compouonds used in 2018)
    a = re.findall(r"\bA([1-7])\b", txt)
    # regex for new c codes (C1-C5 used from 2019-2025)
    c = re.findall(r"\bC([1-6])\b", txt)
    # give list of full codes
    if a:
        return [f"A{x}" for x in a]
    if c:
        return [f"C{x}" for x in c]
    return []

def _build_relative_map(slicks: List[str]) -> Dict[str, str]:
    """map absolute compound codes to relative hardness"""
    # we need exactly three slicks for this to work
    if len(slicks) != 3:
        return {}
    # sort by hardness where smaller number is harder
    ordered = sorted(slicks, key=lambda s: int(s[1:]))
    # hardcode the mapping to the sorted list
    return {ordered[0]: "HARD", ordered[1]: "MEDIUM", ordered[2]: "SOFT"}

def _normalise_compound(raw: Any, rel_map: Dict[str, str]) -> Optional[str]:
    """
    converts the raw tyre label into the final stored compound label
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None

    s = str(raw).strip().upper()

    if s in {"", "NAN", "NONE", "NULL"}:
        return None
    if s in {"INTERMEDIATE", "INTER", "IN", "I"}:
        return "INTERMEDIATE"
    if s in {"WET", "FULLWET", "FW"}:
        return "WET"

    if s in {"SOFT", "MEDIUM", "HARD"}:
        return s

    if s in rel_map:
        return rel_map[s]

    a_code = _ABS_2018_TO_A.get(s)
    if a_code is not None:
        return rel_map.get(a_code)

    return None

def _derive_availablecompounds_c(availablecompounds: str) -> Optional[str]:
    """derive the C compound range equivalent string from an A compound"""
    # dictionary translating old to new
    a_to_c = {"A2": "C1", "A3": "C2", "A4": "C3", "A6": "C4", "A7": "C5"}
    # split the string
    parts = [c.strip() for c in availablecompounds.split(",") if c.strip()]
    #translate everything we can (not all A compounds have an equivalent C compound, in this case its stored as None)
    c_parts = [a_to_c[p] for p in parts if p in a_to_c]
    return ",".join(sorted(set(c_parts))) if c_parts else None

# gap interval computation
def compute_gap_interval_by_position(laps_df: pd.DataFrame) -> pd.DataFrame:
    """
    computes gap to leader and interval to car ahead per lap based on
    position order at lap completion
    """
    # make a copy
    df = laps_df.copy()
    # force numeric types on critical columns
    for col in ("lapnumber", "position", "laptime", "time", "lapstarttime"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # calculate each cars lap end time
    end_time = df.get("time", pd.Series(dtype=float))
    # if there is no time column we try to add start plus laptime
    if end_time.isna().all() and "lapstarttime" in df.columns and "laptime" in df.columns:
        end_time = df["lapstarttime"] + df["laptime"]
    df["_end_s"] = end_time

    #columns we need to build gaps
    tmp = df[["driver", "lapnumber", "position", "_end_s"]].dropna(
        subset=["driver", "lapnumber", "position", "_end_s"]
    ).copy()
    tmp["lapnumber"] = tmp["lapnumber"].astype(int)
    tmp["position"] = tmp["position"].astype(int)
    
    #deduplication just in case
    tmp = (tmp.sort_values(["lapnumber", "position", "_end_s", "driver"]).drop_duplicates(subset=["lapnumber", "position"], keep="first").sort_values(["lapnumber", "position"]))

    # calculate gap to leader by finding the smallest lap-end time, which belongs to the leader, then subtract that from everyone else’s lap-end time
    leader = tmp.groupby("lapnumber")["_end_s"].transform("min")
    tmp["gap_calc"] = tmp["_end_s"] - leader

    # calculate interval to car ahead by finding the previous car in the list
    prev = tmp.groupby("lapnumber")["_end_s"].shift(1)
    tmp["interval_calc"] = tmp["_end_s"] - prev
    # leader interval is zero
    tmp.loc[tmp["position"] == 1, "interval_calc"] = 0.0
    tmp.loc[tmp["interval_calc"] < -1e-6, "interval_calc"] = np.nan

    # join the calculated metrics back onto the main df
    df = df.merge(tmp[["driver", "lapnumber", "gap_calc", "interval_calc"]],
                  on=["driver", "lapnumber"], how="left")
    # toss the temporary column
    df.drop(columns=["_end_s"], inplace=True, errors="ignore")
    return df


# fcy phase helpers

def _extract_fcy_phases(track_status_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """extract virtual safety car and safety car phases from track status data"""
    # handle missing track status df
    if track_status_df is None or track_status_df.empty:
        return []
    # check that we have time and status columns
    if not {"time", "status"}.issubset(track_status_df.columns):
        return []

    # copy and sort by time
    df = track_status_df.copy().sort_values("time")
    df["time"] = df["time"].apply(td_to_s)

    def classify(s):
        s = str(s)
        # four means safety car
        if s in {"4"}:
            return "SC"
        # six and seven indicate virtual safety car
        if s in {"6", "7"}:
            return "VSC"
        return None

    # map raw status numbers to types
    df["fcy_type"] = df["status"].apply(classify)
    phases: List[Dict[str, Any]] = []
    cur_type = None
    cur_start = None

    # iterate through rows to find contiguous phases
    for _, row in df.iterrows():
        t, ft = row["time"], row["fcy_type"]
        if cur_type is None:
            # start a new phase if we find one
            if ft is not None:
                cur_type, cur_start = ft, t
        elif ft != cur_type:
            # save the old phase and maybe start a new one
            phases.append({"start": cur_start, "end": t, "type": cur_type})
            cur_type = cur_start = None
            if ft is not None:
                cur_type, cur_start = ft, t

    # close out the last phase if the race ends under fcy
    if cur_type is not None:
        phases.append({"start": cur_start, "end": None, "type": cur_type})

    # return valid phases only
    return [p for p in phases if p["start"] is not None
            and (p["end"] is None or p["end"] > p["start"])]

def _phase_times_to_laps(phases: List[Dict[str, Any]], leader_laps_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """map fcy phase timestamps to specific lap numbers based on leader progress"""
    #if there are no fcy phases or no leader lap data return none
    if not phases or leader_laps_df is None or leader_laps_df.empty:
        return phases
    # get leader lap data sorted by lap number
    ll = leader_laps_df.sort_values("lapnumber")
    laps_list = ll["lapnumber"].tolist()
    times_list = ll["racetime"].tolist()

    #finds the first lap whose end time is at or after this race time
    def _time_to_lap(t):
        # handle missing times
        if t is None:
            return None
        for ln, rt in zip(laps_list, times_list):
            if rt is not None and rt >= t:
                return int(ln)
        # default to last lap if it is at the very end
        return int(laps_list[-1]) if laps_list else None

    #get the start and end lap for all phases
    for p in phases:
        p["startlap"] = _time_to_lap(p.get("start"))
        p["endlap"] = _time_to_lap(p.get("end")) if p.get("end") is not None else None
    return phases

# position deduplication

def _clean_positions(df: pd.DataFrame, season: int, location: str, session: str) -> Optional[pd.DataFrame]:
    """remove duplicate positions from qualifying results to ensure uniqueness"""
    if "position" not in df.columns:
        return None
    # drop nas
    out = df[pd.notna(df["position"])].copy()
    # force to numeric
    out["position"] = pd.to_numeric(out["position"], errors="coerce")
    out = out[pd.notna(out["position"])].copy()
    out["position"] = out["position"].astype(int)
    # log if we find duplicates and then remove them
    if out.duplicated(subset=["position"]).any():
        logger.warning(f"{season} {location} {session} duplicate positions keeping first")
        out = out.sort_values("position").drop_duplicates(subset=["position"], keep="first")
    return out

# pit timing helpers
def _precompute_pit_timing(laps_df: pd.DataFrame, group_key: str) -> pd.DataFrame:
    """
    Align pit timing so each pit stop is attached to the lap where pit entry occurred.
    """
    work = laps_df.copy()

    #converts pitintime and pitouttime to milliseconds
    if "pitintime" in work.columns:
        work["_pitintimenum_ms"] = work["pitintime"].apply(time_to_ms)
    else:
        work["_pitintimenum_ms"] = np.nan

    if "pitouttime" in work.columns:
        work["_pitouttimenum_ms"] = work["pitouttime"].apply(time_to_ms)
    else:
        work["_pitouttimenum_ms"] = np.nan

    #orders rows to be in driver lap order (laps are stored sequentially for each driver)
    sort_cols = [group_key]
    if "lapnumber" in work.columns:
        sort_cols.append("lapnumber")
    work = work.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    if group_key in work.columns:
        work["_pitouttimenum_next_ms"] = (
            work.groupby(group_key)["_pitouttimenum_ms"].shift(-1)
        )
        work["_pitouttime_next_val"] = (
            work.groupby(group_key)["pitouttime"].shift(-1)
            if "pitouttime" in work.columns else np.nan
        )
    else:
        work["_pitouttimenum_next_ms"] = work["_pitouttimenum_ms"].shift(-1)
        work["_pitouttime_next_val"] = (
            work["pitouttime"].shift(-1)
            if "pitouttime" in work.columns else np.nan
        )

    pitin = work["_pitintimenum_ms"]
    pitout_same = work["_pitouttimenum_ms"]
    pitout_next = work["_pitouttimenum_next_ms"]

    #only counts a pit stop as valid if for when there is a not null pitintime on lap n, there is a not null pitouttime on lap n + 1
    is_same_valid = pitin.notna() & pitout_same.notna() & (pitout_same > pitin)

    #compute pit duration
    work["_pitstopduration_s"] = np.where(
        is_same_valid,
        (pitout_same - pitin) / 1000.0,
        np.where(
            pitin.notna() & pitout_next.notna(),
            (pitout_next - pitin) / 1000.0,
            np.nan,
        ),
    )

    work["_pitout_aligned_ms"] = np.where(
        is_same_valid,
        pitout_same,
        np.where(pitin.notna(), pitout_next, pitout_same),
    )

    if "pitouttime" in work.columns:
        work["_pitout_aligned_val"] = np.where(
            is_same_valid,
            work["pitouttime"],
            np.where(pitin.notna(), work["_pitouttime_next_val"], work["pitouttime"]),
        )
    else:
        work["_pitout_aligned_val"] = np.nan

    return work


# database creation

def create_database(db_path: str) -> sqlite3.Connection:
    """initialise the sqlite database with the full schema"""
    # ensure parent dir exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    #remove old one if it exists
    if os.path.exists(db_path):
        os.remove(db_path)

    #make a new database
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    #create all the tables
    for name, sql in FULL_SCHEMA.items():
        cur.execute(sql)
        logger.info(f"created table {name}")
    #create all the indexes
    for idx_sql in SCHEMA_INDICES:
        cur.execute(idx_sql)
    #save it
    conn.commit()
    logger.info("schema created")
    return conn

# main processing

def process_sessions(input_dir: str, conn: sqlite3.Connection):
    """main processing loop iterating through all seasons and events"""
    cur = conn.cursor()

    # setting up some caches and ids
    driver_cache: Dict[str, int] = {}
    race_cache: Dict[Tuple[int, str], int] = {}
    next_driver_id = 1
    next_race_id = 1
    next_session_id = 1
    next_fcyphase_id = 1

    # find all the season directories
    season_dirs = sorted(
        [p for p in Path(input_dir).iterdir() if p.is_dir() and p.name.isdigit()]
    )
    if not season_dirs:
        logger.warning(f"no season directories found under {input_dir}")

    # pass 1 get all races and drivers
    logger.info("pass 1 races and drivers")

    for season_dir in season_dirs:
        season = int(season_dir.name)
        for gp_dir in sorted(p for p in season_dir.iterdir() if p.is_dir()):
            location = gp_dir.name
            race_session_dir = gp_dir / "Race"

            # make sure the race dir exists
            if not race_session_dir.is_dir():
                continue

            # looking for results parquet
            results_path = race_session_dir / "results.parquet"
            if not results_path.exists():
                continue

            #load results
            results_df = _safe_read_parquet(results_path)
            if results_df is None:
                continue

            # populate race details
            if (season, location) not in race_cache:
                race_date = f"{season}-01-01"
                laps_path = race_session_dir / "laps.parquet"
                # try to get actual date from laps parquet
                if laps_path.exists():
                    ldf = _safe_read_parquet(laps_path)
                    if ldf is not None and "lapstartdate" in ldf.columns and not ldf.empty:
                        v = ldf["lapstartdate"].iloc[0]
                        if pd.notna(v):
                            race_date = pd.to_datetime(v).strftime("%Y-%m-%d") #store date consistently

                # get compound allocations
                dry = get_compound_allocation(season, location)
                avail = f"{dry},I,W" if dry else "A1,A2,A3,I,W"
                avail_c = _derive_availablecompounds_c(avail) if dry else None

                # count the laps
                nolaps = 0
                if "position" in results_df.columns and "laps" in results_df.columns:
                    w = results_df[results_df["position"] == 1]
                    nolaps = int(w["laps"].iloc[0]) if not w.empty else 0

                #insert into database
                cur.execute(
                    """INSERT INTO races
                       (id, date, season, location, availablecompounds, availablecompounds_c,
                        comment, nolaps, nolapsplanned, tracklength)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (next_race_id, race_date, season, location,
                     avail, avail_c, None, nolaps, nolaps, None),
                )
                race_cache[(season, location)] = next_race_id
                next_race_id += 1

            # populate driver details
            sources = [results_df]
            q_res = gp_dir / "Qualifying" / "results.parquet"
            # load qualifying results
            if q_res.exists():
                qdf = _safe_read_parquet(q_res)
                if qdf is not None:
                    sources.append(qdf)

            # loop over all result sources to catch every driver
            for sdf in sources:
                for _, row in sdf.iterrows():
                    abbr = row.get("abbreviation")
                    if not abbr or abbr in driver_cache:
                        continue
                    # try to get a good name
                    name = row.get("broadcastname") or row.get("fullname") or abbr
                    # try to get car number
                    try:
                        carno = int(row.get("drivernumber", 0) or 0)
                    except Exception:
                        carno = 0
                    # shove driver into db
                    cur.execute(
                        "INSERT INTO drivers (id, carno, initials, name) VALUES (?,?,?,?)",
                        (next_driver_id, carno, abbr, name),
                    )
                    driver_cache[abbr] = next_driver_id
                    next_driver_id += 1

    # save pass 1
    conn.commit()
    logger.info(f"pass 1 done {len(race_cache)} races {len(driver_cache)} drivers")

    # pass 2 starterfields and qualifyings
    logger.info("pass 2 starterfields and qualifyings")

    # cache to hold retirement data until pass 4
    retirements_data: Dict[Tuple[int, int], Dict[str, int]] = {}

    for season_dir in season_dirs:
        season = int(season_dir.name)
        for gp_dir in sorted(p for p in season_dir.iterdir() if p.is_dir()):
            location = gp_dir.name
            race_id = race_cache.get((season, location))
            if race_id is None:
                continue

            # insert starterfield data
            results_path = gp_dir / "Race" / "results.parquet"
            if results_path.exists():
                rdf = _safe_read_parquet(results_path)
                if rdf is not None:
                    for _, row in rdf.iterrows():
                        abbr = row.get("abbreviation")
                        did = driver_cache.get(abbr)
                        if not did:
                            continue
                        # build starterfield entry
                        cur.execute(
                            """INSERT INTO starterfields
                               (race_id, driver_id, team, teamcolor, enginemanufacturer,
                                gridposition, status, resultposition, completedlaps, speedtrap)
                               VALUES (?,?,?,?,?,?,?,?,?,?)""",
                            (race_id, did, row.get("teamname"), row.get("teamcolor"), None,
                             row.get("gridposition"), row.get("status"),
                             row.get("position"), row.get("laps"), None),
                        )
                        # track retirement reasons
                        status = str(row.get("status", "")).lower()
                        if "finished" not in status and "+" not in status:
                            key = (season, did)
                            retirements_data.setdefault(key, {"accidents": 0, "failures": 0})
                            # guess at the retirement type using common retirement reasons
                            if any(kw in status for kw in
                                   ["accident", "collision", "crash", "spun", "damage", "contact"]):
                                retirements_data[key]["accidents"] += 1
                            else:
                                retirements_data[key]["failures"] += 1

            # insert qualifying data
            q_path = gp_dir / "Qualifying" / "results.parquet"
            if q_path.exists():
                qdf = _safe_read_parquet(q_path)
                if qdf is not None:
                    # clean out duplicates
                    qdf = _clean_positions(qdf, season, location, "Qualifying")
                    if qdf is not None:
                        for _, row in qdf.iterrows():
                            did = driver_cache.get(row.get("abbreviation"))
                            if not did:
                                continue
                            # log qualifying imes
                            cur.execute(
                                """INSERT INTO qualifyings
                                   (race_id, position, driver_id, q1laptime, q2laptime, q3laptime, speedtrap)
                                   VALUES (?,?,?,?,?,?,?)""",
                                (race_id, int(row.get("position")), did,
                                 td_to_s(row.get("q1")), td_to_s(row.get("q2")),
                                 td_to_s(row.get("q3")), None),
                            )

    # save pass 2
    conn.commit()
    logger.info("pass 2 done")

    # pass 3 sessions telemetry and laps
    logger.info("pass 3 sessions telemetry tables laps fcy")

    for season_dir in season_dirs:
        season = int(season_dir.name)
        for gp_dir in sorted(p for p in season_dir.iterdir() if p.is_dir()):
            location = gp_dir.name
            race_id = race_cache.get((season, location))
            if race_id is None:
                continue

            race_dir = gp_dir / "Race"
            if not race_dir.is_dir():
                continue

            logger.info(f"  processing {season} {location} race id {race_id}")

            # create session entry
            cur.execute(
                """INSERT INTO sessions (id, race_id, session_code, session_name)
                VALUES (?,?,'R','Race')""",
                (next_session_id, race_id),
            )
            session_id = next_session_id
            next_session_id += 1

            # process weather data
            wet_race = 0
            wp = race_dir / "weather_data.parquet"

            if wp.exists():
                wdf = _safe_read_parquet(wp)
                if wdf is not None:
                    base_date = None
                    if "date" in wdf.columns:
                        dates = wdf["date"].dropna()
                        if len(dates):
                            base_date = dates.min()

                    rows_w = []
                    for _, wr in wdf.iterrows():
                        t_ms = time_to_ms(wr.get("time"))
                        date_v = wr.get("date")

                        if t_ms is None and base_date is not None and pd.notna(date_v):
                            try:
                                t_ms = int((pd.Timestamp(date_v) - pd.Timestamp(base_date)).total_seconds() * 1000)
                            except Exception:
                                pass

                        if t_ms is None:
                            continue

                        date_utc = str(date_v) if pd.notna(date_v) else None

                        rain_v = wr.get("rainfall")
                        rain_i = None
                        if pd.notna(rain_v) and rain_v is not None:
                            if isinstance(rain_v, bool):
                                rain_i = 1 if rain_v else 0
                            else:
                                try:
                                    rain_i = int(rain_v)
                                except Exception:
                                    rain_i = 1 if str(rain_v).lower() in ("true", "yes") else 0

                        if rain_i == 1:
                            wet_race = 1

                        rows_w.append((
                            session_id,
                            t_ms,
                            date_utc,
                            wr.get("airtemp"),
                            wr.get("tracktemp"),
                            wr.get("humidity"),
                            wr.get("pressure"),
                            wr.get("windspeed"),
                            wr.get("winddirection"),
                            rain_i,
                        ))

                    if rows_w:
                        conn.executemany(
                            """INSERT OR REPLACE INTO weather_samples
                            (session_id,time_ms,date_utc,air_temp_c,track_temp_c,
                                humidity_pct,pressure_hpa,wind_speed_ms,wind_dir_deg,rainfall)
                            VALUES (?,?,?,?,?,?,?,?,?,?)""",
                            rows_w,
                        )
                        logger.info(f"    weather {len(rows_w)} rows from weather_data.parquet")

            cur.execute("UPDATE races SET wet_race=? WHERE id=?", (wet_race, race_id))

            # process track status events
            ts_df = None
            ts_path = race_dir / "track_status.parquet"
            if ts_path.exists():
                ts_df = _safe_read_parquet(ts_path)
                if ts_df is not None:
                    rows_ts = []
                    for _, r in ts_df.iterrows():
                        t_ms = time_to_ms(r.get("time"))
                        sv = r.get("status")
                        if t_ms is None or sv is None or pd.isna(sv):
                            continue
                        dv = r.get("date")
                        rows_ts.append((
                            session_id,
                            t_ms,
                            str(dv) if pd.notna(dv) else None,
                            str(sv),
                            r.get("message"),
                        ))
                    if rows_ts:
                        conn.executemany(
                            """INSERT INTO track_status_events
                            (session_id,time_ms,date_utc,status_code,message)
                            VALUES (?,?,?,?,?)""",
                            rows_ts,
                        )
                        logger.info(f"    track status events {len(rows_ts)}")

            # process race control messages
            rcm_path = race_dir / "race_control_messages.parquet"
            if rcm_path.exists():
                rcm_df = _safe_read_parquet(rcm_path)
                if rcm_df is not None:
                    rows_rcm = []
                    for _, r in rcm_df.iterrows():
                        uv = r.get("utc")
                        utc_str = uv.isoformat() if pd.notna(uv) and uv is not None else None
                        dno = r.get("racingnumber")
                        try:
                            dno = int(dno) if pd.notna(dno) else None
                        except Exception:
                            dno = None
                        rows_rcm.append((
                            session_id,
                            utc_str,
                            time_to_ms(r.get("time")),
                            r.get("category"),
                            r.get("message"),
                            r.get("status"),
                            r.get("flag"),
                            r.get("scope"),
                            r.get("sector"),
                            r.get("lap"),
                            dno,
                        ))
                    if rows_rcm:
                        conn.executemany(
                            """INSERT INTO race_control_messages
                            (session_id,utc,time_ms,category,message,
                                status,flag,scope,sector,lap,driver_no)
                            VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                            rows_rcm,
                        )
                        logger.info(f"    rcm {len(rows_rcm)} messages")

            # process laps
            laps_path = race_dir / "laps.parquet"
            if not laps_path.exists():
                conn.commit()
                continue

            laps_df = _safe_read_parquet(laps_path)
            if laps_df is None:
                conn.commit()
                continue

            if not {"driver", "lapnumber", "position", "laptime"}.issubset(laps_df.columns):
                conn.commit()
                continue

            # derive compound info
            cur.execute("SELECT availablecompounds FROM races WHERE id=?", (race_id,))
            avail_row = cur.fetchone()
            avail_str = avail_row[0] if avail_row else None
            rel_map = _build_relative_map(_parse_allocated_slicks(avail_str or ""))

            group_key = "drivernumber" if "drivernumber" in laps_df.columns else "driver"

            # ensure time columns are in seconds
            for col in ("laptime", "time", "lapstarttime"):
                if col in laps_df.columns:
                    laps_df[col] = laps_df[col].apply(td_to_s)

            # clean up position data
            laps_df["position"] = pd.to_numeric(laps_df["position"], errors="coerce")
            laps_df = laps_df[laps_df["position"].notna()].copy()
            laps_df["position"] = laps_df["position"].astype(int)

            # normalise tyre compound names
            if "compound" in laps_df.columns:
                laps_df["_compound_rel"] = laps_df["compound"].apply(
                    lambda x: _normalise_compound(x, rel_map)
                )
            else:
                laps_df["_compound_rel"] = None

            # calculate cumulative racetime per driver
            laps_df = laps_df.sort_values(["driver", "lapnumber"])
            laps_df["_racetime"] = laps_df.groupby("driver")["laptime"].cumsum()

            # determine the tyre compound used after the pit stop
            laps_df["_nextcompound"] = laps_df.groupby("driver")["_compound_rel"].shift(-1)
            if "pitintime" in laps_df.columns:
                laps_df["_nextcompound"] = laps_df["_nextcompound"].where(
                    laps_df["pitintime"].notna(),
                    other=None,
                )
            else:
                laps_df["_nextcompound"] = None

            # compute gaps and intervals
            laps_df = compute_gap_interval_by_position(laps_df)

            # align pit stop timings across laps
            if "pitintime" in laps_df.columns or "pitouttime" in laps_df.columns:
                laps_df = _precompute_pit_timing(laps_df, group_key)

            # loop over every lap and insert it in the db
            laps_inserted = 0
            for _, lap in laps_df.iterrows():
                drv = lap.get("driver")
                did = driver_cache.get(str(drv))
                if did is None:
                    continue

                lapno = lap.get("lapnumber")
                pos = lap.get("position")
                if pd.isna(lapno) or pd.isna(pos):
                    continue

                # extract pit timing fields
                pitin_raw = lap.get("pitintime")
                pitin_ms = lap.get("_pitintimenum_ms")
                pitout_ms = lap.get("_pitout_aligned_ms")
                pitout_val = lap.get("_pitout_aligned_val")
                pitdur = lap.get("_pitstopduration_s")

                pitin_str = str(pitin_raw) if pitin_raw is not None and pd.notna(pitin_raw) else None
                pitout_str = str(pitout_val) if pitout_val is not None and pd.notna(pitout_val) else None
                pitin_s = (pitin_ms / 1000.0) if pitin_ms is not None and pd.notna(pitin_ms) else None
                pitout_s = (pitout_ms / 1000.0) if pitout_ms is not None and pd.notna(pitout_ms) else None
                pitdur = float(pitdur) if pitdur is not None and pd.notna(pitdur) else None

                # calculate pit in elapsed
                lap_start_s = lap.get("lapstarttime")
                pit_in_elapsed = None
                if pitin_s is not None and lap_start_s is not None and pd.notna(lap_start_s):
                    pit_in_elapsed = pitin_s - float(lap_start_s)

                # extract sector times
                s1 = td_to_s(lap.get("sector1time"))
                s2 = td_to_s(lap.get("sector2time"))
                s3 = td_to_s(lap.get("sector3time"))
                s1s = time_to_ms(lap.get("sector1sessiontime"))
                s2s = time_to_ms(lap.get("sector2sessiontime"))
                s3s = time_to_ms(lap.get("sector3sessiontime"))

                # fetch flags
                pb_v = lap.get("ispersonalbest")
                acc_v = lap.get("isaccurate")
                del_v = lap.get("deleted")
                dr_v = lap.get("deletedreason")

                ts_val = lap.get("trackstatus")
                ts_str = str(ts_val) if ts_val is not None and pd.notna(ts_val) else None

                try:
                    cur.execute(
                        """INSERT INTO laps
                        (race_id, lapno, position, driver_id,
                            laptime, racetime, gap, interval,
                            compound, tireage, nextcompound,
                            pitintime, pitouttime, pitintime_s, pitouttime_s,
                            pitintimenum, pitouttimenum, pitstopduration, pit_in_elapsed,
                            sector1time, sector2time, sector3time,
                            sector1session_ms, sector2session_ms, sector3session_ms,
                            speed_i1_kph, speed_i2_kph, speed_fl_kph, speed_st_kph,
                            track_status_code,
                            is_personal_best, is_accurate, is_deleted, deleted_reason)
                        VALUES
                        (?,?,?,?,  ?,?,?,?,  ?,?,?,
                            ?,?,?,?,  ?,?,?,?,
                            ?,?,?,  ?,?,?,
                            ?,?,?,?,  ?,
                            ?,?,?,?)""",
                        (
                            race_id, int(lapno), int(pos), did,
                            lap.get("laptime"), lap.get("_racetime"),
                            lap.get("gap_calc"), lap.get("interval_calc"),
                            lap.get("_compound_rel"),
                            int(lap.get("tyrelife")) if pd.notna(lap.get("tyrelife")) else None,
                            lap.get("_nextcompound"),
                            pitin_str, pitout_str, pitin_s, pitout_s,
                            int(pitin_ms) if pitin_ms is not None and pd.notna(pitin_ms) else None,
                            int(pitout_ms) if pitout_ms is not None and pd.notna(pitout_ms) else None,
                            pitdur, pit_in_elapsed,
                            s1, s2, s3, s1s, s2s, s3s,
                            lap.get("speedi1"),
                            lap.get("speedi2"),
                            lap.get("speedfl"),
                            lap.get("speedst"),
                            ts_str,
                            1 if pb_v else 0,
                            1 if acc_v else 0,
                            1 if del_v else 0,
                            str(dr_v) if dr_v is not None and pd.notna(dr_v) else None,
                        ),
                    )
                    laps_inserted += 1
                except sqlite3.IntegrityError:
                    pass

            if laps_inserted > 0:
                logger.info(f"    laps inserted {laps_inserted}")

            # update speedtrap data in starterfields if found
            if "speedst" in laps_df.columns:
                st_df = laps_df.groupby("driver")["speedst"].max().reset_index()
                for _, r in st_df.iterrows():
                    did = driver_cache.get(str(r["driver"]))
                    sv = r["speedst"]
                    if did and pd.notna(sv):
                        cur.execute(
                            """UPDATE starterfields SET speedtrap=?
                            WHERE race_id=? AND driver_id=? AND speedtrap IS NULL""",
                            (sv, race_id, did),
                        )

            # extract fcy phases from track status and sync with laps
            if ts_df is not None:
                phases = _extract_fcy_phases(ts_df)
                if "position" in laps_df.columns:
                    leader_df = (
                        laps_df[laps_df["position"] == 1][["lapnumber", "_racetime"]]
                        .rename(columns={"_racetime": "racetime"})
                        .dropna()
                    )
                    phases = _phase_times_to_laps(phases, leader_df)

                for ph in phases:
                    cur.execute(
                        """INSERT INTO fcyphases
                        (id, race_id, startracetime, endracetime, startlap, endlap, type)
                        VALUES (?,?,?,?,?,?,?)""",
                        (
                            next_fcyphase_id,
                            race_id,
                            ph.get("start"),
                            ph.get("end"),
                            ph.get("startlap"),
                            ph.get("endlap"),
                            ph.get("type"),
                        ),
                    )
                    next_fcyphase_id += 1

            conn.commit()

    logger.info("pass 3 done")

    logger.info("pass 4 retirements and validation")

    # finally add the retirements we cached
    for (season, did), counts in retirements_data.items():
        if counts["accidents"] > 0 or counts["failures"] > 0:
            cur.execute(
                """INSERT INTO retirements (season, driver_id, accidents, failures)
                   VALUES (?,?,?,?)""",
                (season, did, counts["accidents"], counts["failures"]),
            )
    conn.commit()

    #validation check on all tables
    for table in FULL_SCHEMA:
        count = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        logger.info(f"  {table} {count} rows")

def main():
    input_dir = Path("../f1_data")
    output_db = Path("../data/raw/f1_database.sqlite")

    conn = create_database(output_db)

    try:
        process_sessions(input_dir, conn)
        logger.info("all data committed")
    except Exception as e:
        logger.error(f"unexpected error {e}", exc_info=True)
        conn.rollback()
    finally:
        conn.close()
        logger.info("connection closed")

    logger.info("done")

if __name__ == "__main__":
    main()