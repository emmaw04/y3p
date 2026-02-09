import sqlite3
import pandas as pd
import os
import re
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, Optional, Tuple, List

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Audit / Reports ---
REPORT_DIR = Path("build_reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

missing_inputs: List[Dict[str, Any]] = []
unused_inputs: List[Dict[str, Any]] = []
anomalies: List[Dict[str, Any]] = []
used_files_by_race: Dict[int, set[str]] = defaultdict(set)


def record_missing_input(*, race_id: int, season: int, location: str, kind: str, looked_for: List[str], race_dir: str, note: str):
    missing_inputs.append({
        "race_id": race_id,
        "season": season,
        "location": location,
        "kind": kind,
        "race_dir": race_dir,
        "looked_for": "|".join(looked_for),
        "note": note
    })


def record_used_file(race_id: int, path: str):
    used_files_by_race[race_id].add(os.path.basename(path))


def record_unused_parquets_for_race(race_id: int, season: int, location: str, race_dir: str):
    try:
        present = {p.name for p in Path(race_dir).glob("*.parquet")}
        used = used_files_by_race.get(race_id, set())
        unused = sorted(present - used)
        for fn in unused:
            unused_inputs.append({
                "race_id": race_id,
                "season": season,
                "location": location,
                "race_dir": race_dir,
                "unused_file": fn
            })
    except Exception as e:
        anomalies.append({
            "issue": "unused_parquet_scan_failed",
            "race_id": race_id,
            "season": season,
            "location": location,
            "race_dir": race_dir,
            "error": f"{type(e).__name__}: {e}"
        })


# --- Tyre Compound Mappings ---
A_TO_C_MAPPING = {
    "A2": "C1", "A3": "C2", "A4": "C3", "A6": "C4", "A7": "C5"
}
C_TO_A_MAPPING = {v: k for k, v in A_TO_C_MAPPING.items()}


# -------------------------
# Parquet helpers (robust to column casing)
# -------------------------
def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Return the first matching column in df for any candidate name, case-insensitive.
    """
    if df is None or df.empty:
        return None
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def get_val(row: pd.Series, col: Optional[str]) -> Any:
    if col is None:
        return None
    return row.get(col)


def time_to_ms(val: Any) -> Optional[int]:
    """
    Convert a FastF1 time-like value to integer ms since session start.

    Supports:
    - pandas/py Timedelta-like (has .total_seconds())
    - numeric seconds or milliseconds
    """
    if val is None or pd.isna(val):
        return None

    # Timedelta-like
    if hasattr(val, "total_seconds"):
        try:
            return int(val.total_seconds() * 1000)
        except Exception:
            return None

    # Numeric
    try:
        x = float(val)
    except Exception:
        return None

    # Heuristic:
    # - if it's huge, it’s probably already ms
    # - otherwise assume seconds
    if x > 1e6:
        return int(x)
    return int(x * 1000)


def td_to_s(td_val: Any) -> Optional[float]:
    if pd.isna(td_val) or td_val is None:
        return None
    try:
        return float(td_val.total_seconds())
    except Exception:
        return None


# --- DB Helpers ---
def get_db_connection(db_path: str, row_factory: bool = True) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    if row_factory:
        conn.row_factory = sqlite3.Row
    return conn


def copy_schema_and_data(input_db_path: str, output_db_path: str) -> None:
    logger.info(f"Copying schema and data from '{input_db_path}' to '{output_db_path}'...")
    if os.path.exists(output_db_path):
        os.remove(output_db_path)  # Ensure a clean start

    input_conn = get_db_connection(input_db_path, row_factory=False)
    output_conn = get_db_connection(output_db_path, row_factory=False)

    with input_conn:
        sql_script = "".join(input_conn.iterdump())
        output_conn.executescript(sql_script)

    output_conn.close()
    input_conn.close()
    logger.info("Schema and data copied successfully.")


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
    return cur.fetchone() is not None


def column_exists(cursor: sqlite3.Cursor, table_name: str, column_name: str) -> bool:
    cursor.execute(f"PRAGMA table_info({table_name});")
    for col_info in cursor.fetchall():
        if col_info[1] == column_name:
            return True
    return False


def first_existing_column(conn: sqlite3.Connection, table_name: str, candidates: List[str]) -> Optional[str]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table_name});")
    cols = {row[1] for row in cur.fetchall()}
    for c in candidates:
        if c in cols:
            return c
    return None


# --- Schema changes ---
def apply_schema_changes(conn: sqlite3.Connection) -> None:
    logger.info("Applying schema changes to the new database...")
    cursor = conn.cursor()

    # A) Create sessions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
          id INTEGER PRIMARY KEY,
          race_id INTEGER NOT NULL,
          session_code TEXT NOT NULL,   -- always 'R'
          session_name TEXT,            -- always 'Race'
          date_utc TEXT,
          timezone TEXT,
          is_official INTEGER DEFAULT 1,
          UNIQUE (race_id, session_code),
          FOREIGN KEY (race_id) REFERENCES races(id)
        );
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_race ON sessions(race_id);")

    # B) Create weather_samples table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS weather_samples (
          session_id INTEGER NOT NULL,
          time_ms INTEGER NOT NULL,
          date_utc TEXT,
          air_temp_c REAL,
          track_temp_c REAL,
          humidity_pct REAL,
          pressure_hpa REAL,
          wind_speed_ms REAL,
          wind_dir_deg REAL,
          rainfall INTEGER,
          PRIMARY KEY (session_id, time_ms),
          FOREIGN KEY (session_id) REFERENCES sessions(id)
        ) WITHOUT ROWID;
    """)

    # C) Create track_status_events table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS track_status_events (
          id INTEGER PRIMARY KEY,
          session_id INTEGER NOT NULL,
          time_ms INTEGER,
          date_utc TEXT,
          status_code TEXT NOT NULL,
          message TEXT,
          FOREIGN KEY (session_id) REFERENCES sessions(id)
        );
    """)

    # D) Create race_control_messages table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS race_control_messages (
          id INTEGER PRIMARY KEY,
          session_id INTEGER NOT NULL,
          utc TEXT,
          time_ms INTEGER,
          category TEXT,
          message TEXT,
          status TEXT,
          flag TEXT,
          scope TEXT,
          sector INTEGER,
          lap INTEGER,
          driver_no INTEGER,
          FOREIGN KEY (session_id) REFERENCES sessions(id)
        );
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rcm_session_utc ON race_control_messages(session_id, utc);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rcm_session_lap ON race_control_messages(session_id, lap);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rcm_driver_no ON race_control_messages(session_id, driver_no);")

    # E) ALTER races table
    if not column_exists(cursor, "races", "wet_race"):
        cursor.execute("ALTER TABLE races ADD COLUMN wet_race INTEGER DEFAULT 0;")
    if not column_exists(cursor, "races", "availablecompounds_c"):
        cursor.execute("ALTER TABLE races ADD COLUMN availablecompounds_c TEXT;")

    # F) ALTER laps table (only if it exists)
    if table_exists(conn, "laps"):
        columns_to_add_to_laps = [
            ("sector1time", "REAL"), ("sector2time", "REAL"), ("sector3time", "REAL"),
            ("sector1session_ms", "INTEGER"), ("sector2session_ms", "INTEGER"), ("sector3session_ms", "INTEGER"),
            ("speed_i1_kph", "REAL"), ("speed_i2_kph", "REAL"), ("speed_fl_kph", "REAL"), ("speed_st_kph", "REAL"),
            ("track_status_code", "TEXT"), ("is_personal_best", "INTEGER"), ("isaccurate", "INTEGER"),
            ("is_deleted", "INTEGER"), ("deleted_reason", "TEXT"),

            # Pit timing numeric fields:
            ("pitintimenum", "INTEGER"),   # ms since session start
            ("pitouttimenum", "INTEGER"),  # ms since session start
            ("pitstopduration", "REAL"),   # seconds (aligned to PitIn lap, using next lap PitOut)

            # Convenience (seconds):
            ("pitintime_s", "REAL"),
            ("pitouttime_s", "REAL"),
        ]

        # Ensure these exist for compatibility/debug
        if not column_exists(cursor, "laps", "pitintime"):
            cursor.execute("ALTER TABLE laps ADD COLUMN pitintime TEXT;")
        if not column_exists(cursor, "laps", "pitouttime"):
            cursor.execute("ALTER TABLE laps ADD COLUMN pitouttime TEXT;")

        for col_name, col_type in columns_to_add_to_laps:
            if not column_exists(cursor, "laps", col_name):
                cursor.execute(f"ALTER TABLE laps ADD COLUMN {col_name} {col_type};")

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_laps_race_driver ON laps(race_id, driver_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_laps_race_driver_lap ON laps(race_id, driver_id, lapno);")

        conn.commit()
        logger.info("Schema changes applied successfully.")
    else:
        conn.commit()
        logger.warning(
            "Schema changes applied, but 'laps' table does not exist in the copied DB. "
            "Lap updates and cleaning steps that depend on laps will be skipped."
        )


# --- Mappings ---
def get_race_driver_mappings(conn: sqlite3.Connection) -> Tuple[Dict, Dict, Dict, Dict]:
    cursor = conn.cursor()

    cursor.execute("SELECT id, season, location FROM races")
    race_map = {}
    for row in cursor.fetchall():
        normalized_location = re.sub(r"[^a-z0-9]", "", row["location"].lower())
        race_map[(row["season"], normalized_location)] = row["id"]

    cursor.execute("SELECT id, carno, initials, name FROM drivers")
    driver_id_by_carno = {}
    driver_id_by_initials = {}
    driver_id_by_name = {}
    for row in cursor.fetchall():
        if row["carno"] is not None:
            driver_id_by_carno[row["carno"]] = row["id"]
        if row["initials"] is not None:
            driver_id_by_initials[row["initials"]] = row["id"]
        if row["name"] is not None:
            driver_id_by_name[row["name"]] = row["id"]

    return race_map, driver_id_by_carno, driver_id_by_initials, driver_id_by_name


def get_session_id(conn: sqlite3.Connection, race_id: int) -> Optional[int]:
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM sessions WHERE race_id = ? AND session_code = 'R'", (race_id,))
    row = cursor.fetchone()
    return row["id"] if row else None


def insert_session(conn: sqlite3.Connection, race_id: int, date_utc: Optional[str] = None,
                   timezone: Optional[str] = None) -> int:
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO sessions (race_id, session_code, session_name, date_utc, timezone)
            VALUES (?, 'R', 'Race', ?, ?)
        """, (race_id, date_utc, timezone))
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        logger.warning(f"Session for race_id {race_id} already exists. Retrieving existing session_id.")
        existing = get_session_id(conn, race_id)
        if existing is None:
            raise
        return existing





# --- Audit SQL exports ---
def export_audit_csvs(db_path: str) -> None:
    try:
        conn = sqlite3.connect(db_path)
        # Negative pit durations (seconds)
        neg_pits = pd.read_sql_query("""
            SELECT race_id, driver_id, lapno, position, pitintimenum, pitouttimenum, pitstopduration
            FROM laps
            WHERE pitstopduration IS NOT NULL AND pitstopduration < 0
            ORDER BY race_id, driver_id, lapno
        """, conn)

        # Huge pit durations (seconds) - tune if you want
        huge_pits = pd.read_sql_query("""
            SELECT race_id, driver_id, lapno, position, pitintimenum, pitouttimenum, pitstopduration
            FROM laps
            WHERE pitstopduration IS NOT NULL AND pitstopduration > 180
            ORDER BY race_id, driver_id, lapno
        """, conn)

        # Null coverage summary (based on this script's schema additions)
        null_summary = pd.read_sql_query("""
            SELECT
              SUM(CASE WHEN sector1time IS NULL THEN 1 ELSE 0 END) AS sector1time_nulls,
              SUM(CASE WHEN sector2time IS NULL THEN 1 ELSE 0 END) AS sector2time_nulls,
              SUM(CASE WHEN sector3time IS NULL THEN 1 ELSE 0 END) AS sector3time_nulls,
              SUM(CASE WHEN speed_i1_kph IS NULL THEN 1 ELSE 0 END) AS speed_i1_kph_nulls,
              SUM(CASE WHEN speed_i2_kph IS NULL THEN 1 ELSE 0 END) AS speed_i2_kph_nulls,
              SUM(CASE WHEN speed_fl_kph IS NULL THEN 1 ELSE 0 END) AS speed_fl_kph_nulls,
              SUM(CASE WHEN speed_st_kph IS NULL THEN 1 ELSE 0 END) AS speed_st_kph_nulls,
              SUM(CASE WHEN pitstopduration IS NULL THEN 1 ELSE 0 END) AS pitstopduration_nulls,
              COUNT(*) AS total_laps
            FROM laps
        """, conn)

        # Completeness by race
        by_race = pd.read_sql_query("""
            SELECT
              race_id,
              COUNT(*) AS total_laps,
              SUM(CASE WHEN sector1time IS NOT NULL OR sector2time IS NOT NULL OR sector3time IS NOT NULL OR
                            speed_i1_kph IS NOT NULL OR speed_i2_kph IS NOT NULL OR speed_fl_kph IS NOT NULL OR speed_st_kph IS NOT NULL
                       THEN 1 ELSE 0 END) AS laps_with_any_fastf1_fields,
              SUM(CASE WHEN pitstopduration IS NOT NULL THEN 1 ELSE 0 END) AS laps_with_pitstopduration
            FROM laps
            GROUP BY race_id
            ORDER BY race_id
        """, conn)

        conn.close()

        neg_pits.to_csv(REPORT_DIR / "audit_negative_pitstopduration_rows.csv", index=False)
        huge_pits.to_csv(REPORT_DIR / "audit_huge_pitstopduration_rows.csv", index=False)
        null_summary.to_csv(REPORT_DIR / "audit_null_summary.csv", index=False)
        by_race.to_csv(REPORT_DIR / "audit_completeness_by_race.csv", index=False)
    except Exception as e:
        anomalies.append({
            "issue": "audit_sql_export_failed",
            "error": f"{type(e).__name__}: {e}"
        })


# --- Processing per race ---
def process_race_event(
    output_conn: sqlite3.Connection,
    race_row: sqlite3.Row,
    race_map: Dict,
    driver_id_by_carno: Dict,
    driver_id_by_initials: Dict,
    driver_id_by_name: Dict,
    parquet_root: str
) -> None:
    race_id = race_row["id"]
    season = race_row["season"]
    location = race_row["location"]

    race_data_path = os.path.join(parquet_root, str(season), location, "Race")
    if not os.path.isdir(race_data_path):
        logger.warning(f"Skipping race {season} {location}: Race data directory not found at '{race_data_path}'")
        record_missing_input(
            race_id=race_id,
            season=season,
            location=location,
            kind="race_dir",
            looked_for=["<season>/<location>/Race"],
            race_dir=race_data_path,
            note="Race data directory missing"
        )
        return

    logger.info(f"Processing race: {season} {location} (Race ID: {race_id})")

    # --- Insert Session ---
    session_id = insert_session(output_conn, race_id)

    # --- Process weather(.parquet / _data.parquet) ---
    weather_candidates = ["weather.parquet", "weather_data.parquet", "weatherdata.parquet"]
    weather_parquet_path = None
    for fn in weather_candidates:
        p = os.path.join(race_data_path, fn)
        if os.path.exists(p):
            weather_parquet_path = p
            break

    if weather_parquet_path:
        record_used_file(race_id, weather_parquet_path)
        try:
            df_weather = pd.read_parquet(weather_parquet_path)

            col_time = pick_col(df_weather, ["time", "Time"])
            col_date = pick_col(df_weather, ["date", "Date"])
            col_rain = pick_col(df_weather, ["rainfall", "Rainfall"])
            col_air = pick_col(df_weather, ["airtemp", "AirTemp"])
            col_track = pick_col(df_weather, ["tracktemp", "TrackTemp"])
            col_hum = pick_col(df_weather, ["humidity", "Humidity"])
            col_pres = pick_col(df_weather, ["pressure", "Pressure"])
            col_wsp = pick_col(df_weather, ["windspeed", "WindSpeed"])
            col_wdir = pick_col(df_weather, ["winddirection", "WindDirection"])

            # If Time is missing/unusable, fall back to Date deltas from earliest Date
            base_date = None
            if col_date is not None:
                dates = df_weather[col_date].dropna()
                if len(dates) > 0:
                    base_date = dates.min()

            weather_data = []
            wet_race_flag = 0

            for _, row in df_weather.iterrows():
                time_val = get_val(row, col_time)
                date_val = get_val(row, col_date)

                time_ms = time_to_ms(time_val)

                if time_ms is None and base_date is not None and pd.notna(date_val):
                    # Compute relative ms from earliest date in this session
                    try:
                        time_ms = int((pd.Timestamp(date_val) - pd.Timestamp(base_date)).total_seconds() * 1000)
                    except Exception:
                        time_ms = None

                if time_ms is None:
                    continue

                date_utc = str(date_val) if pd.notna(date_val) and date_val is not None else None

                rainfall_val = get_val(row, col_rain)
                rainfall_i = None
                if pd.notna(rainfall_val) and rainfall_val is not None:
                    if isinstance(rainfall_val, (bool,)):
                        rainfall_i = 1 if rainfall_val else 0
                    else:
                        try:
                            rainfall_i = int(rainfall_val)
                        except Exception:
                            s = str(rainfall_val).strip().lower()
                            if s in ("true", "yes", "y"):
                                rainfall_i = 1
                            elif s in ("false", "no", "n"):
                                rainfall_i = 0

                if rainfall_i == 1:
                    wet_race_flag = 1

                weather_data.append((
                    session_id,
                    time_ms,
                    date_utc,
                    get_val(row, col_air),
                    get_val(row, col_track),
                    get_val(row, col_hum),
                    get_val(row, col_pres),
                    get_val(row, col_wsp),
                    get_val(row, col_wdir),
                    rainfall_i
                ))

            if weather_data:
                output_conn.executemany("""
                    INSERT OR REPLACE INTO weather_samples
                        (session_id, time_ms, date_utc, air_temp_c, track_temp_c, humidity_pct, pressure_hpa,
                         wind_speed_ms, wind_dir_deg, rainfall)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, weather_data)
                output_conn.commit()
                logger.info(f"  Inserted {len(weather_data)} weather samples from {os.path.basename(weather_parquet_path)}.")
            else:
                msg = (
                    f"Weather file found ({os.path.basename(weather_parquet_path)}), but 0 rows inserted. "
                    f"Columns={list(df_weather.columns)}; time_col={col_time}; date_col={col_date}"
                )
                logger.warning("  " + msg)
                anomalies.append({
                    "issue": "weather_0_rows_inserted",
                    "race_id": race_id,
                    "season": season,
                    "location": location,
                    "file": os.path.basename(weather_parquet_path),
                    "note": msg
                })

            output_conn.execute("UPDATE races SET wet_race = ? WHERE id = ?", (wet_race_flag, race_id))
            output_conn.commit()

        except Exception as e:
            logger.error(f"  Error processing {os.path.basename(weather_parquet_path)} for {season} {location}: {e}")
            record_missing_input(
                race_id=race_id,
                season=season,
                location=location,
                kind="weather",
                looked_for=[os.path.basename(weather_parquet_path)],
                race_dir=race_data_path,
                note=f"Found but failed to read/insert: {type(e).__name__}: {e}"
            )
    else:
        logger.info(f"  No weather parquet found for {season} {location}. Looked for {weather_candidates}.")
        record_missing_input(
            race_id=race_id,
            season=season,
            location=location,
            kind="weather",
            looked_for=weather_candidates,
            race_dir=race_data_path,
            note="No weather parquet found"
        )

    # --- Process track_status.parquet ---
    track_status_parquet_path = os.path.join(race_data_path, "track_status.parquet")
    if os.path.exists(track_status_parquet_path):
        record_used_file(race_id, track_status_parquet_path)
        try:
            df_track_status = pd.read_parquet(track_status_parquet_path)

            col_time = pick_col(df_track_status, ["time", "Time"])
            col_date = pick_col(df_track_status, ["date", "Date"])
            col_status = pick_col(df_track_status, ["status", "Status"])
            col_msg = pick_col(df_track_status, ["message", "Message"])

            track_status_data = []
            for _, row in df_track_status.iterrows():
                time_ms = time_to_ms(get_val(row, col_time))
                date_val = get_val(row, col_date)
                date_utc = str(date_val) if pd.notna(date_val) and date_val is not None else None

                status_val = get_val(row, col_status)
                msg_val = get_val(row, col_msg)

                if time_ms is None or status_val is None or pd.isna(status_val):
                    continue

                track_status_data.append((
                    session_id,
                    time_ms,
                    date_utc,
                    str(status_val),
                    msg_val
                ))

            if track_status_data:
                output_conn.executemany("""
                    INSERT OR REPLACE INTO track_status_events (session_id, time_ms, date_utc, status_code, message)
                    VALUES (?, ?, ?, ?, ?)
                """, track_status_data)
                output_conn.commit()
                logger.info(f"  Inserted {len(track_status_data)} track status events.")
            else:
                anomalies.append({
                    "issue": "track_status_0_rows_inserted",
                    "race_id": race_id,
                    "season": season,
                    "location": location,
                    "file": "track_status.parquet"
                })
        except Exception as e:
            logger.error(f"  Error processing track_status.parquet for {season} {location}: {e}")
            record_missing_input(
                race_id=race_id,
                season=season,
                location=location,
                kind="track_status",
                looked_for=["track_status.parquet"],
                race_dir=race_data_path,
                note=f"Found but failed to read/insert: {type(e).__name__}: {e}"
            )
    else:
        logger.info(f"  track_status.parquet not found for {season} {location}. Skipping track status events.")
        record_missing_input(
            race_id=race_id,
            season=season,
            location=location,
            kind="track_status",
            looked_for=["track_status.parquet"],
            race_dir=race_data_path,
            note="File missing"
        )

    # --- Process race_control_messages.parquet ---
    rcm_parquet_path = os.path.join(race_data_path, "race_control_messages.parquet")
    if os.path.exists(rcm_parquet_path):
        record_used_file(race_id, rcm_parquet_path)
        try:
            df_rcm = pd.read_parquet(rcm_parquet_path)

            col_utc = pick_col(df_rcm, ["utc", "Utc"])
            col_time = pick_col(df_rcm, ["time", "Time"])
            col_cat = pick_col(df_rcm, ["category", "Category"])
            col_msg = pick_col(df_rcm, ["message", "Message"])
            col_status = pick_col(df_rcm, ["status", "Status"])
            col_flag = pick_col(df_rcm, ["flag", "Flag"])
            col_scope = pick_col(df_rcm, ["scope", "Scope"])
            col_sector = pick_col(df_rcm, ["sector", "Sector"])
            col_lap = pick_col(df_rcm, ["lap", "Lap"])
            col_rn = pick_col(df_rcm, ["racingnumber", "RacingNumber", "driver_no", "DriverNo"])

            rcm_data = []
            for _, row in df_rcm.iterrows():
                utc_val = get_val(row, col_utc)
                utc_str = utc_val.isoformat() if pd.notna(utc_val) and utc_val is not None else None

                time_ms = time_to_ms(get_val(row, col_time))
                driver_no = get_val(row, col_rn)
                try:
                    driver_no = int(driver_no) if pd.notna(driver_no) and driver_no is not None else None
                except Exception:
                    driver_no = None

                rcm_data.append((
                    session_id,
                    utc_str,
                    time_ms,
                    get_val(row, col_cat),
                    get_val(row, col_msg),
                    get_val(row, col_status),
                    get_val(row, col_flag),
                    get_val(row, col_scope),
                    get_val(row, col_sector),
                    get_val(row, col_lap),
                    driver_no
                ))

            if rcm_data:
                output_conn.executemany("""
                    INSERT INTO race_control_messages
                        (session_id, utc, time_ms, category, message, status, flag, scope, sector, lap, driver_no)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, rcm_data)
                output_conn.commit()
                logger.info(f"  Inserted {len(rcm_data)} race control messages.")
            else:
                anomalies.append({
                    "issue": "race_control_messages_0_rows_inserted",
                    "race_id": race_id,
                    "season": season,
                    "location": location,
                    "file": "race_control_messages.parquet"
                })
        except Exception as e:
            logger.error(f"  Error processing race_control_messages.parquet for {season} {location}: {e}")
            record_missing_input(
                race_id=race_id,
                season=season,
                location=location,
                kind="race_control_messages",
                looked_for=["race_control_messages.parquet"],
                race_dir=race_data_path,
                note=f"Found but failed to read/insert: {type(e).__name__}: {e}"
            )
    else:
        logger.info(f"  race_control_messages.parquet not found for {season} {location}. Skipping race control messages.")
        record_missing_input(
            race_id=race_id,
            season=season,
            location=location,
            kind="race_control_messages",
            looked_for=["race_control_messages.parquet"],
            race_dir=race_data_path,
            note="File missing"
        )

    # --- Update laps table ---
    if not table_exists(output_conn, "laps"):
        logger.warning("  Skipping laps update: 'laps' table does not exist in this DB.")
        return

    laps_parquet_path = os.path.join(race_data_path, "laps.parquet")
    if os.path.exists(laps_parquet_path):
        record_used_file(race_id, laps_parquet_path)
        try:
            df_laps = pd.read_parquet(laps_parquet_path)
            updated_laps_count = 0

            # Resolve columns (your laps.parquet uses lowercase: driver, drivernumber, lapnumber, etc.)
            col_driver_no = pick_col(df_laps, ["drivernumber", "DriverNumber", "RacingNumber", "driver_no"])
            col_driver = pick_col(df_laps, ["driver", "Driver"])
            has_driver_info_columns = (col_driver_no is not None) and (col_driver is not None)

            col_lapnumber = pick_col(df_laps, ["lapnumber", "LapNumber", "lapno", "LapNo"])
            col_position = pick_col(df_laps, ["position", "Position"])
            col_s1 = pick_col(df_laps, ["sector1time", "Sector1Time"])
            col_s2 = pick_col(df_laps, ["sector2time", "Sector2Time"])
            col_s3 = pick_col(df_laps, ["sector3time", "Sector3Time"])
            col_s1s = pick_col(df_laps, ["sector1sessiontime", "Sector1SessionTime"])
            col_s2s = pick_col(df_laps, ["sector2sessiontime", "Sector2SessionTime"])
            col_s3s = pick_col(df_laps, ["sector3sessiontime", "Sector3SessionTime"])
            col_sp_i1 = pick_col(df_laps, ["speedi1", "SpeedI1"])
            col_sp_i2 = pick_col(df_laps, ["speedi2", "SpeedI2"])
            col_sp_fl = pick_col(df_laps, ["speedfl", "SpeedFL"])
            col_sp_st = pick_col(df_laps, ["speedst", "SpeedST"])
            col_trackstatus = pick_col(df_laps, ["trackstatus", "TrackStatus"])
            col_pb = pick_col(df_laps, ["ispersonalbest", "IsPersonalBest"])
            col_acc = pick_col(df_laps, ["isaccurate", "IsAccurate"])
            col_deleted = pick_col(df_laps, ["deleted", "Deleted"])
            col_del_reason = pick_col(df_laps, ["deletedreason", "DeletedReason"])
            col_pitin = pick_col(df_laps, ["pitintime", "PitInTime"])
            

            if col_lapnumber is None or col_position is None:
                msg = f"laps.parquet missing lapnumber/position columns. Columns={list(df_laps.columns)}"
                logger.warning("  " + msg)
                anomalies.append({
                    "issue": "laps_missing_key_columns",
                    "race_id": race_id,
                    "season": season,
                    "location": location,
                    "note": msg
                })
                return

            # --- Update drivers.carno from laps.parquet (batch) ---
            if has_driver_info_columns:
                driver_updates: List[Tuple[int, int]] = []  # (carno, driver_id)
                unique_drivers_in_laps = df_laps[[col_driver_no, col_driver]].drop_duplicates()

                for _, driver_info_row in unique_drivers_in_laps.iterrows():
                    driver_number_from_parquet = driver_info_row.get(col_driver_no)
                    driver_initials_from_parquet = driver_info_row.get(col_driver)

                    if pd.isna(driver_number_from_parquet) or pd.isna(driver_initials_from_parquet):
                        continue

                    try:
                        new_carno = int(driver_number_from_parquet)
                    except Exception:
                        continue

                    driver_id_to_update = None
                    if driver_initials_from_parquet in driver_id_by_initials:
                        driver_id_to_update = driver_id_by_initials[driver_initials_from_parquet]
                    elif driver_initials_from_parquet in driver_id_by_name:
                        driver_id_to_update = driver_id_by_name[driver_initials_from_parquet]

                    if driver_id_to_update is not None:
                        cur = output_conn.cursor()
                        cur.execute("SELECT carno FROM drivers WHERE id = ?", (driver_id_to_update,))
                        row = cur.fetchone()
                        existing_carno = row[0] if row else None
                        if existing_carno is None or existing_carno != new_carno:
                            driver_updates.append((new_carno, driver_id_to_update))
                            driver_id_by_carno[new_carno] = driver_id_to_update
                    else:
                        logger.warning(
                            f"    Driver {driver_initials_from_parquet} (Number {new_carno}) "
                            f"not found in drivers table by initials or name. Cannot update carno."
                        )

                if driver_updates:
                    output_conn.executemany("UPDATE drivers SET carno = ? WHERE id = ?", driver_updates)
                    output_conn.commit()
                    logger.info(f"    Updated carno for {len(driver_updates)} drivers from laps.parquet.")
            else:
                logger.warning(
                    f"    Could not find driver/drivernumber columns in laps.parquet for {season} {location}. "
                    f"Skipping carno updates for this race."
                )

            # ------------------------------------------------------------
            # Pit duration FIX:
            # FastF1 PitInTime is on lap L (enter pit at end of lap).
            # PitOutTime is on lap L+1 (exit pit and start next lap).
            # So for a stop on lap L:
            #   pitstopduration(L) = PitOutTime(L+1) - PitInTime(L)
            # ------------------------------------------------------------
            df_laps_work = df_laps.copy()

            # Precompute numeric pit times (ms)
            if col_pitin is not None:
                df_laps_work["_pitintimenum_ms"] = df_laps_work[col_pitin].apply(time_to_ms)
            else:
                df_laps_work["_pitintimenum_ms"] = None

            if col_pitout is not None:
                df_laps_work["_pitouttimenum_ms"] = df_laps_work[col_pitout].apply(time_to_ms)
            else:
                df_laps_work["_pitouttimenum_ms"] = None

            # Choose grouping key
            group_key = None
            if col_driver_no is not None:
                group_key = col_driver_no
            elif col_driver is not None:
                group_key = col_driver

            # Sort so shift(-1) aligns to next lap for each driver
            sort_cols = [col_lapnumber]
            if group_key is not None:
                sort_cols = [group_key, col_lapnumber]

            df_laps_work = df_laps_work.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

            # Compute next-lap pitout for each driver
            if group_key is not None:
                df_laps_work["_pitouttimenum_next_ms"] = df_laps_work.groupby(group_key)["_pitouttimenum_ms"].shift(-1)
            else:
                df_laps_work["_pitouttimenum_next_ms"] = df_laps_work["_pitouttimenum_ms"].shift(-1)

            # Compute duration (seconds), aligned to PitIn lap
            df_laps_work["_pitstopduration_s"] = None
            mask = df_laps_work["_pitintimenum_ms"].notna() & df_laps_work["_pitouttimenum_next_ms"].notna()
            df_laps_work.loc[mask, "_pitstopduration_s"] = (
                (df_laps_work.loc[mask, "_pitouttimenum_next_ms"] - df_laps_work.loc[mask, "_pitintimenum_ms"]) / 1000.0
            )

            # Log anomalies: missing next pitout, negative, huge
            # (We resolve driver_id in the row loop; here we log with whatever id we can.)
            if mask.any():
                bad_mask = mask & (
                    (df_laps_work["_pitstopduration_s"] < 0) |
                    (df_laps_work["_pitstopduration_s"] > 180)
                )
                for _, r in df_laps_work.loc[bad_mask].iterrows():
                    anomalies.append({
                        "issue": "pitstopduration_out_of_range",
                        "race_id": race_id,
                        "season": season,
                        "location": location,
                        "lap": r[col_lapnumber],
                        "driver_key": r[group_key] if group_key is not None else None,
                        "pitintimenum_ms": r["_pitintimenum_ms"],
                        "pitouttimenum_next_ms": r["_pitouttimenum_next_ms"],
                        "pitstopduration_s": r["_pitstopduration_s"],
                    })

            missing_next = df_laps_work["_pitintimenum_ms"].notna() & df_laps_work["_pitouttimenum_next_ms"].isna()
            if missing_next.any():
                # only record a few per race (otherwise too spammy), but keep it complete by writing all rows:
                for _, r in df_laps_work.loc[bad_mask].iterrows():
                    anomalies.append({
                        "issue": "missing_pitout_next_for_pitin",
                        "race_id": race_id,
                        "season": season,
                        "location": location,
                        "lap": getattr(r, col_lapnumber),
                        "driver_key": getattr(r, group_key) if group_key is not None else None,
                        "pitintimenum_ms": getattr(r, "_pitintimenum_ms"),
                    })

            # --- Batch lap updates ---
            update_rows: List[Tuple[Any, ...]] = []

            for _, row in df_laps_work.iterrows():
                lap_number = get_val(row, col_lapnumber)
                position = get_val(row, col_position)

                if pd.isna(lap_number) or pd.isna(position) or lap_number is None or position is None:
                    continue

                try:
                    lap_number_i = int(lap_number)
                    position_i = int(position)
                except Exception:
                    continue

                driver_id = None
                if has_driver_info_columns:
                    driver_number = get_val(row, col_driver_no)
                    if pd.notna(driver_number) and driver_number is not None:
                        try:
                            driver_id = driver_id_by_carno.get(int(driver_number))
                        except Exception:
                            driver_id = None

                if driver_id is None:
                    driver_initials = get_val(row, col_driver)
                    if pd.notna(driver_initials) and driver_initials is not None:
                        if driver_initials in driver_id_by_initials:
                            driver_id = driver_id_by_initials[driver_initials]
                        elif driver_initials in driver_id_by_name:
                            driver_id = driver_id_by_name[driver_initials]

                if driver_id is None:
                    logger.warning(f"    Could not determine driver_id for race {race_id}, lap {lap_number_i}. Skipping.")
                    anomalies.append({
                        "issue": "driver_id_unresolved",
                        "race_id": race_id,
                        "season": season,
                        "location": location,
                        "lap": lap_number_i,
                        "position": position_i,
                        "driver_no": get_val(row, col_driver_no) if col_driver_no else None,
                        "driver": get_val(row, col_driver) if col_driver else None
                    })
                    continue

                # Sector times (seconds)
                sector1time = td_to_s(get_val(row, col_s1))
                sector2time = td_to_s(get_val(row, col_s2))
                sector3time = td_to_s(get_val(row, col_s3))

                # Sector session times (ms since session start)
                sector1session_ms = time_to_ms(get_val(row, col_s1s))
                sector2session_ms = time_to_ms(get_val(row, col_s2s))
                sector3session_ms = time_to_ms(get_val(row, col_s3s))

                # Speeds
                speed_i1_kph = get_val(row, col_sp_i1)
                speed_i2_kph = get_val(row, col_sp_i2)
                speed_fl_kph = get_val(row, col_sp_fl)
                speed_st_kph = get_val(row, col_sp_st)

                track_status_code = None
                ts_val = get_val(row, col_trackstatus)
                if pd.notna(ts_val) and ts_val is not None:
                    track_status_code = str(ts_val)

                pb_val = get_val(row, col_pb)
                acc_val = get_val(row, col_acc)
                del_val = get_val(row, col_deleted)
                del_reason = get_val(row, col_del_reason)

                is_personal_best = 1 if bool(pb_val) else 0
                isaccurate = 1 if bool(acc_val) else 0
                is_deleted = 1 if bool(del_val) else 0
                deleted_reason = del_reason if pd.notna(del_reason) else None

                pit_in_td = get_val(row, col_pitin)
                pit_out_td = get_val(row, col_pitout)

                pitintime_str = str(pit_in_td) if pd.notna(pit_in_td) else None
                pitouttime_str = str(pit_out_td) if pd.notna(pit_out_td) else None

                pitintimenum = row.get("_pitintimenum_ms")
                pitouttimenum = row.get("_pitouttimenum_ms")

                pitintime_s = (pitintimenum / 1000.0) if pitintimenum is not None and pd.notna(pitintimenum) else None
                pitouttime_s = (pitouttimenum / 1000.0) if pitouttimenum is not None and pd.notna(pitouttimenum) else None

                # IMPORTANT: duration aligned to pit-in lap, using next lap pit-out
                pitstopduration = row.get("_pitstopduration_s")
                if pitstopduration is not None and pd.notna(pitstopduration):
                    # Optional extra sanity logging
                    if pitstopduration < 0:
                        anomalies.append({
                            "issue": "negative_pitstopduration_after_alignment",
                            "race_id": race_id,
                            "season": season,
                            "location": location,
                            "driver_id": driver_id,
                            "lap": lap_number_i,
                            "pitintimenum_ms": pitintimenum,
                            "pitouttimenum_next_ms": row.get("_pitouttimenum_next_ms"),
                            "pitstopduration_s": pitstopduration
                        })
                else:
                    pitstopduration = None

                update_rows.append((
                    sector1time, sector2time, sector3time,
                    sector1session_ms, sector2session_ms, sector3session_ms,
                    speed_i1_kph, speed_i2_kph, speed_fl_kph, speed_st_kph,
                    track_status_code, is_personal_best, isaccurate,
                    is_deleted, deleted_reason,
                    pitintime_str, pitouttime_str,
                    pitintimenum, pitouttimenum,
                    pitintime_s, pitouttime_s,
                    pitstopduration, position_i,
                    race_id, driver_id, lap_number_i
                ))

            if update_rows:
                cur = output_conn.cursor()
                before = output_conn.total_changes
                cur.executemany("""
                    UPDATE laps
                    SET
                        sector1time = ?, sector2time = ?, sector3time = ?,
                        sector1session_ms = ?, sector2session_ms = ?, sector3session_ms = ?,
                        speed_i1_kph = ?, speed_i2_kph = ?, speed_fl_kph = ?, speed_st_kph = ?,
                        track_status_code = ?, is_personal_best = ?, isaccurate = ?,
                        is_deleted = ?, deleted_reason = ?,
                        pitintime = ?, pitouttime = ?,
                        pitintimenum = ?, pitouttimenum = ?,
                        pitintime_s = ?, pitouttime_s = ?,
                        pitstopduration = ?, position = ?
                    WHERE race_id = ? AND driver_id = ? AND lapno = ?
                """, update_rows)
                output_conn.commit()
                updated_laps_count = output_conn.total_changes - before

            logger.info(f"  Updated {updated_laps_count} laps with FastF1 data.")

        except Exception as e:
            logger.error(f"  Error processing laps.parquet for {season} {location}: {e}")
            record_missing_input(
                race_id=race_id,
                season=season,
                location=location,
                kind="laps",
                looked_for=["laps.parquet"],
                race_dir=race_data_path,
                note=f"Found but failed to read/update: {type(e).__name__}: {e}"
            )
    else:
        logger.info(f"  laps.parquet not found for {season} {location}. Skipping laps update.")
        record_missing_input(
            race_id=race_id,
            season=season,
            location=location,
            kind="laps",
            looked_for=["laps.parquet"],
            race_dir=race_data_path,
            note="File missing"
        )

    # --- Update races table: availablecompounds_c ---
    current_available_compounds = race_row["availablecompounds"]
    if current_available_compounds:
        compounds_list = [c.strip() for c in current_available_compounds.split(",") if c.strip()]

        converted_to_a = []
        has_c_compounds = False
        for comp in compounds_list:
            if comp in C_TO_A_MAPPING:
                converted_to_a.append(C_TO_A_MAPPING[comp])
                has_c_compounds = True
            else:
                converted_to_a.append(comp)

        if has_c_compounds:
            new_available_compounds = ",".join(sorted(list(set(converted_to_a))))
            output_conn.execute(
                "UPDATE races SET availablecompounds = ? WHERE id = ?",
                (new_available_compounds, race_id),
            )
            logger.info(
                f"  Converted availablecompounds for race {race_id} from C-range to A-range: "
                f"{current_available_compounds} -> {new_available_compounds}"
            )
            current_available_compounds = new_available_compounds

        derived_c_compounds = []
        for comp_a in current_available_compounds.split(","):
            comp_a = comp_a.strip()
            if comp_a in A_TO_C_MAPPING:
                derived_c_compounds.append(A_TO_C_MAPPING[comp_a])

        if derived_c_compounds:
            new_available_compounds_c = ",".join(sorted(list(set(derived_c_compounds))))
            output_conn.execute(
                "UPDATE races SET availablecompounds_c = ? WHERE id = ?",
                (new_available_compounds_c, race_id),
            )
            logger.info(f"  Derived availablecompounds_c for race {race_id}: {new_available_compounds_c}")

        output_conn.commit()

    # At end of race processing: record unused parquet files in that race folder
    record_unused_parquets_for_race(race_id, season, location, race_data_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extend VSE-style SQLite DB with FastF1 Race-session data + cleaning."
    )
    parser.add_argument("--input-db", required=True, help="Path to the existing fastf1_vse_style.sqlite DB.")
    parser.add_argument("--output-db", required=True, help="Path for the NEW fastf1_vse_plus.sqlite DB.")
    parser.add_argument("--parquet-root", required=True, help="Root directory for FastF1 parquet data (e.g., f1_data).")

    args = parser.parse_args()

    input_db_path = os.path.abspath(args.input_db)
    output_db_path = os.path.abspath(args.output_db)
    parquet_root = os.path.abspath(args.parquet_root)

    # Step 0: Copy existing DB
    copy_schema_and_data(input_db_path, output_db_path)

    output_conn = get_db_connection(output_db_path)

    # Step 1: Apply schema changes
    apply_schema_changes(output_conn)

    # Step 2: Ingest Race-session parquet files
    race_map, driver_id_by_carno, driver_id_by_initials, driver_id_by_name = get_race_driver_mappings(output_conn)

    cursor = output_conn.cursor()
    cursor.execute("SELECT id, season, location, availablecompounds FROM races")
    all_races = cursor.fetchall()

    processed_races_count = 0
    for race_row in all_races:
        process_race_event(
            output_conn,
            race_row,
            race_map,
            driver_id_by_carno,
            driver_id_by_initials,
            driver_id_by_name,
            parquet_root,
        )
        processed_races_count += 1

    # Step 4: Validation report
    logger.info("\n--- Validation Report ---")
    logger.info(f"Total races processed: {processed_races_count}")

    cursor.execute("SELECT COUNT(*) FROM sessions")
    sessions_count = cursor.fetchone()[0]
    logger.info(f"Sessions rows count: {sessions_count}")

    cursor.execute("SELECT COUNT(*) FROM weather_samples")
    weather_samples_count = cursor.fetchone()[0]
    logger.info(f"Weather samples count: {weather_samples_count}")

    cursor.execute("SELECT COUNT(*) FROM track_status_events")
    track_status_events_count = cursor.fetchone()[0]
    logger.info(f"Track status events count: {track_status_events_count}")

    cursor.execute("SELECT COUNT(*) FROM race_control_messages")
    race_control_messages_count = cursor.fetchone()[0]
    logger.info(f"Race control messages count: {race_control_messages_count}")

    cursor.execute("SELECT CAST(SUM(wet_race) AS REAL) * 100 / COUNT(*) FROM races")
    wet_race_percentage = cursor.fetchone()[0]
    logger.info(f"Percentage of wet races: {wet_race_percentage:.2f}%")

    if table_exists(output_conn, "laps"):
        cursor.execute("SELECT COUNT(*) FROM laps WHERE sector1time IS NOT NULL OR speed_i1_kph IS NOT NULL")
        laps_updated_count = cursor.fetchone()[0]
        logger.info(f"Number of laps updated with sector times/speeds: {laps_updated_count}")

        cursor.execute("SELECT COUNT(*) FROM laps WHERE pitstopduration IS NOT NULL")
        pitdur_count = cursor.fetchone()[0]
        logger.info(f"Number of laps with computed pitstopduration: {pitdur_count}")

    # Step 5: Write audit CSVs
    try:
        pd.DataFrame(missing_inputs).to_csv(REPORT_DIR / "missing_inputs.csv", index=False)
        pd.DataFrame(unused_inputs).to_csv(REPORT_DIR / "unused_inputs.csv", index=False)
        pd.DataFrame(anomalies).to_csv(REPORT_DIR / "value_anomalies.csv", index=False)
        export_audit_csvs(output_db_path)
        logger.info(f"Wrote audit CSVs to: {REPORT_DIR.resolve()}")
    except Exception as e:
        logger.error(f"Failed to write audit CSVs: {type(e).__name__}: {e}")

    output_conn.close()
    logger.info("Script finished successfully.")


if __name__ == "__main__":
    main()