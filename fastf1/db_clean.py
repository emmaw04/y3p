#!/usr/bin/env python3
"""
db_clean.py

SQLite cleaning pipeline for the F1 strategy database.

This script now produces TWO cleaned database files from one input DB:

A) LESS-CLEAN VERSION (applies ONLY):
   1) Remove ALL laps for (race_id, driver_id) where the race is DRY and the driver
      did NOT complete 1-3 pit stops (outside inclusive range [1..3]).
      DRY definition: race session has weather_samples coverage (COUNT>0) and
      there does NOT exist any weather_samples row with COALESCE(rainfall, 0) != 0.

   2) Remove ALL laps for (race_id, driver_id) where the driver finished later than 15
      (i.e., starterfields.resultposition > 15), including NULL/invalid resultposition.

B) CLEANER VERSION (your current full pipeline):
   - Step 1: Remove ALL laps for DRY races where driver has > N pit stops.
   - Step 2: Remove pit-stop laps in the last fraction of race when no rain in lookback window.
   - Step X: Remove ALL laps for non-top-10 finishers.
   - Red-flag lap removals.
   - Extreme lap/pitstopduration pair removals.

Usage:
  Dry run (no DB files created, only reports):
    python db_clean.py /path/to/db.sqlite --dry-run

  Apply (creates two output DB files):
    python db_clean.py /path/to/db.sqlite --apply

  Optional:
    --out-dir /path/to/output_directory
    --max-pitstops 3
    --verbose
    --sanity-report
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sqlite3
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple


# -------------------------
# Logging
# -------------------------

def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# -------------------------
# DB helpers
# -------------------------

def connect_sqlite(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # Safety + integrity
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")   # better safety/perf for writes
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn

def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1;",
        (table,),
    ).fetchone()
    return row is not None

def require_tables(conn: sqlite3.Connection, tables: Iterable[str]) -> None:
    missing = [t for t in tables if not table_exists(conn, t)]
    if missing:
        raise RuntimeError(f"Missing required tables: {missing}")

# -------------------------
# Reporting helper
# -------------------------

def sanity_report_deleted_pairs(
    conn: sqlite3.Connection,
    bad_pairs: list["BadPair"],
) -> list[sqlite3.Row]:
    """
    Returns rows with race metadata + driver name for each affected (race_id, driver_id).
    """
    if not bad_pairs:
        return []

    conn.execute("DROP TABLE IF EXISTS tmp_bad_pairs;")
    conn.execute("""
        CREATE TEMP TABLE tmp_bad_pairs (
            race_id INTEGER,
            driver_id INTEGER,
            pit_stops INTEGER,
            PRIMARY KEY (race_id, driver_id)
        );
    """)

    conn.executemany(
        "INSERT OR REPLACE INTO tmp_bad_pairs(race_id, driver_id, pit_stops) VALUES (?, ?, ?);",
        [(bp.race_id, bp.driver_id, bp.pit_stops) for bp in bad_pairs],
    )

    sql = """
    SELECT
        bp.race_id,
        r.season,
        r.date AS race_date,
        r.location,
        bp.driver_id,
        d.name AS driver_name,
        bp.pit_stops,
        (
            SELECT COUNT(*)
            FROM laps l
            WHERE l.race_id = bp.race_id
              AND l.driver_id = bp.driver_id
        ) AS laps_rows_affected
    FROM tmp_bad_pairs bp
    JOIN races r ON r.id = bp.race_id
    JOIN drivers d ON d.id = bp.driver_id
    ORDER BY r.season, r.date, r.location, d.name;
    """
    return conn.execute(sql).fetchall()


# -------------------------
# Deletion primitives
# -------------------------

def count_laps_for_pairs(conn: sqlite3.Connection, pairs: Iterable[Tuple[int, int]]) -> int:
    """
    Count how many laps rows would be removed for (race_id, driver_id) pairs.
    """
    pairs = list(pairs)
    if not pairs:
        return 0

    conn.execute("DROP TABLE IF EXISTS tmp_bad_pairs;")
    conn.execute("CREATE TEMP TABLE tmp_bad_pairs (race_id INTEGER, driver_id INTEGER, PRIMARY KEY (race_id, driver_id));")
    conn.executemany("INSERT OR IGNORE INTO tmp_bad_pairs(race_id, driver_id) VALUES (?, ?);", pairs)

    row = conn.execute(
        """
        SELECT COUNT(*) AS n
        FROM laps l
        WHERE EXISTS (
            SELECT 1
            FROM tmp_bad_pairs bp
            WHERE bp.race_id = l.race_id
              AND bp.driver_id = l.driver_id
        );
        """
    ).fetchone()
    return int(row["n"])

def delete_laps_for_pairs(conn: sqlite3.Connection, pairs: Iterable[Tuple[int, int]]) -> int:
    """
    Delete all laps rows for the specified (race_id, driver_id) pairs.
    Returns the number of deleted rows.
    """
    pairs = list(pairs)
    if not pairs:
        return 0

    conn.execute("DROP TABLE IF EXISTS tmp_bad_pairs;")
    conn.execute("CREATE TEMP TABLE tmp_bad_pairs (race_id INTEGER, driver_id INTEGER, PRIMARY KEY (race_id, driver_id));")
    conn.executemany("INSERT OR IGNORE INTO tmp_bad_pairs(race_id, driver_id) VALUES (?, ?);", pairs)

    before = conn.total_changes
    conn.execute(
        """
        DELETE FROM laps
        WHERE EXISTS (
            SELECT 1
            FROM tmp_bad_pairs bp
            WHERE bp.race_id = laps.race_id
              AND bp.driver_id = laps.driver_id
        );
        """
    )
    conn.commit()

    deleted = conn.total_changes - before
    return deleted


# -------------------------
# Step 1 (existing): dry race + > max pitstops
# -------------------------

@dataclass(frozen=True)
class BadPair:
    race_id: int
    driver_id: int
    pit_stops: int
    session_id: int


def find_bad_driver_race_pairs_over_pitstops(
    conn: sqlite3.Connection,
    max_pitstops: int,
) -> list[BadPair]:
    """
    Identify driver-race pairs in dry races where the driver has > max_pitstops pit stops.

    Pit stop indicator (laps row):
      l.pitintime IS NOT NULL AND non-empty
    """
    sql = """
    WITH race_sessions AS (
        SELECT s.id AS session_id, s.race_id
        FROM sessions s
        WHERE s.session_code = 'R'
          AND COALESCE(s.is_official, 1) = 1
    ),
    dry_races AS (
        SELECT rs.race_id, rs.session_id
        FROM race_sessions rs
        WHERE EXISTS (
            SELECT 1
            FROM weather_samples ws
            WHERE ws.session_id = rs.session_id
        )
        AND NOT EXISTS (
            SELECT 1
            FROM weather_samples ws
            WHERE ws.session_id = rs.session_id
              AND COALESCE(ws.rainfall, 0) != 0
        )
    ),
    driver_pit_counts AS (
        SELECT
            l.race_id,
            l.driver_id,
            SUM(
                CASE
                    WHEN l.pitintime IS NOT NULL
                        AND TRIM(l.pitintime) NOT IN ('', 'NaT', 'nan', 'None')
                    THEN 1 ELSE 0
                END
            ) AS pit_stops
        FROM laps l
        GROUP BY l.race_id, l.driver_id
    )
    SELECT
        dpc.race_id,
        dpc.driver_id,
        dpc.pit_stops,
        dr.session_id
    FROM driver_pit_counts dpc
    JOIN dry_races dr
      ON dr.race_id = dpc.race_id
    WHERE dpc.pit_stops > ?
    ORDER BY dpc.race_id, dpc.driver_id;
    """
    rows = conn.execute(sql, (max_pitstops,)).fetchall()
    return [BadPair(int(r["race_id"]), int(r["driver_id"]), int(r["pit_stops"]), int(r["session_id"])) for r in rows]


def run_step1_over_max_pitstops(
    conn: sqlite3.Connection,
    apply: bool,
    max_pitstops: int,
    sanity_report: bool,
) -> int:
    require_tables(conn, ["sessions", "weather_samples", "laps", "races", "drivers"])

    logging.info("Step 1: Remove laps for driver-race pairs in dry races with > %d pit stops", max_pitstops)

    bad_pairs = find_bad_driver_race_pairs_over_pitstops(conn, max_pitstops=max_pitstops)
    if not bad_pairs:
        logging.info("Step 1: No driver-race pairs found. Nothing to delete.")
        return 0

    pair_keys = [(bp.race_id, bp.driver_id) for bp in bad_pairs]
    would_delete = count_laps_for_pairs(conn, pair_keys)
    logging.info("Step 1: Identified %d driver-race pairs. Laps rows affected: %d", len(bad_pairs), would_delete)

    if sanity_report:
        rows = sanity_report_deleted_pairs(conn, bad_pairs)
        print("\n=== SANITY REPORT: driver–race deletions (Step 1) ===")
        for r in rows:
            print(
                f"Race {r['race_id']} | {r['season']} {r['location']} ({r['race_date']}) "
                f"| Driver {r['driver_id']} {r['driver_name']} "
                f"| pit_stops={r['pit_stops']} | laps_rows_affected={r['laps_rows_affected']}"
            )
        print("=== END SANITY REPORT ===\n")

    if not apply:
        logging.info("Step 1 dry-run: no changes applied.")
        return 0

    deleted = delete_laps_for_pairs(conn, pair_keys)
    logging.info("Step 1 applied: deleted %d laps rows.", deleted)
    return deleted


# -------------------------
# LESS-CLEAN STEP L1: dry race + pitstops outside [min..max]
# -------------------------

def find_driver_race_pairs_dry_pitstops_outside_range(
    conn: sqlite3.Connection,
    min_pitstops: int = 1,
    max_pitstops: int = 3,
) -> list[BadPair]:
    """
    Identify driver-race pairs in DRY races where pit_stops NOT in [min_pitstops..max_pitstops] (inclusive).

    DRY definition:
      - weather_samples coverage exists for the race session (COUNT > 0)
      - AND no weather_samples row has COALESCE(rainfall, 0) != 0
    """
    if min_pitstops > max_pitstops:
        raise ValueError("min_pitstops must be <= max_pitstops")

    sql = """
    WITH race_sessions AS (
        SELECT s.id AS session_id, s.race_id
        FROM sessions s
        WHERE s.session_code = 'R'
          AND COALESCE(s.is_official, 1) = 1
    ),
    dry_races AS (
        SELECT rs.race_id, rs.session_id
        FROM race_sessions rs
        WHERE EXISTS (
            SELECT 1
            FROM weather_samples ws
            WHERE ws.session_id = rs.session_id
        )
        AND NOT EXISTS (
            SELECT 1
            FROM weather_samples ws
            WHERE ws.session_id = rs.session_id
              AND COALESCE(ws.rainfall, 0) != 0
        )
    ),
    driver_pit_counts AS (
        SELECT
            l.race_id,
            l.driver_id,
            SUM(
                CASE
                    WHEN l.pitintime IS NOT NULL
                        AND TRIM(l.pitintime) NOT IN ('', 'NaT', 'nan', 'None')
                    THEN 1 ELSE 0
                END
            ) AS pit_stops
        FROM laps l
        GROUP BY l.race_id, l.driver_id
    )
    SELECT
        dpc.race_id,
        dpc.driver_id,
        dpc.pit_stops,
        dr.session_id
    FROM driver_pit_counts dpc
    JOIN dry_races dr
      ON dr.race_id = dpc.race_id
    WHERE dpc.pit_stops < ?
       OR dpc.pit_stops > ?
    ORDER BY dpc.race_id, dpc.driver_id;
    """
    rows = conn.execute(sql, (min_pitstops, max_pitstops)).fetchall()
    return [BadPair(int(r["race_id"]), int(r["driver_id"]), int(r["pit_stops"]), int(r["session_id"])) for r in rows]


def remove_dry_pairs_pitstops_outside_range(
    conn: sqlite3.Connection,
    apply: bool,
    min_pitstops: int = 1,
    max_pitstops: int = 3,
    sanity_report: bool = False,
) -> int:
    require_tables(conn, ["sessions", "weather_samples", "laps", "races", "drivers"])

    logging.info(
        "Less-clean Step L1: Remove laps for dry-race driver-pairs with pit_stops outside [%d..%d].",
        min_pitstops, max_pitstops
    )

    bad_pairs = find_driver_race_pairs_dry_pitstops_outside_range(conn, min_pitstops, max_pitstops)
    if not bad_pairs:
        logging.info("Less-clean Step L1: No pairs found. Nothing to delete.")
        return 0

    pair_keys = [(bp.race_id, bp.driver_id) for bp in bad_pairs]
    would_delete = count_laps_for_pairs(conn, pair_keys)
    logging.info("Less-clean Step L1: Identified %d driver-race pairs. Laps rows affected: %d", len(bad_pairs), would_delete)

    if sanity_report:
        rows = sanity_report_deleted_pairs(conn, bad_pairs)
        print("\n=== SANITY REPORT: Less-clean L1 (dry + pitstops outside range) ===")
        for r in rows:
            print(
                f"Race {r['race_id']} | {r['season']} {r['location']} ({r['race_date']}) "
                f"| Driver {r['driver_id']} {r['driver_name']} "
                f"| pit_stops={r['pit_stops']} | laps_rows_affected={r['laps_rows_affected']}"
            )
        print("=== END SANITY REPORT ===\n")

    if not apply:
        logging.info("Less-clean Step L1 dry-run: no changes applied.")
        return 0

    deleted = delete_laps_for_pairs(conn, pair_keys)
    logging.info("Less-clean Step L1 applied: deleted %d laps rows.", deleted)
    return deleted


# -------------------------
# Existing steps you already added (kept as-is)
# -------------------------

import math


def remove_pitstop_laps_in_last_race_fraction(
    conn: sqlite3.Connection,
    apply: bool,
    tail_fraction: float = 0.10,              # last 10%
    rain_lookback_minutes: int = 5,           # "minutes leading up"
    sanity_report: bool = False,
) -> int:
    require_tables(conn, ["laps", "races", "sessions", "weather_samples", "drivers"])

    if not (0.0 < tail_fraction < 1.0):
        raise ValueError("tail_fraction must be between 0 and 1 (exclusive).")

    lookback_ms = int(rain_lookback_minutes * 60 * 1000)

    conn.execute("DROP TABLE IF EXISTS tmp_pit_tail_laps;")
    conn.execute("""
        CREATE TEMP TABLE tmp_pit_tail_laps (
            race_id INTEGER NOT NULL,
            driver_id INTEGER NOT NULL,
            lapno INTEGER NOT NULL,
            PRIMARY KEY (race_id, driver_id, lapno)
        );
    """)

    insert_sql = """
    INSERT OR IGNORE INTO tmp_pit_tail_laps (race_id, driver_id, lapno)
    SELECT
        l.race_id,
        l.driver_id,
        l.lapno
    FROM laps l
    JOIN races r
      ON r.id = l.race_id
    JOIN sessions s
      ON s.race_id = r.id
     AND s.session_code = 'R'
     AND COALESCE(s.is_official, 1) = 1
    WHERE
        -- Only apply to seasons 2019-2024
        r.season BETWEEN 2019 AND 2024

        -- Need race length
        AND r.nolapsplanned IS NOT NULL
        AND r.nolapsplanned > 0

        -- Need lap number
        AND l.lapno IS NOT NULL

        -- Pit stop occurred on this lap
        AND l.pitintime IS NOT NULL
        AND TRIM(l.pitintime) NOT IN ('', 'NaT', 'nan', 'None')

        -- We need a time anchor to check rain; use driver racetime (sec) -> ms
        AND l.racetime IS NOT NULL

        -- Only after (1-tail_fraction) race progression (lap-based)
        AND l.lapno >= (CAST(((1.0 - ?) * r.nolapsplanned) AS INTEGER) + 1)

        -- Require weather coverage in the lookback window (avoid deleting if weather missing)
        AND EXISTS (
            SELECT 1
            FROM weather_samples ws
            WHERE ws.session_id = s.id
              AND ws.time_ms BETWEEN CAST(l.racetime * 1000 AS INTEGER) - ? AND CAST(l.racetime * 1000 AS INTEGER)
        )

        -- Only delete if there was NO rain in the lookback window
        AND NOT EXISTS (
            SELECT 1
            FROM weather_samples ws
            WHERE ws.session_id = s.id
              AND ws.time_ms BETWEEN CAST(l.racetime * 1000 AS INTEGER) - ? AND CAST(l.racetime * 1000 AS INTEGER)
              AND COALESCE(ws.rainfall, 0) != 0
        )
    ;
    """

    conn.execute(insert_sql, (tail_fraction, lookback_ms, lookback_ms))

    would_delete = conn.execute("SELECT COUNT(*) AS n FROM tmp_pit_tail_laps;").fetchone()["n"]
    logging.info(
        "Step 2: seasons 2019-2024, pit-stop laps in last %.0f%% with NO rain in last %d min: %d rows matched.",
        tail_fraction * 100.0,
        rain_lookback_minutes,
        int(would_delete),
    )

    if sanity_report and would_delete:
        preview = conn.execute("""
            SELECT
              t.race_id,
              r.season,
              r.location,
              r.date AS race_date,
              t.driver_id,
              d.name AS driver_name,
              COUNT(*) AS n_laps_to_delete,
              MIN(t.lapno) AS first_deleted_lap,
              MAX(t.lapno) AS last_deleted_lap,
              r.nolapsplanned AS nolapsplanned
            FROM tmp_pit_tail_laps t
            JOIN races r ON r.id = t.race_id
            JOIN drivers d ON d.id = t.driver_id
            GROUP BY t.race_id, t.driver_id
            ORDER BY r.season, r.date, r.location, d.name
            LIMIT 200;
        """).fetchall()

        print("\n=== SANITY REPORT: Step 2 (2019-2024, last fraction, no-rain lookback) [top 200 groups] ===")
        for rr in preview:
            threshold = int((1.0 - tail_fraction) * rr["nolapsplanned"]) + 1
            print(
                f"Race {rr['race_id']} | {rr['season']} {rr['location']} ({rr['race_date']}) "
                f"| Driver {rr['driver_id']} {rr['driver_name']} "
                f"| deleted_laps={rr['n_laps_to_delete']} (lap {rr['first_deleted_lap']}..{rr['last_deleted_lap']}) "
                f"| nolapsplanned={rr['nolapsplanned']} | threshold_lap={threshold}"
            )
        print("=== END SANITY REPORT ===\n")

    if not apply:
        logging.info("Step 2 dry-run: no deletions applied.")
        return 0

    before = conn.total_changes
    conn.execute("""
        DELETE FROM laps
        WHERE EXISTS (
            SELECT 1
            FROM tmp_pit_tail_laps t
            WHERE t.race_id = laps.race_id
              AND t.driver_id = laps.driver_id
              AND t.lapno = laps.lapno
        );
    """)
    conn.commit()

    deleted = conn.total_changes - before
    logging.info("Step 2 applied: deleted %d laps rows.", deleted)
    return deleted


def remove_laps_for_non_top10_finishers(
    conn: sqlite3.Connection,
    apply: bool,
    sanity_report: bool = False,
    report_limit: int = 200,
) -> int:
    """
    Step X:
    Delete ALL laps for (race_id, driver_id) pairs where the driver did NOT finish in the top 10.
    """
    require_tables(conn, ["laps", "starterfields", "races", "drivers"])

    conn.execute("DROP TABLE IF EXISTS tmp_non_top10_pairs;")
    conn.execute("""
        CREATE TEMP TABLE tmp_non_top10_pairs (
            race_id INTEGER NOT NULL,
            driver_id INTEGER NOT NULL,
            resultposition INTEGER,
            PRIMARY KEY (race_id, driver_id)
        );
    """)

    insert_sql = """
    INSERT OR IGNORE INTO tmp_non_top10_pairs (race_id, driver_id, resultposition)
    SELECT
        sf.race_id,
        sf.driver_id,
        sf.resultposition
    FROM starterfields sf
    WHERE
        sf.resultposition IS NULL
        OR CAST(sf.resultposition AS INTEGER) < 1
        OR CAST(sf.resultposition AS INTEGER) > 10
    ;
    """
    conn.execute(insert_sql)

    n_pairs = conn.execute("SELECT COUNT(*) AS n FROM tmp_non_top10_pairs;").fetchone()["n"]
    n_laps = conn.execute("""
        SELECT COUNT(*) AS n
        FROM laps l
        WHERE EXISTS (
            SELECT 1 FROM tmp_non_top10_pairs t
            WHERE t.race_id = l.race_id AND t.driver_id = l.driver_id
        );
    """).fetchone()["n"]

    logging.info("Step X: Non-top-10 driver-race pairs: %d pairs; %d laps rows affected.", int(n_pairs), int(n_laps))

    if sanity_report and n_pairs:
        rows = conn.execute(f"""
            SELECT
                t.race_id,
                r.season,
                r.location,
                r.date AS race_date,
                t.driver_id,
                d.name AS driver_name,
                t.resultposition,
                (
                    SELECT COUNT(*)
                    FROM laps l
                    WHERE l.race_id = t.race_id AND l.driver_id = t.driver_id
                ) AS laps_rows_affected
            FROM tmp_non_top10_pairs t
            JOIN races r ON r.id = t.race_id
            JOIN drivers d ON d.id = t.driver_id
            ORDER BY r.season, r.date, r.location, d.name
            LIMIT ?;
        """, (report_limit,)).fetchall()

        print(f"\n=== SANITY REPORT: Step X (delete non-top-10 finishers) [top {report_limit} pairs] ===")
        for r in rows:
            print(
                f"Race {r['race_id']} | {r['season']} {r['location']} ({r['race_date']}) "
                f"| Driver {r['driver_id']} {r['driver_name']} "
                f"| resultposition={r['resultposition']} | laps_rows_affected={r['laps_rows_affected']}"
            )
        if n_pairs > report_limit:
            print(f"... ({int(n_pairs) - report_limit} more pairs not shown)")
        print("=== END SANITY REPORT ===\n")

    if not apply:
        logging.info("Step X dry-run: no deletions applied.")
        return 0

    before = conn.total_changes
    conn.execute("""
        DELETE FROM laps
        WHERE EXISTS (
            SELECT 1
            FROM tmp_non_top10_pairs t
            WHERE t.race_id = laps.race_id
              AND t.driver_id = laps.driver_id
        );
    """)
    conn.commit()

    deleted = conn.total_changes - before
    logging.info("Step X applied: deleted %d laps rows.", deleted)
    return deleted


def remove_red_flag_laps(
    conn: sqlite3.Connection,
    apply: bool,
    sanity_report: bool = False,
    report_limit: int = 200,
) -> int:
    """
    Remove all laps that occurred during a Red Flag.
    Heuristic: laps.track_status_code contains the digit '5'
    """
    require_tables(conn, ["laps", "races", "drivers"])

    conn.execute("DROP TABLE IF EXISTS tmp_red_flag_laps;")
    conn.execute("""
        CREATE TEMP TABLE tmp_red_flag_laps (
            race_id INTEGER NOT NULL,
            driver_id INTEGER NOT NULL,
            lapno INTEGER NOT NULL,
            track_status_code TEXT,
            PRIMARY KEY (race_id, driver_id, lapno)
        );
    """)

    conn.execute("""
        INSERT OR IGNORE INTO tmp_red_flag_laps (race_id, driver_id, lapno, track_status_code)
        SELECT
            l.race_id,
            l.driver_id,
            l.lapno,
            l.track_status_code
        FROM laps l
        WHERE
            l.track_status_code IS NOT NULL
            AND CAST(l.track_status_code AS TEXT) LIKE '%5%';
    """)

    n_laps = conn.execute("SELECT COUNT(*) AS n FROM tmp_red_flag_laps;").fetchone()["n"]
    logging.info("Red-flag filter: %d laps rows matched (track_status_code contains '5').", int(n_laps))

    if sanity_report and n_laps:
        rows = conn.execute(f"""
            SELECT
                t.race_id,
                r.season,
                r.location,
                r.date AS race_date,
                t.driver_id,
                d.name AS driver_name,
                COUNT(*) AS n_laps_to_delete,
                MIN(t.lapno) AS first_deleted_lap,
                MAX(t.lapno) AS last_deleted_lap
            FROM tmp_red_flag_laps t
            JOIN races r ON r.id = t.race_id
            JOIN drivers d ON d.id = t.driver_id
            GROUP BY t.race_id, t.driver_id
            ORDER BY r.season, r.date, r.location, d.name
            LIMIT ?;
        """, (report_limit,)).fetchall()

        print(f"\n=== SANITY REPORT: Red-flag lap deletions [top {report_limit} race-driver groups] ===")
        for r in rows:
            print(
                f"Race {r['race_id']} | {r['season']} {r['location']} ({r['race_date']}) "
                f"| Driver {r['driver_id']} {r['driver_name']} "
                f"| redflag_laps={r['n_laps_to_delete']} (lap {r['first_deleted_lap']}..{r['last_deleted_lap']})"
            )
        print("=== END SANITY REPORT ===\n")

    if not apply:
        logging.info("Red-flag filter dry-run: no deletions applied.")
        return 0

    before = conn.total_changes
    conn.execute("""
        DELETE FROM laps
        WHERE EXISTS (
            SELECT 1
            FROM tmp_red_flag_laps t
            WHERE t.race_id = laps.race_id
              AND t.driver_id = laps.driver_id
              AND t.lapno = laps.lapno
        );
    """)
    conn.commit()

    deleted = conn.total_changes - before
    logging.info("Red-flag filter applied: deleted %d laps rows.", deleted)
    return deleted


def remove_driver_race_pairs_with_extreme_laps(
    conn: sqlite3.Connection,
    apply: bool,
    laptime_threshold_s: float = 200.0,
    pitstop_threshold_s: float = 50.0,
    sanity_report: bool = False,
    report_limit: int = 200,
) -> int:
    """
    Remove ALL laps for driver-race pairs where the driver had:
      - any laptime > laptime_threshold_s, OR
      - any pitstopduration > pitstop_threshold_s
    """
    require_tables(conn, ["laps", "races", "drivers"])

    conn.execute("DROP TABLE IF EXISTS tmp_extreme_pairs;")
    conn.execute("""
        CREATE TEMP TABLE tmp_extreme_pairs (
            race_id INTEGER NOT NULL,
            driver_id INTEGER NOT NULL,
            max_laptime REAL,
            max_pitstopduration REAL,
            offending_laps_count INTEGER,
            PRIMARY KEY (race_id, driver_id)
        );
    """)

    conn.execute("""
        INSERT OR IGNORE INTO tmp_extreme_pairs (race_id, driver_id, max_laptime, max_pitstopduration, offending_laps_count)
        SELECT
            l.race_id,
            l.driver_id,
            MAX(CASE WHEN l.laptime IS NOT NULL THEN l.laptime ELSE NULL END) AS max_laptime,
            MAX(CASE WHEN l.pitstopduration IS NOT NULL THEN l.pitstopduration ELSE NULL END) AS max_pitstopduration,
            SUM(
                CASE
                    WHEN (l.laptime IS NOT NULL AND l.laptime > ?)
                      OR (l.pitstopduration IS NOT NULL AND l.pitstopduration > ?)
                    THEN 1 ELSE 0
                END
            ) AS offending_laps_count
        FROM laps l
        GROUP BY l.race_id, l.driver_id
        HAVING offending_laps_count > 0;
    """, (laptime_threshold_s, pitstop_threshold_s))

    n_pairs = conn.execute("SELECT COUNT(*) AS n FROM tmp_extreme_pairs;").fetchone()["n"]
    n_laps = conn.execute("""
        SELECT COUNT(*) AS n
        FROM laps l
        WHERE EXISTS (
            SELECT 1 FROM tmp_extreme_pairs t
            WHERE t.race_id = l.race_id AND t.driver_id = l.driver_id
        );
    """).fetchone()["n"]

    logging.info(
        "Extreme lap filter: %d driver-race pairs flagged; %d laps rows affected (laptime>%.1fs or pitstop>%.1fs).",
        int(n_pairs), int(n_laps), laptime_threshold_s, pitstop_threshold_s
    )

    if sanity_report and n_pairs:
        rows = conn.execute(f"""
            SELECT
                t.race_id,
                r.season,
                r.location,
                r.date AS race_date,
                t.driver_id,
                d.name AS driver_name,
                t.offending_laps_count,
                t.max_laptime,
                t.max_pitstopduration,
                (
                    SELECT COUNT(*)
                    FROM laps l
                    WHERE l.race_id = t.race_id AND l.driver_id = t.driver_id
                ) AS laps_rows_affected
            FROM tmp_extreme_pairs t
            JOIN races r ON r.id = t.race_id
            JOIN drivers d ON d.id = t.driver_id
            ORDER BY r.season, r.date, r.location, d.name
            LIMIT ?;
        """, (report_limit,)).fetchall()

        print(f"\n=== SANITY REPORT: Extreme lap deletions [top {report_limit} driver-race pairs] ===")
        for r in rows:
            print(
                f"Race {r['race_id']} | {r['season']} {r['location']} ({r['race_date']}) "
                f"| Driver {r['driver_id']} {r['driver_name']} "
                f"| offending_laps={r['offending_laps_count']} "
                f"| max_laptime={r['max_laptime']}s | max_pitstop={r['max_pitstopduration']}s "
                f"| laps_rows_affected={r['laps_rows_affected']}"
            )
        print("=== END SANITY REPORT ===\n")

    if not apply:
        logging.info("Extreme lap filter dry-run: no deletions applied.")
        return 0

    before = conn.total_changes
    conn.execute("""
        DELETE FROM laps
        WHERE EXISTS (
            SELECT 1
            FROM tmp_extreme_pairs t
            WHERE t.race_id = laps.race_id
              AND t.driver_id = laps.driver_id
        );
    """)
    conn.commit()

    deleted = conn.total_changes - before
    logging.info("Extreme lap filter applied: deleted %d laps rows.", deleted)
    return deleted


# -------------------------
# LESS-CLEAN STEP L2: remove finishers later than 15
# -------------------------

def remove_laps_for_finishers_later_than(
    conn: sqlite3.Connection,
    apply: bool,
    cutoff_position: int = 15,
    sanity_report: bool = False,
    report_limit: int = 200,
) -> int:
    """
    Less-clean Step L2:
    Delete ALL laps for (race_id, driver_id) pairs where resultposition is:
      - NULL, or
      - < 1, or
      - > cutoff_position

    For your requirement: cutoff_position = 15 (remove finishers later than 15).
    """
    require_tables(conn, ["laps", "starterfields", "races", "drivers"])

    conn.execute("DROP TABLE IF EXISTS tmp_late_finish_pairs;")
    conn.execute("""
        CREATE TEMP TABLE tmp_late_finish_pairs (
            race_id INTEGER NOT NULL,
            driver_id INTEGER NOT NULL,
            resultposition INTEGER,
            PRIMARY KEY (race_id, driver_id)
        );
    """)

    conn.execute("""
        INSERT OR IGNORE INTO tmp_late_finish_pairs (race_id, driver_id, resultposition)
        SELECT
            sf.race_id,
            sf.driver_id,
            sf.resultposition
        FROM starterfields sf
        WHERE
            sf.resultposition IS NULL
            OR CAST(sf.resultposition AS INTEGER) < 1
            OR CAST(sf.resultposition AS INTEGER) > ?
        ;
    """, (cutoff_position,))

    n_pairs = conn.execute("SELECT COUNT(*) AS n FROM tmp_late_finish_pairs;").fetchone()["n"]
    n_laps = conn.execute("""
        SELECT COUNT(*) AS n
        FROM laps l
        WHERE EXISTS (
            SELECT 1 FROM tmp_late_finish_pairs t
            WHERE t.race_id = l.race_id AND t.driver_id = l.driver_id
        );
    """).fetchone()["n"]

    logging.info(
        "Less-clean Step L2: Finish position > %d (or invalid/NULL): %d pairs; %d laps rows affected.",
        cutoff_position, int(n_pairs), int(n_laps)
    )

    if sanity_report and n_pairs:
        rows = conn.execute(f"""
            SELECT
                t.race_id,
                r.season,
                r.location,
                r.date AS race_date,
                t.driver_id,
                d.name AS driver_name,
                t.resultposition,
                (
                    SELECT COUNT(*)
                    FROM laps l
                    WHERE l.race_id = t.race_id AND l.driver_id = t.driver_id
                ) AS laps_rows_affected
            FROM tmp_late_finish_pairs t
            JOIN races r ON r.id = t.race_id
            JOIN drivers d ON d.id = t.driver_id
            ORDER BY r.season, r.date, r.location, d.name
            LIMIT ?;
        """, (report_limit,)).fetchall()

        print(f"\n=== SANITY REPORT: Less-clean L2 (delete finishers later than {cutoff_position}) [top {report_limit}] ===")
        for r in rows:
            print(
                f"Race {r['race_id']} | {r['season']} {r['location']} ({r['race_date']}) "
                f"| Driver {r['driver_id']} {r['driver_name']} "
                f"| resultposition={r['resultposition']} | laps_rows_affected={r['laps_rows_affected']}"
            )
        if int(n_pairs) > report_limit:
            print(f"... ({int(n_pairs) - report_limit} more pairs not shown)")
        print("=== END SANITY REPORT ===\n")

    if not apply:
        logging.info("Less-clean Step L2 dry-run: no deletions applied.")
        return 0

    before = conn.total_changes
    conn.execute("""
        DELETE FROM laps
        WHERE EXISTS (
            SELECT 1
            FROM tmp_late_finish_pairs t
            WHERE t.race_id = laps.race_id
              AND t.driver_id = laps.driver_id
        );
    """)
    conn.commit()

    deleted = conn.total_changes - before
    logging.info("Less-clean Step L2 applied: deleted %d laps rows.", deleted)
    return deleted


# -------------------------
# Pipelines
# -------------------------

def run_less_clean_pipeline(conn: sqlite3.Connection, apply: bool, sanity_report: bool) -> None:
    """
    Applies ONLY the two steps requested for the less-clean dataset:
      L1: dry race + pitstops outside [1..3]
      L2: finish position > 15
    """
    require_tables(conn, ["sessions", "weather_samples", "laps", "races", "drivers", "starterfields"])
    remove_dry_pairs_pitstops_outside_range(conn, apply=apply, min_pitstops=1, max_pitstops=3, sanity_report=sanity_report)
    remove_laps_for_finishers_later_than(conn, apply=apply, cutoff_position=15, sanity_report=sanity_report)


def run_cleaner_pipeline(conn: sqlite3.Connection, apply: bool, max_pitstops: int, sanity_report: bool) -> None:
    """
    Your current full pipeline (unchanged logic).
    """
    require_tables(conn, ["sessions", "weather_samples", "laps", "races", "drivers"])

    run_step1_over_max_pitstops(conn, apply=apply, max_pitstops=max_pitstops, sanity_report=sanity_report)
    remove_pitstop_laps_in_last_race_fraction(conn, apply=apply, tail_fraction=0.10, sanity_report=sanity_report)
    remove_laps_for_non_top10_finishers(conn, apply=apply, sanity_report=sanity_report)
    remove_red_flag_laps(conn, apply=apply, sanity_report=sanity_report)
    remove_driver_race_pairs_with_extreme_laps(conn, apply=apply, laptime_threshold_s=200.0, pitstop_threshold_s=50.0, sanity_report=sanity_report)


# -------------------------
# Output DB creation
# -------------------------

def derive_output_paths(input_db: str, out_dir: Optional[str]) -> tuple[str, str]:
    in_abs = os.path.abspath(input_db)
    base_dir = out_dir if out_dir is not None else os.path.dirname(in_abs)
    os.makedirs(base_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(in_abs))[0]
    less_clean_path = os.path.join(base_dir, f"{stem}__less_clean.sqlite")
    cleaner_path = os.path.join(base_dir, f"{stem}__clean.sqlite")
    return less_clean_path, cleaner_path


def copy_db(src: str, dst: str) -> None:
    if os.path.abspath(src) == os.path.abspath(dst):
        raise ValueError("Refusing to overwrite the input DB in-place. Choose a different --out-dir.")
    shutil.copy2(src, dst)


# -------------------------
# CLI
# -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create two cleaned variants of an F1 strategy SQLite database.")
    p.add_argument("db_path", help="Path to the input SQLite .db/.sqlite file")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--apply", action="store_true", help="Create two cleaned DB copies and apply deletions to each")
    mode.add_argument("--dry-run", action="store_true", help="Show what would change without creating/modifying any DB files")
    p.add_argument("--out-dir", default=None, help="Directory to write output DBs (default: alongside input DB)")
    p.add_argument("--max-pitstops", type=int, default=3, help="Max allowed pit stops in dry races for the CLEANER DB step 1 (default: 3)")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    p.add_argument("--sanity-report", action="store_true", help="Print per-pair grouped reports for affected deletions")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    input_db = args.db_path

    if args.dry_run:
        logging.info("DRY-RUN on input DB (no files created): %s", input_db)
        conn = connect_sqlite(input_db)
        try:
            logging.info("=== Less-clean pipeline (dry-run) ===")
            run_less_clean_pipeline(conn, apply=False, sanity_report=args.sanity_report)

            logging.info("=== Cleaner pipeline (dry-run) ===")
            run_cleaner_pipeline(conn, apply=False, max_pitstops=args.max_pitstops, sanity_report=args.sanity_report)
        finally:
            conn.close()
            logging.info("Done (dry-run).")
        return

    # APPLY: create two DB copies, then apply the respective pipelines to each copy.
    less_clean_path, cleaner_path = derive_output_paths(input_db, args.out_dir)

    logging.info("Creating LESS-CLEAN DB: %s", less_clean_path)
    copy_db(input_db, less_clean_path)

    logging.info("Creating CLEANER DB: %s", cleaner_path)
    copy_db(input_db, cleaner_path)

    # Apply less-clean pipeline
    conn_less = connect_sqlite(less_clean_path)
    try:
        logging.info("Applying less-clean pipeline to: %s", less_clean_path)
        with conn_less:
            run_less_clean_pipeline(conn_less, apply=True, sanity_report=args.sanity_report)
    finally:
        conn_less.close()

    # Apply cleaner pipeline
    conn_clean = connect_sqlite(cleaner_path)
    try:
        logging.info("Applying cleaner pipeline to: %s", cleaner_path)
        with conn_clean:
            run_cleaner_pipeline(conn_clean, apply=True, max_pitstops=args.max_pitstops, sanity_report=args.sanity_report)
    finally:
        conn_clean.close()

    logging.info("Done. Output files:\n- %s\n- %s", less_clean_path, cleaner_path)


if __name__ == "__main__":
    main()