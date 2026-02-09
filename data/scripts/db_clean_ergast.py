#!/usr/bin/env python3
"""
db_clean_minimal_two_tiers.py

Creates TWO cleaned variants of a reduced-schema F1 strategy SQLite database.

ASSUMED TABLES (reduced schema):
- drivers(id, carno, initials, name)
- races(id, date, season, location, availablecompounds, comment, nolaps, nolapsplanned, tracklength)
- laps(race_id, lapno, position, driver_id, laptime, racetime, gap, interval, compound, tireage,
      pitintime, pitstopduration, nextcompound, startlapprog_vsc, endlapprog_vsc, age_vsc,
      startlapprog_sc, endlapprog_sc, age_sc)
- starterfields(race_id, driver_id, ..., resultposition, ...)
- fcyphases(race_id, startracetime, endracetime, startlap, endlap, type)

OUTPUTS (from one input DB):
A) LESS-CLEAN DB:
   L1) Remove ALL laps for (race_id, driver_id) where pit stops NOT in [1..3] (inclusive).
   L2) Remove ALL laps for (race_id, driver_id) where resultposition is NULL/invalid or > 15.

B) CLEANER DB:
   Uses the existing "full" reduced-schema pipeline:
   1) Remove ALL laps for pairs with > max_pitstops pit stops (no weather logic).
   2) Remove pit-stop laps in last tail_fraction of race (lap-based).
   3) Remove ALL laps for non-top-10 finishers.
   4) Remove inferred red-flag laps via big racetime gaps between lapno and lapno+1.
   5) Remove ALL laps for driver-race pairs with extreme anomalies.

USAGE:
  Dry run (no DB files created; prints what would change):
    python db_clean_minimal_two_tiers.py /path/to/db.sqlite --dry-run --sanity-report

  Apply (creates two output DB files and applies respective pipelines):
    python db_clean_minimal_two_tiers.py /path/to/db.sqlite --apply --out-dir /path/to/out

OPTIONS:
  --max-pitstops       (cleaner DB Step 1 threshold; default 3)
  --tail-fraction      (cleaner DB Step 2; default 0.10)
  --gap-threshold-s    (cleaner DB Step 4; default 600)
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
from typing import Iterable, Optional, Sequence, Tuple


# -------------------------
# Logging
# -------------------------

def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


# -------------------------
# DB helpers
# -------------------------

def connect_sqlite(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn

def get_triggers_for_table(conn: sqlite3.Connection, table: str) -> list[tuple[str, str]]:
    rows = conn.execute("""
        SELECT name, sql
        FROM sqlite_master
        WHERE type='trigger' AND tbl_name = ?
          AND sql IS NOT NULL;
    """, (table,)).fetchall()
    return [(r["name"], r["sql"]) for r in rows]

def drop_triggers(conn: sqlite3.Connection, triggers: list[tuple[str, str]]) -> None:
    for name, _sql in triggers:
        conn.execute(f'DROP TRIGGER IF EXISTS "{name}";')

def restore_triggers(conn: sqlite3.Connection, triggers: list[tuple[str, str]]) -> None:
    for _name, sql in triggers:
        # sql already contains CREATE TRIGGER ...
        conn.execute(sql)

def repair_fcyphases_fk(conn: sqlite3.Connection) -> None:
    """
    Fix fcyphases FOREIGN KEY (race_id) REFERENCES race(id)
    -> should reference races(id).

    This rebuilds the table safely.
    """
    if not table_exists(conn, "fcyphases"):
        return

    # Check whether the FK is already correct
    fk_rows = conn.execute("PRAGMA foreign_key_list(fcyphases);").fetchall()
    # fk_rows columns: (id, seq, table, from, to, on_update, on_delete, match)
    if fk_rows and fk_rows[0]["table"] == "races":
        return  # already fixed

    logging.warning("Repairing fcyphases FK: race(id) -> races(id) (rebuilding table).")

    conn.execute("PRAGMA foreign_keys = OFF;")
    try:
        conn.execute("BEGIN;")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS fcyphases__new (
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
            );
        """)

        # Copy data across
        conn.execute("""
            INSERT INTO fcyphases__new (id, race_id, startracetime, endracetime, startraceprog, endraceprog, startlap, endlap, type)
            SELECT id, race_id, startracetime, endracetime, startraceprog, endraceprog, startlap, endlap, type
            FROM fcyphases;
        """)

        # Swap tables
        conn.execute("DROP TABLE fcyphases;")
        conn.execute("ALTER TABLE fcyphases__new RENAME TO fcyphases;")

        conn.execute("COMMIT;")
    except Exception:
        conn.execute("ROLLBACK;")
        raise
    finally:
        # Restore default behaviour for rest of script (your connect_sqlite enables it anyway)
        conn.execute("PRAGMA foreign_keys = ON;")


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


def _is_pitstop_expr() -> str:
    """
    Definition of "pit stop happened on this lap" for the reduced schema.

    A lap is treated as a pitstop lap if:
      - pitintime is non-empty (and not common NA sentinels), OR
      - pitstopduration is present and > 0
    """
    return """
    (
      (l.pitintime IS NOT NULL AND TRIM(l.pitintime) NOT IN ('', 'NaT', 'nan', 'None'))
      OR (l.pitstopduration IS NOT NULL AND l.pitstopduration > 0)
    )
    """


# -------------------------
# Shared deletion primitives
# -------------------------

def count_laps_for_pairs(conn: sqlite3.Connection, pairs: Iterable[Tuple[int, int]]) -> int:
    pairs = list(pairs)
    if not pairs:
        return 0

    conn.execute("DROP TABLE IF EXISTS tmp_pairs;")
    conn.execute("""
        CREATE TEMP TABLE tmp_pairs (
            race_id  INTEGER NOT NULL,
            driver_id INTEGER NOT NULL,
            PRIMARY KEY (race_id, driver_id)
        );
    """)
    conn.executemany("INSERT OR IGNORE INTO tmp_pairs(race_id, driver_id) VALUES (?, ?);", pairs)

    row = conn.execute("""
        SELECT COUNT(*) AS n
        FROM laps l
        WHERE EXISTS (
            SELECT 1 FROM tmp_pairs t
            WHERE t.race_id = l.race_id AND t.driver_id = l.driver_id
        );
    """).fetchone()

    return int(row["n"])


def delete_laps_for_pairs(conn: sqlite3.Connection, pairs: Iterable[Tuple[int, int]]) -> int:
    pairs = list(pairs)
    if not pairs:
        return 0

    conn.execute("DROP TABLE IF EXISTS tmp_pairs;")
    conn.execute("""
        CREATE TEMP TABLE tmp_pairs (
            race_id  INTEGER NOT NULL,
            driver_id INTEGER NOT NULL,
            PRIMARY KEY (race_id, driver_id)
        );
    """)
    conn.executemany("INSERT OR IGNORE INTO tmp_pairs(race_id, driver_id) VALUES (?, ?);", pairs)

    before = conn.total_changes
    conn.execute("""
        DELETE FROM laps
        WHERE EXISTS (
            SELECT 1 FROM tmp_pairs t
            WHERE t.race_id = laps.race_id AND t.driver_id = laps.driver_id
        );
    """)
    conn.commit()

    return conn.total_changes - before


def sanity_report_pairs(
    conn: sqlite3.Connection,
    pairs_with_extra: Sequence[Tuple[int, int, Optional[int]]],
    title: str,
    limit: int = 200,
) -> None:
    """
    Human-readable preview for (race_id, driver_id) pairs, joining races + drivers.
    'extra' can hold pit_stops or resultposition, etc.
    """
    if not pairs_with_extra:
        return

    require_tables(conn, ["races", "drivers"])

    conn.execute("DROP TABLE IF EXISTS tmp_report_pairs;")
    conn.execute("""
        CREATE TEMP TABLE tmp_report_pairs (
            race_id   INTEGER NOT NULL,
            driver_id INTEGER NOT NULL,
            extra     INTEGER,
            PRIMARY KEY (race_id, driver_id)
        );
    """)
    conn.executemany(
        "INSERT OR REPLACE INTO tmp_report_pairs(race_id, driver_id, extra) VALUES (?, ?, ?);",
        [(r, d, x) for (r, d, x) in pairs_with_extra],
    )

    rows = conn.execute("""
        SELECT
            t.race_id,
            r.season,
            r.location,
            r.date AS race_date,
            t.driver_id,
            d.name AS driver_name,
            t.extra AS extra_value,
            (
                SELECT COUNT(*)
                FROM laps l
                WHERE l.race_id = t.race_id AND l.driver_id = t.driver_id
            ) AS laps_rows_affected
        FROM tmp_report_pairs t
        JOIN races r ON r.id = t.race_id
        JOIN drivers d ON d.id = t.driver_id
        ORDER BY r.season, r.date, r.location, d.name
        LIMIT ?;
    """, (limit,)).fetchall()

    print(f"\n=== SANITY REPORT: {title} [top {limit}] ===")
    for rr in rows:
        print(
            f"Race {rr['race_id']} | {rr['season']} {rr['location']} ({rr['race_date']}) "
            f"| Driver {rr['driver_id']} {rr['driver_name']} "
            f"| extra={rr['extra_value']} | laps_rows_affected={rr['laps_rows_affected']}"
        )
    if len(pairs_with_extra) > limit:
        print(f"... ({len(pairs_with_extra) - limit} more pairs not shown)")
    print("=== END SANITY REPORT ===\n")


# -------------------------
# LESS-CLEAN pipeline steps
# -------------------------

@dataclass(frozen=True)
class PitCountPair:
    race_id: int
    driver_id: int
    pit_stops: int


def find_pairs_pitstops_outside_range(
    conn: sqlite3.Connection,
    min_pitstops: int = 1,
    max_pitstops: int = 3,
) -> list[PitCountPair]:
    require_tables(conn, ["laps"])
    if min_pitstops > max_pitstops:
        raise ValueError("min_pitstops must be <= max_pitstops")

    sql = f"""
    WITH driver_pit_counts AS (
        SELECT
            l.race_id,
            l.driver_id,
            SUM(CASE WHEN {_is_pitstop_expr()} THEN 1 ELSE 0 END) AS pit_stops
        FROM laps l
        GROUP BY l.race_id, l.driver_id
    )
    SELECT race_id, driver_id, pit_stops
    FROM driver_pit_counts
    WHERE pit_stops < ? OR pit_stops > ?
    ORDER BY race_id, driver_id;
    """
    rows = conn.execute(sql, (min_pitstops, max_pitstops)).fetchall()
    return [PitCountPair(int(r["race_id"]), int(r["driver_id"]), int(r["pit_stops"])) for r in rows]


def remove_pairs_pitstops_outside_range(
    conn: sqlite3.Connection,
    apply: bool,
    min_pitstops: int = 1,
    max_pitstops: int = 3,
    sanity_report: bool = False,
) -> int:
    logging.info("Less-clean L1: Remove ALL laps where pit_stops NOT in [%d..%d].", min_pitstops, max_pitstops)

    pairs = find_pairs_pitstops_outside_range(conn, min_pitstops=min_pitstops, max_pitstops=max_pitstops)
    if not pairs:
        logging.info("Less-clean L1: no pairs found.")
        return 0

    pair_keys = [(p.race_id, p.driver_id) for p in pairs]
    would_delete = count_laps_for_pairs(conn, pair_keys)
    logging.info("Less-clean L1: %d pairs flagged; %d laps rows affected.", len(pairs), would_delete)

    if sanity_report:
        sanity_report_pairs(
            conn,
            [(p.race_id, p.driver_id, p.pit_stops) for p in pairs],
            title="Less-clean L1 (pit_stops outside [1..3]) — extra=pit_stops",
        )

    if not apply:
        logging.info("Less-clean L1 dry-run: no deletions applied.")
        return 0

    deleted = delete_laps_for_pairs(conn, pair_keys)
    logging.info("Less-clean L1 applied: deleted %d laps rows.", deleted)
    return deleted


def remove_laps_for_finishers_later_than(
    conn: sqlite3.Connection,
    apply: bool,
    cutoff_position: int = 15,
    sanity_report: bool = False,
    report_limit: int = 200,
) -> int:
    """
    Less-clean L2:
    Delete ALL laps for (race_id, driver_id) where resultposition is:
      - NULL, or
      - < 1, or
      - > cutoff_position
    """
    require_tables(conn, ["laps", "starterfields", "races", "drivers"])
    logging.info("Less-clean L2: Remove ALL laps where resultposition is invalid/NULL or > %d.", cutoff_position)

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
            OR CAST(sf.resultposition AS INTEGER) > ?;
    """, (cutoff_position,))

    n_pairs = int(conn.execute("SELECT COUNT(*) AS n FROM tmp_late_finish_pairs;").fetchone()["n"])
    n_laps = int(conn.execute("""
        SELECT COUNT(*) AS n
        FROM laps l
        WHERE EXISTS (
            SELECT 1 FROM tmp_late_finish_pairs t
            WHERE t.race_id = l.race_id AND t.driver_id = l.driver_id
        );
    """).fetchone()["n"])

    logging.info("Less-clean L2: %d pairs flagged; %d laps rows affected.", n_pairs, n_laps)

    if sanity_report and n_pairs:
        rows = conn.execute("""
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

        print(f"\n=== SANITY REPORT: Less-clean L2 (finish > {cutoff_position}) [top {report_limit}] ===")
        for rr in rows:
            print(
                f"Race {rr['race_id']} | {rr['season']} {rr['location']} ({rr['race_date']}) "
                f"| Driver {rr['driver_id']} {rr['driver_name']} "
                f"| resultposition={rr['resultposition']} | laps_rows_affected={rr['laps_rows_affected']}"
            )
        if n_pairs > report_limit:
            print(f"... ({n_pairs - report_limit} more pairs not shown)")
        print("=== END SANITY REPORT ===\n")

    if not apply:
        logging.info("Less-clean L2 dry-run: no deletions applied.")
        return 0

    before = conn.total_changes
    conn.execute("""
        DELETE FROM laps
        WHERE EXISTS (
            SELECT 1 FROM tmp_late_finish_pairs t
            WHERE t.race_id = laps.race_id
              AND t.driver_id = laps.driver_id
        );
    """)
    conn.commit()

    deleted = conn.total_changes - before
    logging.info("Less-clean L2 applied: deleted %d laps rows.", deleted)
    return deleted


def run_less_clean_pipeline(conn: sqlite3.Connection, apply: bool, sanity_report: bool) -> None:
    require_tables(conn, ["laps", "races", "drivers", "starterfields"])
    repair_fcyphases_fk(conn)
    remove_wet_races(conn, apply=apply, sanity_report=sanity_report)
    remove_pairs_pitstops_outside_range(conn, apply=apply, min_pitstops=1, max_pitstops=3, sanity_report=sanity_report)
    remove_laps_for_finishers_later_than(conn, apply=apply, cutoff_position=15, sanity_report=sanity_report)


# -------------------------
# CLEANER pipeline (your existing reduced-schema steps)
# -------------------------

@dataclass(frozen=True)
class PitStopHeavyPair:
    race_id: int
    driver_id: int
    pit_stops: int


def remove_wet_races(
    conn: sqlite3.Connection,
    apply: bool,
    sanity_report: bool = False,
    report_limit: int = 200,
) -> int:
    require_tables(conn, ["laps", "races"])

    logging.info("Step WET: Remove all races where INTERMEDIATE/WET tyres were used at any point.")

    conn.execute("DROP TABLE IF EXISTS tmp_wet_race_ids;")
    conn.execute("""
        CREATE TEMP TABLE tmp_wet_race_ids (
            race_id INTEGER PRIMARY KEY
        );
    """)

    # Ergast tyre codes: 'I' (intermediate) and 'W' (wet)
    conn.execute("""
        INSERT OR IGNORE INTO tmp_wet_race_ids(race_id)
        SELECT DISTINCT l.race_id
        FROM laps l
        WHERE
            UPPER(TRIM(COALESCE(l.compound, ''))) IN ('W', 'I')
            OR UPPER(TRIM(COALESCE(l.nextcompound, ''))) IN ('W', 'I');
    """)

    n_races = int(conn.execute("SELECT COUNT(*) AS n FROM tmp_wet_race_ids;").fetchone()["n"])
    n_laps = int(conn.execute("""
        SELECT COUNT(*) AS n
        FROM laps l
        WHERE EXISTS (SELECT 1 FROM tmp_wet_race_ids t WHERE t.race_id = l.race_id);
    """).fetchone()["n"])

    logging.info("Step WET: flagged %d races; %d laps rows affected.", n_races, n_laps)

    if sanity_report and n_races:
        rows = conn.execute("""
            SELECT
                r.id AS race_id,
                r.season,
                r.location,
                r.date AS race_date,
                (SELECT COUNT(*) FROM laps l WHERE l.race_id = r.id) AS n_laps
            FROM races r
            WHERE EXISTS (SELECT 1 FROM tmp_wet_race_ids t WHERE t.race_id = r.id)
            ORDER BY r.season, r.date, r.location
            LIMIT ?;
        """, (report_limit,)).fetchall()

        print(f"\n=== SANITY REPORT: Wet races to delete [top {report_limit}] ===")
        for rr in rows:
            print(
                f"Race {rr['race_id']} | {rr['season']} {rr['location']} ({rr['race_date']}) | laps={rr['n_laps']}"
            )
        if n_races > report_limit:
            print(f"... ({n_races - report_limit} more races not shown)")
        print("=== END SANITY REPORT ===\n")

    if not apply:
        logging.info("Step WET dry-run: no deletions applied.")
        return 0

    # If you've run repair_fcyphases_fk(conn) earlier, you shouldn't need FK off,
    # but keeping this makes bulk deletes more robust across slightly inconsistent DBs.
    fk_prev = int(conn.execute("PRAGMA foreign_keys;").fetchone()[0])
    conn.execute("PRAGMA foreign_keys = OFF;")
    try:
        before = conn.total_changes

        # Delete in child->parent order
        conn.execute("""
            DELETE FROM laps
            WHERE EXISTS (SELECT 1 FROM tmp_wet_race_ids t WHERE t.race_id = laps.race_id);
        """)

        if table_exists(conn, "fcyphases"):
            conn.execute("""
                DELETE FROM fcyphases
                WHERE EXISTS (SELECT 1 FROM tmp_wet_race_ids t WHERE t.race_id = fcyphases.race_id);
            """)

        if table_exists(conn, "starterfields"):
            conn.execute("""
                DELETE FROM starterfields
                WHERE EXISTS (SELECT 1 FROM tmp_wet_race_ids t WHERE t.race_id = starterfields.race_id);
            """)

        if table_exists(conn, "qualifyings"):
            conn.execute("""
                DELETE FROM qualifyings
                WHERE EXISTS (SELECT 1 FROM tmp_wet_race_ids t WHERE t.race_id = qualifyings.race_id);
            """)

        conn.execute("""
            DELETE FROM races
            WHERE EXISTS (SELECT 1 FROM tmp_wet_race_ids t WHERE t.race_id = races.id);
        """)

        conn.commit()
        deleted = conn.total_changes - before
        logging.info("Step WET applied: total rows deleted across tables = %d.", deleted)
        return deleted
    finally:
        conn.execute(f"PRAGMA foreign_keys = {fk_prev};")



def find_driver_race_pairs_over_pitstops(conn: sqlite3.Connection, max_pitstops: int) -> list[PitStopHeavyPair]:
    require_tables(conn, ["laps"])

    sql = f"""
    WITH driver_pit_counts AS (
        SELECT
            l.race_id,
            l.driver_id,
            SUM(CASE WHEN {_is_pitstop_expr()} THEN 1 ELSE 0 END) AS pit_stops
        FROM laps l
        GROUP BY l.race_id, l.driver_id
    )
    SELECT race_id, driver_id, pit_stops
    FROM driver_pit_counts
    WHERE pit_stops > ?
    ORDER BY race_id, driver_id;
    """
    rows = conn.execute(sql, (max_pitstops,)).fetchall()
    return [PitStopHeavyPair(int(r["race_id"]), int(r["driver_id"]), int(r["pit_stops"])) for r in rows]


def remove_pitstop_laps_in_last_race_fraction(
    conn: sqlite3.Connection,
    apply: bool,
    tail_fraction: float = 0.10,
    sanity_report: bool = False,
    report_limit: int = 200,
) -> int:
    require_tables(conn, ["laps", "races", "drivers"])

    if not (0.0 < tail_fraction < 1.0):
        raise ValueError("tail_fraction must be between 0 and 1 (exclusive).")

    conn.execute("DROP TABLE IF EXISTS tmp_pit_tail_laps;")
    conn.execute("""
        CREATE TEMP TABLE tmp_pit_tail_laps (
            race_id INTEGER NOT NULL,
            driver_id INTEGER NOT NULL,
            lapno INTEGER NOT NULL,
            PRIMARY KEY (race_id, driver_id, lapno)
        );
    """)

    insert_sql = f"""
    INSERT OR IGNORE INTO tmp_pit_tail_laps (race_id, driver_id, lapno)
    SELECT
        l.race_id,
        l.driver_id,
        l.lapno
    FROM laps l
    JOIN races r ON r.id = l.race_id
    WHERE
        l.lapno IS NOT NULL
        AND COALESCE(r.nolapsplanned, r.nolaps) IS NOT NULL
        AND COALESCE(r.nolapsplanned, r.nolaps) > 0
        AND ({_is_pitstop_expr()})
        AND l.lapno >= (CAST(((1.0 - ?) * COALESCE(r.nolapsplanned, r.nolaps)) AS INTEGER) + 1)
    ;
    """
    conn.execute(insert_sql, (tail_fraction,))

    would_delete = int(conn.execute("SELECT COUNT(*) AS n FROM tmp_pit_tail_laps;").fetchone()["n"])
    logging.info("Step 2: pit-stop laps in last %.0f%% of race: %d rows matched.", tail_fraction * 100.0, would_delete)

    if sanity_report and would_delete:
        rows = conn.execute("""
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
              COALESCE(r.nolapsplanned, r.nolaps) AS race_laps
            FROM tmp_pit_tail_laps t
            JOIN races r ON r.id = t.race_id
            JOIN drivers d ON d.id = t.driver_id
            GROUP BY t.race_id, t.driver_id
            ORDER BY r.season, r.date, r.location, d.name
            LIMIT ?;
        """, (report_limit,)).fetchall()

        print(f"\n=== SANITY REPORT: Step 2 (pit-stop laps in last fraction) [top {report_limit} groups] ===")
        for rr in rows:
            threshold = int((1.0 - tail_fraction) * rr["race_laps"]) + 1
            print(
                f"Race {rr['race_id']} | {rr['season']} {rr['location']} ({rr['race_date']}) "
                f"| Driver {rr['driver_id']} {rr['driver_name']} "
                f"| deleted_laps={rr['n_laps_to_delete']} (lap {rr['first_deleted_lap']}..{rr['last_deleted_lap']}) "
                f"| race_laps={rr['race_laps']} | threshold_lap={threshold}"
            )
        print("=== END SANITY REPORT ===\n")

    if not apply:
        logging.info("Step 2 dry-run: no deletions applied.")
        return 0

    before = conn.total_changes
    conn.execute("""
        DELETE FROM laps
        WHERE EXISTS (
            SELECT 1 FROM tmp_pit_tail_laps t
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

    conn.execute("""
        INSERT OR IGNORE INTO tmp_non_top10_pairs (race_id, driver_id, resultposition)
        SELECT
            sf.race_id,
            sf.driver_id,
            sf.resultposition
        FROM starterfields sf
        WHERE
            sf.resultposition IS NULL
            OR CAST(sf.resultposition AS INTEGER) < 1
            OR CAST(sf.resultposition AS INTEGER) > 10;
    """)

    n_pairs = int(conn.execute("SELECT COUNT(*) AS n FROM tmp_non_top10_pairs;").fetchone()["n"])
    n_laps = int(conn.execute("""
        SELECT COUNT(*) AS n
        FROM laps l
        WHERE EXISTS (
            SELECT 1 FROM tmp_non_top10_pairs t
            WHERE t.race_id = l.race_id AND t.driver_id = l.driver_id
        );
    """).fetchone()["n"])

    logging.info("Step 3: Non-top-10 pairs: %d pairs; %d laps rows affected.", n_pairs, n_laps)

    if sanity_report and n_pairs:
        rows = conn.execute("""
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

        print(f"\n=== SANITY REPORT: Step 3 (delete non-top-10 finishers) [top {report_limit} pairs] ===")
        for rr in rows:
            print(
                f"Race {rr['race_id']} | {rr['season']} {rr['location']} ({rr['race_date']}) "
                f"| Driver {rr['driver_id']} {rr['driver_name']} "
                f"| resultposition={rr['resultposition']} | laps_rows_affected={rr['laps_rows_affected']}"
            )
        if n_pairs > report_limit:
            print(f"... ({n_pairs - report_limit} more pairs not shown)")
        print("=== END SANITY REPORT ===\n")

    if not apply:
        logging.info("Step 3 dry-run: no deletions applied.")
        return 0

    before = conn.total_changes
    conn.execute("""
        DELETE FROM laps
        WHERE EXISTS (
            SELECT 1 FROM tmp_non_top10_pairs t
            WHERE t.race_id = laps.race_id
              AND t.driver_id = laps.driver_id
        );
    """)
    conn.commit()

    deleted = conn.total_changes - before
    logging.info("Step 3 applied: deleted %d laps rows.", deleted)
    return deleted


def remove_red_flag_laps_inferred_from_time_gaps(
    conn: sqlite3.Connection,
    apply: bool,
    gap_threshold_s: float = 600.0,
    sanity_report: bool = False,
    report_limit: int = 200,
) -> int:
    require_tables(conn, ["laps", "races", "drivers"])

    conn.execute("DROP TABLE IF EXISTS tmp_red_flag_laps;")
    conn.execute("""
        CREATE TEMP TABLE tmp_red_flag_laps (
            race_id INTEGER NOT NULL,
            lapno   INTEGER NOT NULL,
            gap_s   REAL,
            PRIMARY KEY (race_id, lapno)
        );
    """)

    conn.execute("""
        WITH LapMaxRaceTime AS (
            SELECT
                race_id,
                lapno,
                MAX(racetime) AS max_racetime
            FROM laps
            WHERE racetime IS NOT NULL
              AND lapno IS NOT NULL
            GROUP BY race_id, lapno
        )
        INSERT OR IGNORE INTO tmp_red_flag_laps (race_id, lapno, gap_s)
        SELECT
            t1.race_id,
            t1.lapno,
            (t2.max_racetime - t1.max_racetime) AS gap_s
        FROM LapMaxRaceTime t1
        JOIN LapMaxRaceTime t2
          ON t1.race_id = t2.race_id
         AND t1.lapno + 1 = t2.lapno
        WHERE (t2.max_racetime - t1.max_racetime) > ?;
    """, (float(gap_threshold_s),))

    n_flagged = int(conn.execute("SELECT COUNT(*) AS n FROM tmp_red_flag_laps;").fetchone()["n"])
    n_rows = int(conn.execute("""
        SELECT COUNT(*) AS n
        FROM laps l
        WHERE EXISTS (
            SELECT 1 FROM tmp_red_flag_laps t
            WHERE t.race_id = l.race_id AND t.lapno = l.lapno
        );
    """).fetchone()["n"])

    logging.info(
        "Step 4: Red-flag inference (gap > %.1fs): flagged %d (race_id, lapno); %d laps rows affected.",
        gap_threshold_s, n_flagged, n_rows
    )

    if sanity_report and n_flagged:
        rows = conn.execute("""
            SELECT
                t.race_id,
                r.season,
                r.location,
                r.date AS race_date,
                t.lapno,
                t.gap_s,
                (
                    SELECT COUNT(*)
                    FROM laps l
                    WHERE l.race_id = t.race_id AND l.lapno = t.lapno
                ) AS laps_rows_affected
            FROM tmp_red_flag_laps t
            JOIN races r ON r.id = t.race_id
            ORDER BY r.season, r.date, r.location, t.lapno
            LIMIT ?;
        """, (report_limit,)).fetchall()

        print(f"\n=== SANITY REPORT: Step 4 (inferred red-flag laps; gap>{gap_threshold_s:.1f}s) [top {report_limit}] ===")
        for rr in rows:
            print(
                f"Race {rr['race_id']} | {rr['season']} {rr['location']} ({rr['race_date']}) "
                f"| lap={rr['lapno']} | gap_s={rr['gap_s']:.1f} | laps_rows_affected={rr['laps_rows_affected']}"
            )
        if n_flagged > report_limit:
            print(f"... ({n_flagged - report_limit} more events not shown)")
        print("=== END SANITY REPORT ===\n")

    if not apply:
        logging.info("Step 4 dry-run: no deletions applied.")
        return 0

    before = conn.total_changes
    conn.execute("""
        DELETE FROM laps
        WHERE EXISTS (
            SELECT 1 FROM tmp_red_flag_laps t
            WHERE t.race_id = laps.race_id
              AND t.lapno = laps.lapno
        );
    """)
    conn.commit()

    deleted = conn.total_changes - before
    logging.info("Step 4 applied: deleted %d laps rows.", deleted)
    return deleted


def remove_driver_race_pairs_with_extreme_laps(
    conn: sqlite3.Connection,
    apply: bool,
    laptime_threshold_s: float = 200.0,
    pitstop_threshold_s: float = 50.0,
    sanity_report: bool = False,
    report_limit: int = 200,
) -> int:
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

    n_pairs = int(conn.execute("SELECT COUNT(*) AS n FROM tmp_extreme_pairs;").fetchone()["n"])
    n_laps = int(conn.execute("""
        SELECT COUNT(*) AS n
        FROM laps l
        WHERE EXISTS (
            SELECT 1 FROM tmp_extreme_pairs t
            WHERE t.race_id = l.race_id AND t.driver_id = l.driver_id
        );
    """).fetchone()["n"])

    logging.info(
        "Step 5: Extreme anomalies: %d pairs flagged; %d laps rows affected (laptime>%.1fs or pitstop>%.1fs).",
        n_pairs, n_laps, laptime_threshold_s, pitstop_threshold_s
    )

    if sanity_report and n_pairs:
        rows = conn.execute("""
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

        print(f"\n=== SANITY REPORT: Step 5 (extreme driver-race deletions) [top {report_limit}] ===")
        for rr in rows:
            print(
                f"Race {rr['race_id']} | {rr['season']} {rr['location']} ({rr['race_date']}) "
                f"| Driver {rr['driver_id']} {rr['driver_name']} "
                f"| offending_laps={rr['offending_laps_count']} "
                f"| max_laptime={rr['max_laptime']}s | max_pitstop={rr['max_pitstopduration']}s "
                f"| laps_rows_affected={rr['laps_rows_affected']}"
            )
        if n_pairs > report_limit:
            print(f"... ({n_pairs - report_limit} more pairs not shown)")
        print("=== END SANITY REPORT ===\n")

    if not apply:
        logging.info("Step 5 dry-run: no deletions applied.")
        return 0

    before = conn.total_changes
    conn.execute("""
        DELETE FROM laps
        WHERE EXISTS (
            SELECT 1 FROM tmp_extreme_pairs t
            WHERE t.race_id = laps.race_id
              AND t.driver_id = laps.driver_id
        );
    """)
    conn.commit()

    deleted = conn.total_changes - before
    logging.info("Step 5 applied: deleted %d laps rows.", deleted)
    return deleted


def run_cleaner_pipeline(
    conn: sqlite3.Connection,
    apply: bool,
    max_pitstops: int,
    tail_fraction: float,
    gap_threshold_s: float,
    sanity_report: bool,
) -> None:
    require_tables(conn, ["laps", "races", "drivers", "starterfields"])
    repair_fcyphases_fk(conn)
    remove_wet_races(conn, apply=apply, sanity_report=sanity_report)

    # Step 1
    logging.info("Step 1: Remove ALL laps for pairs with > %d pit stops.", max_pitstops)
    heavy_pairs = find_driver_race_pairs_over_pitstops(conn, max_pitstops=max_pitstops)

    if heavy_pairs:
        pair_keys = [(p.race_id, p.driver_id) for p in heavy_pairs]
        would_delete = count_laps_for_pairs(conn, pair_keys)
        logging.info("Step 1: %d pairs flagged; %d laps rows affected.", len(heavy_pairs), would_delete)

        if sanity_report:
            sanity_report_pairs(
                conn,
                [(p.race_id, p.driver_id, p.pit_stops) for p in heavy_pairs],
                title="Step 1 (pairs with too many pit stops) — extra=pit_stops",
            )

        if apply:
            deleted = delete_laps_for_pairs(conn, pair_keys)
            logging.info("Step 1 applied: deleted %d laps rows.", deleted)
        else:
            logging.info("Step 1 dry-run: no deletions applied.")
    else:
        logging.info("Step 1: no pairs found.")

    # Step 2
    remove_pitstop_laps_in_last_race_fraction(
        conn,
        apply=apply,
        tail_fraction=tail_fraction,
        sanity_report=sanity_report,
    )

    # Step 3
    remove_laps_for_non_top10_finishers(
        conn,
        apply=apply,
        sanity_report=sanity_report,
    )

    # Step 4 (inferred)
    remove_red_flag_laps_inferred_from_time_gaps(
        conn,
        apply=apply,
        gap_threshold_s=gap_threshold_s,
        sanity_report=sanity_report,
    )

    # Step 5
    remove_driver_race_pairs_with_extreme_laps(
        conn,
        apply=apply,
        laptime_threshold_s=200.0,
        pitstop_threshold_s=50.0,
        sanity_report=sanity_report,
    )


# -------------------------
# Output DB creation (same pattern as your first script)
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
    p = argparse.ArgumentParser(description="Create two cleaned variants of a reduced-schema F1 strategy SQLite database.")
    p.add_argument("db_path", help="Path to the input SQLite .db/.sqlite file")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--apply", action="store_true", help="Create two cleaned DB copies and apply deletions to each")
    mode.add_argument("--dry-run", action="store_true", help="Show what would change without creating/modifying any DB files")

    p.add_argument("--out-dir", default=None, help="Directory to write output DBs (default: alongside input DB)")
    p.add_argument("--max-pitstops", type=int, default=3, help="Cleaner DB Step 1 threshold (default: 3)")
    p.add_argument("--tail-fraction", type=float, default=0.10, help="Cleaner DB Step 2 tail fraction (default: 0.10)")
    p.add_argument("--gap-threshold-s", type=float, default=600.0, help="Cleaner DB Step 4 red-flag gap threshold (default: 600s)")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    p.add_argument("--sanity-report", action="store_true", help="Print human-readable previews of affected deletions")

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
            run_cleaner_pipeline(
                conn,
                apply=False,
                max_pitstops=args.max_pitstops,
                tail_fraction=args.tail_fraction,
                gap_threshold_s=args.gap_threshold_s,
                sanity_report=args.sanity_report,
            )
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
        run_less_clean_pipeline(conn_less, apply=True, sanity_report=args.sanity_report)
    finally:
        conn_less.close()

    # Apply cleaner pipeline
    conn_clean = connect_sqlite(cleaner_path)
    try:
        logging.info("Applying cleaner pipeline to: %s", cleaner_path)
        run_cleaner_pipeline(
            conn_clean,
            apply=True,
            max_pitstops=args.max_pitstops,
            tail_fraction=args.tail_fraction,
            gap_threshold_s=args.gap_threshold_s,
            sanity_report=args.sanity_report,
        )
    finally:
        conn_clean.close()

    logging.info("Done. Output files:\n- %s\n- %s", less_clean_path, cleaner_path)


if __name__ == "__main__":
    main()