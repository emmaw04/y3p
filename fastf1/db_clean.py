#!/usr/bin/env python3
"""
this script cleans up our f1 timing database. 
it spits out two versions:
1. a 'less clean' one with just basic filtering (pit stops and top 15).
2. a 'cleaner' one with more aggressive filtering (top 10, red flags, late pit stops, etc).
"""

import argparse
import logging
import os
import shutil
import sqlite3

# -------------------------
# setup & helpers
# -------------------------

def setup_logging():
    # keep it simple and verbose as requested
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

def connect_db(path):
    # standard sqlite connection with some speed boosts
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn

def delete_by_pairs(conn, pairs):
    # generic helper to delete laps for specific (race_id, driver_id) combos
    if not pairs:
        return 0
    
    conn.execute("DROP TABLE IF EXISTS tmp_bad_pairs;")
    conn.execute("CREATE TEMP TABLE tmp_bad_pairs (race_id INTEGER, driver_id INTEGER, PRIMARY KEY (race_id, driver_id));")
    conn.executemany("INSERT OR IGNORE INTO tmp_bad_pairs(race_id, driver_id) VALUES (?, ?);", pairs)

    before = conn.total_changes
    conn.execute("""
        DELETE FROM laps
        WHERE EXISTS (
            SELECT 1 FROM tmp_bad_pairs bp
            WHERE bp.race_id = laps.race_id AND bp.driver_id = laps.driver_id
        );
    """)
    conn.commit()
    return conn.total_changes - before

# -------------------------
# filtering logic
# -------------------------

def get_dry_pitstop_outliers(conn, min_pits=None, max_pits=3):
    # finds drivers who had too many (or too few) pits in a dry race
    # if min_pits is None, we only check the upper bound.
    
    # helper sql to find dry races (no rain recorded)
    sql_dry_races = """
        SELECT rs.race_id
        FROM sessions rs
        WHERE rs.session_code = 'R' AND COALESCE(rs.is_official, 1) = 1
        AND EXISTS (SELECT 1 FROM weather_samples ws WHERE ws.session_id = rs.id)
        AND NOT EXISTS (SELECT 1 FROM weather_samples ws WHERE ws.session_id = rs.id AND COALESCE(ws.rainfall, 0) != 0)
    """

    sql = f"""
    WITH dry_races AS ({sql_dry_races})
    SELECT l.race_id, l.driver_id, 
           COUNT(CASE WHEN l.pitintime IS NOT NULL AND TRIM(l.pitintime) NOT IN ('', 'NaT', 'nan', 'None') THEN 1 END) as pits
    FROM laps l
    JOIN dry_races dr ON dr.race_id = l.race_id
    GROUP BY l.race_id, l.driver_id
    """
    
    # filter in python for simplicity
    rows = conn.execute(sql).fetchall()
    bad_pairs = []
    for r in rows:
        p = r['pits']
        if (min_pits is not None and p < min_pits) or (p > max_pits):
            bad_pairs.append((r['race_id'], r['driver_id']))
            
    return bad_pairs

def remove_late_pitstops(conn):
    # removes pit stops in the last 10% of a dry race (usually weird outliers)
    # only for 2019-2024 seasons as per previous logic
    sql = """
    DELETE FROM laps
    WHERE rowid IN (
        SELECT l.rowid
        FROM laps l
        JOIN races r ON r.id = l.race_id
        JOIN sessions s ON s.race_id = r.id AND s.session_code = 'R'
        WHERE r.season BETWEEN 2019 AND 2024
        AND l.pitintime IS NOT NULL AND TRIM(l.pitintime) NOT IN ('', 'NaT', 'nan', 'None')
        AND l.lapno >= (CAST((0.9 * r.nolapsplanned) AS INTEGER) + 1)
        AND NOT EXISTS (
            SELECT 1 FROM weather_samples ws 
            WHERE ws.session_id = s.id 
            AND ws.time_ms BETWEEN CAST(l.racetime * 1000 AS INTEGER) - 300000 AND CAST(l.racetime * 1000 AS INTEGER)
            AND COALESCE(ws.rainfall, 0) != 0
        )
    )
    """
    before = conn.total_changes
    conn.execute(sql)
    conn.commit()
    return conn.total_changes - before

def remove_by_position(conn, max_pos):
    # nukes everyone who finished outside the top X
    sql = """
    SELECT race_id, driver_id FROM starterfields
    WHERE resultposition IS NULL OR CAST(resultposition AS INTEGER) < 1 OR CAST(resultposition AS INTEGER) > ?
    """
    pairs = [(r['race_id'], r['driver_id']) for r in conn.execute(sql, (max_pos,)).fetchall()]
    return delete_by_pairs(conn, pairs)

def remove_red_flags(conn):
    # gets rid of laps where the red flag was out (status 5)
    before = conn.total_changes
    conn.execute("DELETE FROM laps WHERE track_status_code LIKE '%5%'")
    conn.commit()
    return conn.total_changes - before

def remove_extremes(conn):
    # filters out drivers who had crazy lap times (>200s) or pit durations (>50s)
    sql = """
    SELECT race_id, driver_id FROM laps
    GROUP BY race_id, driver_id
    HAVING MAX(CASE WHEN laptime IS NOT NULL THEN laptime ELSE 0 END) > 200 
        OR MAX(CASE WHEN pitstopduration IS NOT NULL THEN pitstopduration ELSE 0 END) > 50
    """
    pairs = [(r['race_id'], r['driver_id']) for r in conn.execute(sql).fetchall()]
    return delete_by_pairs(conn, pairs)

# -------------------------
# pipelines
# -------------------------

def run_less_clean(conn):
    # just the basics for the supervisor
    logging.info("starting the 'less-clean' pipeline...")
    
    # 1. dry race + pit stops NOT in [1..3]
    p1 = get_dry_pitstop_outliers(conn, min_pits=1, max_pits=3)
    d1 = delete_by_pairs(conn, p1)
    logging.info(f"dry pitstop check (1-3 pits): flagged {len(p1)} driver/race pairs. deleted {d1} laps.")
    
    # 2. finished > 15th
    d2 = remove_by_position(conn, 15)
    logging.info(f"position check (>15th): deleted {d2} laps.")

def run_cleaner(conn):
    # the full aggressive cleaning
    logging.info("starting the 'cleaner' pipeline...")
    
    # 1. dry race + pit stops > 3 (we allow 0 here, distinct from less-clean)
    p1 = get_dry_pitstop_outliers(conn, min_pits=None, max_pits=3) 
    d1 = delete_by_pairs(conn, p1)
    logging.info(f"dry pitstop check (>3 pits): flagged {len(p1)} driver/race pairs. deleted {d1} laps.")
    
    # 2. late pit stops in dry conditions
    d2 = remove_late_pitstops(conn)
    logging.info(f"late pitstop check (last 10% w/o rain): deleted {d2} laps.")
    
    # 3. finished > 10th
    d3 = remove_by_position(conn, 10)
    logging.info(f"position check (>10th): deleted {d3} laps.")
    
    # 4. red flags
    d4 = remove_red_flags(conn)
    logging.info(f"red flag check: deleted {d4} laps.")
    
    # 5. extreme outliers
    d5 = remove_extremes(conn)
    logging.info(f"extreme outliers check (>200s lap or >50s pit): deleted {d5} laps.")

# -------------------------
# main entry
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("db_path", help="path to the sqlite db")
    args = parser.parse_args()

    setup_logging()
    
    # figure out where we're writing to (always same dir as input)
    in_db = os.path.abspath(args.db_path)
    base_dir = os.path.dirname(in_db)
    
    stem = os.path.splitext(os.path.basename(in_db))[0]
    lc_path = os.path.join(base_dir, f"{stem}__less_clean.sqlite")
    c_path = os.path.join(base_dir, f"{stem}__clean.sqlite")

    # process less-clean
    logging.info(f"spinning up {lc_path}...")
    shutil.copy2(in_db, lc_path)
    conn_lc = connect_db(lc_path)
    try:
        run_less_clean(conn_lc)
    finally:
        conn_lc.close()

    # process cleaner
    logging.info(f"spinning up {c_path}...")
    shutil.copy2(in_db, c_path)
    conn_c = connect_db(c_path)
    try:
        run_cleaner(conn_c)
    finally:
        conn_c.close()

    logging.info("all done! datasets are ready for review.")

if __name__ == "__main__":
    main()
