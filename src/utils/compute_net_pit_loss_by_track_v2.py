#!/usr/bin/env python3

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = "data/raw/f1_database.sqlite"

# put your holdout race ids here if you want to exclude them
EXCLUDE_RACE_IDS = set()



def normalize_compound(x):
    if x is None or pd.isna(x):
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
    return s


conn = sqlite3.connect(DB_PATH)

# -----------------------------
# actual pit stops with next-lap pit exit
# -----------------------------
stops_sql = """
SELECT
    l.race_id,
    r.location AS race_track,
    l.driver_id,
    l.lapno AS pit_lap,
    l.compound,
    l.nextcompound,
    l.pitstopduration,
    l.racetime,
    l.pitintime_s,
    l.pitintimenum AS pit_in_ms,
    n.pitouttimenum AS pit_out_ms,
    l.sector2session_ms AS s2_end_ms,
    l.sector3session_ms AS lap_end_ms,
    n.sector1session_ms AS next_s1_end_ms
FROM laps l
JOIN races r
  ON r.id = l.race_id
JOIN laps n
  ON n.race_id = l.race_id
 AND n.driver_id = l.driver_id
 AND n.lapno = l.lapno + 1
WHERE l.pitintimenum IS NOT NULL
  AND n.pitouttimenum IS NOT NULL
  AND l.sector2session_ms IS NOT NULL
  AND l.sector3session_ms IS NOT NULL
  AND n.sector1session_ms IS NOT NULL;
"""

stops = pd.read_sql_query(stops_sql, conn)

# -----------------------------
# non-pit laps for track-level S1/S3 medians, derived from session_ms
# s1_s = current sector1session_ms - previous lap sector3session_ms
# s3_s = current sector3session_ms - current sector2session_ms
# -----------------------------
laps_sql = """
SELECT
    l.race_id,
    r.location AS race_track,
    l.driver_id,
    l.lapno,
    l.compound,
    l.racetime,
    l.sector1session_ms,
    l.sector2session_ms,
    l.sector3session_ms,
    p.sector3session_ms AS prev_lap_end_ms,
    l.is_deleted,
    l.is_accurate,
    l.pitintimenum
FROM laps l
JOIN races r
  ON r.id = l.race_id
JOIN laps p
  ON p.race_id = l.race_id
 AND p.driver_id = l.driver_id
 AND p.lapno = l.lapno - 1
WHERE l.sector1session_ms IS NOT NULL
  AND l.sector2session_ms IS NOT NULL
  AND l.sector3session_ms IS NOT NULL
  AND p.sector3session_ms IS NOT NULL;
"""

laps = pd.read_sql_query(laps_sql, conn)
conn.close()

print(f"raw stop rows: {len(stops)}")
print(f"raw timed laps: {len(laps)}")

if EXCLUDE_RACE_IDS:
    stops = stops.loc[~stops["race_id"].isin(EXCLUDE_RACE_IDS)].copy()
    laps = laps.loc[~laps["race_id"].isin(EXCLUDE_RACE_IDS)].copy()
    print(f"after holdout exclusion, stop rows: {len(stops)}")
    print(f"after holdout exclusion, timed laps: {len(laps)}")

# -----------------------------
# derive stop-level quantities
# -----------------------------
for c in ["pit_in_ms", "pit_out_ms", "s2_end_ms", "lap_end_ms", "next_s1_end_ms", "pitstopduration"]:
    stops[c] = pd.to_numeric(stops[c], errors="coerce")

stops["compound_norm"] = stops["compound"].map(normalize_compound)
stops["nextcompound_norm"] = stops["nextcompound"].map(normalize_compound)

# dry-only stops
stops = stops.loc[
    ~stops["compound_norm"].isin({"INTERMEDIATE", "WET"})
    & ~stops["nextcompound_norm"].isin({"INTERMEDIATE", "WET"})
].copy()
print(f"after dry-stop filter: {len(stops)}")



# exclude FCY/SC pit entries
# do this in python by re-reading fcyphases if needed would be awkward here,
# so keep v2 simple first; you can add FCY filtering later once baseline works

stops["entry_frac"] = (stops["pit_in_ms"] - stops["s2_end_ms"]) / (stops["lap_end_ms"] - stops["s2_end_ms"])
stops["exit_frac"] = (stops["pit_out_ms"] - stops["lap_end_ms"]) / (stops["next_s1_end_ms"] - stops["lap_end_ms"])
stops["abs_pit_window_s"] = (stops["pit_out_ms"] - stops["pit_in_ms"]) / 1000.0

stops = stops.loc[
    stops["entry_frac"].between(0.0, 1.0, inclusive="both")
    & stops["exit_frac"].between(0.0, 1.0, inclusive="both")
    & stops["abs_pit_window_s"].between(10.0, 60.0, inclusive="both")
].copy()
print(f"after fraction/window sanity filters: {len(stops)}")

# -----------------------------
# derive clean sector times from session_ms
# -----------------------------
for c in ["sector1session_ms", "sector2session_ms", "sector3session_ms", "prev_lap_end_ms", "racetime"]:
    laps[c] = pd.to_numeric(laps[c], errors="coerce")

laps["compound_norm"] = laps["compound"].map(normalize_compound)

# non-pit, dry, accurate laps
laps = laps.loc[
    laps["pitintimenum"].isna()
    & ~laps["compound_norm"].isin({"INTERMEDIATE", "WET"})
    & (laps["is_deleted"].fillna(0).astype(int) == 0)
    & (laps["is_accurate"].fillna(1).astype(int) == 1)
].copy()
print(f"after clean non-pit dry filter: {len(laps)}")

laps["s1_s"] = (laps["sector1session_ms"] - laps["prev_lap_end_ms"]) / 1000.0
laps["s3_s"] = (laps["sector3session_ms"] - laps["sector2session_ms"]) / 1000.0

laps = laps.loc[
    laps["s1_s"].between(5.0, 60.0, inclusive="both")
    & laps["s3_s"].between(5.0, 60.0, inclusive="both")
].copy()
print(f"after sector sanity filters: {len(laps)}")

# -----------------------------
# aggregate by track
# -----------------------------
stop_agg = (
    stops.groupby("race_track", as_index=True)
    .agg(
        n_stops=("pit_lap", "size"),
        entry_frac_med=("entry_frac", "median"),
        exit_frac_med=("exit_frac", "median"),
        abs_pit_window_med_s=("abs_pit_window_s", "median"),
    )
)

sector_agg = (
    laps.groupby("race_track", as_index=True)
    .agg(
        n_clean_laps=("lapno", "size"),
        s1_med_s=("s1_s", "median"),
        s3_med_s=("s3_s", "median"),
    )
)

table = stop_agg.join(sector_agg, how="inner").copy()

print(f"tracks with stop medians: {len(stop_agg)}")
print(f"tracks with sector medians: {len(sector_agg)}")
print(f"tracks in final joined table: {len(table)}")

table["no_stop_window_s"] = (
    (1.0 - table["entry_frac_med"]) * table["s3_med_s"]
    + table["exit_frac_med"] * table["s1_med_s"]
)

table["net_pit_loss_s"] = table["abs_pit_window_med_s"] - table["no_stop_window_s"]

table = table.sort_index().reset_index()

out_csv = Path("net_pit_loss_by_track.csv")
table.to_csv(out_csv, index=False)

print("\nTrack-level results:\n")
print(
    table[
        [
            "race_track",
            "n_stops",
            "n_clean_laps",
            "entry_frac_med",
            "exit_frac_med",
            "abs_pit_window_med_s",
            "s1_med_s",
            "s3_med_s",
            "no_stop_window_s",
            "net_pit_loss_s",
        ]
    ].round(3).to_string(index=False)
)

net_dict = {
    row["race_track"]: round(float(row["net_pit_loss_s"]), 3)
    for _, row in table.iterrows()
}

default_net = round(float(table["net_pit_loss_s"].median()), 3) if len(table) else np.nan

print("\nPaste this into dataset_builder.py:\n")
print("NET_PIT_LOSS_BY_TRACK_S: dict[str, float] = {")
for k, v in net_dict.items():
    print(f'    "{k}": {v},')
print("}")
print(f"\nDEFAULT_NET_PIT_LOSS_S = {default_net}")
print(f"\nWrote full table to {out_csv}")