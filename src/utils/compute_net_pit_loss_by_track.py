#!/usr/bin/env python3
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.data import HOLDOUT_RACE_IDS

DB_PATH = "data/raw/f1_database.sqlite"

def normalise_compound(x):
    """
    standardise compound names
    """
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

#get all pit stops
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

#get all clean dry non pit laps where the S1 and S3 times are not null
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

#sanity check
print(f"raw stop rows: {len(stops)}")
print(f"raw timed laps: {len(laps)}")

#exclude holdout races from timings
if HOLDOUT_RACE_IDS:
    stops = stops.loc[~stops["race_id"].isin(HOLDOUT_RACE_IDS)].copy()
    laps = laps.loc[~laps["race_id"].isin(HOLDOUT_RACE_IDS)].copy()
    print(f"after holdout exclusion, stop rows: {len(stops)}")
    print(f"after holdout exclusion, timed laps: {len(laps)}")

#convert timing columns to numeric
for c in ["pit_in_ms", "pit_out_ms", "s2_end_ms", "lap_end_ms", "next_s1_end_ms", "pitstopduration"]:
    stops[c] = pd.to_numeric(stops[c], errors="coerce")

#ensure compounds are consistent
stops["compound_norm"] = stops["compound"].map(normalise_compound)
stops["nextcompound_norm"] = stops["nextcompound"].map(normalise_compound)

#keep dry only pit stops (switch from dry tyre to dry tyre)
stops = stops.loc[
    ~stops["compound_norm"].isin({"INTERMEDIATE", "WET"})
    & ~stops["nextcompound_norm"].isin({"INTERMEDIATE", "WET"})
].copy()
print(f"after dry-stop filter: {len(stops)}") #sanity check

stops["entry_frac"] = (stops["pit_in_ms"] - stops["s2_end_ms"]) / (stops["lap_end_ms"] - stops["s2_end_ms"]) #tells you how far through sector 3 the car was when it entered the pits
stops["exit_frac"] = (stops["pit_out_ms"] - stops["lap_end_ms"]) / (stops["next_s1_end_ms"] - stops["lap_end_ms"]) #tells you how far through sector 1 of the next lap the driver was when they exited the pits
stops["abs_pit_window_s"] = (stops["pit_out_ms"] - stops["pit_in_ms"]) / 1000.0 #total observed time from pit entry to exit in seconds

#keeps pit stops with a valid pit duration and pit stops that have a valid pitintime and pitouttime
stops = stops.loc[
    stops["entry_frac"].between(0.0, 1.0, inclusive="both")
    & stops["exit_frac"].between(0.0, 1.0, inclusive="both")
    & stops["abs_pit_window_s"].between(10.0, 60.0, inclusive="both")
].copy()
print(f"after fraction/window sanity filters: {len(stops)}")

# derive clean sector times from session_ms
for c in ["sector1session_ms", "sector2session_ms", "sector3session_ms", "prev_lap_end_ms", "racetime"]:
    laps[c] = pd.to_numeric(laps[c], errors="coerce")

laps["compound_norm"] = laps["compound"].map(normalise_compound)

# keep dry non pit laps that are accurate
laps = laps.loc[
    laps["pitintimenum"].isna()
    & ~laps["compound_norm"].isin({"INTERMEDIATE", "WET"})
    & (laps["is_deleted"].fillna(0).astype(int) == 0)
    & (laps["is_accurate"].fillna(1).astype(int) == 1)
].copy()
print(f"after clean non-pit dry filter: {len(laps)}") #sanity

#get sector 1 and sector 3 durations
laps["s1_s"] = (laps["sector1session_ms"] - laps["prev_lap_end_ms"]) / 1000.0
laps["s3_s"] = (laps["sector3session_ms"] - laps["sector2session_ms"]) / 1000.0

#remove non pit laps where sector 1 and 3 durations are implausible
laps = laps.loc[
    laps["s1_s"].between(5.0, 60.0, inclusive="both")
    & laps["s3_s"].between(5.0, 60.0, inclusive="both")
].copy()
print(f"after sector sanity filters: {len(laps)}")

# aggregate stop data by track
stop_agg = (
    stops.groupby("race_track", as_index=True)
    .agg(
        n_stops=("pit_lap", "size"), #number of dry pit stops
        entry_frac_med=("entry_frac", "median"), #median pit entry detection
        exit_frac_med=("exit_frac", "median"), #median pit exit detection
        abs_pit_window_med_s=("abs_pit_window_s", "median"), #median absolute pit window
    )
)

#aggregate sector data by track
sector_agg = (
    laps.groupby("race_track", as_index=True)
    .agg(
        n_clean_laps=("lapno", "size"), #number of clean dry laps per track
        s1_med_s=("s1_s", "median"), #median sector 1 time
        s3_med_s=("s3_s", "median"), #median sector 2 time
    )
)

table = stop_agg.join(sector_agg, how="inner").copy() #join stop data and sector data by track

table["no_stop_window_s"] = ( #calculate how long it would take in s for a driver to cover the same amount of track the pit lane covers (given the driver hasn't pitted)
    (1.0 - table["entry_frac_med"]) * table["s3_med_s"]
    + table["exit_frac_med"] * table["s1_med_s"]
)

#compute net pit loss
table["net_pit_loss_s"] = table["abs_pit_window_med_s"] - table["no_stop_window_s"]

table = table.sort_index().reset_index()

net_dict = {
    row["race_track"]: round(float(row["net_pit_loss_s"]), 3)
    for _, row in table.iterrows()
}

#compute median pit loss across tracks as a fallback value
default_net = round(float(table["net_pit_loss_s"].median()), 3) if len(table) else np.nan

print("NET_PIT_LOSS_BY_TRACK_S: dict[str, float] = {")
for k, v in net_dict.items():
    print(f'    "{k}": {v},')
print("}")