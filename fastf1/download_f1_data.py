"""
automates the retrieval and serialisation of Formula 1 session telemetry, lap timing, and metadata from the FastF1 API.
Stores data in a hierarchical way Year / Event / Session / [Feature].parquet
stores all information in a folder called 'f1_data'
"""

import os
import time
import json
import logging
import datetime
from pathlib import Path
from typing import Any, List, Optional, Sequence, Dict

import fastf1
import pandas as pd
import numpy as np
from tqdm import tqdm

# configuration for what seasons and sessions to download
YEARS: Sequence[int] = range(2018, 2026)
SESSION_TYPES: List[str] = ['Q', 'R']
DATA_DIR: Path = Path("f1_data")
CACHE_DIR: Path = Path("f1_cache")

# set up logging so progress and errors show in the terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# make sure the cache folder exists before fastf1 starts using it
# this helps avoid redownloading data every time
CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))


class F1DataEncoder(json.JSONEncoder):
    """Handles serialisation of F1-specific types for metadata storage."""

    # this makes sure some pandas and numpy types can still be saved into json
    def default(self, obj: Any) -> Any:
        # timedeltas are turned into strings so json can store them
        if isinstance(obj, pd.Timedelta):
            return str(obj)
        # timestamps and dates are turned into strings
        if isinstance(obj, (pd.Timestamp, datetime.datetime, datetime.date)):
            return obj.isoformat()
        # numpy scalar values get converted into normal python values
        if hasattr(obj, 'item') and callable(getattr(obj, 'item')):
            return obj.item()
        return super().default(obj)


def save_session_metadata(session: fastf1.core.Session, base_path: Path):
    """Serialises session attributes to a JSON file."""

    # this stores a few useful scalar session attributes in a json file
    meta: Dict[str, Any] = {}
    scalar_attrs = [
        'name', 'date', 'api_path', 'session_info',
        'drivers', 'session_start_time', 't0_date',
    ]

    # loop through the chosen attributes and save any that exist
    for attr in scalar_attrs:
        try:
            value = getattr(session, attr, None)
            if value is not None:
                meta[attr] = value
        except Exception:
            pass

    # write the metadata to disk in the session folder
    out_path = base_path / 'session_metadata.json'
    with open(out_path, 'w') as f:
        json.dump(meta, f, indent=4, cls=F1DataEncoder)


def save_dataframe_attributes(session: fastf1.core.Session, base_path: Path):
    """Exports structured DataFrames (laps, results, weather) to Parquet format."""

    # these are the main session level tables we want to save
    df_attrs = [
        'laps', 'results', 'weather_data', 'track_status',
        'session_status', 'race_control_messages',
    ]

    # loop through each dataframe attribute and save it if it exists and is not empty
    for attr in df_attrs:
        try:
            data = getattr(session, attr, None)
            if data is not None and isinstance(data, pd.DataFrame) and not data.empty:
                data = data.copy()
                # force column names to lowercase for consistency later
                data.columns = data.columns.astype(str).str.lower()
                data.to_parquet(base_path / f'{attr}.parquet')
        except Exception as e:
            logger.error(f"Failed to export '{attr}': {e}")


def save_telemetry_data(session: fastf1.core.Session, base_path: Path):
    """Iterates through per-driver telemetry dictionaries and saves to individual files."""

    # telemetry is stored separately for each driver
    # we save car telemetry and position telemetry if available
    for attr in ['car_data', 'pos_data']:
        try:
            data_dict = getattr(session, attr, None)
            if not data_dict:
                continue

            # each entry in the dictionary is one driver number and that driver's telemetry dataframe
            for driver_num, telem in data_dict.items():
                if telem is not None and not telem.empty:
                    telem = telem.copy()
                    telem.columns = telem.columns.astype(str).str.lower()
                    telem.to_parquet(base_path / f'{attr}_driver_{driver_num}.parquet')
        except Exception as e:
            logger.error(f"Failed to process telemetry '{attr}': {e}")


def save_session(session: fastf1.core.Session, path: Path):
    """Main entry point for session serialisation."""

    # make the output folder for this session
    path.mkdir(parents=True, exist_ok=True)

    # save the metadata, tabular data, and per driver telemetry
    save_session_metadata(session, path)
    save_dataframe_attributes(session, path)
    save_telemetry_data(session, path)


def download_season(year: int):
    """Processes all requested session types for a given F1 season."""

    logger.info(f"Starting ingestion for {year} season")

    # first get the season schedule
    schedule = None
    while schedule is None:
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as e:
            logger.warning(f"Network error fetching {year} schedule: {e}. Retrying in 120s...")
            time.sleep(120)

    # go through every event in the schedule
    for _, event in tqdm(schedule.iterrows(), total=len(schedule), desc=f"Season {year}"):
        event_name = event['EventName']

        safe_event_name = event_name.replace('/', '-')

        # then download each session type for that event
        for session_type in SESSION_TYPES:
            try:
                session = fastf1.get_session(year, event_name, session_type)

                # load the full session data we care about
                # this includes laps, telemetry, weather, and messages
                session.load(
                    laps=True,
                    telemetry=True,
                    weather=True,
                    messages=True,
                )

                # build the output path using year, event, and session name
                session_path = DATA_DIR / str(year) / safe_event_name / session.name

                save_session(session, session_path)

                logger.info(f"Successfully cached {event_name} - {session_type}")

            except Exception as e:
                if "is not a valid session type" in str(e):
                    continue

                logger.error(f"Ingestion failed for {event_name} ({session_type}): {e}")

            finally:
                # small pause between requests to reduce risk of rate limiting
                time.sleep(5)


def main():
    # make sure the main output directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # download each season one by one
    for year in YEARS:
        download_season(year)

    logger.info("F1 Data Ingestion Complete.")

if __name__ == "__main__":
    main()