"""
SafeStride — Step 2: Fetch & Clean Chicago Crime Data
=====================================================
Fetches crime incident data from the Chicago Open Data Portal (Socrata SODA API),
cleans it for geospatial safety analysis, and saves a production-ready CSV.

Dataset : "Crimes - 2001 to Present"
Dataset ID : ijzp-q8t2
Endpoint : https://data.cityofchicago.org/resource/ijzp-q8t2.json
Auth     : No API key required for public datasets (throttled to ~1000 req/hr)

Why Socrata/SODA?
  The SODA API supports SoQL (SQL-like) queries — we push date filtering and
  column selection to the server so we only download what we need, saving
  bandwidth and memory.

Usage:
  python scripts/fetch_crime_data.py
"""

import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
# We set every tuneable at the top so nothing is buried in the logic.

BASE_URL = "https://data.cityofchicago.org/resource/ijzp-q8t2.json"

# Columns we actually need for safety scoring.
# Selecting only these columns keeps the response small and speeds up parsing.
COLUMNS = [
    "date",           # ISO-8601 timestamp of the incident
    "primary_type",   # Crime category (THEFT, BATTERY, etc.)
    "description",    # Sub-category detail
    "location_description",  # Where it happened (STREET, SIDEWALK, etc.)
    "latitude",       # WGS-84 latitude  — needed for geospatial join
    "longitude",      # WGS-84 longitude — needed for geospatial join
    "arrest",         # Whether an arrest was made
    "domestic",       # Whether it was domestic
]

# Dynamic date filter: always the last 2 years from *right now*.
# We never hardcode dates because the script should stay valid over time.
TWO_YEARS_AGO = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%dT%H:%M:%S")

# Pagination settings.
# Socrata caps $limit at 50 000 per request; we match that to minimise round-trips.
# MAX_ROWS is a safety cap so we don't accidentally download millions of rows
# during development. Remove or raise it for production.
PAGE_SIZE = 50_000
MAX_ROWS  = 200_000

# Output path — relative to repo root, works from any working directory.
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "data")
OUTPUT_FILE  = os.path.join(OUTPUT_DIR, "crimes_cleaned.csv")

# Retry settings for transient network failures.
MAX_RETRIES   = 3
RETRY_DELAY_S = 5  # seconds between retries


# ──────────────────────────────────────────────
# DATA FETCHING
# ──────────────────────────────────────────────
def fetch_crime_data() -> pd.DataFrame:
    """
    Page through the SODA API and return one combined DataFrame.

    We use $offset pagination (not cursor-based) because:
      1. The SODA API supports it natively with $offset + $limit.
      2. It's simple and predictable for bounded datasets.
      3. We add $order=date DESC so newest records come first.
    """
    all_pages: list[pd.DataFrame] = []
    offset = 0

    # Build the column-select clause once.
    # $select tells the server to return only these columns (like SQL SELECT).
    select_clause = ",".join(COLUMNS)

    print(f"📡  Fetching Chicago crime data (last 2 years, since {TWO_YEARS_AGO[:10]})")
    print(f"    Endpoint : {BASE_URL}")
    print(f"    Page size: {PAGE_SIZE:,} rows")
    print(f"    Max rows : {MAX_ROWS:,}")
    print()

    while offset < MAX_ROWS:
        # Build query parameters for this page.
        # $where filters server-side so we never download old data.
        # $order ensures deterministic pagination (without it, rows can shift between pages).
        params = {
            "$select":  select_clause,
            "$where":   f"date > '{TWO_YEARS_AGO}'",
            "$order":   "date DESC",
            "$limit":   PAGE_SIZE,
            "$offset":  offset,
        }

        # Retry loop — transient timeouts and 5xx errors are common on public APIs.
        page_df = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = requests.get(BASE_URL, params=params, timeout=60)
                resp.raise_for_status()  # Raises HTTPError for 4xx/5xx

                data = resp.json()

                # An empty list means we've exhausted all matching rows.
                if not data:
                    print(f"    ℹ️  Page {offset // PAGE_SIZE + 1}: empty response — all data fetched.")
                    return _combine_pages(all_pages)

                page_df = pd.DataFrame(data)
                break  # Success — exit retry loop

            except requests.exceptions.Timeout:
                print(f"    ⏳ Timeout on page {offset // PAGE_SIZE + 1}, attempt {attempt}/{MAX_RETRIES}.")
                if attempt < MAX_RETRIES:
                    print(f"       Retrying in {RETRY_DELAY_S}s...")
                    time.sleep(RETRY_DELAY_S)
                else:
                    print("    ❌ Max retries exceeded on timeout. Returning data collected so far.")
                    return _combine_pages(all_pages)

            except requests.exceptions.ConnectionError as exc:
                print(f"    ❌ Connection error: {exc}")
                if attempt < MAX_RETRIES:
                    print(f"       Retrying in {RETRY_DELAY_S}s...")
                    time.sleep(RETRY_DELAY_S)
                else:
                    print("    ❌ Max retries exceeded. Returning data collected so far.")
                    return _combine_pages(all_pages)

            except requests.exceptions.HTTPError as exc:
                print(f"    ❌ HTTP error: {exc}")
                # 4xx errors (bad query) won't fix themselves — bail immediately.
                if resp.status_code < 500:
                    print("       Client error — aborting.")
                    sys.exit(1)
                if attempt < MAX_RETRIES:
                    print(f"       Server error — retrying in {RETRY_DELAY_S}s...")
                    time.sleep(RETRY_DELAY_S)
                else:
                    print("    ❌ Max retries exceeded. Returning data collected so far.")
                    return _combine_pages(all_pages)

            except ValueError as exc:
                # requests.Response.json() raises ValueError if body isn't valid JSON.
                print(f"    ❌ JSON decode error: {exc}")
                print("       The API returned non-JSON data. Aborting.")
                sys.exit(1)

        # If we get here page_df should be set, but guard just in case.
        if page_df is None or page_df.empty:
            print("    ℹ️  Received empty page — stopping pagination.")
            break

        all_pages.append(page_df)
        total_so_far = sum(len(p) for p in all_pages)
        print(f"    ✅ Page {offset // PAGE_SIZE + 1}: got {len(page_df):,} rows  |  Total so far: {total_so_far:,}")

        # If the page returned fewer rows than PAGE_SIZE, there are no more pages.
        if len(page_df) < PAGE_SIZE:
            break

        offset += PAGE_SIZE

    return _combine_pages(all_pages)


def _combine_pages(pages: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate a list of DataFrames, handling the empty-list edge case."""
    if not pages:
        print("    ⚠️  No data was fetched. Check your date filter or network connection.")
        sys.exit(1)
    return pd.concat(pages, ignore_index=True)


# ──────────────────────────────────────────────
# DATA CLEANING
# ──────────────────────────────────────────────
def clean_crime_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw API data into analysis-ready format.

    Why each step matters:
      - datetime parsing:  lets us do time-series analysis & temporal joins
      - hour / day_of_week: pre-computed features for ML models (crime varies by time)
      - float conversion:   lat/lon arrive as strings from JSON; floats needed for geometry
      - null/zero drop:     rows without coordinates can't be placed on the road network
      - dedup:              the API occasionally returns duplicate incident IDs
    """
    rows_before = len(df)
    print(f"\n🧹  Cleaning {rows_before:,} raw rows...")

    # 1. Parse the ISO-8601 date string into a proper datetime object.
    #    errors='coerce' turns unparseable dates into NaT instead of crashing.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 2. Extract hour (0-23) — crime peaks at specific hours; this is a strong ML feature.
    df["hour"] = df["date"].dt.hour

    # 3. Extract day of week (0=Mon, 6=Sun) — weekday vs weekend patterns differ.
    df["day_of_week"] = df["date"].dt.dayofweek

    # 4. Convert lat/lon from strings to float64.
    #    errors='coerce' turns non-numeric values (like empty strings) into NaN.
    df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # 5. Drop rows where coordinates are missing — can't map them to roads.
    df.dropna(subset=["latitude", "longitude"], inplace=True)

    # 6. Drop rows where lat or lon is exactly 0.0 — these are placeholder values,
    #    not real Chicago coordinates (Chicago is ~41.8°N, -87.6°W).
    df = df[(df["latitude"] != 0.0) & (df["longitude"] != 0.0)]

    # 7. Drop rows with unparseable dates (NaT).
    df.dropna(subset=["date"], inplace=True)

    # 8. Remove exact-duplicate rows. The API can return duplicates when
    #    records are updated between pagination requests.
    df.drop_duplicates(inplace=True)

    # 9. Reset index so it runs 0..N-1 cleanly after all the drops.
    df.reset_index(drop=True, inplace=True)

    rows_after = len(df)
    dropped = rows_before - rows_after
    print(f"    Dropped {dropped:,} rows ({dropped / rows_before * 100:.1f}%)")
    print(f"    Remaining: {rows_after:,} clean rows")

    return df


# ──────────────────────────────────────────────
# SAVE & REPORT
# ──────────────────────────────────────────────
def save_and_report(df: pd.DataFrame) -> None:
    """Save the cleaned DataFrame to CSV and print a summary report."""

    # Create the output directory if it doesn't exist.
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save without the pandas index column — downstream code should not depend on it.
    df.to_csv(OUTPUT_FILE, index=False)
    file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"\n💾  Saved to {OUTPUT_FILE}")
    print(f"    File size: {file_size_mb:.1f} MB")

    # ── Summary report ──
    print(f"\n📊  Dataset Summary")
    print(f"    Total rows   : {len(df):,}")
    print(f"    Date range   : {df['date'].min()} → {df['date'].max()}")
    print(f"    Null lat/lon : {df['latitude'].isna().sum()} / {df['longitude'].isna().sum()}")

    print(f"\n    Columns & dtypes:")
    for col in df.columns:
        print(f"      {col:30s} {str(df[col].dtype)}")

    print(f"\n    Preview (first 5 rows):")
    print(df.head().to_string(index=False))


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  SafeStride — Step 2: Fetch & Clean Chicago Crime Data")
    print("=" * 60)

    raw_df     = fetch_crime_data()
    cleaned_df = clean_crime_data(raw_df)
    save_and_report(cleaned_df)

    print("\n" + "=" * 60)
    print("  ✅ Step 2 complete!")
    print("=" * 60)
