"""
Generate CSV inventory of call-recording blobs from Azure Blob Storage.

Container : call-centre
Prefix    : atpl/cross-sell/intermediate-input/
Filename pattern:
    <LoanAccountNo>_<DD-MM-YYYY>-<HH-MM-SS>_<Language>_<FormatTag>_<DurationSeconds>.wav
    e.g. BH3058CD0000763_18-08-2026-12-47-43_Hindi_mp3_980.wav

Outputs:
    1. all_calls.csv            -> every successfully parsed blob
    2. calls_under_20min.csv    -> subset where duration_seconds <= 1200
    3. parse_errors.csv         -> filenames that couldn't be parsed, with reason

Duration/date/language are derived PURELY from the filename (no per-blob
Azure API calls), so this scales fine to 3L+ files. Listing blobs itself
is a single paginated call to Azure (list_blobs), which is unavoidable
and inherently sequential — but row-building/parsing is parallelized
with a thread pool.
"""

import csv
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from azure.storage.blob import ContainerClient

# ---------------------------------------------------------------------------
# CONFIG - edit these
# ---------------------------------------------------------------------------
AZURE_CONNECTION_STRING = os.environ.get(
    "AZURE_STORAGE_CONNECTION_STRING",
    "PASTE_YOUR_AZURE_CONNECTION_STRING_HERE",
)
CONTAINER_NAME = "call-centre"
BLOB_PREFIX = "atpl/cross-sell/intermediate-input/"

# If your container is private, full URL will still be built (SAS not
# added here). Add a SAS token suffix in build_blob_url() if you need
# publicly clickable links.
ACCOUNT_URL_OVERRIDE = None  # e.g. "https://<account>.blob.core.windows.net" or None to auto-detect

MAX_WORKERS = 32          # thread pool size for parsing
UNDER_LIMIT_SECONDS = 20 * 60  # 20 minutes

OUTPUT_ALL_CSV = "all_calls.csv"
OUTPUT_UNDER20_CSV = "calls_under_20min.csv"
OUTPUT_ERRORS_CSV = "parse_errors.csv"

# ---------------------------------------------------------------------------
# Filename pattern
# BH3058CD0000763_18-08-2026-12-47-43_Hindi_mp3_980.wav
#   group1 = loan_account_no        (BH3058CD0000763)
#   group2 = call_date              (18-08-2026)
#   group3 = call_time              (12-47-43)
#   group4 = language               (Hindi)
#   group5 = format_tag             (mp3)
#   group6 = duration_seconds       (980)
# ---------------------------------------------------------------------------
FILENAME_RE = re.compile(
    r"^(?P<loan_account_no>[A-Za-z0-9]+)"
    r"_(?P<call_date>\d{2}-\d{2}-\d{4})"
    r"-(?P<call_time>\d{2}-\d{2}-\d{2})"
    r"_(?P<language>[A-Za-z]+)"
    r"_(?P<format_tag>[A-Za-z0-9]+)"
    r"_(?P<duration_seconds>\d+)"
    r"\.(?P<extension>[A-Za-z0-9]+)$"
)

csv_lock = threading.Lock()


def build_blob_url(account_url: str, container: str, blob_name: str) -> str:
    """Build the full URL of a blob (no SAS token added by default)."""
    return f"{account_url.rstrip('/')}/{container}/{blob_name}"


def parse_filename(file_name: str):
    """
    Parse a blob's file name into structured fields.
    Returns (row_dict, None) on success, or (None, error_reason) on failure.
    """
    m = FILENAME_RE.match(file_name)
    if not m:
        return None, "filename_pattern_mismatch"

    d = m.groupdict()

    # Validate date
    try:
        call_date_obj = datetime.strptime(d["call_date"], "%d-%m-%Y")
        call_date_iso = call_date_obj.strftime("%Y-%m-%d")
    except ValueError:
        return None, "invalid_date"

    # Validate time
    try:
        time_str = d["call_time"].replace("-", ":")
        datetime.strptime(time_str, "%H:%M:%S")
    except ValueError:
        return None, "invalid_time"

    # Validate duration
    try:
        duration_seconds = int(d["duration_seconds"])
    except ValueError:
        return None, "invalid_duration"

    duration_minutes = round(duration_seconds / 60.0, 2)

    row = {
        "loan_account_no": d["loan_account_no"],
        "call_date": d["call_date"],          # original DD-MM-YYYY
        "call_date_iso": call_date_iso,        # normalized YYYY-MM-DD
        "call_time": time_str,                 # HH:MM:SS
        "language": d["language"],
        "format_tag": d["format_tag"],
        "duration_seconds": duration_seconds,
        "duration_minutes": duration_minutes,
        "extension": d["extension"],
        "file_name": file_name,
    }
    return row, None


def process_blob(blob_name: str, prefix_len: int, account_url: str):
    """Worker function: parse one blob name and build its row (incl. URL)."""
    file_name = blob_name[prefix_len:] if blob_name.startswith(BLOB_PREFIX) else os.path.basename(blob_name)

    row, error = parse_filename(file_name)
    full_url = build_blob_url(account_url, CONTAINER_NAME, blob_name)

    if error:
        return None, {"file_name": file_name, "blob_path": blob_name, "reason": error}

    row["blob_path"] = blob_name
    row["full_url"] = full_url
    return row, None


def main():
    if "PASTE_YOUR" in AZURE_CONNECTION_STRING:
        print("ERROR: Set AZURE_STORAGE_CONNECTION_STRING env var or edit the script config.")
        sys.exit(1)

    print(f"Connecting to container '{CONTAINER_NAME}' ...")
    container_client = ContainerClient.from_connection_string(
        conn_str=AZURE_CONNECTION_STRING, container_name=CONTAINER_NAME
    )

    account_url = ACCOUNT_URL_OVERRIDE or container_client.url.split(f"/{CONTAINER_NAME}")[0]

    print(f"Listing blobs under prefix '{BLOB_PREFIX}' ... (this can take a while for 3L+ files)")
    blob_names = []
    for blob in container_client.list_blobs(name_starts_with=BLOB_PREFIX):
        # skip "directory placeholder" entries if any (size 0, ends with /)
        if blob.name.endswith("/"):
            continue
        blob_names.append(blob.name)

    total = len(blob_names)
    print(f"Found {total} blobs. Parsing with {MAX_WORKERS} threads ...")

    prefix_len = len(BLOB_PREFIX)

    all_rows = []
    error_rows = []
    done_count = 0
    progress_lock = threading.Lock()

    def _worker(name):
        return process_blob(name, prefix_len, account_url)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_worker, name): name for name in blob_names}
        for future in as_completed(futures):
            row, error = future.result()
            if row:
                all_rows.append(row)
            if error:
                error_rows.append(error)

            with progress_lock:
                done_count += 1
                if done_count % 10000 == 0 or done_count == total:
                    print(f"  processed {done_count}/{total}")

    print(f"Parsed OK: {len(all_rows)}  |  Errors: {len(error_rows)}")

    # ---- write all_calls.csv ----
    fieldnames = [
        "loan_account_no",
        "call_date",
        "call_date_iso",
        "call_time",
        "language",
        "format_tag",
        "duration_seconds",
        "duration_minutes",
        "extension",
        "file_name",
        "blob_path",
        "full_url",
    ]

    with open(OUTPUT_ALL_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Written: {OUTPUT_ALL_CSV} ({len(all_rows)} rows)")

    # ---- write calls_under_20min.csv ----
    under20_rows = [r for r in all_rows if r["duration_seconds"] <= UNDER_LIMIT_SECONDS]
    with open(OUTPUT_UNDER20_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(under20_rows)
    print(f"Written: {OUTPUT_UNDER20_CSV} ({len(under20_rows)} rows, <= 20 min)")

    # ---- write parse_errors.csv ----
    if error_rows:
        with open(OUTPUT_ERRORS_CSV, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["file_name", "blob_path", "reason"])
            writer.writeheader()
            writer.writerows(error_rows)
        print(f"Written: {OUTPUT_ERRORS_CSV} ({len(error_rows)} rows)")
    else:
        print("No parse errors.")


if __name__ == "__main__":
    main()
