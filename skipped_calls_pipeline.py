"""
Pipeline: List blobs from Azure Blob Storage (call-centre container) for the
'cross-sell/archive/skipped/<date>/' path across multiple dates, and build a
CSV with parsed metadata from each file name.

Path pattern (everything fixed except the date folder):
    imarque/cross-sell/archive/skipped/<DD-MM-YYYY>/<filename>.mp3

File name pattern:
    <LoanAccountNo>_<AudioFileDate-Time>_<LANGUAGE>.mp3
    e.g. UP3040TW00811_22-08-2026-10-15-42_HINDI.mp3

Output CSV columns:
    Date_processed   -> the folder date (from DATES list below)
    File_name        -> full file name
    Loan_account_no  -> first part of file name (before first "_")
    Audio_file_date  -> second part of file name (between 1st and 2nd "_")
    Language         -> last part of file name (before ".mp3")

Usage:
    1. Fill in AZURE_CONNECTION_STRING, CONTAINER_NAME.
    2. Fill in the DATES list below with all the dates you want to process.
    3. Run: python skipped_calls_pipeline.py
    4. Output CSV will be saved as skipped_calls_output.csv
"""

import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from azure.storage.blob import ContainerClient

# ============================================================
# CONFIG - EDIT THESE
# ============================================================

# Azure connection string (from Storage Account -> Access Keys)
AZURE_CONNECTION_STRING = "PASTE_YOUR_AZURE_CONNECTION_STRING_HERE"

# Container name
CONTAINER_NAME = "call-centre"

# Fixed path prefix (everything except the date folder)
BASE_PREFIX = "imarque/cross-sell/archive/skipped/"

# ---- DATES LIST -> EDIT THIS with all the dates you want to process ----
# Format must match the folder name exactly, e.g. "28-08-2026"
DATES = [
    "28-08-2026",
    # "27-08-2026",
    # "26-08-2026",
    # add more dates here...
]
# ==========================================================

# Number of parallel worker threads (tune based on network/blob throttling)
MAX_WORKERS = 16

# Output CSV file path
OUTPUT_CSV = "/mnt/user-data/outputs/skipped_calls_output.csv"


def list_blobs_for_date(container_client: ContainerClient, date_str: str):
    """
    Lists all blobs under the given date folder and parses each file name
    into the required CSV row fields.
    Returns a list of dict rows.
    """
    prefix = f"{BASE_PREFIX}{date_str}/"
    rows = []

    try:
        blob_list = container_client.list_blobs(name_starts_with=prefix)
        for blob in blob_list:
            full_name = blob.name
            file_name = full_name.split("/")[-1]

            # Skip if not a valid file (e.g. empty "directory" placeholder)
            if not file_name or not file_name.lower().endswith(".mp3"):
                continue

            # Strip extension before splitting on "_"
            name_no_ext = file_name.rsplit(".", 1)[0]
            parts = name_no_ext.split("_")

            if len(parts) < 3:
                # Unexpected file name format - still capture it, blank out missing fields
                loan_account_no = parts[0] if len(parts) > 0 else ""
                audio_file_date = parts[1] if len(parts) > 1 else ""
                language = parts[-1] if len(parts) > 0 else ""
            else:
                loan_account_no = parts[0]
                audio_file_date = parts[1]
                language = parts[-1]  # last part after final "_"

            rows.append({
                "Date_processed": date_str,
                "File_name": file_name,
                "Loan_account_no": loan_account_no,
                "Audio_file_date": audio_file_date,
                "Language": language,
            })

    except Exception as e:
        print(f"[ERROR] Failed listing blobs for date {date_str}: {e}")

    return rows


def main():
    if "PASTE_YOUR_AZURE_CONNECTION_STRING_HERE" in AZURE_CONNECTION_STRING:
        raise ValueError(
            "Please set AZURE_CONNECTION_STRING at the top of the script before running."
        )

    if not DATES:
        raise ValueError("Please add at least one date to the DATES list before running.")

    container_client = ContainerClient.from_connection_string(
        conn_str=AZURE_CONNECTION_STRING,
        container_name=CONTAINER_NAME,
    )

    all_rows = []

    print(f"Processing {len(DATES)} date folder(s) using {MAX_WORKERS} threads...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_date = {
            executor.submit(list_blobs_for_date, container_client, date_str): date_str
            for date_str in DATES
        }

        for future in as_completed(future_to_date):
            date_str = future_to_date[future]
            try:
                rows = future.result()
                print(f"  [{date_str}] -> {len(rows)} files found")
                all_rows.extend(rows)
            except Exception as e:
                print(f"[ERROR] Date {date_str} generated an exception: {e}")

    if not all_rows:
        print("No files found for the given dates. CSV will not be created.")
        return

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    fieldnames = ["Date_processed", "File_name", "Loan_account_no", "Audio_file_date", "Language"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nDone. {len(all_rows)} total rows written to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
