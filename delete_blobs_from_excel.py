"""
Delete Azure Blob files whose names are listed in an Excel file.

WHAT THIS SCRIPT DOES
----------------------
1. Reads the 'file_name' column from your Excel (~1,00,000 rows).
2. Lists ALL blobs under the given folder/prefix in your container (~3,00,000 files).
3. Finds the intersection (blobs that exist AND are present in the Excel list).
4. Deletes ONLY those matched blobs, using a thread pool for speed.
5. Prints exact counts: total in excel, total in blob folder, matched/found,
   deleted successfully, failed, and not-found-in-blob.
6. Writes a report (CSV) listing found / deleted / failed / not-found file names.

SAFETY
------
- DRY_RUN = True by default. It will NOT delete anything, only show you the
  counts and write the report. Set DRY_RUN = False only after you've checked
  the report and are sure.
- Only files that are BOTH in the Excel AND actually present in the blob
  folder are touched. Nothing else in the folder is deleted.

INSTALL (run once)
-------------------
pip install azure-storage-blob openpyxl pandas
"""

import csv
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobServiceClient

# ============================================================
# ============  PATHS / CONFIG - EDIT THESE  ==================
# ============================================================

# Path to your excel file (local path on your machine/server)
EXCEL_PATH = r"/path/to/your/file_names.xlsx"

# Name of the column in the excel that contains the file names
EXCEL_COLUMN = "file_name"

# Azure Storage connection string (from Azure Portal -> Storage Account -> Access Keys)
AZURE_CONNECTION_STRING = "PASTE_YOUR_AZURE_STORAGE_CONNECTION_STRING_HERE"

# Container name (the top-level container, NOT the full path)
CONTAINER_NAME = "call-centre"

# Folder path (prefix) INSIDE the container where the files live
# Note: no leading slash, must end with "/"
BLOB_FOLDER_PREFIX = "imarque/cross-sell/raw-input-2026-07-final/"

# Where to save the output report (CSV)
REPORT_PATH = r"/path/to/output/delete_report.csv"

# How many parallel threads to use for delete calls
MAX_WORKERS = 32

# SAFETY SWITCH: keep True until you've reviewed the report at least once.
# True  -> only simulates, prints counts, writes report, deletes NOTHING
# False -> actually deletes the matched blobs
DRY_RUN = True

# ============================================================


def load_excel_filenames(path: str, column: str) -> set:
    """Read the excel and return a set of cleaned (stripped) file names."""
    print(f"[1/4] Reading Excel: {path}")
    df = pd.read_excel(path, usecols=[column], engine="openpyxl")

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found in Excel. Found columns: {list(df.columns)}"
        )

    # Drop blanks/NaN, strip whitespace, drop duplicates
    names = (
        df[column]
        .dropna()
        .astype(str)
        .str.strip()
    )
    names = names[names != ""]
    name_set = set(names.tolist())

    print(f"      Excel rows read        : {len(df)}")
    print(f"      Unique file names found : {len(name_set)}")
    return name_set


def list_blob_filenames(container_client, prefix: str) -> dict:
    """
    List all blobs under prefix.
    Returns a dict mapping: bare_file_name -> full_blob_name (with folder path)

    bare_file_name = whatever appears after the last '/' in the blob path.
    This is what gets matched against the Excel 'file_name' column.
    If your excel file_name column actually stores the FULL path
    (folder + filename), see the NOTE below this function.
    """
    print(f"[2/4] Listing blobs under prefix: {prefix}")
    blob_map = {}
    count = 0
    for blob in container_client.list_blobs(name_starts_with=prefix):
        bare_name = blob.name.split("/")[-1]
        if bare_name == "":  # skip folder markers
            continue
        blob_map[bare_name] = blob.name
        count += 1
        if count % 50000 == 0:
            print(f"      ...listed {count} blobs so far")

    print(f"      Total blobs in folder   : {count}")
    return blob_map


# NOTE: If your Excel 'file_name' column contains the FULL blob path
# (e.g. "imarque/cross-sell/raw-input-2026-07-final/abc.csv") instead of
# just "abc.csv", then instead of matching on bare_name, match directly on
# blob.name. In that case, change list_blob_filenames() to:
#     blob_map[blob.name] = blob.name
# and skip the .split("/")[-1] step. Everything else stays the same.


def delete_one_blob(container_client, blob_name: str):
    """Delete a single blob. Returns (blob_name, success_bool, error_message)."""
    try:
        container_client.delete_blob(blob_name)
        return (blob_name, True, "")
    except ResourceNotFoundError:
        return (blob_name, False, "not found at delete time")
    except Exception as e:
        return (blob_name, False, str(e))


def main():
    if not os.path.exists(EXCEL_PATH):
        print(f"ERROR: Excel file not found at {EXCEL_PATH}")
        sys.exit(1)

    if "PASTE_YOUR" in AZURE_CONNECTION_STRING:
        print("ERROR: Please set AZURE_CONNECTION_STRING at the top of the script.")
        sys.exit(1)

    # --- Step 1: read excel file names ---
    excel_names = load_excel_filenames(EXCEL_PATH, EXCEL_COLUMN)

    # --- Step 2: connect to Azure and list blobs in folder ---
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)

    blob_map = list_blob_filenames(container_client, BLOB_FOLDER_PREFIX)

    # --- Step 3: find matches (found in both excel + blob folder) ---
    print("[3/4] Matching excel file names against blob folder...")
    found_names = excel_names & blob_map.keys()          # present in excel AND blob
    not_found_names = excel_names - blob_map.keys()      # in excel but NOT in blob folder

    found_full_paths = [blob_map[name] for name in found_names]

    print(f"      Found (to delete)       : {len(found_names)}")
    print(f"      In excel but NOT in blob: {len(not_found_names)}")

    if DRY_RUN:
        print("\n*** DRY_RUN = True -> no files will be deleted. ***")
        print("*** Review the report, then set DRY_RUN = False to actually delete. ***\n")

    # --- Step 4: delete matched blobs using multithreading ---
    deleted = []
    failed = []

    if not DRY_RUN and found_full_paths:
        print(f"[4/4] Deleting {len(found_full_paths)} blobs using {MAX_WORKERS} threads...")
        lock = threading.Lock()
        progress = {"done": 0}

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(delete_one_blob, container_client, blob_name): blob_name
                for blob_name in found_full_paths
            }
            for future in as_completed(futures):
                blob_name, success, err = future.result()
                with lock:
                    progress["done"] += 1
                    if success:
                        deleted.append(blob_name)
                    else:
                        failed.append((blob_name, err))
                    if progress["done"] % 5000 == 0:
                        print(f"      ...processed {progress['done']} / {len(found_full_paths)}")
    else:
        print("[4/4] Skipped actual deletion (DRY_RUN mode or nothing to delete).")

    # --- Write report ---
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["status", "file_name"])
        for name in found_full_paths:
            if DRY_RUN:
                writer.writerow(["found_would_delete", name])
        for name in deleted:
            writer.writerow(["deleted", name])
        for name, err in failed:
            writer.writerow([f"delete_failed: {err}", name])
        for name in not_found_names:
            writer.writerow(["not_found_in_blob", name])

    # --- Final summary ---
    print("\n========== SUMMARY ==========")
    print(f"Total rows in Excel          : {len(excel_names)} (unique file names)")
    print(f"Total blobs in target folder  : {len(blob_map)}")
    print(f"Matched / Found for deletion   : {len(found_names)}")
    print(f"Not found in blob folder       : {len(not_found_names)}")
    if not DRY_RUN:
        print(f"Successfully deleted           : {len(deleted)}")
        print(f"Failed to delete                : {len(failed)}")
    print(f"Report written to              : {REPORT_PATH}")
    print("==============================\n")


if __name__ == "__main__":
    main()
