"""
Copy 2 lakh+ audio blobs from one folder to another inside the same Azure
Blob Storage container, using multithreading + server-side copy (fast,
no bytes downloaded/uploaded through this machine).

============================
FILL THESE IN BEFORE RUNNING
============================
"""

EXCEL_FILE_PATH   = r"final_atpl_calls_process.xlsx"   # <-- your Excel file path
EXCEL_SHEET_NAME  = 0                                   # <-- sheet name or index, 0 = first sheet
URL_COLUMN_NAME   = "full_url"                          # <-- column with the full blob URLs

AZURE_CONNECTION_STRING = "REPLACE_WITH_YOUR_AZURE_STORAGE_CONNECTION_STRING"

CONTAINER_NAME       = "call-centre"
DEST_FOLDER_PREFIX   = "atpl/cross-sell/filtered_final_27-august/"  # trailing slash matters

MAX_WORKERS = 32   # thread count - tune based on your bandwidth / Azure throttling limits

# Where to write the run's result log (success/failure per row)
RESULT_LOG_PATH = "copy_results_log.csv"

# ============================================================

import pandas as pd
from azure.storage.blob import BlobServiceClient
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, unquote
import threading
import time
import csv

# Thread-safe counters
lock = threading.Lock()
stats = {"success": 0, "failed": 0, "skipped": 0}
results = []  # (source_blob_path, dest_blob_path, status, message)


def extract_blob_path(full_url: str, container_name: str) -> str:
    """
    Extracts the blob path (everything after the container name) from a full
    blob URL, regardless of the account name / domain.

    Example:
    https://xxxxaccount.blob.core.windows.net/call-centre/atpl/cross-sell/intermediate-input/file.wav
    -> atpl/cross-sell/intermediate-input/file.wav
    """
    parsed = urlparse(full_url)
    path = unquote(parsed.path)  # decode %20 etc.
    path = path.lstrip("/")      # remove leading slash

    prefix = container_name + "/"
    if path.startswith(prefix):
        return path[len(prefix):]

    raise ValueError(f"Container name '{container_name}' not found in URL path: {full_url}")


def copy_one_blob(blob_service_client: BlobServiceClient, source_url: str, idx: int, total: int):
    try:
        source_blob_path = extract_blob_path(source_url, CONTAINER_NAME)
        file_name = source_blob_path.rsplit("/", 1)[-1]
        dest_blob_path = DEST_FOLDER_PREFIX + file_name

        dest_blob_client = blob_service_client.get_blob_client(
            container=CONTAINER_NAME, blob=dest_blob_path
        )

        # Skip if already copied (avoids re-copying on re-runs)
        if dest_blob_client.exists():
            with lock:
                stats["skipped"] += 1
            results.append((source_blob_path, dest_blob_path, "skipped", "already exists"))
            return

        # Server-side copy: Azure copies directly, no data through this machine
        dest_blob_client.start_copy_from_url(source_url)

        with lock:
            stats["success"] += 1
        results.append((source_blob_path, dest_blob_path, "started", ""))

    except Exception as e:
        with lock:
            stats["failed"] += 1
        results.append((source_url, "", "failed", str(e)))

    finally:
        with lock:
            done = stats["success"] + stats["failed"] + stats["skipped"]
        if done % 500 == 0 or done == total:
            print(f"[{done}/{total}] success={stats['success']} "
                  f"skipped={stats['skipped']} failed={stats['failed']}")


def main():
    print("Reading Excel file...")
    df = pd.read_excel(EXCEL_FILE_PATH, sheet_name=EXCEL_SHEET_NAME)

    if URL_COLUMN_NAME not in df.columns:
        raise ValueError(f"Column '{URL_COLUMN_NAME}' not found. Available columns: {list(df.columns)}")

    urls = df[URL_COLUMN_NAME].dropna().astype(str).tolist()
    total = len(urls)
    print(f"Found {total} URLs to copy.")

    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(copy_one_blob, blob_service_client, url, i, total)
            for i, url in enumerate(urls, start=1)
        ]
        for f in as_completed(futures):
            pass  # progress already printed inside copy_one_blob

    elapsed = time.time() - start_time
    print("\n=== DONE ===")
    print(f"Total: {total} | Success(started): {stats['success']} | "
          f"Skipped(existing): {stats['skipped']} | Failed: {stats['failed']}")
    print(f"Time taken: {elapsed:.1f} seconds")

    # Write result log
    with open(RESULT_LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_blob_path", "dest_blob_path", "status", "message"])
        writer.writerows(results)
    print(f"Result log written to: {RESULT_LOG_PATH}")


if __name__ == "__main__":
    main()
