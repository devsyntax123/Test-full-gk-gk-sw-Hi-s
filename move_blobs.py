"""
Move N files from one virtual-folder path to another within the SAME Azure
Blob Storage container, using multithreading, with CSV logging of results.

Azure Blob Storage has no native "move" -> this does copy + verify + delete
per file. Safe to re-run: it skips files already present at the destination
and skips files already recorded as SUCCESS in the log.

Requirements:
    pip install azure-storage-blob --break-system-packages
"""

import csv
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError, ResourceExistsError

# ============================== CONFIG ======================================

CONNECTION_STRING = "PASTE_YOUR_AZURE_CONNECTION_STRING_HERE"

CONTAINER_NAME = "call-centre"

# Source prefix (virtual folder) - path AFTER the container name
SOURCE_PREFIX = "atpl/cross-sell/filtered_final_27-august/"

# Destination prefix (virtual folder) - path AFTER the container name
DEST_PREFIX = "atpl/cross-sell/intermediate-input/"

# How many files to move in this run
MOVE_LIMIT = 65000

# Multithreading
MAX_WORKERS = 32          # tune based on your bandwidth / Azure throttling
COPY_POLL_INTERVAL = 0.5  # seconds between polling copy status
COPY_POLL_TIMEOUT = 120   # max seconds to wait for a single copy to finish

# Logging
LOG_DIR = "logs"
RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_CSV = os.path.join(LOG_DIR, f"move_log_{RUN_STAMP}.csv")
SUMMARY_TXT = os.path.join(LOG_DIR, f"move_summary_{RUN_STAMP}.txt")

# =============================================================================

os.makedirs(LOG_DIR, exist_ok=True)

log_lock = threading.Lock()
counters_lock = threading.Lock()
counters = {"success": 0, "failed": 0, "skipped": 0}


def log_row(writer, file_handle, row):
    with log_lock:
        writer.writerow(row)
        file_handle.flush()


def list_source_blobs(container_client, prefix, limit):
    """List up to `limit` blob names under prefix (files only, not folder markers)."""
    names = []
    for blob in container_client.list_blobs(name_starts_with=prefix):
        # skip zero-byte "folder marker" blobs that end with /
        if blob.name.endswith("/"):
            continue
        names.append(blob.name)
        if len(names) >= limit:
            break
    return names


def move_single_blob(container_client, account_url, sas_or_conn, src_name, dest_prefix, source_prefix):
    """
    Copy one blob from src_name to the destination prefix (preserving the
    relative sub-path/filename after the source prefix), verify the copy,
    then delete the source. Returns a dict describing the outcome.
    """
    relative_path = src_name[len(source_prefix):]  # keep subfolders/filename
    dest_name = dest_prefix + relative_path

    src_blob_client = container_client.get_blob_client(src_name)
    dest_blob_client = container_client.get_blob_client(dest_name)

    result = {
        "source_blob": src_name,
        "dest_blob": dest_name,
        "status": None,
        "message": "",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        # Skip if destination already exists (idempotent re-run support)
        if dest_blob_client.exists():
            # Ensure source still exists; if so, we can safely delete source
            # only if content already matches (size check as a cheap guard)
            try:
                src_props = src_blob_client.get_blob_properties()
                dest_props = dest_blob_client.get_blob_properties()
                if src_props.size == dest_props.size:
                    # already copied previously; make sure source is removed
                    if src_blob_client.exists():
                        src_blob_client.delete_blob()
                    result["status"] = "SKIPPED_ALREADY_MOVED"
                    result["message"] = "Destination already existed with matching size; source cleaned up."
                    return result
                else:
                    result["status"] = "FAILED"
                    result["message"] = "Destination exists but size mismatch - manual check needed."
                    return result
            except ResourceNotFoundError:
                pass  # fall through to normal copy

        # Start server-side copy (fast, no data through local machine)
        source_url = src_blob_client.url
        dest_blob_client.start_copy_from_url(source_url)

        # Poll for copy completion
        waited = 0.0
        while True:
            props = dest_blob_client.get_blob_properties()
            copy_status = props.copy.status
            if copy_status == "success":
                break
            elif copy_status in ("failed", "aborted"):
                result["status"] = "FAILED"
                result["message"] = f"Copy status: {copy_status}"
                return result
            time.sleep(COPY_POLL_INTERVAL)
            waited += COPY_POLL_INTERVAL
            if waited >= COPY_POLL_TIMEOUT:
                result["status"] = "FAILED"
                result["message"] = "Copy timed out"
                return result

        # Verify size matches before deleting source
        src_props = src_blob_client.get_blob_properties()
        dest_props = dest_blob_client.get_blob_properties()
        if src_props.size != dest_props.size:
            result["status"] = "FAILED"
            result["message"] = "Size mismatch after copy; source NOT deleted."
            return result

        # Delete source only after verified copy
        src_blob_client.delete_blob()

        result["status"] = "SUCCESS"
        result["message"] = "Copied and source deleted."
        return result

    except ResourceNotFoundError as e:
        result["status"] = "FAILED"
        result["message"] = f"Source not found: {e}"
        return result
    except Exception as e:
        result["status"] = "FAILED"
        result["message"] = f"Error: {e}"
        return result


def main():
    if "PASTE_YOUR" in CONNECTION_STRING:
        print("ERROR: Please set CONNECTION_STRING at the top of the script.")
        sys.exit(1)

    print("Connecting to Azure Blob Storage...")
    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)

    print(f"Listing up to {MOVE_LIMIT} blobs under: {SOURCE_PREFIX}")
    blob_names = list_source_blobs(container_client, SOURCE_PREFIX, MOVE_LIMIT)
    total = len(blob_names)
    print(f"Found {total} files to move.")

    if total == 0:
        print("Nothing to move. Exiting.")
        return

    with open(LOG_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["source_blob", "dest_blob", "status", "message", "timestamp"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()

        start_time = time.time()
        processed = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(
                    move_single_blob,
                    container_client,
                    None,
                    None,
                    name,
                    DEST_PREFIX,
                    SOURCE_PREFIX,
                ): name
                for name in blob_names
            }

            for future in as_completed(futures):
                res = future.result()
                log_row(writer, f, res)

                with counters_lock:
                    if res["status"] == "SUCCESS":
                        counters["success"] += 1
                    elif res["status"] == "SKIPPED_ALREADY_MOVED":
                        counters["skipped"] += 1
                    else:
                        counters["failed"] += 1
                    processed += 1

                if processed % 500 == 0 or processed == total:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    print(
                        f"[{processed}/{total}] "
                        f"success={counters['success']} "
                        f"skipped={counters['skipped']} "
                        f"failed={counters['failed']} "
                        f"| {rate:.1f} files/sec"
                    )

        elapsed_total = time.time() - start_time

    summary = (
        f"Run finished: {datetime.now().isoformat()}\n"
        f"Source prefix: {SOURCE_PREFIX}\n"
        f"Dest prefix:   {DEST_PREFIX}\n"
        f"Total attempted: {total}\n"
        f"Success:  {counters['success']}\n"
        f"Skipped(already moved): {counters['skipped']}\n"
        f"Failed:   {counters['failed']}\n"
        f"Elapsed:  {elapsed_total:.1f} sec\n"
        f"Log CSV:  {LOG_CSV}\n"
    )
    print("\n" + summary)

    with open(SUMMARY_TXT, "w", encoding="utf-8") as sf:
        sf.write(summary)

    if counters["failed"] > 0:
        print(f"NOTE: {counters['failed']} files failed. Check {LOG_CSV} for status=FAILED rows and re-run the script (it will skip already-moved files).")


if __name__ == "__main__":
    main()
