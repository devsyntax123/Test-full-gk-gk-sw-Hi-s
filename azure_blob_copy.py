"""
==============================================================================
 AZURE BLOB COPY SCRIPT - MULTITHREADED WITH PERIODIC CSV LOGGING
==============================================================================
MENTION AT STARTING (READ BEFORE RUNNING):

1. Fill in the CONFIGURATION section below:
   - CONNECTION_STRING : your Azure Storage account connection string
   - SOURCE_CONTAINER   : container name that holds the source path
   - SOURCE_PREFIX      : "call-centre/imarque/cross-sell/raw-input-2026-07-final/"
   - DEST_CONTAINER     : container name for destination
                          (if source & destination are in the SAME container,
                           just set DEST_CONTAINER = SOURCE_CONTAINER)
   - DEST_PREFIX        : "call-centre-raw-input/imarque/cross-sell/raw-input/"

2. This script does SERVER-SIDE COPY (start_copy_from_url), so files are
   copied directly within Azure - no download/upload through this machine.
   This is fast and cheap even for 2.5 lakh (250,000+) files.

3. Multithreading: uses a ThreadPoolExecutor. MAX_WORKERS controls
   concurrency (default 32). Tune based on your network / Azure throttling
   limits.

4. Logging: a CSV log file is written incrementally. Every LOG_FLUSH_INTERVAL
   seconds (default 30s) OR every LOG_BATCH_SIZE records (default 500,
   whichever comes first), the script flushes completed copy records
   (file_name, source_path, destination_path, status, timestamp) to the CSV.
   This means you can tail the CSV while the job is running to see progress.

5. Resumability: if the script is interrupted and re-run, it will SKIP any
   blob that already exists at the destination (checked via a quick
   metadata call) - so it's safe to re-run. Set SKIP_EXISTING = False to
   disable this and force re-copy of everything.

6. Run with:  python3 azure_blob_copy.py
   No command-line / parser arguments are used - everything is configured
   in the CONFIGURATION section directly below.
==============================================================================
"""

import csv
import os
import sys
import time
import threading
import queue
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError, HttpResponseError

# ==============================================================================
# CONFIGURATION - EDIT THESE VALUES
# ==============================================================================

CONNECTION_STRING = "PASTE_YOUR_AZURE_STORAGE_CONNECTION_STRING_HERE"

SOURCE_CONTAINER = "PASTE_SOURCE_CONTAINER_NAME_HERE"
SOURCE_PREFIX = "call-centre/imarque/cross-sell/raw-input-2026-07-final/"

DEST_CONTAINER = "PASTE_DEST_CONTAINER_NAME_HERE"   # same as SOURCE_CONTAINER if applicable
DEST_PREFIX = "call-centre-raw-input/imarque/cross-sell/raw-input/"

MAX_WORKERS = 32                 # number of parallel copy threads
SKIP_EXISTING = True             # skip files already present at destination
COPY_POLL_INTERVAL_SECONDS = 0.5 # how often to poll async copy status
COPY_TIMEOUT_SECONDS = 300       # max wait for a single blob copy to finish

LOG_DIR = "./logs"
LOG_FILE_PREFIX = "blob_copy_log"
LOG_FLUSH_INTERVAL = 30          # seconds
LOG_BATCH_SIZE = 500             # records

FAILED_LOG_FILE_PREFIX = "blob_copy_failures"

# ==============================================================================
# INTERNAL STATE
# ==============================================================================

os.makedirs(LOG_DIR, exist_ok=True)
_run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
SUCCESS_LOG_PATH = os.path.join(LOG_DIR, f"{LOG_FILE_PREFIX}_{_run_ts}.csv")
FAILURE_LOG_PATH = os.path.join(LOG_DIR, f"{FAILED_LOG_FILE_PREFIX}_{_run_ts}.csv")

log_queue = queue.Queue()
failure_queue = queue.Queue()

counters_lock = threading.Lock()
counters = {
    "total": 0,
    "copied": 0,
    "skipped": 0,
    "failed": 0,
}

stop_logging_flag = threading.Event()


def init_csv(path, headers):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)


def logger_thread_func():
    """Background thread: periodically flushes queued log records to CSV."""
    success_headers = ["file_name", "source_path", "destination_path", "status", "timestamp_utc"]
    failure_headers = ["file_name", "source_path", "destination_path", "error", "timestamp_utc"]
    init_csv(SUCCESS_LOG_PATH, success_headers)
    init_csv(FAILURE_LOG_PATH, failure_headers)

    last_flush = time.time()
    success_buffer = []
    failure_buffer = []

    while not stop_logging_flag.is_set() or not log_queue.empty() or not failure_queue.empty():
        try:
            while True:
                record = log_queue.get_nowait()
                success_buffer.append(record)
                if len(success_buffer) >= LOG_BATCH_SIZE:
                    break
        except queue.Empty:
            pass

        try:
            while True:
                record = failure_queue.get_nowait()
                failure_buffer.append(record)
                if len(failure_buffer) >= LOG_BATCH_SIZE:
                    break
        except queue.Empty:
            pass

        now = time.time()
        should_flush = (
            len(success_buffer) >= LOG_BATCH_SIZE
            or len(failure_buffer) >= LOG_BATCH_SIZE
            or (now - last_flush) >= LOG_FLUSH_INTERVAL
        )

        if should_flush:
            if success_buffer:
                with open(SUCCESS_LOG_PATH, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerows(success_buffer)
                success_buffer = []
            if failure_buffer:
                with open(FAILURE_LOG_PATH, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerows(failure_buffer)
                failure_buffer = []
            last_flush = now

            with counters_lock:
                print(
                    f"[progress] total={counters['total']} "
                    f"copied={counters['copied']} "
                    f"skipped={counters['skipped']} "
                    f"failed={counters['failed']}"
                )

        time.sleep(1)

    # final flush on shutdown
    if success_buffer:
        with open(SUCCESS_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(success_buffer)
    if failure_buffer:
        with open(FAILURE_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(failure_buffer)


def copy_single_blob(blob_service_client, blob_name):
    """Copy one blob from source path to destination path (server-side copy)."""
    file_name = blob_name.split("/")[-1]
    relative_path = blob_name[len(SOURCE_PREFIX):] if blob_name.startswith(SOURCE_PREFIX) else blob_name
    dest_blob_name = DEST_PREFIX + relative_path

    source_client = blob_service_client.get_blob_client(container=SOURCE_CONTAINER, blob=blob_name)
    dest_client = blob_service_client.get_blob_client(container=DEST_CONTAINER, blob=dest_blob_name)

    ts = datetime.now(timezone.utc).isoformat()

    try:
        if SKIP_EXISTING:
            try:
                dest_client.get_blob_properties()
                with counters_lock:
                    counters["skipped"] += 1
                    counters["total"] += 1
                log_queue.put([file_name, blob_name, dest_blob_name, "SKIPPED_EXISTS", ts])
                return
            except ResourceNotFoundError:
                pass  # doesn't exist yet, proceed to copy

        source_url = source_client.url
        dest_client.start_copy_from_url(source_url)

        # poll for copy completion
        waited = 0.0
        while waited < COPY_TIMEOUT_SECONDS:
            props = dest_client.get_blob_properties()
            copy_status = props.copy.status
            if copy_status == "success":
                with counters_lock:
                    counters["copied"] += 1
                    counters["total"] += 1
                log_queue.put([file_name, blob_name, dest_blob_name, "COPIED", ts])
                return
            elif copy_status in ("failed", "aborted"):
                raise Exception(f"Copy status: {copy_status}")
            time.sleep(COPY_POLL_INTERVAL_SECONDS)
            waited += COPY_POLL_INTERVAL_SECONDS

        raise Exception("Copy timed out")

    except Exception as e:
        with counters_lock:
            counters["failed"] += 1
            counters["total"] += 1
        failure_queue.put([file_name, blob_name, dest_blob_name, str(e), ts])


def list_source_blobs(blob_service_client):
    """Generator yielding blob names under SOURCE_PREFIX."""
    container_client = blob_service_client.get_container_client(SOURCE_CONTAINER)
    for blob in container_client.list_blobs(name_starts_with=SOURCE_PREFIX):
        yield blob.name


def main():
    print("==============================================================")
    print(" Azure Blob Copy - starting")
    print(f" Source:      container='{SOURCE_CONTAINER}' prefix='{SOURCE_PREFIX}'")
    print(f" Destination: container='{DEST_CONTAINER}' prefix='{DEST_PREFIX}'")
    print(f" Success log: {SUCCESS_LOG_PATH}")
    print(f" Failure log: {FAILURE_LOG_PATH}")
    print(f" Max workers: {MAX_WORKERS}")
    print("==============================================================")

    if "PASTE_" in CONNECTION_STRING or "PASTE_" in SOURCE_CONTAINER or "PASTE_" in DEST_CONTAINER:
        print("ERROR: Please fill in CONNECTION_STRING, SOURCE_CONTAINER, and DEST_CONTAINER "
              "in the CONFIGURATION section before running.")
        sys.exit(1)

    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)

    # start background logger thread
    logger_thread = threading.Thread(target=logger_thread_func, daemon=True)
    logger_thread.start()

    print("Listing source blobs (this may take a while for 2.5 lakh+ files)...")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        submitted = 0
        for blob_name in list_source_blobs(blob_service_client):
            futures.append(executor.submit(copy_single_blob, blob_service_client, blob_name))
            submitted += 1
            if submitted % 5000 == 0:
                print(f"[submit] queued {submitted} files so far...")

        print(f"All {submitted} files submitted. Waiting for completion...")

        for future in as_completed(futures):
            # exceptions are already handled inside copy_single_blob
            pass

    stop_logging_flag.set()
    logger_thread.join()

    elapsed = time.time() - start_time
    print("==============================================================")
    print(" DONE")
    print(f" Total processed : {counters['total']}")
    print(f" Copied          : {counters['copied']}")
    print(f" Skipped         : {counters['skipped']}")
    print(f" Failed          : {counters['failed']}")
    print(f" Elapsed         : {elapsed:.1f} sec")
    print(f" Success CSV     : {SUCCESS_LOG_PATH}")
    print(f" Failure CSV     : {FAILURE_LOG_PATH}")
    print("==============================================================")


if __name__ == "__main__":
    main()
