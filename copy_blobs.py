"""
Multithreaded Azure Blob Storage copy script.

Reads an Excel file containing a `file_path` column, and copies each blob
from the source path (within the `call-centre` container) to:

    imarque/cross-sell/result-31-07-2026-filtered/keep/<file_name>

Uses server-side copy (start_copy_from_url) so data never leaves Azure's
network / gets downloaded locally — this is the fastest possible method
for blob-to-blob copy within the same storage account.

Concurrency is handled via ThreadPoolExecutor since the copy calls are
I/O-bound (mostly waiting on HTTP responses from Azure), and the SDK
releases the GIL during network calls.
"""

import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import PurePosixPath

import pandas as pd
from azure.core.exceptions import ResourceNotFoundError, ResourceExistsError
from azure.storage.blob import BlobServiceClient

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CONNECTION_STRING = "PASTE_YOUR_CONNECTION_STRING_HERE"

CONTAINER_NAME = "call-centre"

# Excel file containing the list of files to copy
EXCEL_PATH = "files_to_copy.xlsx"          # <-- edit path
EXCEL_SHEET = 0                             # <-- edit sheet name/index if needed

# Column in the excel that holds the source blob path (relative to container)
SOURCE_COLUMN = "file_path"
NAME_COLUMN = "file_name"

# Destination prefix (relative to container) where files should be copied
DEST_PREFIX = "imarque/cross-sell/result-31-07-2026-filtered/keep"

MAX_WORKERS = 64          # tune based on network/throttling; 32-128 is typical
POLL_COPY_STATUS = False  # True = wait & confirm each copy completed (slower, safer)
LOG_FILE = "copy_log.log"
FAILED_CSV = "failed_copies.csv"
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def build_dest_path(dest_prefix: str, src_path: str, file_name: str) -> str:
    """
    Destination blob path = DEST_PREFIX/<file_name>
    (flattening into the 'keep' folder, using just the file name)
    """
    return str(PurePosixPath(dest_prefix) / file_name)


def copy_one(container_client, src_blob_path: str, dest_blob_path: str, source_url_base: str):
    """
    Perform a single server-side copy. Returns (src, dest, status, message)
    """
    src_client = container_client.get_blob_client(src_blob_path)
    dest_client = container_client.get_blob_client(dest_blob_path)

    source_url = f"{source_url_base}/{src_blob_path}"

    try:
        # Quick existence check on source avoids a confusing copy failure later
        if not src_client.exists():
            return (src_blob_path, dest_blob_path, "FAILED", "Source blob not found")

        copy_props = dest_client.start_copy_from_url(source_url)

        if POLL_COPY_STATUS:
            # Poll until copy finishes (usually near-instant for same-account copies)
            while True:
                props = dest_client.get_blob_properties()
                status = props.copy.status
                if status == "success":
                    break
                elif status in ("failed", "aborted"):
                    return (src_blob_path, dest_blob_path, "FAILED", f"Copy status: {status}")
                time.sleep(0.5)

        return (src_blob_path, dest_blob_path, "OK", "")

    except ResourceNotFoundError as e:
        return (src_blob_path, dest_blob_path, "FAILED", f"Not found: {e}")
    except ResourceExistsError as e:
        return (src_blob_path, dest_blob_path, "FAILED", f"Already exists: {e}")
    except Exception as e:
        return (src_blob_path, dest_blob_path, "FAILED", str(e))


def main():
    log.info("Reading excel: %s", EXCEL_PATH)
    df = pd.read_excel(EXCEL_PATH, sheet_name=EXCEL_SHEET)

    if SOURCE_COLUMN not in df.columns:
        raise ValueError(f"Column '{SOURCE_COLUMN}' not found in excel. Found: {list(df.columns)}")

    df = df.dropna(subset=[SOURCE_COLUMN]).reset_index(drop=True)
    total = len(df)
    log.info("Total rows to process: %s", total)

    # Set up blob service client
    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)

    account_url = blob_service_client.url.rstrip("/")
    source_url_base = f"{account_url}/{CONTAINER_NAME}"

    # Build (src, dest) pairs
    tasks = []
    for _, row in df.iterrows():
        src_path = str(row[SOURCE_COLUMN]).strip().lstrip("/")
        file_name = str(row[NAME_COLUMN]).strip() if NAME_COLUMN in df.columns else PurePosixPath(src_path).name
        dest_path = build_dest_path(DEST_PREFIX, src_path, file_name)
        tasks.append((src_path, dest_path))

    log.info("Starting copy with %s worker threads", MAX_WORKERS)

    success_count = 0
    fail_count = 0
    failed_rows = []

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {
            executor.submit(copy_one, container_client, src, dest, source_url_base): (src, dest)
            for src, dest in tasks
        }

        for i, future in enumerate(as_completed(future_to_task), start=1):
            src, dest, status, message = future.result()

            if status == "OK":
                success_count += 1
            else:
                fail_count += 1
                failed_rows.append({"source": src, "destination": dest, "error": message})
                log.warning("FAILED: %s -> %s | %s", src, dest, message)

            if i % 1000 == 0 or i == total:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                log.info(
                    "Progress: %s/%s | success=%s fail=%s | %.1f files/sec | elapsed=%.1fs",
                    i, total, success_count, fail_count, rate, elapsed,
                )

    elapsed_total = time.time() - start_time
    log.info("DONE. Total=%s Success=%s Failed=%s Time=%.1fs", total, success_count, fail_count, elapsed_total)

    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(FAILED_CSV, index=False)
        log.info("Failed copies written to %s", FAILED_CSV)


if __name__ == "__main__":
    main()
