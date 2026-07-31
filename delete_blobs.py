"""
Delete Azure Blob Storage files that match file_name values listed in a CSV/Excel file.

- Lists all blobs under a given prefix (folder) in the container.
- Matches blobs by filename (the part after the last '/') against the
  'file_name' column in your CSV/Excel.
- Supports DRY RUN (default) — no deletion happens unless you set DRY_RUN = False.
- Uses a thread pool for fast deletion when there are many files (e.g. 60k).

Usage:
    1. Fill in CONNECTION_STRING, CONTAINER_NAME, PREFIX, EXCEL_PATH below
       (or pass them as environment variables / CLI args — see bottom of file).
    2. Run once with DRY_RUN = True to review what WOULD be deleted.
    3. Review the generated 'dry_run_matches.csv' report.
    4. Set DRY_RUN = False to actually delete.
"""

import os
import sys
import csv
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from azure.storage.blob import ContainerClient
from azure.core.exceptions import ResourceNotFoundError

# ------------------------- CONFIG -------------------------------------
CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "PASTE_YOUR_CONNECTION_STRING_HERE")
CONTAINER_NAME = "your-container-name"
PREFIX = "call-centre-raw-input/imarque/cross-sell/raw-input/"

EXCEL_PATH = "/path/to/your/file_list.xlsx"   # .xlsx, .xls, or .csv all work
FILE_NAME_COLUMN = "file_name"

DRY_RUN = True          # <-- IMPORTANT: keep True until you've reviewed the report
MAX_WORKERS = 32         # parallel delete threads; 16-64 is usually a good range
REPORT_PATH = "dry_run_matches.csv"
NOT_FOUND_REPORT_PATH = "file_names_not_found.csv"
# ------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def load_target_filenames(path: str, column: str) -> set:
    """Load the file_name column from csv/xlsx into a set for O(1) lookups."""
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Available columns: {list(df.columns)}")

    names = (
        df[column]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )
    unique_names = set(names)
    log.info(f"Loaded {len(names)} rows ({len(unique_names)} unique file names) from {path}")
    return unique_names


def list_blobs_under_prefix(container_client: ContainerClient, prefix: str):
    """
    Generator that lists all blobs under a prefix.
    Uses the SDK's built-in pagination so it scales to tens of thousands of blobs
    without loading everything into memory at once.
    """
    log.info(f"Listing blobs under prefix: {prefix} ...")
    count = 0
    for blob in container_client.list_blobs(name_starts_with=prefix):
        count += 1
        if count % 5000 == 0:
            log.info(f"  ...listed {count} blobs so far")
        yield blob
    log.info(f"Finished listing. Total blobs under prefix: {count}")


def match_blobs(container_client: ContainerClient, prefix: str, target_names: set):
    """
    Walk all blobs under the prefix once, match filename (basename) against target_names.
    Returns list of full blob names to delete, and the set of target names that were found.
    """
    matches = []
    found_names = set()

    for blob in list_blobs_under_prefix(container_client, prefix):
        basename = blob.name.rsplit("/", 1)[-1]
        if basename in target_names:
            matches.append(blob.name)
            found_names.add(basename)

    return matches, found_names


def delete_blobs_parallel(container_client: ContainerClient, blob_names: list, max_workers: int):
    """Delete blobs concurrently using a thread pool. Returns (deleted, failed)."""
    deleted = []
    failed = []

    def _delete_one(name):
        try:
            container_client.delete_blob(name)
            return ("ok", name)
        except ResourceNotFoundError:
            return ("already_gone", name)
        except Exception as e:
            return ("error", name, str(e))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_delete_one, name): name for name in blob_names}
        total = len(futures)
        done = 0
        for future in as_completed(futures):
            result = future.result()
            done += 1
            if result[0] in ("ok", "already_gone"):
                deleted.append(result[1])
            else:
                failed.append((result[1], result[2]))
                log.warning(f"Failed to delete {result[1]}: {result[2]}")

            if done % 1000 == 0 or done == total:
                log.info(f"  ...deleted {done}/{total}")

    return deleted, failed


def main():
    if "PASTE_YOUR" in CONNECTION_STRING:
        log.error("Please set CONNECTION_STRING (or env var AZURE_STORAGE_CONNECTION_STRING).")
        sys.exit(1)

    target_names = load_target_filenames(EXCEL_PATH, FILE_NAME_COLUMN)

    container_client = ContainerClient.from_connection_string(
        conn_str=CONNECTION_STRING, container_name=CONTAINER_NAME
    )

    matches, found_names = match_blobs(container_client, PREFIX, target_names)
    not_found = target_names - found_names

    log.info(f"Matched {len(matches)} blobs against {len(target_names)} target file names.")
    log.info(f"{len(not_found)} file names from your list were NOT found in the container.")

    # Write reports regardless of dry run or not, so you always have a record
    with open(REPORT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["blob_name_to_delete"])
        for name in matches:
            writer.writerow([name])
    log.info(f"Wrote match report to {REPORT_PATH}")

    with open(NOT_FOUND_REPORT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name_not_found_in_blob"])
        for name in sorted(not_found):
            writer.writerow([name])
    log.info(f"Wrote not-found report to {NOT_FOUND_REPORT_PATH}")

    if DRY_RUN:
        log.info("DRY RUN complete. No blobs were deleted.")
        log.info(f"Review '{REPORT_PATH}' to confirm the exact blobs that would be deleted.")
        log.info("Set DRY_RUN = False to actually delete them.")
        return

    if not matches:
        log.info("Nothing to delete.")
        return

    log.info(f"Starting DELETION of {len(matches)} blobs with {MAX_WORKERS} parallel workers...")
    deleted, failed = delete_blobs_parallel(container_client, matches, MAX_WORKERS)

    log.info(f"Done. Deleted: {len(deleted)}, Failed: {len(failed)}")
    if failed:
        fail_report = "delete_failures.csv"
        with open(fail_report, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["blob_name", "error"])
            writer.writerows(failed)
        log.info(f"Wrote failure details to {fail_report}")


if __name__ == "__main__":
    main()
