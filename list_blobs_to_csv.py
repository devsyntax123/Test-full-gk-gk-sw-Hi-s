"""
Azure Blob Storage - List file names & paths (excluding 'keep' marker files) to CSV
--------------------------------------------------------------------------------------
Lists blobs under multiple prefixes (paths) inside a single container, excludes any
"keep" marker files, and writes the results to separate CSV files (one per prefix).

This does NOT copy, move, or delete anything - it only lists/reads blob metadata.

Multithreading strategy:
- Azure's list_blobs() is a paginated, sequential API call per prefix - it cannot be
  parallelized *within* a single prefix's pagination stream.
- However, since there are multiple independent prefixes, each prefix is listed in
  its own worker thread concurrently (ThreadPoolExecutor), so all 4 paths are
  scanned in parallel instead of one after another.
- Each thread writes to its own CSV file, so there is no lock contention or shared
  file writes needed - safe and fast even for 300,000+ blobs total.
- Within a thread, blobs are streamed and written in batches (not held fully in
  memory) to keep memory usage low regardless of blob count.

Requirements:
    pip install azure-storage-blob
"""

import csv
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from azure.storage.blob import BlobServiceClient

# ============================================================================
# CONFIG - EDIT THESE
# ============================================================================

CONNECTION_STRING = ""   # <-- your Azure Storage connection string (you said you'll add this)

CONTAINER_NAME = "call-centre"

# Prefixes (paths) to scan inside the container
PREFIXES = [
    "imarque/cross-sell/archive/raw-input/30-07-2026/success/",
    "imarque/cross-sell/archive/raw-input/31-07-2026/success/",
    "imarque/cross-sell/archive/raw-input/31-07-2026/failure/",
    "imarque/cross-sell/intermediate-input/",
]

# Files to exclude - any blob whose filename (last path segment, case-insensitive)
# starts with one of these markers will be SKIPPED from the CSV.
# Adjust this list/logic if your "keep file" convention is different
# (e.g. exact filename ".keep", "KEEP.txt", "_keep", etc.)
KEEP_FILE_PREFIXES = ("keep", ".keep", "_keep")

# Output directory for CSVs
OUTPUT_DIR = "."

# How many blob records to buffer before writing a batch to CSV (memory control)
BATCH_SIZE = 5000

# Number of parallel worker threads (one per prefix is enough here since we have
# 4 prefixes; increase only if you add more prefixes)
MAX_WORKERS = 4

# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class PrefixResult:
    prefix: str
    csv_path: str
    total_scanned: int
    total_written: int
    total_skipped_keep: int
    elapsed_seconds: float


def is_keep_file(blob_name: str) -> bool:
    """Return True if this blob should be excluded as a 'keep' marker file."""
    filename = blob_name.rstrip("/").split("/")[-1]
    if not filename:
        # this is a "directory placeholder" blob (name ends with /), skip it too
        return True
    filename_lower = filename.lower()
    return any(filename_lower.startswith(marker) for marker in KEEP_FILE_PREFIXES)


def safe_filename_from_prefix(prefix: str) -> str:
    """Turn a blob prefix path into a safe CSV filename."""
    cleaned = prefix.strip("/").replace("/", "_")
    return f"{cleaned}.csv"


def list_prefix_to_csv(blob_service_client: BlobServiceClient, container_name: str, prefix: str) -> PrefixResult:
    """
    List all blobs under a given prefix, exclude 'keep' files, and stream results
    to a dedicated CSV file in batches. Runs inside a worker thread.
    """
    start = time.time()
    container_client = blob_service_client.get_container_client(container_name)

    csv_filename = safe_filename_from_prefix(prefix)
    csv_path = f"{OUTPUT_DIR.rstrip('/')}/{csv_filename}"

    total_scanned = 0
    total_written = 0
    total_skipped_keep = 0
    batch = []

    logger.info(f"Starting scan of prefix: {prefix}")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "file_path", "size_bytes", "last_modified"])

        # list_blobs is a lazy/paginated iterator under the hood - it fetches
        # pages from Azure as you iterate, so this streams rather than loading
        # everything into memory at once.
        blob_iter = container_client.list_blobs(name_starts_with=prefix)

        for blob in blob_iter:
            total_scanned += 1

            if is_keep_file(blob.name):
                total_skipped_keep += 1
                continue

            file_name = blob.name.rstrip("/").split("/")[-1]
            batch.append([
                file_name,
                blob.name,                # full path within container
                blob.size,
                blob.last_modified.isoformat() if blob.last_modified else "",
            ])
            total_written += 1

            if len(batch) >= BATCH_SIZE:
                writer.writerows(batch)
                batch.clear()
                logger.info(f"[{prefix}] scanned={total_scanned:,} written={total_written:,}")

        if batch:
            writer.writerows(batch)

    elapsed = time.time() - start
    logger.info(
        f"Finished prefix: {prefix} | scanned={total_scanned:,} "
        f"written={total_written:,} skipped_keep={total_skipped_keep:,} "
        f"in {elapsed:.1f}s -> {csv_path}"
    )

    return PrefixResult(
        prefix=prefix,
        csv_path=csv_path,
        total_scanned=total_scanned,
        total_written=total_written,
        total_skipped_keep=total_skipped_keep,
        elapsed_seconds=elapsed,
    )


def write_summary_csv(results: list[PrefixResult]) -> str:
    """Write a summary CSV with counts per prefix."""
    summary_path = f"{OUTPUT_DIR.rstrip('/')}/_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "prefix", "csv_file", "total_scanned", "total_written",
            "total_skipped_keep", "elapsed_seconds",
        ])
        for r in results:
            writer.writerow([
                r.prefix, r.csv_path, r.total_scanned, r.total_written,
                r.total_skipped_keep, f"{r.elapsed_seconds:.1f}",
            ])
    return summary_path


def main():
    if not CONNECTION_STRING:
        raise ValueError("CONNECTION_STRING is empty - please set your Azure Storage connection string.")

    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)

    overall_start = time.time()
    results: list[PrefixResult] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="prefix-worker") as executor:
        future_to_prefix = {
            executor.submit(list_prefix_to_csv, blob_service_client, CONTAINER_NAME, prefix): prefix
            for prefix in PREFIXES
        }

        for future in as_completed(future_to_prefix):
            prefix = future_to_prefix[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                logger.error(f"Prefix '{prefix}' generated an exception: {exc}")
                raise

    # Keep summary in same order as PREFIXES for readability
    results.sort(key=lambda r: PREFIXES.index(r.prefix))
    summary_path = write_summary_csv(results)

    overall_elapsed = time.time() - overall_start
    total_scanned = sum(r.total_scanned for r in results)
    total_written = sum(r.total_written for r in results)
    total_skipped = sum(r.total_skipped_keep for r in results)

    logger.info("=" * 70)
    logger.info(f"ALL DONE in {overall_elapsed:.1f}s")
    logger.info(f"Total blobs scanned : {total_scanned:,}")
    logger.info(f"Total written to CSV: {total_written:,}")
    logger.info(f"Total 'keep' skipped: {total_skipped:,}")
    logger.info(f"Summary CSV         : {summary_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
