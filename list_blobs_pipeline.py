"""
Azure Blob Storage — Multithreaded Filename/Path Lister
=========================================================

Purpose:
  List blob names/paths across multiple container+prefix combinations
  (list-only, no downloads/reads), write them to a single CSV, and
  report the total UNIQUE filename count.

Why multithreading helps here:
  The Azure SDK's list_blobs() call is paginated and I/O-bound (one
  HTTP request per page, ~5000 blobs/page by default). With 3 lakh+
  blobs spread across multiple prefixes, running each prefix's listing
  in its own thread lets network wait-time overlap across prefixes.
  NOTE: within a single prefix, listing is inherently sequential
  (each page needs the previous page's continuation token), so
  threading here parallelizes ACROSS prefixes, not within one.

Fill in your connection string below before running.
"""

import csv
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from queue import Queue, Empty

from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import AzureError

# ─────────────────────────────────────────────────────────────────────────
# CONFIG — fill these in
# ─────────────────────────────────────────────────────────────────────────

CONNECTION_STRING = "PASTE_YOUR_AZURE_STORAGE_CONNECTION_STRING_HERE"

# Each entry: (container_name, prefix_path)
TARGETS = [
    ("call-centre", "imarque/cross-sell/archive/raw-input/30-07-2026/success/"),
    ("call-centre", "imarque/cross-sell/archive/raw-input/31-07-2026/success/"),
    ("call-centre-raw-input", "imarque/cross-sell/raw-input/"),
]

OUTPUT_CSV = "blob_inventory.csv"
OUTPUT_SUMMARY = "blob_inventory_summary.txt"

MAX_WORKERS = min(8, len(TARGETS))   # one thread per prefix is enough; raising
                                       # this further won't help since each
                                       # prefix listing is itself sequential
PAGE_SIZE = 5000                      # max page size Azure allows
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SEC = 2

# ─────────────────────────────────────────────────────────────────────────


@dataclass
class PrefixResult:
    container: str
    prefix: str
    count: int = 0
    error: str = None
    elapsed_sec: float = 0.0


def list_prefix(blob_service_client: BlobServiceClient, container: str, prefix: str,
                 writer_queue: Queue, progress_lock: threading.Lock,
                 progress: dict) -> PrefixResult:
    """List all blobs under one container+prefix, list-only (no downloads).
    Streams results into a thread-safe queue for the CSV writer thread.
    Retries transient Azure errors with backoff.
    """
    result = PrefixResult(container=container, prefix=prefix)
    start = time.time()

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            container_client = blob_service_client.get_container_client(container)
            local_count = 0

            # list_blobs() is a lazy paginated iterator — walking it
            # issues one HTTP GET per ~PAGE_SIZE blobs under the hood.
            blob_iter = container_client.list_blobs(
                name_starts_with=prefix,
                results_per_page=PAGE_SIZE,
            ).by_page()

            for page in blob_iter:
                for blob in page:
                    writer_queue.put((container, blob.name, blob.size, blob.last_modified))
                    local_count += 1

                    if local_count % 20000 == 0:
                        with progress_lock:
                            progress[(container, prefix)] = local_count
                            _print_progress(progress)

            result.count = local_count
            result.elapsed_sec = time.time() - start
            with progress_lock:
                progress[(container, prefix)] = local_count
                _print_progress(progress)
            return result

        except AzureError as e:
            if attempt == RETRY_ATTEMPTS:
                result.error = str(e)
                result.elapsed_sec = time.time() - start
                return result
            time.sleep(RETRY_BACKOFF_SEC * attempt)

    return result


def _print_progress(progress: dict):
    line = " | ".join(f"{c}/{p.rstrip('/').split('/')[-1]}: {n:,}" for (c, p), n in progress.items())
    sys.stdout.write("\r" + line + "   ")
    sys.stdout.flush()


def csv_writer_thread(writer_queue: Queue, output_path: str, stop_event: threading.Event, rows_written: list):
    """Single dedicated writer thread — avoids concurrent file writes from
    multiple listing threads, which would corrupt the CSV or need locking
    around every row anyway."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["container", "blob_path", "filename", "size_bytes", "last_modified"])
        while not (stop_event.is_set() and writer_queue.empty()):
            try:
                container, blob_path, size, last_modified = writer_queue.get(timeout=0.5)
            except Empty:
                continue
            filename = blob_path.rsplit("/", 1)[-1]
            writer.writerow([container, blob_path, filename, size, last_modified])
            rows_written[0] += 1
            writer_queue.task_done()


def main():
    if "PASTE_YOUR" in CONNECTION_STRING:
        print("ERROR: Set CONNECTION_STRING at the top of the script before running.")
        sys.exit(1)

    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)

    writer_queue: Queue = Queue(maxsize=100_000)
    stop_event = threading.Event()
    progress_lock = threading.Lock()
    progress = {}
    rows_written = [0]

    writer_t = threading.Thread(
        target=csv_writer_thread,
        args=(writer_queue, OUTPUT_CSV, stop_event, rows_written),
        daemon=True,
    )
    writer_t.start()

    print(f"Listing {len(TARGETS)} container/prefix targets with {MAX_WORKERS} worker threads...\n")

    overall_start = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(list_prefix, blob_service_client, c, p, writer_queue, progress_lock, progress): (c, p)
            for c, p in TARGETS
        }
        for future in as_completed(futures):
            results.append(future.result())

    # signal writer to finish once all queued rows are flushed
    stop_event.set()
    writer_t.join()

    overall_elapsed = time.time() - overall_start
    print("\n\nListing complete.\n")

    # ── Summary ──────────────────────────────────────────────────────
    total_listed = sum(r.count for r in results)
    errors = [r for r in results if r.error]

    # Unique filename count (by full blob path, which is the true unique key;
    # also report unique by bare filename in case that's what's meant)
    unique_paths = set()
    unique_filenames = set()
    with open(OUTPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            unique_paths.add((row["container"], row["blob_path"]))
            unique_filenames.add(row["filename"])

    summary_lines = []
    summary_lines.append("Azure Blob Listing Summary")
    summary_lines.append("=" * 40)
    for r in results:
        status = f"ERROR: {r.error}" if r.error else "OK"
        summary_lines.append(f"[{status}] {r.container}/{r.prefix}")
        summary_lines.append(f"    blobs listed: {r.count:,}   time: {r.elapsed_sec:.1f}s")
    summary_lines.append("-" * 40)
    summary_lines.append(f"Total rows listed:              {total_listed:,}")
    summary_lines.append(f"Total unique blob paths:         {len(unique_paths):,}")
    summary_lines.append(f"Total unique bare filenames:      {len(unique_filenames):,}")
    summary_lines.append(f"(paths vs filenames differ if the same filename appears under multiple folders)")
    summary_lines.append(f"Total wall time:                 {overall_elapsed:.1f}s")
    summary_lines.append(f"CSV output:                      {OUTPUT_CSV}")
    if errors:
        summary_lines.append(f"\nWARNING: {len(errors)} prefix(es) failed after {RETRY_ATTEMPTS} retries. Re-run to retry those.")

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")


if __name__ == "__main__":
    main()
