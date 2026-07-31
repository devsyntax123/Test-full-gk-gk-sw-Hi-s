"""
List all blobs under a virtual folder path in an Azure Blob container,
parse the filename (agreement no + audio duration), and write to CSV.

Container : call-centre
Prefix    : imarque/cross-sell/results-28-07-26/

Filename pattern:
AS3072CD0822914_12-07-2026-13-15-40_HINDI_mp3_206.wav
  -> agreement_no       = AS3072CD0822914   (part before first "_")
  -> audio_duration_sec = 206               (numeric part before extension, last "_" segment)

Handles 600,000+ files using:
  - Azure SDK's native paged listing (server-side, cheap)
  - ThreadPoolExecutor for parallel parsing + CSV writing in chunks
  - Streaming writes (doesn't hold all rows in memory at once)

Usage:
  python list_blobs_to_csv.py
"""

import csv
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from azure.storage.blob import ContainerClient

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
CONNECTION_STRING = os.environ.get(
    "AZURE_STORAGE_CONNECTION_STRING",
    "PUT_YOUR_CONNECTION_STRING_HERE",
)
CONTAINER_NAME = "call-centre"
PREFIX = "imarque/cross-sell/results-28-07-26/"

OUTPUT_CSV = "blob_inventory.csv"

# How many blob names to hand to a worker thread at once for parsing.
# (Listing itself is a single sequential paged generator — Azure doesn't
# support parallel "list" calls against one prefix — but we parallelize
# the CPU-bound filename parsing + batch CSV writing to keep the pipe full.)
BATCH_SIZE = 2000
MAX_WORKERS = 16

# Regex to parse: AGREEMENTNO_DATE-TIME_LANG_FORMAT_DURATION.wav
# Example: AS3072CD0822914_12-07-2026-13-15-40_HINDI_mp3_206.wav
FILENAME_RE = re.compile(
    r"^(?P<agreement_no>[^_]+)_.*_(?P<duration>\d+)\.\w+$"
)

csv_lock = threading.Lock()
counter_lock = threading.Lock()
processed_count = 0


def parse_filename(filename: str):
    """Extract agreement_no and audio_duration_sec from a filename."""
    match = FILENAME_RE.match(filename)
    if match:
        return match.group("agreement_no"), match.group("duration")
    return "", ""


def process_batch(batch):
    """Parse a batch of (name, path) tuples -> list of CSV rows."""
    rows = []
    for name, path in batch:
        agreement_no, duration = parse_filename(name)
        rows.append([name, path, agreement_no, duration])
    return rows


def write_rows(writer, rows):
    with csv_lock:
        writer.writerows(rows)


def main():
    if "PUT_YOUR_CONNECTION_STRING_HERE" in CONNECTION_STRING:
        print("ERROR: Set AZURE_STORAGE_CONNECTION_STRING env var or edit the script.")
        sys.exit(1)

    start = time.time()

    container_client = ContainerClient.from_connection_string(
        conn_str=CONNECTION_STRING, container_name=CONTAINER_NAME
    )

    print(f"Listing blobs under '{CONTAINER_NAME}/{PREFIX}' ...")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "file_path", "agreement_no", "audio_duration_sec"])

        # Azure SDK's list_blobs() returns a lazy pageable iterator that
        # fetches pages (default 5000 items) from the server as you iterate.
        # We buffer names into batches and hand each batch to a thread pool
        # for the parsing + write step, so parsing/writing overlaps with
        # network I/O of fetching the next page.
        blob_iter = container_client.list_blobs(name_starts_with=PREFIX)

        batch = []
        futures = []
        global processed_count

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

            def flush_batch(current_batch):
                fut = executor.submit(process_batch, current_batch)
                futures.append(fut)
                # Drain completed futures periodically so memory doesn't grow
                # unbounded and results get written promptly.
                if len(futures) >= MAX_WORKERS * 2:
                    drain(futures, writer)

            def drain(fut_list, writer):
                global processed_count
                done_futs = [fu for fu in fut_list if fu.done()]
                for fu in done_futs:
                    rows = fu.result()
                    write_rows(writer, rows)
                    with counter_lock:
                        processed_count += len(rows)
                    fut_list.remove(fu)

            for blob in blob_iter:
                name = blob.name.split("/")[-1]   # just filename
                path = blob.name                   # full path within container
                batch.append((name, path))

                if len(batch) >= BATCH_SIZE:
                    flush_batch(batch)
                    batch = []

                if processed_count and processed_count % 50000 < BATCH_SIZE:
                    print(f"  ... processed ~{processed_count} files "
                          f"({time.time()-start:.1f}s elapsed)")

            if batch:
                flush_batch(batch)

            # Final drain — wait for all remaining futures
            for fu in as_completed(futures):
                rows = fu.result()
                write_rows(writer, rows)
                with counter_lock:
                    processed_count += len(rows)

    elapsed = time.time() - start
    print(f"\nDone. {processed_count} files written to {OUTPUT_CSV}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
