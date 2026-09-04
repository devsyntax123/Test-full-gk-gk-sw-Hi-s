"""
Copy blobs within the SAME Azure Storage container from a source path/prefix
to a destination path/prefix, using multiple threads. Built for large sets
(tested against ~1.5 lakh / 150,000 files, e.g. audio call recordings like
'raw-input_29th_to_31st/BH3058CD0000115_29-08-2026-18-07-24_HINDI.mp3').

Why local state instead of "check if destination exists" for dedup:
    The destination is watched by a Blob Trigger (e.g. Azure Function / Event Grid),
    which may process and DELETE files shortly after they land. So "does the file
    already exist at destination?" is NOT a reliable way to tell if it was already
    copied. Instead, this script keeps a local JSON file recording the names of
    blobs it has already copied, and skips those on every run -- regardless of
    whether they still exist at the destination.

Built for scale:
    - Lists blobs in large pages (5000/page) instead of the default, to cut
      down round trips for big folders.
    - Copy is server-side (start_copy_from_url) -- no file data passes through
      this machine, so it's fast and cheap regardless of file size.
    - State file writes are BATCHED (flushed every STATE_FLUSH_EVERY copies or
      STATE_FLUSH_SECONDS seconds, whichever comes first) instead of writing to
      disk after every single file. With 150k files, writing on every single
      copy would be a lot of unnecessary disk I/O. A final flush always runs
      at the end (and on Ctrl+C) so no progress is lost.
    - Progress is logged periodically (every PROGRESS_EVERY files) instead of
      once per file, so the console stays readable at this scale.

Usage:
    Edit the CONFIG section below, then run:
        python blob_copy.py

Requires:
    pip install azure-storage-blob --break-system-packages
"""

import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError, AzureError

# ============================================================
# CONFIG - replace these values before running
# ============================================================
CONNECTION_STRING = "DefaultEndpointsProtocol=...;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net"

CONTAINER_NAME = "my-container"          # same container for source and destination
SOURCE_PREFIX = "raw-input_29th_to_31st/"  # source path/prefix to list blobs from
DEST_PREFIX = "processed/outgoing/"      # destination path/prefix to copy into

THREAD_COUNT = 16                        # number of worker threads (16-32 is reasonable for 1.5 lakh files)
STATE_FILE = "copied_state.json"         # local file tracking already-copied blobs

OVERWRITE = False                        # True = always (re)copy even if destination currently exists
IGNORE_STATE = False                     # True = ignore local state file dedup (still writes new state)

LIST_PAGE_SIZE = 5000                    # blobs fetched per listing request
PROGRESS_EVERY = 500                     # log a progress line every N completed copies
STATE_FLUSH_EVERY = 200                  # write state file to disk every N new copies
STATE_FLUSH_SECONDS = 15                 # ...or every N seconds, whichever comes first
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(threadName)s: %(message)s",
)
log = logging.getLogger("blob_copy")

# Azure SDK logs every HTTP request/response at INFO level by default, which
# drowns out this script's own log lines at this scale. Silence it down.
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)


class StateStore:
    """
    Tracks which source blob names have already been copied, persisted to a
    JSON file on disk. Thread-safe. This is the source of truth for dedup --
    NOT the presence/absence of the file at the destination.

    Writes are batched (see mark_copied) so 150k files doesn't mean 150k
    individual disk writes.
    """

    def __init__(self, path: str, flush_every: int, flush_seconds: float):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._copied: set[str] = set()
        self._dirty_count = 0
        self._last_flush = time.monotonic()
        self._flush_every = flush_every
        self._flush_seconds = flush_seconds
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path, "r") as f:
                    data = json.load(f)
                self._copied = set(data.get("copied", []))
                log.info(f"Loaded state: {len(self._copied)} blob(s) already marked as copied.")
            except (json.JSONDecodeError, OSError) as e:
                log.warning(f"Could not read state file ({e}); starting with empty state.")
                self._copied = set()
        else:
            log.info("No existing state file found; starting fresh.")

    def is_copied(self, blob_name: str) -> bool:
        with self._lock:
            return blob_name in self._copied

    def mark_copied(self, blob_name: str):
        with self._lock:
            self._copied.add(blob_name)
            self._dirty_count += 1
            now = time.monotonic()
            if (self._dirty_count >= self._flush_every) or (now - self._last_flush >= self._flush_seconds):
                self._flush_locked()

    def flush(self):
        """Force a flush regardless of batching thresholds. Call at the end and on interrupt."""
        with self._lock:
            self._flush_locked()

    def _flush_locked(self):
        tmp_path = self.path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump({"copied": sorted(self._copied)}, f)
        os.replace(tmp_path, self.path)  # atomic on POSIX
        self._dirty_count = 0
        self._last_flush = time.monotonic()

    def copied_count(self) -> int:
        with self._lock:
            return len(self._copied)


def copy_one_blob(
    blob_service_client: BlobServiceClient,
    container: str,
    source_blob_name: str,
    dest_blob_name: str,
    state: StateStore,
    overwrite: bool,
) -> tuple[str, bool, str]:
    """
    Copy a single blob server-side (no data through this machine) using
    start_copy_from_url, then poll until the copy completes.
    Returns (source_blob_name, success, message).
    """
    if state.is_copied(source_blob_name):
        return source_blob_name, True, "skipped (already copied per local state)"

    try:
        source_client = blob_service_client.get_blob_client(container=container, blob=source_blob_name)
        dest_client = blob_service_client.get_blob_client(container=container, blob=dest_blob_name)

        if not overwrite and dest_client.exists():
            # Destination happens to exist right now. We still don't know if
            # the trigger already consumed a PREVIOUS copy, so this is only a
            # courtesy check, not the dedup mechanism. We mark as copied and skip.
            state.mark_copied(source_blob_name)
            return source_blob_name, True, "skipped (destination currently exists, overwrite=False)"

        source_url = source_client.url
        dest_client.start_copy_from_url(source_url)

        # Poll until the async server-side copy finishes.
        while True:
            props = dest_client.get_blob_properties()
            status = props.copy.status
            if status == "success":
                break
            elif status in ("failed", "aborted"):
                raise AzureError(f"Copy ended with status={status}")
            time.sleep(0.5)

        state.mark_copied(source_blob_name)
        return source_blob_name, True, "copied"

    except ResourceExistsError as e:
        state.mark_copied(source_blob_name)
        return source_blob_name, True, f"skipped (already exists, treated as done): {e}"
    except AzureError as e:
        return source_blob_name, False, f"FAILED: {e}"
    except Exception as e:
        return source_blob_name, False, f"FAILED (unexpected): {e}"


def list_all_source_blobs(container_client, source_prefix: str) -> list[str]:
    log.info(f"Listing blobs under '{source_prefix}' (page size={LIST_PAGE_SIZE})...")
    blobs = []
    page_num = 0
    for page in container_client.list_blobs(name_starts_with=source_prefix, results_per_page=LIST_PAGE_SIZE).by_page():
        page_num += 1
        page_blobs = [b.name for b in page if not b.name.endswith("/")]
        blobs.extend(page_blobs)
        log.info(f"  page {page_num}: running total {len(blobs)} blob(s)")
    return blobs


def main():
    if not CONNECTION_STRING or CONNECTION_STRING.startswith("DefaultEndpointsProtocol=..."):
        log.error("Set CONNECTION_STRING at the top of the script before running.")
        sys.exit(1)

    source_prefix = SOURCE_PREFIX if SOURCE_PREFIX.endswith("/") else SOURCE_PREFIX + "/"
    dest_prefix = DEST_PREFIX if DEST_PREFIX.endswith("/") else DEST_PREFIX + "/"

    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)

    t0 = time.monotonic()
    source_blobs = list_all_source_blobs(container_client, source_prefix)
    log.info(f"Found {len(source_blobs)} blob(s) under source prefix in {time.monotonic() - t0:.1f}s.")

    if not source_blobs:
        log.info("Nothing to copy. Exiting.")
        return

    state = StateStore(STATE_FILE, STATE_FLUSH_EVERY, STATE_FLUSH_SECONDS)
    if IGNORE_STATE:
        log.warning("IGNORE_STATE=True: will attempt to copy everything regardless of prior state.")

    jobs = []
    for src_name in source_blobs:
        relative = src_name[len(source_prefix):]
        dest_name = dest_prefix + relative

        if not IGNORE_STATE and state.is_copied(src_name):
            continue
        jobs.append((src_name, dest_name))

    log.info(f"{len(jobs)} blob(s) to copy after filtering already-copied ones (threads={THREAD_COUNT}).")

    results = {"copied": 0, "skipped": 0, "failed": 0}
    completed = 0

    try:
        with ThreadPoolExecutor(max_workers=THREAD_COUNT, thread_name_prefix="copy") as executor:
            future_to_name = {
                executor.submit(
                    copy_one_blob, blob_service_client, CONTAINER_NAME, src, dest, state, OVERWRITE
                ): src
                for src, dest in jobs
            }

            for future in as_completed(future_to_name):
                src_name = future_to_name[future]
                completed += 1
                try:
                    name, success, message = future.result()
                except Exception as e:
                    success, message = False, f"FAILED (executor exception): {e}"
                    name = src_name

                if success and "skipped" in message:
                    results["skipped"] += 1
                elif success:
                    results["copied"] += 1
                else:
                    results["failed"] += 1
                    log.error(f"{name}: {message}")  # always log failures immediately

                if completed % PROGRESS_EVERY == 0 or completed == len(jobs):
                    log.info(
                        f"Progress: {completed}/{len(jobs)} | "
                        f"copied={results['copied']} skipped={results['skipped']} failed={results['failed']}"
                    )
    finally:
        # Always persist final state, even on Ctrl+C / crash mid-run.
        state.flush()
        log.info(f"State file flushed. Total blobs marked copied so far: {state.copied_count()}")

    log.info(
        f"Done. Copied={results['copied']} Skipped={results['skipped']} "
        f"Failed={results['failed']} TotalConsidered={len(jobs)}"
    )
    if results["failed"] > 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
