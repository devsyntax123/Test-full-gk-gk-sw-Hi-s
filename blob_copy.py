"""
Copy blobs within the SAME Azure Storage container from a source path/prefix
to a destination path/prefix, using multiple threads.

Why local state instead of "check if destination exists" for dedup:
    The destination is watched by a Blob Trigger (e.g. Azure Function / Event Grid),
    which may process and DELETE files shortly after they land. So "does the file
    already exist at destination?" is NOT a reliable way to tell if it was already
    copied. Instead, this script keeps a local JSON file recording the names of
    blobs it has already copied, and skips those on every run -- regardless of
    whether they still exist at the destination.

Usage:
    Edit the CONFIG section near the top of this file, then run:
        python blob_copy.py

Requires:
    pip install azure-storage-blob --break-system-packages
"""

import json
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError, AzureError

# ============================================================
# CONFIG - replace these values before running
# ============================================================
CONNECTION_STRING = "DefaultEndpointsProtocol=...;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net"

CONTAINER_NAME = "my-container"          # same container for source and destination
SOURCE_PREFIX = "raw/incoming/"          # source path/prefix to list blobs from
DEST_PREFIX = "processed/outgoing/"      # destination path/prefix to copy into

THREAD_COUNT = 8                         # number of worker threads
STATE_FILE = "copied_state.json"         # local file tracking already-copied blobs

OVERWRITE = False                        # True = always (re)copy even if destination currently exists
IGNORE_STATE = False                     # True = ignore local state file dedup (still writes new state)
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(threadName)s: %(message)s",
)
log = logging.getLogger("blob_copy")


class StateStore:
    """
    Tracks which source blob names have already been copied, persisted to a
    JSON file on disk. Thread-safe. This is the source of truth for dedup --
    NOT the presence/absence of the file at the destination.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._copied: set[str] = set()
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
        # Persist immediately so progress survives a crash / interruption
        # partway through a large batch.
        with self._lock:
            self._copied.add(blob_name)
            self._flush_locked()

    def _flush_locked(self):
        tmp_path = self.path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump({"copied": sorted(self._copied)}, f, indent=2)
        os.replace(tmp_path, self.path)  # atomic on POSIX


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
        copy_props = dest_client.start_copy_from_url(source_url)

        # Poll until the async server-side copy finishes.
        import time
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


def main():
    if not CONNECTION_STRING or CONNECTION_STRING.startswith("DefaultEndpointsProtocol=..."):
        log.error("Set CONNECTION_STRING at the top of the script before running.")
        sys.exit(1)

    source_prefix = SOURCE_PREFIX if SOURCE_PREFIX.endswith("/") else SOURCE_PREFIX + "/"
    dest_prefix = DEST_PREFIX if DEST_PREFIX.endswith("/") else DEST_PREFIX + "/"

    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)

    log.info(f"Listing blobs under '{source_prefix}' in container '{CONTAINER_NAME}'...")
    source_blobs = [b.name for b in container_client.list_blobs(name_starts_with=source_prefix)
                     if not b.name.endswith("/")]  # skip virtual "folder" markers
    log.info(f"Found {len(source_blobs)} blob(s) under source prefix.")

    if not source_blobs:
        log.info("Nothing to copy. Exiting.")
        return

    state = StateStore(STATE_FILE)
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

    with ThreadPoolExecutor(max_workers=THREAD_COUNT, thread_name_prefix="copy") as executor:
        future_to_name = {
            executor.submit(
                copy_one_blob, blob_service_client, CONTAINER_NAME, src, dest, state, OVERWRITE
            ): src
            for src, dest in jobs
        }

        for future in as_completed(future_to_name):
            src_name = future_to_name[future]
            try:
                name, success, message = future.result()
            except Exception as e:
                success, message = False, f"FAILED (executor exception): {e}"
                name = src_name

            if success and "skipped" in message:
                results["skipped"] += 1
                log.info(f"{name}: {message}")
            elif success:
                results["copied"] += 1
                log.info(f"{name}: {message}")
            else:
                results["failed"] += 1
                log.error(f"{name}: {message}")

    log.info(
        f"Done. Copied={results['copied']} Skipped={results['skipped']} "
        f"Failed={results['failed']} TotalConsidered={len(jobs)}"
    )
    if results["failed"] > 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
