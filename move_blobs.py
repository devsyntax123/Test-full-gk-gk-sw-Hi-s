"""
move_blobs.py

Moves blobs from:
    call-centre/imarque/cross-sell/result-31-07-2026-filtered/
to:
    call-centre/imarque/cross-sell/intermediate-input/

within the SAME Azure Storage container, using multi-threading.

Key behaviour (per requirements):
  - Source may contain ~3,00,000 (3 lakh) files.
  - Only the first 1,00,000 (1 lakh) blobs (by listing order) are moved in this run.
    Re-run the script later to move the next batch (see "RESUME" notes below).
  - "Move" = copy blob to destination (server-side copy, no data download) + verify + delete source.
  - Verbose Azure SDK / HTTP payload logs are SUPPRESSED unless a blob's move fails,
    in which case that blob's operation is logged normally, and (optionally) you can
    flip on full SDK debug logging just to diagnose the failure.
  - A CSV log is written with the status of every blob (moved / failed / skipped).
  - A separate rotating text log captures run-level events, warnings and errors.

USAGE
-----
    export AZURE_STORAGE_CONNECTION_STRING="<your connection string>"
    python move_blobs.py --container <container-name>

    # optional overrides
    python move_blobs.py \
        --container mycontainer \
        --source-prefix "call-centre/imarque/cross-sell/result-31-07-2026-filtered/" \
        --dest-prefix   "call-centre/imarque/cross-sell/intermediate-input/" \
        --batch-size 100000 \
        --max-workers 32 \
        --csv-log move_log.csv \
        --run-log move_run.log

RESUME BEHAVIOUR
-----------------
Every blob that is successfully moved is recorded in the CSV log (append mode,
one CSV per day by default if you don't override --csv-log). On the next run,
the script loads the CSV log(s) matching --csv-log-glob and skips any source
blob names already marked "MOVED", so you can safely re-run it repeatedly,
each time picking up the next 1,00,000 files, until the source is empty.
"""

import argparse
import csv
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob

from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError, HttpResponseError


# --------------------------------------------------------------------------
# Logging setup
# --------------------------------------------------------------------------
# Two separate loggers:
#   1. "run" logger  -> human-readable progress / warnings / errors -> file + console
#   2. Azure SDK's own loggers ("azure", "azure.core.pipeline.policies.http_logging_policy")
#      are kept at WARNING level so the noisy per-request payload logs (headers,
#      bodies, request IDs etc.) do NOT show up UNLESS something fails, in which
#      case we temporarily bump them to DEBUG around that one operation and dump
#      those lines into the run log for diagnosis.

def build_run_logger(run_log_path: str) -> logging.Logger:
    logger = logging.getLogger("blob_mover")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(threadName)-12s | %(message)s"
    )

    file_handler = RotatingFileHandler(
        run_log_path, maxBytes=50 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def silence_azure_sdk_logs(run_log_path: str):
    """
    Keep Azure SDK HTTP/pipeline logs OFF (WARNING+) by default so we don't
    dump request/response payloads for every one of ~1 lakh operations.
    Also attach a dedicated file handler capturing ONLY WARNING/ERROR+ SDK
    logs (e.g. throttling, auth errors) into the same run log, so genuine
    problems are never silently lost.
    """
    azure_logger = logging.getLogger("azure")
    azure_logger.setLevel(logging.WARNING)  # suppress INFO/DEBUG payload spam

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | AZURE-SDK | %(message)s"
    )
    handler = RotatingFileHandler(
        run_log_path, maxBytes=50 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    handler.setFormatter(fmt)
    handler.setLevel(logging.WARNING)
    azure_logger.addHandler(handler)
    azure_logger.propagate = False


def enable_verbose_azure_logging_temporarily():
    """
    Context manager-ish helper: bump Azure SDK loggers to DEBUG so full
    request/response (including payload) logs are captured -- used ONLY
    around a failed operation for diagnosis, then reverted.
    """
    azure_logger = logging.getLogger("azure")
    previous_level = azure_logger.level
    azure_logger.setLevel(logging.DEBUG)
    return previous_level


def restore_azure_logging(previous_level):
    logging.getLogger("azure").setLevel(previous_level)


# --------------------------------------------------------------------------
# CSV logging (thread-safe)
# --------------------------------------------------------------------------
CSV_HEADERS = [
    "timestamp_utc",
    "source_blob",
    "destination_blob",
    "status",       # MOVED / FAILED / SKIPPED_ALREADY_MOVED
    "size_bytes",
    "duration_seconds",
    "error_message",
]


class CsvLogger:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._lock = threading.Lock()
        is_new = not os.path.exists(csv_path)
        # open once, append mode, keep handle open for the whole run
        self._fh = open(csv_path, mode="a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=CSV_HEADERS)
        if is_new:
            self._writer.writeheader()
            self._fh.flush()

    def write(self, row: dict):
        with self._lock:
            self._writer.writerow(row)
            self._fh.flush()  # flush per-row: safe for long unattended runs

    def close(self):
        with self._lock:
            self._fh.close()


def load_already_moved(csv_glob_pattern: str) -> set:
    """
    Scan any existing CSV log(s) matching the glob pattern and return the
    set of source blob names already marked MOVED, so re-runs can skip them.
    """
    moved = set()
    for path in glob(csv_glob_pattern):
        try:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("status") == "MOVED":
                        moved.add(row["source_blob"])
        except Exception:
            # a partially written/corrupt old log shouldn't crash the run
            continue
    return moved


# --------------------------------------------------------------------------
# Core move logic
# --------------------------------------------------------------------------
@dataclass
class MoveResult:
    source_blob: str
    destination_blob: str
    status: str
    size_bytes: int = 0
    duration_seconds: float = 0.0
    error_message: str = ""


def move_single_blob(
    connection_string: str,
    container_name: str,
    source_blob_name: str,
    destination_blob_name: str,
    logger: logging.Logger,
    copy_poll_interval: float = 0.5,
    copy_timeout_seconds: int = 300,
) -> MoveResult:
    """
    Moves one blob: server-side copy (no download/upload of bytes) -> poll
    until copy completes -> verify -> delete source.
    On any failure, briefly enables verbose Azure SDK logging so the payload
    detail is captured in the run log, then reverts it.
    """
    start = time.monotonic()

    # New client per call is fine here since we're inside worker threads and
    # BlobServiceClient / BlobClient are relatively lightweight; alternatively
    # a shared client can be reused (see note in main()).
    service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = service_client.get_container_client(container_name)

    source_client = container_client.get_blob_client(source_blob_name)
    dest_client = container_client.get_blob_client(destination_blob_name)

    try:
        source_url = source_client.url

        # Kick off server-side copy
        dest_client.start_copy_from_url(source_url)

        # Poll for copy completion
        elapsed = 0.0
        while True:
            props = dest_client.get_blob_properties()
            copy_status = props.copy.status  # "pending" | "success" | "failed" | "aborted"

            if copy_status == "success":
                size_bytes = props.size
                break
            elif copy_status in ("failed", "aborted"):
                raise HttpResponseError(
                    f"Copy ended with status='{copy_status}' "
                    f"(copy_status_description={props.copy.status_description})"
                )

            time.sleep(copy_poll_interval)
            elapsed += copy_poll_interval
            if elapsed > copy_timeout_seconds:
                raise TimeoutError(
                    f"Copy did not complete within {copy_timeout_seconds}s"
                )

        # Delete source only after confirmed successful copy
        source_client.delete_blob()

        duration = time.monotonic() - start
        logger.info(
            f"MOVED  '{source_blob_name}' -> '{destination_blob_name}' "
            f"({size_bytes} bytes, {duration:.2f}s)"
        )
        return MoveResult(
            source_blob=source_blob_name,
            destination_blob=destination_blob_name,
            status="MOVED",
            size_bytes=size_bytes,
            duration_seconds=duration,
        )

    except ResourceNotFoundError as e:
        duration = time.monotonic() - start
        logger.error(f"FAILED (not found) '{source_blob_name}': {e}")
        return MoveResult(
            source_blob=source_blob_name,
            destination_blob=destination_blob_name,
            status="FAILED",
            duration_seconds=duration,
            error_message=str(e),
        )

    except Exception as e:
        # Failure path: turn on verbose Azure SDK logging just for a retry
        # attempt so we capture full payload/diagnostic detail for this blob.
        duration = time.monotonic() - start
        prev_level = enable_verbose_azure_logging_temporarily()
        try:
            logger.error(
                f"FAILED '{source_blob_name}' -> '{destination_blob_name}': {e}. "
                f"Retrying once with verbose Azure SDK logging enabled..."
            )
            try:
                dest_client.start_copy_from_url(source_client.url)
                elapsed = 0.0
                while True:
                    props = dest_client.get_blob_properties()
                    copy_status = props.copy.status
                    if copy_status == "success":
                        size_bytes = props.size
                        source_client.delete_blob()
                        duration = time.monotonic() - start
                        logger.info(
                            f"MOVED (on retry) '{source_blob_name}' -> "
                            f"'{destination_blob_name}' ({size_bytes} bytes, {duration:.2f}s)"
                        )
                        return MoveResult(
                            source_blob=source_blob_name,
                            destination_blob=destination_blob_name,
                            status="MOVED",
                            size_bytes=size_bytes,
                            duration_seconds=duration,
                        )
                    elif copy_status in ("failed", "aborted"):
                        raise HttpResponseError(f"Retry copy status='{copy_status}'")
                    time.sleep(copy_poll_interval)
                    elapsed += copy_poll_interval
                    if elapsed > copy_timeout_seconds:
                        raise TimeoutError("Retry copy timed out")
            except Exception as retry_err:
                duration = time.monotonic() - start
                logger.error(
                    f"FAILED PERMANENTLY '{source_blob_name}' -> "
                    f"'{destination_blob_name}': {retry_err}"
                )
                return MoveResult(
                    source_blob=source_blob_name,
                    destination_blob=destination_blob_name,
                    status="FAILED",
                    duration_seconds=duration,
                    error_message=str(retry_err),
                )
        finally:
            restore_azure_logging(prev_level)


# --------------------------------------------------------------------------
# Listing
# --------------------------------------------------------------------------
def list_source_blobs(container_client, source_prefix: str, batch_size: int, already_moved: set, logger):
    """
    Lazily lists blobs under source_prefix and yields up to batch_size
    NEW (not-already-moved) blob names. Uses the SDK's paged iterator so
    it does not load all 3 lakh names into memory at once.
    """
    count_yielded = 0
    count_seen = 0
    for blob in container_client.list_blobs(name_starts_with=source_prefix):
        count_seen += 1
        # skip "directory marker" zero-byte blobs that equal the prefix itself
        if blob.name == source_prefix:
            continue
        if blob.name in already_moved:
            continue
        yield blob.name
        count_yielded += 1
        if count_yielded >= batch_size:
            logger.info(
                f"Reached batch size {batch_size} after scanning {count_seen} "
                f"source blobs. Stopping listing for this run."
            )
            return
    logger.info(
        f"Source listing exhausted: scanned {count_seen} blobs, "
        f"yielded {count_yielded} for this run (source may be fully drained)."
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Move blobs between prefixes in an Azure container.")
    parser.add_argument(
        "--container",
        required=True,
        help="Azure Storage container name (source and destination prefixes are inside this one container).",
    )
    parser.add_argument(
        "--source-prefix",
        default="call-centre/imarque/cross-sell/result-31-07-2026-filtered/",
    )
    parser.add_argument(
        "--dest-prefix",
        default="call-centre/imarque/cross-sell/intermediate-input/",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100_000,
        help="Number of blobs to move in this run (default: 1,00,000).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=32,
        help="Thread pool size for concurrent moves.",
    )
    parser.add_argument(
        "--csv-log",
        default=f"move_log_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv",
        help="CSV file to record every move result (append mode).",
    )
    parser.add_argument(
        "--csv-log-glob",
        default="move_log_*.csv",
        help="Glob used to find PAST csv logs so already-moved blobs are skipped on resume.",
    )
    parser.add_argument(
        "--run-log",
        default="move_run.log",
        help="Human-readable run log file (progress/warnings/errors).",
    )
    args = parser.parse_args()

    connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        print("ERROR: Set the AZURE_STORAGE_CONNECTION_STRING environment variable.", file=sys.stderr)
        sys.exit(1)

    logger = build_run_logger(args.run_log)
    silence_azure_sdk_logs(args.run_log)

    logger.info("=" * 80)
    logger.info("Starting blob move run")
    logger.info(f"Container       : {args.container}")
    logger.info(f"Source prefix   : {args.source_prefix}")
    logger.info(f"Dest prefix     : {args.dest_prefix}")
    logger.info(f"Batch size      : {args.batch_size}")
    logger.info(f"Max workers     : {args.max_workers}")
    logger.info(f"CSV log         : {args.csv_log}")
    logger.info(f"Run log         : {args.run_log}")

    already_moved = load_already_moved(args.csv_log_glob)
    logger.info(f"Found {len(already_moved)} blobs already marked MOVED in past CSV logs; these will be skipped.")

    service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = service_client.get_container_client(args.container)

    csv_logger = CsvLogger(args.csv_log)

    blob_names = list(
        list_source_blobs(container_client, args.source_prefix, args.batch_size, already_moved, logger)
    )
    total = len(blob_names)
    logger.info(f"Prepared {total} blobs to move in this run.")

    if total == 0:
        logger.info("Nothing to move. Exiting.")
        csv_logger.close()
        return

    moved_count = 0
    failed_count = 0
    lock = threading.Lock()
    start_time = time.monotonic()

    def worker(src_name: str):
        rel_path = src_name[len(args.source_prefix):] if src_name.startswith(args.source_prefix) else src_name
        dest_name = args.dest_prefix + rel_path
        result = move_single_blob(
            connection_string=connection_string,
            container_name=args.container,
            source_blob_name=src_name,
            destination_blob_name=dest_name,
            logger=logger,
        )
        csv_logger.write({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source_blob": result.source_blob,
            "destination_blob": result.destination_blob,
            "status": result.status,
            "size_bytes": result.size_bytes,
            "duration_seconds": f"{result.duration_seconds:.3f}",
            "error_message": result.error_message,
        })
        return result

    with ThreadPoolExecutor(max_workers=args.max_workers, thread_name_prefix="mover") as executor:
        futures = {executor.submit(worker, name): name for name in blob_names}
        for i, future in enumerate(as_completed(futures), start=1):
            try:
                result = future.result()
                with lock:
                    if result.status == "MOVED":
                        moved_count += 1
                    else:
                        failed_count += 1
            except Exception as e:
                # Should not normally happen since move_single_blob catches internally,
                # but guard against unexpected errors so one bad blob can't kill the run.
                src_name = futures[future]
                logger.error(f"UNEXPECTED ERROR moving '{src_name}': {e}")
                with lock:
                    failed_count += 1
                csv_logger.write({
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "source_blob": src_name,
                    "destination_blob": "",
                    "status": "FAILED",
                    "size_bytes": 0,
                    "duration_seconds": "",
                    "error_message": str(e),
                })

            if i % 1000 == 0 or i == total:
                elapsed = time.monotonic() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Progress: {i}/{total} processed | moved={moved_count} "
                    f"failed={failed_count} | {rate:.1f} blobs/sec"
                )

    csv_logger.close()

    elapsed_total = time.monotonic() - start_time
    logger.info("-" * 80)
    logger.info(f"Run complete in {elapsed_total:.1f}s")
    logger.info(f"Total processed : {total}")
    logger.info(f"Moved           : {moved_count}")
    logger.info(f"Failed          : {failed_count}")
    logger.info(f"CSV log written : {args.csv_log}")
    if failed_count > 0:
        logger.warning(
            f"{failed_count} blobs FAILED to move. Check '{args.run_log}' for details "
            f"(verbose Azure SDK payload logs were captured for each failure) "
            f"and re-run the script -- failed blobs are NOT marked MOVED, so they "
            f"will be retried automatically on the next run."
        )
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
