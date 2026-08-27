"""
Delete all blobs in an Azure Blob Storage folder (virtual directory),
except specified file(s), using multithreading for large-scale deletion.

Requirements:
    pip install azure-storage-blob --break-system-packages

Usage:
    1. Fill in CONNECTION_STRING, CONTAINER_NAME, FOLDER_PREFIX, KEEP_FILES below.
    2. Run: python delete_blobs.py
"""

import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from azure.storage.blob import BlobServiceClient

# ------------------- CONFIG -------------------
CONNECTION_STRING = "PASTE_YOUR_AZURE_STORAGE_CONNECTION_STRING_HERE"
CONTAINER_NAME = "call-centre"
FOLDER_PREFIX = "atpl/cross-sell/intermediate-input/"   # trailing slash matters
KEEP_FILES = {
    "atpl/cross-sell/intermediate-input/keep_this_file.csv",
    # add more full blob paths here if you want to keep multiple files
}

MAX_WORKERS = 32          # number of parallel delete threads
BATCH_LOG_EVERY = 5000    # progress logging frequency
DRY_RUN = True            # set to False only after verifying the list looks correct
# ------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("delete_blobs.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def list_blobs_to_delete(container_client, prefix, keep_files):
    """Generator that yields blob names to delete, skipping KEEP_FILES."""
    count_total = 0
    count_kept = 0
    for blob in container_client.list_blobs(name_starts_with=prefix):
        count_total += 1
        if blob.name in keep_files:
            count_kept += 1
            logger.info(f"Keeping file: {blob.name}")
            continue
        yield blob.name
    logger.info(f"Scanned {count_total} blobs, keeping {count_kept}.")


def delete_blob(container_client, blob_name, dry_run=False):
    try:
        if dry_run:
            return blob_name, True, "dry-run (not deleted)"
        container_client.delete_blob(blob_name)
        return blob_name, True, "deleted"
    except Exception as e:
        return blob_name, False, str(e)


def main():
    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)

    logger.info(f"Starting deletion. Container='{CONTAINER_NAME}', Prefix='{FOLDER_PREFIX}'")
    logger.info(f"DRY_RUN={DRY_RUN}  (set DRY_RUN=False to actually delete)")

    deleted_count = 0
    failed_count = 0
    processed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        blob_generator = list_blobs_to_delete(container_client, FOLDER_PREFIX, KEEP_FILES)

        # Submit in a streaming fashion so we don't load 4+ lakh names into memory at once
        for blob_name in blob_generator:
            future = executor.submit(delete_blob, container_client, blob_name, DRY_RUN)
            futures[future] = blob_name

            # To avoid unbounded memory growth from too many pending futures,
            # periodically drain completed ones.
            if len(futures) >= MAX_WORKERS * 20:
                for f in list(as_completed(futures, timeout=None)):
                    name, success, msg = f.result()
                    processed += 1
                    if success:
                        deleted_count += 1
                    else:
                        failed_count += 1
                        logger.error(f"FAILED: {name} -> {msg}")
                    del futures[f]
                    if processed % BATCH_LOG_EVERY == 0:
                        logger.info(f"Progress: {processed} processed, {deleted_count} deleted, {failed_count} failed")
                    if len(futures) < MAX_WORKERS * 10:
                        break  # go back to submitting more

        # Drain remaining futures
        for f in as_completed(futures):
            name, success, msg = f.result()
            processed += 1
            if success:
                deleted_count += 1
            else:
                failed_count += 1
                logger.error(f"FAILED: {name} -> {msg}")
            if processed % BATCH_LOG_EVERY == 0:
                logger.info(f"Progress: {processed} processed, {deleted_count} deleted, {failed_count} failed")

    logger.info("=" * 50)
    logger.info(f"DONE. Total processed: {processed}, Deleted: {deleted_count}, Failed: {failed_count}")
    if DRY_RUN:
        logger.info("This was a DRY RUN — no files were actually deleted. Set DRY_RUN=False to delete for real.")


if __name__ == "__main__":
    main()
