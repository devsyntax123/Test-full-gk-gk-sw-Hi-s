"""
Process call recording filenames and build an aggreement-wise summary.

Expected filename format:
    BH3058CD0000727_27-07-2026-18-23-08_HINDI_mp3_40.wav
    <agreement_no>_<dd-mm-YYYY-HH-MM-SS>_<LANGUAGE>_<ext>_<duration_sec>.wav

Just edit INPUT_CSV_PATH and OUTPUT_XLSX_PATH below and run:
    python process_calls.py
"""

import re
import pandas as pd

# ==== EDIT THESE PATHS ====
INPUT_CSV_PATH = "/path/to/input.csv"
OUTPUT_XLSX_PATH = "/path/to/output.xlsx"
# ===========================

DURATION_THRESHOLD_SEC = 1200  # 20 minutes


def parse_filename(file_name: str):
    """
    Parse a single file_name into its components.
    Returns dict with agreement_no, call_datetime_str, language, duration_sec
    or None if the filename doesn't match the expected pattern.
    """
    # Strip extension if present (.wav etc.) - the real extension is the
    # trailing .wav, but the format also embeds an inner "ext" token (mp3).
    base = re.sub(r"\.wav$", "", file_name.strip(), flags=re.IGNORECASE)

    parts = base.split("_")
    # Expect: [agreement_no, datetime, language, ext, duration]
    if len(parts) < 5:
        return None

    agreement_no = parts[0]
    datetime_str = parts[1]
    language = parts[2]
    # ext = parts[3]  # e.g. mp3 - not needed for output
    duration_str = parts[-1]  # last part = duration in seconds

    if not duration_str.isdigit():
        return None

    duration_sec = int(duration_str)

    return {
        "agreement_no": agreement_no,
        "call_datetime": datetime_str,
        "language": language,
        "duration_sec": duration_sec,
    }


def main():
    df = pd.read_csv(INPUT_CSV_PATH)

    if "file_name" not in df.columns:
        raise ValueError("Input CSV must have a 'file_name' column")

    parsed_rows = []
    bad_rows = []

    for fname in df["file_name"]:
        parsed = parse_filename(str(fname))
        if parsed is None:
            bad_rows.append(fname)
            continue
        parsed["file_name"] = fname
        parsed_rows.append(parsed)

    if bad_rows:
        print(f"Warning: {len(bad_rows)} file_name(s) could not be parsed and were skipped:")
        for b in bad_rows[:20]:
            print(f"   - {b}")
        if len(bad_rows) > 20:
            print(f"   ... and {len(bad_rows) - 20} more")

    parsed_df = pd.DataFrame(parsed_rows)

    if parsed_df.empty:
        raise ValueError("No file names could be parsed. Check the filename format.")

    parsed_df["is_gt_20min"] = parsed_df["duration_sec"] > DURATION_THRESHOLD_SEC

    # Aggregate per agreement_no
    summary = (
        parsed_df.groupby("agreement_no")
        .apply(
            lambda g: pd.Series(
                {
                    "total_no_of_audio_per_agmnt_no": len(g),
                    "total_no_of_calls_gt_20min": int(g["is_gt_20min"].sum()),
                    "total_duration_sec": int(g["duration_sec"].sum()),
                    "total_duration_gt_20min_sec": int(
                        g.loc[g["is_gt_20min"], "duration_sec"].sum()
                    ),
                }
            )
        )
        .reset_index()
    )

    # Ensure integer dtypes (groupby/apply can upcast to float)
    int_cols = [
        "total_no_of_audio_per_agmnt_no",
        "total_no_of_calls_gt_20min",
        "total_duration_sec",
        "total_duration_gt_20min_sec",
    ]
    summary[int_cols] = summary[int_cols].astype(int)

    summary = summary.sort_values("agreement_no").reset_index(drop=True)

    summary.to_excel(OUTPUT_XLSX_PATH, index=False, sheet_name="Summary")
    print(f"Done. Wrote {len(summary)} agreement(s) to {OUTPUT_XLSX_PATH}")


if __name__ == "__main__":
    main()
