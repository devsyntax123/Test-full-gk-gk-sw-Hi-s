"""
Filter a large call-log file (5L+ rows) to keep only rows whose agreement_no
appears in a separate agreement list AND whose audio_duration_sec < 1200 (20 min).

One agreement_no can have multiple matching audio instances -> all are kept.
"""

import pandas as pd

# ---------- CONFIG: update these paths/sheet names as needed ----------
BIG_FILE = "big_file.xlsx"          # the 5L+ row file (file_name, file_path, agreement_no, audio_duration_sec)
BIG_SHEET = 0                        # sheet name/index, or use pd.read_csv if it's a CSV

AGREEMENT_FILE = "agreement_list.xlsx"   # file with only agreement_no column
AGREEMENT_SHEET = 0

OUTPUT_FILE = "output_filtered.xlsx"

DURATION_LIMIT_SEC = 20 * 60   # 20 minutes = 1200 seconds
# -----------------------------------------------------------------------


def read_any(path, sheet=0):
    """Read xlsx or csv depending on extension."""
    if str(path).lower().endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path, sheet_name=sheet)


def main():
    # 1. Load both files
    df_big = read_any(BIG_FILE, BIG_SHEET)
    df_agreements = read_any(AGREEMENT_FILE, AGREEMENT_SHEET)

    # 2. Normalize agreement_no (strip spaces, ensure string type) to avoid mismatch issues
    df_big["agreement_no"] = df_big["agreement_no"].astype(str).str.strip()
    df_agreements["agreement_no"] = df_agreements["agreement_no"].astype(str).str.strip()

    agreement_set = set(df_agreements["agreement_no"])

    # 3. Ensure duration column is numeric
    df_big["audio_duration_sec"] = pd.to_numeric(df_big["audio_duration_sec"], errors="coerce")

    # 4. Apply both conditions:
    #    - agreement_no is in the given list
    #    - duration is less than 20 minutes (1200 sec)
    mask = (
        df_big["agreement_no"].isin(agreement_set)
        & (df_big["audio_duration_sec"] < DURATION_LIMIT_SEC)
    )

    df_result = df_big[mask].copy()

    # 5. Save output (all original columns preserved, multiple rows per agreement_no kept)
    df_result.to_excel(OUTPUT_FILE, index=False)

    # 6. Summary
    print(f"Total rows in big file        : {len(df_big)}")
    print(f"Agreement numbers to search    : {len(agreement_set)}")
    print(f"Matched rows (<20 min)         : {len(df_result)}")
    print(f"Unique agreement_no in output  : {df_result['agreement_no'].nunique()}")
    print(f"Output saved to                : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
