import pandas as pd

# Load the input CSV
df = pd.read_csv("input.csv")  # replace with your actual file name

# Threshold for "greater than 20 minutes" in seconds
THRESHOLD = 20 * 60  # 1200 seconds

# Flag rows where duration > 20 min
df["is_gt_20min"] = df["duration_seconds"] > THRESHOLD

# Group by loan_account_no and aggregate
result = df.groupby("loan_account_no").apply(
    lambda g: pd.Series({
        "total_no_of_audio": g["file_name"].count(),
        "total_no_of_calls_gt_20min": g["is_gt_20min"].sum(),
        "total_duration_sec": g["duration_seconds"].sum(),
        "total_duration_gt_20min_sec": g.loc[g["is_gt_20min"], "duration_seconds"].sum()
    })
).reset_index()

# Ensure integer types for count columns
result["total_no_of_audio"] = result["total_no_of_audio"].astype(int)
result["total_no_of_calls_gt_20min"] = result["total_no_of_calls_gt_20min"].astype(int)
result["total_duration_sec"] = result["total_duration_sec"].astype(int)
result["total_duration_gt_20min_sec"] = result["total_duration_gt_20min_sec"].astype(int)

# Save to output CSV
result.to_csv("output.csv", index=False)

print(result)
