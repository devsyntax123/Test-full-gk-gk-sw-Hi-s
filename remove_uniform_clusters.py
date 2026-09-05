"""
Excel/CSV file name: all_images.csv

LOGIC:
For each unique cluster_id:
  - If the cluster has only 1 row -> ALWAYS KEEP (never removed).
  - If the cluster has 2+ rows:
        - If ALL rows have the SAME customer_name_norm -> REMOVE the entire cluster (all its rows)
        - OR if ALL rows have the SAME fathers_name_norm -> REMOVE the entire cluster (all its rows)
        - Otherwise (mixed names in both columns) -> KEEP the cluster

Example:
  cluster_id | customer_name_norm            | fathers_name_norm            | Result
  C1 (4 rows)| John, John, John, John        | Ram, Shyam, Ravi, Amit        | remove (customer name all same)
  C2 (4 rows)| John, John, John, Mike        | Ram, Ram, Ram, Ram            | remove (father name all same)
  C3 (4 rows)| John, John, John, Mike        | Ram, Shyam, Ravi, Amit        | keep (neither fully same)
  C4 (1 row) | John                          | Ram                           | keep (single row cluster)
"""

import pandas as pd

# ---- CONFIG ----
INPUT_FILE = "all_images.csv"
OUTPUT_FILE = "all_images_cleaned.csv"

CLUSTER_COL = "cluster_id"
CUSTOMER_COL = "customer_name_norm"
FATHER_COL = "fathers_name_norm"

# ---- LOAD ----
df = pd.read_csv(INPUT_FILE)

# ---- IDENTIFY CLUSTERS TO REMOVE ----
def is_fully_uniform(series):
    """Return True if all non-null values in the series are identical
    and there is more than 1 row in the group."""
    if len(series) <= 1:
        return False
    return series.nunique(dropna=False) == 1

# Group by cluster_id and check uniformity for each target column
grouped = df.groupby(CLUSTER_COL)

clusters_to_remove = set()

for cluster_id, group in grouped:
    if len(group) <= 1:
        # Single row cluster -> never remove
        continue

    customer_uniform = is_fully_uniform(group[CUSTOMER_COL])
    father_uniform = is_fully_uniform(group[FATHER_COL])

    if customer_uniform or father_uniform:
        clusters_to_remove.add(cluster_id)

# ---- FILTER OUT THOSE CLUSTERS ----
cleaned_df = df[~df[CLUSTER_COL].isin(clusters_to_remove)].copy()

# ---- REPORT ----
print(f"Total rows in original file      : {len(df)}")
print(f"Total unique clusters            : {df[CLUSTER_COL].nunique()}")
print(f"Clusters removed (fully uniform) : {len(clusters_to_remove)}")
print(f"Rows removed                     : {len(df) - len(cleaned_df)}")
print(f"Rows remaining                   : {len(cleaned_df)}")

if clusters_to_remove:
    print("\nRemoved cluster_ids:")
    print(sorted(clusters_to_remove))

# ---- SAVE ----
cleaned_df.to_csv(OUTPUT_FILE, index=False)
print(f"\nCleaned file saved as: {OUTPUT_FILE}")
