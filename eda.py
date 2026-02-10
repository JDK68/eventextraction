import os
import glob
import pandas as pd

DATA_DIR = "Data/raw"  

paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
print(f"Found {len(paths)} CSV files\n")

def read_csv_safe(path):
    try:
        df = pd.read_csv(path)
        return df, ","
    except Exception:
        df = pd.read_csv(path, sep=";")
        return df, ";"

profiles = []

for p in paths:
    df, sep = read_csv_safe(p)
    prof = {
        "file": os.path.basename(p),
        "sep": sep,
        "rows": len(df),
        "cols": df.shape[1],
        "events": df["event_id"].nunique() if "event_id" in df.columns else None,
        "labels": df["label"].nunique() if "label" in df.columns else None,
        "missing_label_pct": df["label"].isna().mean() * 100 if "label" in df.columns else None,
        "missing_event_id_pct": df["event_id"].isna().mean() * 100 if "event_id" in df.columns else None,
    }
    profiles.append(prof)

profile_df = pd.DataFrame(profiles).sort_values("rows", ascending=False)
print(profile_df.to_string(index=False))

print("\n=== GLOBAL LABEL DISTRIBUTION ===")

all_dfs = []
for p in paths:
    df, _ = read_csv_safe(p)
    df["source_file"] = os.path.basename(p)
    all_dfs.append(df)

full_df = pd.concat(all_dfs, ignore_index=True)

label_counts = full_df["label"].value_counts()
label_pct = full_df["label"].value_counts(normalize=True) * 100

label_summary = pd.DataFrame({
    "count": label_counts,
    "pct": label_pct.round(2)
})

print(label_summary)

print("\n=== LABEL DISTRIBUTION PER SOURCE ===")

label_per_source = (
    full_df
    .groupby(["source_file", "label"])
    .size()
    .unstack(fill_value=0)
)

print(label_per_source)

print("\n=== LABEL UNIQUENESS PER EVENT ===")

event_label_counts = (
    full_df
    .dropna(subset=["event_id"])
    .groupby(["event_id", "label"])
    .size()
    .reset_index(name="count")
)

multi_candidate = event_label_counts[event_label_counts["count"] > 1]

print("Events with multiple nodes for the same label:")
print(multi_candidate.head(20))
print(f"\nTotal such cases: {len(multi_candidate)}")

print("\n=== INTRA-LABEL FEATURE COMPARISON ===")

FEATURES = [
    "text_length",
    "word_count",
    "digit_ratio",
    "letter_ratio",
    "depth",
    "has_link"
]

LABELS_TO_CHECK = ["Date", "Location", "Name"]

for label in LABELS_TO_CHECK:
    print(f"\n--- Comparing label '{label}' vs 'Other' ---")
    
    pos = full_df[full_df["label"] == label]
    neg = full_df[full_df["label"] == "Other"]
    
    for feat in FEATURES:
        if feat not in full_df.columns:
            continue
        
        pos_mean = pos[feat].mean()
        neg_mean = neg[feat].mean()
        
        print(
            f"{feat:15s} | "
            f"{label}: {pos_mean:8.2f} | "
            f"Other: {neg_mean:8.2f}"
        )
