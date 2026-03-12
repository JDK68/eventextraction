import pandas as pd
from pathlib import Path

raw_dir = Path("../data/raw")
all_files = sorted(raw_dir.glob("*.csv"))

event_labels = {
    "Date",
    "DateTime",
    "StartTime",
    "EndTime",
    "StartEndTime",
    "Time",
    "TimeLocation",
    "Location",
    "Name",
    "NameLocation",
    "Description",
}

rows = []

for fp in all_files:
    df = pd.read_csv(fp)

    site_id = fp.stem
    num_nodes = len(df)

    if "label" not in df.columns:
        continue

    df["is_event_content"] = df["label"].astype(str).isin(event_labels)
    num_positive_nodes = int(df["is_event_content"].sum())

    if "event_id" in df.columns:
        positive_df = df.loc[df["is_event_content"] & df["event_id"].notna()].copy()

        positive_event_ids = positive_df["event_id"].astype(str).nunique()

        if len(positive_df) > 0:
            avg_nodes_per_event = positive_df.groupby("event_id").size().mean()
        else:
            avg_nodes_per_event = 0.0
    else:
        positive_event_ids = None
        avg_nodes_per_event = None

    rows.append(
        {
            "site_id": site_id,
            "num_nodes": num_nodes,
            "num_positive_nodes": num_positive_nodes,
            "num_events": positive_event_ids,
            "avg_positive_nodes_per_event": avg_nodes_per_event,
        }
    )

summary = pd.DataFrame(rows).sort_values("num_events", ascending=False)

print(summary.to_string(index=False))

print("\nTotals / averages:")
print(
    summary[
        ["num_nodes", "num_positive_nodes", "num_events", "avg_positive_nodes_per_event"]
    ].describe()
)