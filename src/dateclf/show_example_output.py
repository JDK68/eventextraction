import pandas as pd
from pathlib import Path

pred_path = Path("runs/event/nacacnet.org_pattern_labeled/preds.csv")

df = pd.read_csv(pred_path)

# Trier par score
df = df.sort_values("score", ascending=False)

print("\n=== Top predictions (all nodes) ===\n")
print(df[["page_id","score","is_event_content"]].head(10))

print("\n=== Top TRUE event nodes detected ===\n")

events = df[df["is_event_content"] == 1].sort_values("score", ascending=False)

print(events[["page_id","score"]].head(10))