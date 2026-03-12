import pandas as pd
from pathlib import Path

pred_path = Path("runs/event/nacacnet.org_pattern_labeled/preds.csv")

df = pd.read_csv(pred_path)

df = df.sort_values("score", ascending=False)

cols = [c for c in ["page_id", "score", "is_event_content", "text_context"] if c in df.columns]


print("\n=== Top predictions (all nodes) ===\n")
print(df[cols].head(10))


print("\n=== Top TRUE event nodes detected ===\n")
events = df[df["is_event_content"] == 1].sort_values("score", ascending=False)
print(events[cols].head(10))


print("\n=== Top FALSE POSITIVES ===\n")
fp = df[df["is_event_content"] == 0].sort_values("score", ascending=False)
print(fp[cols].head(10))


print("\n=== Top 5 per page ===\n")
for page_id, g in df.groupby("page_id"):
    print(f"\n--- PAGE: {page_id} ---")
    g = g.sort_values("score", ascending=False)
    print(g[cols].head(5))