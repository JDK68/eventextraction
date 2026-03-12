import pandas as pd
from pathlib import Path

site = "nacacnet.org_pattern_labeled"

event_path = Path(f"runs/event/{site}/preds.csv")
anchor_path = Path(f"runs/anchor/{site}/preds.csv")

df_event = pd.read_csv(event_path).sort_values("score", ascending=False)
df_anchor = pd.read_csv(anchor_path).sort_values("score", ascending=False)

print("\n=== EVENT_CONTENT top 20 ===\n")
print(df_event[["text_context", "label", "event_id", "is_event_content", "score"]].head(20).to_string(index=False))

print("\n=== EVENT_ANCHOR top 20 ===\n")
print(df_anchor[["text_context", "label", "event_id", "is_event_anchor", "score"]].head(20).to_string(index=False))