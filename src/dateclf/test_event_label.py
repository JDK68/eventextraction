from pathlib import Path
from data import load_config, load_raw_merged, add_is_event_content_label

# charger config
config = load_config("../../config.yaml")

# charger dataset
df = load_raw_merged(config)

# créer le label event
df = add_is_event_content_label(df)

# afficher distribution
print(df["is_event_content"].value_counts())

# afficher ratio
print("\nRatio:")
print(df["is_event_content"].value_counts(normalize=True))