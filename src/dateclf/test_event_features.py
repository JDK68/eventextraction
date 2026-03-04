from data import (
    load_config,
    load_raw_merged,
    add_is_event_content_label,
    build_feature_matrix_for_event
)

# load config
config = load_config("../../config.yaml")

# load dataset
df = load_raw_merged(config)

# create event label
df = add_is_event_content_label(df)

# build feature matrix
X, y, groups = build_feature_matrix_for_event(df)

print("Feature matrix shape:")
print(X.shape)

print("\nTarget distribution:")
print(y.value_counts())

print("\nNumber of sites:")
print(groups.nunique())