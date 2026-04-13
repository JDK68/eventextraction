# Event Extraction from Heterogenous Websites

**Honours BSc Capstone Project — University of Ottawa (2025–2026)**

Automatic extraction of structured events (Date, Time, Location) from heterogeneous college fair web pages using a machine learning pipeline — no site-specific XPath rules, no hardcoded selectors.

## Overview

College fair websites list events in wildly different HTML structures — tables, card grids, nested divs, flat lists. This project builds a **generalizable ML pipeline** that detects and clusters event nodes from any unseen site's DOM, evaluated under strict **Leave-One-Site-Out (LOSO)** cross-validation across 16 real-world sites (~200 events total).

### Pipeline Architecture

```
Raw HTML (DOM nodes as CSV)
        │
        ▼
┌─────────────────────┐
│  Feature Engineering │  27 DOM-based features per node
│  (features.py)       │  depth, tag, parent_tag, text signals,
│                      │  neighbor context, CSS attributes
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  Binary Detection    │  LightGBM: "is this node part of an event?"
│  (train.py)          │  → top-K candidates by probability score
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  DOM Clustering      │  Group candidates into individual events
│  (train.py)          │  using rendering_order gaps, parent_index,
│                      │  depth + post-processing heuristics
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  Strict Evaluation   │  Per-event: detection, same cluster,
│  (train.py)          │  exclusivity, field coverage (≥33%)
└─────────────────────┘
```

### Strict Metric — Perfect Reconstitution

An event counts as "perfectly reconstituted" only if **all four** conditions hold:

1. **Detection** — at least one of its ground-truth nodes appears in the top-K candidates
2. **Same cluster** — all its detected nodes land in the same predicted cluster
3. **Exclusivity** — that cluster contains nodes from *only* this event (no mixing)
4. **Field coverage** — ≥33% of its Date, Time, and Location nodes are present in the cluster

## Project Structure

```
eventdetection/
├── phase2/
│   ├── data/
│   │   └── raw/                    # 16 site CSVs (not tracked in git)
│   │       ├── gpacac.net_pattern_labeled.csv
│   │       ├── hawaiiacac.org_pattern_labeled.csv
│   │       ├── iacac.knack.com_pattern_labeled.csv
│   │       ├── members.sacac.org_column_labeled.csv
│   │       ├── members.sacac.org_pattern_labeled.csv
│   │       ├── nacacnet.org_pattern_labeled.csv
│   │       ├── neacac_1_pattern_labeled 1.csv
│   │       ├── neacac_fall.net_pattern_labeled.csv
│   │       ├── neacac_spring.net_pattern_labeled.csv
│   │       ├── outofstatecollegefairs.org_pattern_labeled.csv
│   │       ├── pnacac_spring.org_pattern_labeled.csv
│   │       ├── rmacac.org_pattern_labeled.csv
│   │       ├── sacrao.org_column_labeled.csv
│   │       ├── wacac.org_pattern_labeled.csv
│   │       └── ...
│   ├── config.yaml                 # Hyperparameters, paths, label definitions
│   └── src/
│       └── dateclf/
│           ├── __init__.py
│           ├── data.py             # Data loading (load_config, load_raw_merged, add_is_event_member_label)
│           ├── split.py            # LOSO cross-validation folds (loso_folds_event)
│           ├── features.py         # Feature engineering (add_dom_neighbor_features)
│           ├── sampling.py         # Negative undersampling (undersample_negatives)
│           ├── metrics.py          # Node-level and event-level metrics
│           └── train.py            # Main pipeline: train, cluster, evaluate
└── README.md
```

## Data Format

Each site is a CSV where every row represents one DOM node from the parsed HTML page:

| Column | Description |
|--------|-------------|
| `rendering_order` | DFS traversal order of the node in the DOM |
| `tag` | HTML tag (Span, Div, A, H5, P, etc.) |
| `attributes` | Raw CSS classes, IDs, and other attributes |
| `text_context` | Text content of the node |
| `depth` | Depth in the DOM tree |
| `parent_index` | `rendering_order` of the parent node |
| `parent_tag` | HTML tag of the parent |
| `text_length`, `word_count` | Text statistics |
| `letter_ratio`, `digit_ratio`, `whitespace_ratio` | Character composition |
| `contains_date`, `contains_time` | Boolean date/time pattern detection |
| `has_class`, `has_id` | Whether node has CSS class/ID |
| `attr_has_word_date/time/location` | Keyword detection in attributes |
| `label` | Ground-truth label (Date, Time, StartEndTime, Location, Name, Description, Other) |
| `event_id` | Ground-truth event assignment (NaN for non-event nodes) |

## Features (27 total)

The model uses three categories of features:

**Structural** — `depth`, `tag`, `parent_tag`, `num_siblings`, `has_class`, `has_id`

**Textual** — `text_length`, `word_count`, `letter_ratio`, `digit_ratio`, `whitespace_ratio`, `contains_date`, `contains_time`, `starts_with_digit`, `ends_with_digit`

**Contextual** — `attr_has_word_date/time/location`, `text_has_word_date/time/location`, `prev_contains_date`, `next_contains_date`, `prev_contains_time`, `next_contains_time`, `same_parent_as_prev`, `same_parent_as_next`

## Setup

### Requirements

- Python 3.10+
- LightGBM
- pandas, scikit-learn, PyYAML

```bash
pip install lightgbm pandas scikit-learn pyyaml
```

### Configuration

All hyperparameters live in `phase2/config.yaml`:

```yaml
seed: 42

data:
  raw_dir: ../data/raw
  file_glob: "*.csv"
  sep: ","
  encoding: "utf-8"
  site_id_source: "filename"
  label_col: "label"

targets:
  event_content_positive_labels:
    - "Date"
    - "DateTime"
    - "StartTime"
    - "EndTime"
    - "StartEndTime"
    - "Time"
    - "TimeLocation"
    - "Location"
    - "NameLocation"
    - "Name"
    - "Description"

sampling:
  keep_neg_ratio: 0.15

model:
  learning_rate: 0.05
  num_leaves: 63
  min_data_in_leaf: 30
  feature_fraction: 0.9
  bagging_fraction: 0.8
  bagging_freq: 1
  lambda_l2: 1.0
  num_boost_round: 400
```

## Usage

### Run the full LOSO evaluation

```powershell
cd phase2/src
python -m dateclf.train
```

This runs 16-fold LOSO cross-validation and prints per-site results including node-level precision/recall, clustering ARI, and the strict perfect reconstitution rate.

### Key output metrics

| Metric | Score |
|--------|-------|
| **Detection** | 99.5% ± 1.3% |
| **Clustering** | 62.6% ± 41.0% |
| **Date Accuracy** | 62.6% ± 41.0% |
| **Time Accuracy** | 62.2% ± 41.5% |
| **Location Accuracy** | 62.6% ± 41.0% |
| **Perfect Reconstitution** | **62.2% ± 41.5%** |

Detection is nearly perfect — the bottleneck is clustering exclusivity.

## Key Findings

### Structural Discovery: Two Types of Sites

Exploratory analysis revealed the 16 sites fall into two fundamentally different HTML patterns:

| Type | Sites | Structure | Clustering signal |
|------|-------|-----------|------------------|
| **Row-per-event** | iacac, outofstate, pnacac, rmacac | Each `<tr>` = 1 event | Clear parent_index gap separation |
| **Card-based** | sacac, nacacnet, neacac, wacac, gpacac, hawaii | Nested div cards | Overlapping gaps, needs different strategy |

This structural heterogeneity is the core challenge — no single gap threshold works for all sites.

### BIO Sequence Labeling (Explored)

An alternative approach treating DOM nodes as a sequence (B-Event / I-Event / O tags) was validated via EDA:
- 0 event overlaps across all tested sites
- 0 revisits (strictly sequential)
- 95% of intra-event gaps ≤ 3 nodes

BIO achieved **100% on neacac sites** but struggled with boundary detection (B label = only 6.2% of nodes). The approach is promising as an auxiliary signal combined with the main pipeline.

## Git Workflow

```powershell
cd C:\Users\Jad\eventdetection
git add -A
git commit -m "description"
git push origin HEAD:main --force
```

## Authors

**Jad Kebal** — Computer Science Honours BSc, University of Ottawa.

**Mohamed Amine Ablouh** — Computer Science Honours BSc, University of Ottawa.

**Darine Fourar Laidi** — Computer Science Honours BSc, University of Ottawa.

Supervised capstone project — Applied ML/AI specialization with emphasis on NLP and web information extraction.

## License

Academic project — University of Ottawa, 2025–2026.
