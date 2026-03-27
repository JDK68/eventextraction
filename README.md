# Event Extraction from Heterogeneous Websites

**ML Capstone Project - University of Ottawa**

Automated extraction of structured event information (Date, Time, Location) from heterogeneous website DOM structures using machine learning.

---

## 🎯 Results

**Overall Performance (16 websites, 187 events):**

| Metric | Score |
|--------|-------|
| **Perfect End-to-End Reconstitution** | **75.5%** |
| Event Detection | 99.5% |
| Clustering Accuracy | 88.3% |
| Date Extraction | 93.9% |
| Time Extraction | 96.4% |
| Location Extraction | 87.0% |

**Per-site breakdown:**
- 13/16 sites achieve 100% perfect reconstitution
- 1/16 site achieves 95.5%
- 2/16 sites remain challenging (0-7%)

---

## 📋 Overview

This system extracts structured event data from college fair websites with diverse HTML structures, without relying on site-specific XPath rules or templates.

### Problem
College fair aggregator websites use heterogeneous DOM structures. Traditional rule-based extraction requires custom patterns for each site.

### Solution
A machine learning pipeline that:
1. **Detects** event-related DOM nodes (LightGBM classifier)
2. **Clusters** nodes into events (spatial clustering)
3. **Extracts** structured fields (Date, Time, Location)
4. **Validates** end-to-end reconstitution quality

### Key Innovation
**Proximity expansion** post-processing captures short nodes (1-15 chars) like "7", "mar", "thu" that score poorly but are semantically part of events. This improved time extraction from 78% to 96.4%.

---

## 🏗️ Architecture
```
Input: Raw HTML → DOM nodes with features
  ↓
1. Detection (LightGBM)
   - 27 structural + semantic features
   - Top-K selection (adaptive: 30-100 nodes)
   ↓
2. Noise Removal
   - Filter navigation/footer patterns
   - Remove shallow/short fragments
   ↓
3. Clustering (Spatial)
   - rendering_order + parent_index + depth
   - Gap-based segmentation
   ↓
4. Proximity Expansion
   - Capture nearby short Date/Time/Location nodes
   - Window: ±5 rendering positions
   ↓
5. Field Extraction
   - Multi-cluster extraction (handles bad clustering)
   - Set-based validation (33% overlap threshold)
   ↓
Output: Structured events with Date, Time, Location
```

---

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip

### Setup
```bash
# Clone repository
git clone https://github.com/JDK68/eventextraction.git
cd eventextraction/phase2

# Install dependencies
pip install -r requirements.txt
```

**Dependencies:**
- pandas
- numpy
- lightgbm
- scikit-learn
- pyyaml

---

## 🚀 Usage

### Train and Evaluate
```bash
cd src
python -m dateclf.train
```

**Output:**
- Prints metrics for each fold (LOSO cross-validation)
- Saves models and results to `runs/simple/`
- Generates summary JSON with aggregate metrics

### View Results
```bash
cd ..
python view_results.py nacacnet.org_pattern_labeled
```

Shows detailed event-by-event extraction with clustering status and field validation.

### Generate HTML Demo
```bash
python export_full_demo.py
start comprehensive_demo.html
```

Creates interactive demo showing:
- Perfect site (95.5%)
- Problematic site (0%)
- Medium site (6.7%)
- System architecture and challenges

---

## 🔬 Methodology

### Dataset
- **16 websites** from college fair aggregators
- **187 manually labeled events**
- **3,030 DOM nodes** total
- Labels: Event membership, Date, Time, Location, Name, Description

### Evaluation: Leave-One-Site-Out (LOSO) Cross-Validation
- Train on 15 sites, test on 1 held-out site
- Ensures generalization to unseen website structures
- No data leakage between train/test

### Features (27 total)
**Structural:**
- `depth`, `tag`, `parent_tag`, `num_siblings`

**Text Statistics:**
- `text_length`, `word_count`, `letter_ratio`, `digit_ratio`, `whitespace_ratio`

**Patterns (CORE):**
- `contains_date`, `contains_time`, `starts_with_digit`, `ends_with_digit`

**Attributes:**
- `has_class`, `has_id`, `attr_has_word_date`, `attr_has_word_time`, `attr_has_word_location`

**Context:**
- `prev_contains_date`, `next_contains_date`, `same_parent_as_prev`, etc.

### Clustering Algorithm
Gap-based spatial clustering using:
- `rendering_order` (DOM traversal order)
- `parent_index` (structural hierarchy)
- `depth` (tree depth)

**Parameters (tuned):**
- `gap_order=10`: Max rendering gap within event
- `gap_parent=8`: Max parent index gap
- `gap_depth=2`: Max depth difference

### Validation Criteria
Event achieves **perfect reconstitution** if:
1. ✅ At least 1 node detected in top-K
2. ✅ All detected nodes in same cluster
3. ✅ ≥33% of expected Date nodes extracted
4. ✅ ≥33% of expected Time nodes extracted
5. ✅ ≥33% of expected Location nodes extracted

---

## 📊 Detailed Results

### Component Performance

| Component | Accuracy | Details |
|-----------|----------|---------|
| Detection | 99.5% | Event nodes in top-K |
| Clustering | 88.3% | ARI (Adjusted Rand Index) |
| Date Extraction | 93.9% | ≥33% overlap with ground truth |
| Time Extraction | 96.4% | **+18% with proximity expansion** |
| Location Extraction | 87.0% | Challenges with long venue names |

### Before/After Proximity Expansion

| Metric | Before | After | Δ |
|--------|--------|-------|---|
| Perfect Reconstitution | 57.5% | **75.5%** | +18% |
| Date Accuracy | 87.6% | 93.9% | +6.3% |
| Time Accuracy | 78.1% | 96.4% | **+18.3%** |

---

## ⚠️ Limitations

### Phase 2 Scope
This system evaluates **detection and clustering** of event nodes. Field classification (Date/Time/Location) uses ground truth labels for evaluation purposes.

**In production, this would require:**
- A supervised field classifier (Phase 3)
- OR rule-based patterns (regex for dates/times)

### Known Failure Modes

**wacac.org (0% reconstitution):**
- Location nodes score poorly (long venue names: "university of california, riverside")
- Don't enter top-K selection
- Proximity expansion only captures short nodes (≤15 chars)

**kyacac.org (0%):**
- Clustering failure due to unusual DOM structure
- All nodes detected but not grouped correctly

**members.sacac.org (6.7%):**
- Split dates across 3+ nodes: "thu", "january", "29"
- Clustering rate 43-73% (inconsistent)

### Data Quality Issues
Analysis revealed labeling errors in ground truth:
- Some descriptive text mislabeled as "Date"
- Single digits labeled as "Time"
- These inconsistencies introduce noise into training/validation

**Implication:** The 75.5% performance is achieved **despite** noisy labels. With cleaner annotations, the system would likely achieve 80-85%.

---

## 🔮 Future Work

### Phase 3: Field Classification
- Train supervised classifier for Date/Time/Location
- Use regex patterns as features
- Temporal/semantic validation

### Improvements
- **Location detection:** Boost long text nodes in scoring
- **Clustering:** Adaptive gap parameters per site
- **Data quality:** Re-label ground truth with stricter guidelines

### Production Deployment
- Real-time API for event extraction
- Confidence scores per field
- Human-in-the-loop validation interface

---

## 📁 Project Structure
```
phase2/
├── data/
│   └── raw/              # 16 labeled CSV files
├── src/
│   └── dateclf/
│       ├── train.py      # Main pipeline (~550 lines)
│       ├── data.py       # Data loading
│       ├── features.py   # Feature extraction
│       ├── metrics.py    # Evaluation metrics
│       ├── sampling.py   # Negative undersampling
│       └── split.py      # LOSO cross-validation
├── runs/                 # Results (auto-generated)
├── view_results.py       # Terminal viewer
├── export_full_demo.py   # HTML demo generator
├── comprehensive_demo.html  # Interactive demo
├── config.yaml           # Configuration
├── requirements.txt      # Dependencies
└── README.md            # This file
```

---

## 📝 Citation
```bibtex
@mastersthesis{eventextraction2026,
  authors = {Jad Kebal, Mohamed Amine Ablouh, Darine Fourar Laidi},
  title = {Event Extraction from Heterogeneous Websites using Machine Learning},
  school = {University of Ottawa},
  year = {2026},
  type = {Capstone Project}
}
```

---

## 🙏 Acknowledgments

- **Supervisor:** [Tom Cesari]
- **Clients:** [Philippe Sabourin, Emre Bengu]
- **Dataset:** College fair websites (NACAC, IACAC, WACAC, etc.)
- **Tools:** LightGBM, scikit-learn, pandas

---

## 📧 Contact

**Jad Kebal**
- GitHub: [@JDK68](https://github.com/JDK68)
- University: University of Ottawa
- Program: Computer Science (2026)

---

**Last Updated:** March 2026
