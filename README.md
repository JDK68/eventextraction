\# Phase 2 — Node-Level Field Detection



This repository contains Phase 2 (implementation) of the project.



\## Scope (fixed decisions)

\- Node-level field detection only

\- One binary classifier per field (Date, Time, Location, Name)

\- LightGBM as baseline

\- Leave-One-Site-Out cross-validation

\- Class imbalance handled via class weighting + undersampling

\- No event reconstruction, no ranking loss, no NER, no deep learning

\- Evaluation via Precision@K and Recall@K



\## Goal

Implement a clean, reproducible, end-to-end pipeline starting with the \*\*Date\*\* classifier.

