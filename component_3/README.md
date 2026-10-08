# Temporal Product Quality & Emerging Issue Detection

## Overview
This repository contains the implementation of Component 3 of the Smart Phone & Laptop Comparison Assistant.

The main purpose of this component is to analyse how customer opinions about specific product aspects change over time and to identify newly emerging complaint topics from timestamped online reviews.

The component focuses on smartphones and laptops using review data from Amazon Reviews 2023.

---

## Research Component
**Component 3: Temporal Product Quality & Emerging Issue Detection**

This component receives aspect-level review information from the earlier aspect and sentiment extraction stage and converts it into temporal evidence that can be used by the final personalized ranking component.

---

## Main Input

The minimum required input is:

- Review ID
- Product ID
- Review Timestamp
- Review Text
- Product Aspect
- Sentiment Label
- Sentiment Score
- Sentiment Confidence

Optional input:

- Evidence-quality / usefulness score from Component 2

---

## Main Processing Pipeline

1. Data cleaning and timestamp normalization
2. Product × Aspect × Quarter aggregation
3. Bayesian smoothing for sparse review periods
4. Uncertainty / confidence estimation
5. EWMA-based temporal trend analysis
6. PELT change-point detection
7. Negative aspect review filtering
8. Sentence-BERT semantic representation
9. BERTopic complaint-topic discovery
10. Temporal topic-growth analysis
11. Emerging-issue classification
12. Risk and confidence generation

---

## PP1 Key Functions

### Function 1 — Temporal Product Quality & Trend Engine

**Input:**  
Product ID, Timestamp, Aspect, Sentiment Score, Confidence

**Processing:**  
Quarterly Aggregation → Bayesian Smoothing → EWMA → PELT

**Output:**  
- Recent review-perceived quality
- Improving / Stable / Declining trend
- Significant change point
- Confidence / uncertainty

---

### Function 2 — Emerging Complaint Detection Engine

**Input:**  
Negative aspect reviews, Product ID, Aspect, Quarter

**Processing:**  
Sentence-BERT → BERTopic → Topic Prevalence → Temporal Growth Analysis

**Output:**  
- Specific complaint topic
- Emerging-issue status
- Issue risk
- Confidence

Emerging-issue status:

- `0` — No meaningful issue
- `1` — Existing issue but not newly increasing
- `2` — New or rapidly increasing issue

---

## Final Output

For each product aspect, the component will generate:

- Recent Review-Perceived Quality
- Temporal Trend
- Significant Change Point
- Emerging-Issue Status
- Specific Emerging Complaint
- Issue Risk
- Confidence / Uncertainty

These outputs will be passed to Component 4 for personalized and explainable product ranking.

---

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Ruptures
- Sentence-Transformers
- BERTopic
- UMAP
- HDBSCAN
- Jupyter Notebook

---

## Baseline Approaches

The proposed framework will be compared against:

- Lifetime Average
- Recent Fixed-Window Average
- Simple Moving Average
- EWMA-only Approach

---

## Evaluation Plan

The implementation will be evaluated using:

- Accuracy
- Macro-F1
- Precision
- Recall
- F1 Score
- Detection Delay
- Coverage / Calibration
- Topic Coherence
- Topic Stability
- Human Validation
- Ablation Study

Chronological / rolling-origin validation will be used to avoid temporal data leakage.

---

## Component Dependencies

### Required
**Component 1 → Component 3**
- Product ID
- Timestamp
- Aspect
- Sentiment
- Confidence

### Optional
**Component 2 → Component 3**
- Evidence-quality / usefulness information

### Output
**Component 3 → Component 4**
- Recent quality
- Trend
- Change point
- Emerging issue
- Risk
- Confidence

---

