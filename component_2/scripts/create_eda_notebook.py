import json
from pathlib import Path

OUTPUT_PATH = Path("../notebooks/02_eda.ipynb")

def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }

def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }

cells = []

cells.append(md("""# C2 — Exploratory Data Analysis and Usefulness Target Selection

**Component 2:** Helpfulness-Guided Review Evidence Selection with Minority Preservation

This notebook performs the initial EDA for the PP1 development pilot prepared in `01_dataset_inspection.ipynb`.

## Main Goals

1. Check data quality and clean the smartphone pilot.
2. Analyse helpful-vote behaviour and sparsity.
3. Inspect ratings, verified purchases, timestamps and review length.
4. Check whether review age is related to helpful votes.
5. Compare candidate usefulness-target formulations:
   - raw helpful votes,
   - log-transformed helpful votes,
   - product-relative helpfulness,
   - age-adjusted helpfulness.
6. Prepare a chronological development split for the next modelling notebook.
7. Save reproducible EDA tables, figures and a cleaned pilot dataset.

> **Important:** This is preliminary PP1 development analysis. The pilot is not the final research evaluation dataset.
"""))

cells.append(code("""from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import skew, spearmanr

pd.set_option("display.max_columns", 50)
pd.set_option("display.max_colwidth", 120)

print("EDA environment ready")
print("Pandas:", pd.__version__)
print("NumPy:", np.__version__)
"""))

cells.append(md("""## 1. Resolve Project Paths and Load the Pilot Dataset
"""))

cells.append(code("""def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "data").exists() and (candidate / "notebooks").exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate the Component 2 project root."
    )

PROJECT_ROOT = find_project_root(Path.cwd().resolve())

PILOT_PATH = (
    PROJECT_ROOT
    / "data"
    / "pilot"
    / "smartphone_pilot_raw.parquet"
)

CLEANED_PILOT_PATH = (
    PROJECT_ROOT
    / "data"
    / "pilot"
    / "smartphone_pilot_cleaned.parquet"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "eda"
FIGURES_DIR = RESULTS_DIR / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("Project root:", PROJECT_ROOT)
print("Pilot dataset:", PILOT_PATH)
print("Results directory:", RESULTS_DIR)

if not PILOT_PATH.exists():
    raise FileNotFoundError(
        f"Pilot dataset not found at: {PILOT_PATH}"
    )
"""))

cells.append(code("""raw_df = pd.read_parquet(PILOT_PATH)

print("Raw pilot shape:", raw_df.shape)
print("\\nColumns:")
print(raw_df.columns.tolist())

display(raw_df.head())
"""))

cells.append(md("""## 2. Initial Data-Quality Audit
"""))

cells.append(code("""required_columns = [
    "rating",
    "title",
    "text",
    "asin",
    "parent_asin",
    "timestamp",
    "helpful_vote",
    "verified_purchase",
]

missing_columns = [
    col
    for col in required_columns
    if col not in raw_df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns missing: {missing_columns}"
    )

audit = {
    "raw_rows": len(raw_df),
    "missing_text": int(raw_df["text"].isna().sum()),
    "empty_text": int(
        raw_df["text"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    ),
    "duplicate_product_text": int(
        raw_df.duplicated(
            subset=["parent_asin", "text"]
        ).sum()
    ),
    "missing_rating": int(
        raw_df["rating"].isna().sum()
    ),
    "missing_helpful_vote": int(
        raw_df["helpful_vote"].isna().sum()
    ),
    "missing_timestamp": int(
        raw_df["timestamp"].isna().sum()
    ),
    "missing_verified_purchase": int(
        raw_df["verified_purchase"].isna().sum()
    ),
}

audit_df = pd.DataFrame(
    {
        "metric": list(audit.keys()),
        "value": list(audit.values()),
    }
)

display(audit_df)
"""))

cells.append(code("""df = raw_df.copy()

df["text"] = (
    df["text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["title"] = (
    df["title"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["rating"] = pd.to_numeric(
    df["rating"],
    errors="coerce",
)

df["helpful_vote"] = pd.to_numeric(
    df["helpful_vote"],
    errors="coerce",
)

df["timestamp"] = pd.to_numeric(
    df["timestamp"],
    errors="coerce",
)

before = len(df)

df = df[df["text"].ne("")].copy()

df = df[
    df["rating"].between(
        1,
        5,
        inclusive="both",
    )
].copy()

df = df[
    df["helpful_vote"].fillna(-1) >= 0
].copy()

df = df[
    df["timestamp"].notna()
].copy()

df = df.drop_duplicates(
    subset=["parent_asin", "text"],
    keep="first",
).copy()

df["review_date"] = pd.to_datetime(
    df["timestamp"],
    unit="ms",
    errors="coerce",
    utc=True,
)

df = df[
    df["review_date"].notna()
].copy()

print("Rows before cleaning:", before)
print("Rows after cleaning:", len(df))
print("Rows removed:", before - len(df))
"""))

cells.append(md("""## 3. Create Basic Review Features
"""))

cells.append(code("""df["review_length_words"] = (
    df["text"]
    .str.split()
    .str.len()
    .astype(int)
)

df["review_length_chars"] = (
    df["text"]
    .str.len()
    .astype(int)
)

df["verified_purchase"] = (
    df["verified_purchase"]
    .fillna(False)
    .astype(bool)
)

print("Cleaned shape:", df.shape)

display(
    df[
        [
            "parent_asin",
            "rating",
            "helpful_vote",
            "verified_purchase",
            "review_date",
            "review_length_words",
            "text",
        ]
    ].head()
)
"""))

cells.append(md("""## 4. Helpful-Vote Distribution
"""))

cells.append(code("""helpful_summary = (
    df["helpful_vote"]
    .describe(
        percentiles=[
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )
)

zero_helpful_count = int(
    (df["helpful_vote"] == 0).sum()
)

zero_helpful_pct = float(
    (df["helpful_vote"] == 0).mean()
    * 100
)

positive_helpful_count = int(
    (df["helpful_vote"] > 0).sum()
)

raw_skew = float(
    skew(
        df["helpful_vote"],
        bias=False,
    )
)

display(
    helpful_summary.to_frame(
        "helpful_vote"
    )
)

print(
    "Zero helpful-vote reviews:",
    zero_helpful_count,
)

print(
    f"Zero helpful-vote percentage: "
    f"{zero_helpful_pct:.2f}%"
)

print(
    "Positive helpful-vote reviews:",
    positive_helpful_count,
)

print(
    f"Raw helpful-vote skewness: "
    f"{raw_skew:.3f}"
)
"""))

cells.append(code("""fig, ax = plt.subplots(figsize=(9, 5))

ax.hist(
    df["helpful_vote"],
    bins=40,
)

ax.set_title(
    "Raw Helpful-Vote Distribution"
)

ax.set_xlabel("Helpful votes")
ax.set_ylabel("Review count")

plt.tight_layout()

raw_hist_path = (
    FIGURES_DIR
    / "helpful_votes_raw_histogram.png"
)

plt.savefig(
    raw_hist_path,
    dpi=160,
    bbox_inches="tight",
)

plt.show()

print("Saved:", raw_hist_path)
"""))

cells.append(code("""df["log_helpful"] = np.log1p(
    df["helpful_vote"]
)

fig, ax = plt.subplots(figsize=(9, 5))

ax.hist(
    df["log_helpful"],
    bins=40,
)

ax.set_title(
    "Log-Transformed Helpful-Vote Distribution"
)

ax.set_xlabel(
    "log(1 + helpful votes)"
)

ax.set_ylabel("Review count")

plt.tight_layout()

log_hist_path = (
    FIGURES_DIR
    / "helpful_votes_log_histogram.png"
)

plt.savefig(
    log_hist_path,
    dpi=160,
    bbox_inches="tight",
)

plt.show()

log_skew = float(
    skew(
        df["log_helpful"],
        bias=False,
    )
)

print(
    f"Log-helpful skewness: "
    f"{log_skew:.3f}"
)

print("Saved:", log_hist_path)
"""))

cells.append(code("""display(
    df[
        [
            "parent_asin",
            "helpful_vote",
            "rating",
            "review_date",
            "review_length_words",
            "text",
        ]
    ]
    .sort_values(
        "helpful_vote",
        ascending=False,
    )
    .head(15)
)
"""))

cells.append(md("""## 5. Rating and Verified-Purchase Distributions
"""))

cells.append(code("""rating_distribution = (
    df["rating"]
    .value_counts()
    .sort_index()
    .rename("review_count")
    .to_frame()
)

verified_distribution = (
    df["verified_purchase"]
    .value_counts(
        dropna=False
    )
    .rename("review_count")
    .to_frame()
)

print("Rating distribution:")
display(rating_distribution)

print(
    "Verified-purchase distribution:"
)
display(verified_distribution)

verified_pct = float(
    df["verified_purchase"]
    .mean()
    * 100
)

print(
    f"Verified-purchase percentage: "
    f"{verified_pct:.2f}%"
)
"""))

cells.append(code("""fig, ax = plt.subplots(figsize=(7, 4))

rating_counts = (
    df["rating"]
    .value_counts()
    .sort_index()
)

ax.bar(
    rating_counts.index.astype(str),
    rating_counts.values,
)

ax.set_title("Rating Distribution")
ax.set_xlabel("Rating")
ax.set_ylabel("Review count")

plt.tight_layout()

rating_plot_path = (
    FIGURES_DIR
    / "rating_distribution.png"
)

plt.savefig(
    rating_plot_path,
    dpi=160,
    bbox_inches="tight",
)

plt.show()

print("Saved:", rating_plot_path)
"""))

cells.append(md("""## 6. Timestamp Range and Approximate Review Age
"""))

cells.append(code("""earliest_review = (
    df["review_date"].min()
)

latest_review = (
    df["review_date"].max()
)

print(
    "Earliest review:",
    earliest_review,
)

print(
    "Latest review:",
    latest_review,
)

print(
    "Time span:",
    latest_review - earliest_review,
)

reference_date = (
    latest_review
    + pd.Timedelta(days=1)
)

df["age_days"] = (
    (
        reference_date
        - df["review_date"]
    )
    .dt.total_seconds()
    .div(86400)
    .clip(lower=1)
)

display(
    df["age_days"]
    .describe()
    .to_frame("age_days")
)
"""))

cells.append(code("""age_helpful_corr = spearmanr(
    df["age_days"],
    df["helpful_vote"],
    nan_policy="omit",
)

print(
    "Spearman correlation between "
    "review age and helpful votes:"
)

print(
    f"rho = "
    f"{age_helpful_corr.statistic:.4f}"
)

print(
    f"p-value = "
    f"{age_helpful_corr.pvalue:.6g}"
)
"""))

cells.append(md("""## 7. Review-Length Analysis
"""))

cells.append(code("""length_summary = (
    df[
        [
            "review_length_words",
            "review_length_chars",
        ]
    ]
    .describe(
        percentiles=[
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )
)

display(length_summary)

length_helpful_corr = spearmanr(
    df["review_length_words"],
    df["helpful_vote"],
    nan_policy="omit",
)

print(
    "Spearman correlation between "
    "review length and helpful votes:"
)

print(
    f"rho = "
    f"{length_helpful_corr.statistic:.4f}"
)

print(
    f"p-value = "
    f"{length_helpful_corr.pvalue:.6g}"
)
"""))

cells.append(code("""fig, ax = plt.subplots(figsize=(9, 5))

ax.hist(
    df["review_length_words"],
    bins=50,
)

ax.set_title(
    "Review Length Distribution"
)

ax.set_xlabel("Words per review")
ax.set_ylabel("Review count")

plt.tight_layout()

length_plot_path = (
    FIGURES_DIR
    / "review_length_distribution.png"
)

plt.savefig(
    length_plot_path,
    dpi=160,
    bbox_inches="tight",
)

plt.show()

print("Saved:", length_plot_path)
"""))

cells.append(md("""## 8. Product-Level EDA
"""))

cells.append(code("""product_summary = (
    df.groupby("parent_asin")
    .agg(
        review_count=(
            "text",
            "size",
        ),
        avg_rating=(
            "rating",
            "mean",
        ),
        mean_helpful_vote=(
            "helpful_vote",
            "mean",
        ),
        median_helpful_vote=(
            "helpful_vote",
            "median",
        ),
        max_helpful_vote=(
            "helpful_vote",
            "max",
        ),
        zero_helpful_pct=(
            "helpful_vote",
            lambda x:
                (x == 0)
                .mean()
                * 100,
        ),
        verified_purchase_pct=(
            "verified_purchase",
            lambda x:
                x.mean()
                * 100,
        ),
        avg_review_length_words=(
            "review_length_words",
            "mean",
        ),
        earliest_review=(
            "review_date",
            "min",
        ),
        latest_review=(
            "review_date",
            "max",
        ),
    )
    .reset_index()
    .sort_values(
        "review_count",
        ascending=False,
    )
)

display(product_summary)

product_summary_path = (
    RESULTS_DIR
    / "product_summary.csv"
)

product_summary.to_csv(
    product_summary_path,
    index=False,
)

print(
    "Saved:",
    product_summary_path,
)
"""))

cells.append(md("""## 9. Candidate Usefulness Targets

Four candidate target formulations are compared:

- `raw_helpful`
- `log_helpful`
- `product_relative`
- `age_adjusted`

These are preliminary PP1 candidates, not automatically final research targets.
"""))

cells.append(code("""df["raw_helpful"] = (
    df["helpful_vote"]
    .astype(float)
)

df["log_helpful"] = (
    np.log1p(
        df["helpful_vote"]
    )
)

df["product_relative"] = (
    df.groupby(
        "parent_asin"
    )["log_helpful"]
    .rank(
        method="average",
        pct=True,
    )
)

effective_age_days = (
    df["age_days"]
    .clip(lower=30)
)

df["age_adjusted"] = (
    df["helpful_vote"]
    / effective_age_days
    * 365.25
)

target_columns = [
    "raw_helpful",
    "log_helpful",
    "product_relative",
    "age_adjusted",
]

display(
    df[target_columns].head()
)
"""))

cells.append(code("""def safe_skew(series):
    clean = series.dropna()

    if (
        len(clean) < 3
        or clean.nunique() <= 1
    ):
        return np.nan

    return float(
        skew(
            clean,
            bias=False,
        )
    )


def target_diagnostics(
    data,
    target,
):
    s = data[target].dropna()

    return {
        "target": target,
        "count": int(len(s)),
        "unique_values": int(
            s.nunique()
        ),
        "zero_pct": float(
            (s == 0)
            .mean()
            * 100
        ),
        "mean": float(
            s.mean()
        ),
        "median": float(
            s.median()
        ),
        "std": float(
            s.std()
        ),
        "min": float(
            s.min()
        ),
        "max": float(
            s.max()
        ),
        "skewness": safe_skew(s),
    }


target_diagnostics_df = (
    pd.DataFrame(
        [
            target_diagnostics(
                df,
                target,
            )
            for target
            in target_columns
        ]
    )
)

display(
    target_diagnostics_df
)

target_diagnostics_path = (
    RESULTS_DIR
    / "target_diagnostics.csv"
)

target_diagnostics_df.to_csv(
    target_diagnostics_path,
    index=False,
)

print(
    "Saved:",
    target_diagnostics_path,
)
"""))

cells.append(code("""correlation_rows = []

for target in target_columns:

    corr = spearmanr(
        df["raw_helpful"],
        df[target],
        nan_policy="omit",
    )

    correlation_rows.append(
        {
            "target": target,
            "spearman_vs_raw_helpful":
                corr.statistic,
            "p_value":
                corr.pvalue,
        }
    )

target_rank_correlation_df = (
    pd.DataFrame(
        correlation_rows
    )
)

display(
    target_rank_correlation_df
)

rank_corr_path = (
    RESULTS_DIR
    / "target_rank_correlations.csv"
)

target_rank_correlation_df.to_csv(
    rank_corr_path,
    index=False,
)

print(
    "Saved:",
    rank_corr_path,
)
"""))

cells.append(code("""for target in target_columns:

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.hist(
        df[target].dropna(),
        bins=40,
    )

    ax.set_title(
        f"Candidate Target Distribution: "
        f"{target}"
    )

    ax.set_xlabel(target)
    ax.set_ylabel("Review count")

    plt.tight_layout()

    path = (
        FIGURES_DIR
        / f"target_{target}_histogram.png"
    )

    plt.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.show()

    print("Saved:", path)
"""))

cells.append(md("""## 10. Target-Selection Evidence Summary
"""))

cells.append(code("""raw_diag = (
    target_diagnostics_df
    .set_index("target")
    .loc["raw_helpful"]
)

log_diag = (
    target_diagnostics_df
    .set_index("target")
    .loc["log_helpful"]
)

print(
    "=== TARGET-SELECTION EVIDENCE ==="
)

print(
    f"Zero helpful-vote percentage: "
    f"{zero_helpful_pct:.2f}%"
)

print(
    f"Raw helpful skewness: "
    f"{raw_diag['skewness']:.3f}"
)

print(
    f"Log helpful skewness: "
    f"{log_diag['skewness']:.3f}"
)

print(
    "Age vs helpful-vote "
    "Spearman rho:",
    f"{age_helpful_corr.statistic:.4f}",
)

print(
    "Number of products:",
    df["parent_asin"].nunique(),
)

print("\\nInterpretation prompts:")

print(
    "- If raw helpful votes are extremely "
    "skewed, log_helpful is a stronger "
    "regression candidate."
)

print(
    "- If vote scales differ strongly "
    "between products, product_relative "
    "is useful as a ranking-oriented "
    "comparison target."
)

print(
    "- If review age is meaningfully "
    "correlated with helpful votes, "
    "age bias should be considered."
)

print(
    "- A zero-heavy target may require "
    "ranking/classification-oriented "
    "baselines in addition to regression."
)
"""))

cells.append(md("""### Research Decision Record — Fill After Running

After inspecting the actual outputs, replace the placeholders below.

> **Selected preliminary usefulness target:** `<target name>`  
> **Reason:** `<evidence from EDA>`  
> **Alternative retained for sensitivity analysis:** `<target name>`  
> **Limitation:** `<important limitation>`

Do not commit this notebook with the placeholders unchanged.
"""))

cells.append(md("""## 11. Chronological Development Split

For this PP1 pilot:

- 70% earliest reviews → train
- next 15% → validation
- final 15% → test

The split is applied separately within each product.
"""))

cells.append(code("""def chronological_split_group(
    group,
):
    group = (
        group
        .sort_values(
            "review_date"
        )
        .copy()
    )

    n = len(group)

    train_end = max(
        1,
        int(
            math.floor(
                n * 0.70
            )
        ),
    )

    val_end = max(
        train_end + 1,
        int(
            math.floor(
                n * 0.85
            )
        ),
    )

    val_end = min(
        val_end,
        n,
    )

    split = np.array(
        ["test"] * n,
        dtype=object,
    )

    split[:train_end] = "train"

    split[
        train_end:val_end
    ] = "validation"

    if n == 1:
        split[:] = "train"

    elif n == 2:
        split[0] = "train"
        split[1] = "test"

    group["split"] = split

    return group


split_parts = []

for _, group in df.groupby(
    "parent_asin",
    sort=False,
):
    split_parts.append(
        chronological_split_group(
            group
        )
    )

split_df = pd.concat(
    split_parts,
    ignore_index=True,
)

print(
    "Overall split counts:"
)

display(
    split_df["split"]
    .value_counts()
    .to_frame(
        "review_count"
    )
)

print(
    "Per-product split counts:"
)

display(
    pd.crosstab(
        split_df["parent_asin"],
        split_df["split"],
    )
)
"""))

cells.append(md("""## 12. Save Cleaned Pilot and EDA Evidence
"""))

cells.append(code("""columns_to_save = [
    "rating",
    "title",
    "text",
    "asin",
    "parent_asin",
    "timestamp",
    "helpful_vote",
    "verified_purchase",
    "review_date",
    "review_length_words",
    "review_length_chars",
    "age_days",
    "raw_helpful",
    "log_helpful",
    "product_relative",
    "age_adjusted",
]

cleaned_for_save = (
    df[columns_to_save]
    .copy()
)

cleaned_for_save.to_parquet(
    CLEANED_PILOT_PATH,
    index=False,
)

split_output_path = (
    RESULTS_DIR
    / "chronological_split_assignments.csv"
)

split_df[
    [
        "asin",
        "parent_asin",
        "review_date",
        "split",
    ]
].to_csv(
    split_output_path,
    index=False,
)

print(
    "Saved cleaned pilot:",
    CLEANED_PILOT_PATH,
)

print(
    "Saved split assignments:",
    split_output_path,
)

print(
    "Cleaned rows:",
    len(cleaned_for_save),
)
"""))

cells.append(code("""eda_summary = {
    "raw_rows":
        int(len(raw_df)),

    "cleaned_rows":
        int(len(df)),

    "rows_removed":
        int(
            len(raw_df)
            - len(df)
        ),

    "products":
        int(
            df["parent_asin"]
            .nunique()
        ),

    "zero_helpful_count":
        int(
            zero_helpful_count
        ),

    "zero_helpful_pct":
        float(
            zero_helpful_pct
        ),

    "raw_helpful_skewness":
        float(
            raw_skew
        ),

    "log_helpful_skewness":
        float(
            log_skew
        ),

    "earliest_review":
        str(
            earliest_review
        ),

    "latest_review":
        str(
            latest_review
        ),

    "verified_purchase_pct":
        float(
            verified_pct
        ),

    "age_helpful_spearman_rho":
        float(
            age_helpful_corr.statistic
        ),

    "age_helpful_spearman_p":
        float(
            age_helpful_corr.pvalue
        ),

    "length_helpful_spearman_rho":
        float(
            length_helpful_corr.statistic
        ),

    "length_helpful_spearman_p":
        float(
            length_helpful_corr.pvalue
        ),
}

eda_summary_path = (
    RESULTS_DIR
    / "eda_summary.json"
)

with open(
    eda_summary_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        eda_summary,
        f,
        indent=2,
    )

print(
    "Saved:",
    eda_summary_path,
)

print(
    json.dumps(
        eda_summary,
        indent=2,
    )
)
"""))

cells.append(md("""## 13. PP1 Evidence Checklist

After all cells run successfully:

- [ ] Review the data-quality audit.
- [ ] Record the zero-helpful-vote percentage.
- [ ] Compare raw and log skewness.
- [ ] Record the age–helpful correlation.
- [ ] Review product-level differences.
- [ ] Fill the **Research Decision Record** with the actual preliminary target choice.
- [ ] Update the research diary with actual findings.
- [ ] Update the Planner task with notebook/result evidence.
- [ ] Check `git status`.
- [ ] Commit the notebook and appropriate result artifacts.
- [ ] Push the milestone to the Component 2 branch.

### Suggested commit message

`Perform initial EDA and evaluate usefulness target options`
"""))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name":
                ".venv (3.13.3)",
            "language":
                "python",
            "name":
                "python3",
        },
        "language_info": {
            "name":
                "python",
            "version":
                "3.13.3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PATH.write_text(
    json.dumps(
        notebook,
        ensure_ascii=False,
        indent=1,
    ),
    encoding="utf-8",
)

print(
    f"Created: {OUTPUT_PATH}"
)