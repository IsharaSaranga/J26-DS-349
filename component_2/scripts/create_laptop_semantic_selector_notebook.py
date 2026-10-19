
from pathlib import Path
import json
import textwrap


ROOT = Path(__file__).resolve().parent.parent

NOTEBOOK_PATH = (
    ROOT / "notebooks"
    / "12_laptop_semantic_evidence_selection.ipynb"
)

NOTEBOOK_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

cells = []


def add_md(content):
    source = textwrap.dedent(content).strip() + "\n"

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    })


def add_code(content):
    source = textwrap.dedent(content).strip() + "\n"

    compile(source, "<notebook-cell>", "exec")

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    })


add_md("""
# Milestone 12 — Laptop Semantic Representation and Evidence Selection

**Component 2:** Helpfulness-Guided Review Evidence Selection
with Minority Preservation

## Objective

Prepare validated laptop review representations and integration
inputs for the existing smartphone minority-preserving selector.

## Stage A — Integration Preparation

1. Load the frozen laptop dataset.
2. Validate Function 1 predictions.
3. Generate Sentence-BERT embeddings.
4. Verify review-to-embedding alignment.
5. Generate provisional C1 fixtures.
6. Identify fixture-based supported minority groups.
7. Save integration artifacts and run validation.

## Important Research Boundary

Real C1 laptop aspect–sentiment outputs are not yet integrated.

Any generated aspect–sentiment labels are explicitly synthetic
integration fixtures. They must not be reported as actual
aspect-based sentiment analysis findings.

Function 1 scores for training reviews are in-sample.
They must not be treated as independent held-out predictions.

Stage B will reuse the validated smartphone selector and
fidelity-handling logic rather than introducing a separate
laptop-specific selection algorithm.
""")


add_md("## 1. Imports, Configuration and Project Paths")

add_code("""
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer


def find_project_root(start):
    for candidate in [start, *start.parents]:
        if (
            (candidate / "data" / "pilot").exists()
            and (candidate / "notebooks").exists()
        ):
            return candidate

    raise FileNotFoundError(
        "Component 2 project root not found."
    )


ROOT = find_project_root(
    Path.cwd().resolve()
)

DATA_PATH = (
    ROOT / "data" / "pilot"
    / "laptop_pilot_cleaned.parquet"
)

PREDICTIONS_PATH = (
    ROOT / "results" / "usefulness" / "laptop"
    / "laptop_all_review_predictions.csv"
)

SEMANTIC_DIR = (
    ROOT / "results" / "semantic" / "laptop"
)

SELECTOR_DIR = (
    ROOT / "results" / "selector" / "laptop"
)

SEMANTIC_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SELECTOR_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

EXPECTED_REVIEWS = 997
EXPECTED_PRODUCTS = 10
EXPECTED_EMBEDDING_DIM = 384

MIN_SUPPORT = 3
MIN_CONFIDENCE = 0.60
MAX_MINORITY_SHARE = 0.40

K = 5
MAX_PROTECTED_REVIEWS = 2
FIDELITY_THRESHOLD = 0.35

print("Project root:", ROOT)
print("Semantic results:", SEMANTIC_DIR)
print("Selector results:", SELECTOR_DIR)
""")


add_md("## 2. Load and Validate Laptop Reviews and Usefulness Scores")

add_code("""
assert DATA_PATH.exists(), (
    "Milestone 10 dataset is missing."
)

assert PREDICTIONS_PATH.exists(), (
    "Milestone 11 prediction file is missing."
)

reviews = pd.read_parquet(DATA_PATH)

predictions = pd.read_csv(
    PREDICTIONS_PATH,
    dtype={
        "review_id": str,
        "parent_asin": str,
    },
)

reviews["review_id"] = (
    reviews["review_id"].astype(str)
)

reviews["parent_asin"] = (
    reviews["parent_asin"].astype(str)
)

required_prediction_columns = {
    "review_id",
    "parent_asin",
    "split",
    "predicted_log_helpful",
    "predicted_usefulness",
    "prediction_provenance",
}

assert required_prediction_columns.issubset(
    predictions.columns
)

assert reviews["review_id"].is_unique
assert predictions["review_id"].is_unique

assert set(reviews["review_id"]) == set(
    predictions["review_id"]
)

df = reviews.merge(
    predictions[
        [
            "review_id",
            "parent_asin",
            "split",
            "predicted_log_helpful",
            "predicted_usefulness",
            "prediction_provenance",
        ]
    ],
    on=["review_id", "parent_asin"],
    how="left",
    validate="one_to_one",
)

assert len(df) == EXPECTED_REVIEWS

assert (
    df["parent_asin"].nunique()
    ==
    EXPECTED_PRODUCTS
)

assert not df[
    "predicted_usefulness"
].isna().any()

assert np.isfinite(
    df["predicted_usefulness"].to_numpy()
).all()

assert (
    df["predicted_usefulness"]
    >= 0
).all()

assert set(df["split"]) == {
    "train",
    "validation",
    "test",
}

assert set(
    df["prediction_provenance"]
) == {
    "in_sample_training",
    "validation_used_for_model_selection",
    "held_out_test",
}

df["text"] = (
    df["text"]
    .fillna("")
    .astype(str)
)

df = (
    df.sort_values(
        ["parent_asin", "review_id"]
    )
    .reset_index(drop=True)
)

df["embedding_row"] = np.arange(
    len(df),
    dtype=int,
)

print("Reviews:", len(df))
print("Products:", df["parent_asin"].nunique())

display(
    df["prediction_provenance"]
    .value_counts()
    .to_frame("review_count")
)

print(
    "Laptop usefulness prediction join: PASS"
)
""")


add_md("## 3. Generate Sentence-BERT Embeddings")

add_code("""
EMBEDDINGS_PATH = (
    SEMANTIC_DIR / "laptop_review_embeddings.npz"
)

EMBEDDING_INDEX_PATH = (
    SEMANTIC_DIR / "laptop_embedding_index.csv"
)

model = SentenceTransformer(MODEL_NAME)

embeddings = model.encode(
    df["text"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

embeddings = np.asarray(
    embeddings,
    dtype=np.float32,
)

assert embeddings.shape == (
    EXPECTED_REVIEWS,
    EXPECTED_EMBEDDING_DIM,
)

assert np.isfinite(embeddings).all()

norms = np.linalg.norm(
    embeddings,
    axis=1,
)

assert np.allclose(
    norms,
    1.0,
    atol=1e-3,
)

np.savez_compressed(
    EMBEDDINGS_PATH,
    embeddings=embeddings,
    review_ids=df["review_id"].to_numpy(
        dtype=str
    ),
)

embedding_index = df[
    [
        "embedding_row",
        "review_id",
        "parent_asin",
    ]
].copy()

embedding_index.to_csv(
    EMBEDDING_INDEX_PATH,
    index=False,
)

print("Embedding matrix:", embeddings.shape)
print("Saved:", EMBEDDINGS_PATH)
print("Saved:", EMBEDDING_INDEX_PATH)
""")


add_md("## 4. Verify Embedding Alignment and Representativeness")

add_code("""
saved = np.load(
    EMBEDDINGS_PATH,
    allow_pickle=False,
)

saved_vectors = saved["embeddings"]
saved_ids = saved["review_ids"]

assert np.array_equal(
    saved_ids,
    df["review_id"].to_numpy(dtype=str),
)

np.testing.assert_allclose(
    saved_vectors,
    embeddings,
    rtol=1e-5,
    atol=1e-6,
)

representativeness_rows = []

for product, group in df.groupby("parent_asin"):
    indices = group["embedding_row"].to_numpy()

    product_embeddings = embeddings[indices]

    similarity = (
        product_embeddings
        @ product_embeddings.T
    )

    np.fill_diagonal(similarity, 0.0)

    n = len(group)

    if n > 1:
        average_similarity = (
            similarity.sum(axis=1) / (n - 1)
        )
    else:
        average_similarity = np.zeros(n)

    for review_id, score in zip(
        group["review_id"],
        average_similarity,
    ):
        representativeness_rows.append({
            "review_id": review_id,
            "parent_asin": product,
            "semantic_representativeness": float(
                score
            ),
        })

representativeness_df = pd.DataFrame(
    representativeness_rows
)

assert (
    len(representativeness_df)
    ==
    EXPECTED_REVIEWS
)

assert representativeness_df[
    "review_id"
].is_unique

REPRESENTATIVENESS_PATH = (
    SEMANTIC_DIR
    / "laptop_semantic_representativeness.csv"
)

representativeness_df.to_csv(
    REPRESENTATIVENESS_PATH,
    index=False,
)

print("Embedding alignment: PASS")
print("Representativeness rows:", len(
    representativeness_df
))

display(representativeness_df.head())
""")


add_md("""
## 5. Prepare Provisional C1 Integration Fixtures

These fixtures are deterministic software-testing inputs.

Aspect presence uses a small keyword dictionary.
Sentiment is assigned from the overall review rating.

This is **not** a trained ABSA model and must not
be described as real C1 output.

The fixture confidence value is artificial and does
not represent model-calibrated confidence.
""")

add_code("""
ASPECT_KEYWORDS = {
    "battery": [
        "battery", "charge", "charging",
        "battery life",
    ],
    "performance": [
        "speed", "fast", "slow",
        "processor", "cpu", "performance",
        "lag", "ram",
    ],
    "display": [
        "screen", "display", "resolution",
        "brightness", "touchscreen",
    ],
    "keyboard": [
        "keyboard", "typing", "keypad",
        "trackpad", "touchpad",
    ],
    "portability": [
        "lightweight", "portable", "weight",
        "heavy", "travel",
    ],
    "storage": [
        "storage", "ssd", "hard drive",
        "disk space",
    ],
    "price": [
        "price", "cost", "value",
        "expensive", "affordable",
    ],
    "build_quality": [
        "build quality", "durable",
        "hinge", "plastic", "sturdy",
    ],
}


def fixture_aspects(text):
    text = str(text).lower()

    found = []

    for aspect, terms in ASPECT_KEYWORDS.items():
        if any(
            re.search(
                r"(?<!\\w)" + re.escape(term) + r"(?!\\w)",
                text,
            )
            for term in terms
        ):
            found.append(aspect)

    return found or ["general"]


def fixture_sentiment(rating):
    rating = float(rating)

    if rating >= 4:
        return "positive"

    if rating <= 2:
        return "negative"

    return "neutral"


fixture_rows = []

for row in df.itertuples(index=False):
    aspects = fixture_aspects(row.text)
    sentiment = fixture_sentiment(row.rating)

    for aspect in aspects:
        fixture_rows.append({
            "review_id": row.review_id,
            "parent_asin": row.parent_asin,
            "aspect": aspect,
            "sentiment": sentiment,
            "confidence": 0.70,
            "source": "synthetic_c1_fixture",
            "label_method": (
                "keyword_aspect_rating_sentiment"
            ),
        })

c1_fixture = pd.DataFrame(fixture_rows)

assert not c1_fixture.empty

assert set(c1_fixture["review_id"]) == set(
    df["review_id"]
)

assert (
    c1_fixture[
        ["review_id", "aspect", "sentiment"]
    ]
    .duplicated()
    .sum()
    ==
    0
)

FIXTURE_PATH = (
    SELECTOR_DIR
    / "laptop_synthetic_c1_fixture.csv"
)

c1_fixture.to_csv(
    FIXTURE_PATH,
    index=False,
)

print("Synthetic C1 fixture rows:", len(
    c1_fixture
))

print("Synthetic C1 fixture source only.")

display(c1_fixture.head())
""")


add_md("""
## 6. Identify Fixture-Based Supported Minority Groups

A group is provisionally qualified only when:

- support >= 3,
- confidence >= 0.60,
- within-aspect sentiment share <= 0.40.

These are the established preliminary C2 thresholds.

The groups in this notebook are **fixture-derived**, not
validated real customer minority opinions.
""")

add_code("""
group_table = (
    c1_fixture
    .groupby(
        [
            "parent_asin",
            "aspect",
            "sentiment",
        ],
        as_index=False,
    )
    .agg(
        support=(
            "review_id",
            "nunique",
        ),
        mean_confidence=(
            "confidence",
            "mean",
        ),
    )
)

aspect_totals = (
    c1_fixture
    .groupby(
        ["parent_asin", "aspect"],
        as_index=False,
    )
    .agg(
        aspect_review_count=(
            "review_id",
            "nunique",
        ),
    )
)

group_table = group_table.merge(
    aspect_totals,
    on=["parent_asin", "aspect"],
    how="left",
    validate="many_to_one",
)

group_table["within_aspect_share"] = (
    group_table["support"]
    /
    group_table["aspect_review_count"]
)

group_table["qualified"] = (
    (group_table["support"] >= MIN_SUPPORT)
    &
    (
        group_table["mean_confidence"]
        >= MIN_CONFIDENCE
    )
    &
    (
        group_table["within_aspect_share"]
        <= MAX_MINORITY_SHARE
    )
)

group_table["evidence_source"] = (
    "synthetic_c1_fixture"
)

GROUP_TABLE_PATH = (
    SELECTOR_DIR
    / "laptop_fixture_minority_groups.csv"
)

group_table.to_csv(
    GROUP_TABLE_PATH,
    index=False,
)

print("Fixture-based aspect-sentiment groups:", len(
    group_table
))

print("Qualified fixture groups:", int(
    group_table["qualified"].sum()
))

display(
    group_table.sort_values(
        ["qualified", "support"],
        ascending=[False, False],
    ).head(20)
)
""")


add_md("## 7. Save Validated Selector Integration Inputs")

add_code("""
integration = df[
    [
        "review_id",
        "parent_asin",
        "text",
        "rating",
        "helpful_vote",
        "split",
        "prediction_provenance",
        "predicted_usefulness",
        "embedding_row",
    ]
].merge(
    representativeness_df[
        [
            "review_id",
            "semantic_representativeness",
        ]
    ],
    on="review_id",
    how="left",
    validate="one_to_one",
)

assert len(integration) == EXPECTED_REVIEWS

assert integration["review_id"].is_unique

assert not integration[
    "semantic_representativeness"
].isna().any()

integration["semantic_model"] = MODEL_NAME

integration["c1_source"] = (
    "synthetic_c1_fixture"
)

INTEGRATION_PATH = (
    SELECTOR_DIR
    / "laptop_selector_integration_inputs.csv"
)

integration.to_csv(
    INTEGRATION_PATH,
    index=False,
)

manifest = {
    "milestone": 12,
    "stage": "A_semantic_integration",
    "category": "laptop",
    "review_count": len(df),
    "product_count": df["parent_asin"].nunique(),
    "semantic_model": MODEL_NAME,
    "embedding_dimension": embeddings.shape[1],
    "normalized_embeddings": True,
    "c1_data_source": "synthetic_c1_fixture",
    "real_c1_integrated": False,
    "minority_thresholds": {
        "min_support": MIN_SUPPORT,
        "min_confidence": MIN_CONFIDENCE,
        "max_within_aspect_share": MAX_MINORITY_SHARE,
    },
    "selector_parameters_for_stage_b": {
        "budget": K,
        "maximum_protected_reviews": MAX_PROTECTED_REVIEWS,
        "fidelity_threshold": FIDELITY_THRESHOLD,
    },
    "function_1_prediction_limitation": (
        "Training predictions are in-sample; "
        "validation predictions were used for model "
        "selection; only test predictions are held out."
    ),
    "status": (
        "prepared_for_smartphone_selector_adaptation"
    ),
}

MANIFEST_PATH = (
    SELECTOR_DIR
    / "laptop_semantic_integration_manifest.json"
)

MANIFEST_PATH.write_text(
    json.dumps(
        manifest,
        indent=2,
    ),
    encoding="utf-8",
)

print("Integration inputs:", INTEGRATION_PATH)
print("Manifest:", MANIFEST_PATH)
""")


add_md("## 8. Final Stage A Validation")

add_code("""
assert len(df) == EXPECTED_REVIEWS

assert df["parent_asin"].nunique() == (
    EXPECTED_PRODUCTS
)

assert embeddings.shape == (
    EXPECTED_REVIEWS,
    EXPECTED_EMBEDDING_DIM,
)

assert len(representativeness_df) == (
    EXPECTED_REVIEWS
)

assert len(integration) == EXPECTED_REVIEWS

assert set(integration["review_id"]) == set(
    df["review_id"]
)

assert set(c1_fixture["review_id"]) == set(
    df["review_id"]
)

assert c1_fixture["source"].eq(
    "synthetic_c1_fixture"
).all()

assert group_table["within_aspect_share"].between(
    0, 1
).all()

assert all(path.exists() for path in [
    EMBEDDINGS_PATH,
    EMBEDDING_INDEX_PATH,
    REPRESENTATIVENESS_PATH,
    FIXTURE_PATH,
    GROUP_TABLE_PATH,
    INTEGRATION_PATH,
    MANIFEST_PATH,
])

print()
print(
    "Milestone 12 Stage A validation passed."
)

print("Reviews:", len(df))
print("Products:", df["parent_asin"].nunique())
print("Embedding shape:", embeddings.shape)

print(
    "Qualified fixture minority groups:",
    int(group_table["qualified"].sum())
)

print("Real C1 integrated: False")

print(
    "Next: Adapt the existing smartphone "
    "selector and fidelity refinement."
)
""")


add_md("""
## 9. Stage B — Existing Selector Integration

After the Stage A artifacts have been validated:

1. Reuse the existing smartphone selector's scoring
   and constraint-handling implementation.
2. Use the laptop review IDs, predicted usefulness,
   semantic embeddings, and fixture aspect–sentiment groups.
3. Apply the established fixed budget, supported-minority
   reservation, and fidelity-handling rules.
4. Compare against the same selector baselines.
5. Validate deterministic reruns, protected-review
   retention, and explicit constraint outcomes.

Do not claim that synthetic fixture-based minority results
are actual laptop minority-opinion findings.

Do not claim that in-sample training-review scores
establish unbiased downstream selection performance.
""")


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.write_text(
    json.dumps(
        notebook,
        indent=1,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print()
print("Milestone 12 Stage A notebook generated:")
print(NOTEBOOK_PATH)
print("Notebook cell syntax validation passed.")
