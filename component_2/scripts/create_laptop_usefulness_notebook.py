
from pathlib import Path
import json
import textwrap


ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = ROOT / "notebooks"
NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = NOTEBOOK_DIR / "11_laptop_usefulness_baseline.ipynb"

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

    # Validate notebook cell Python syntax before saving.
    compile(source, "<notebook-cell>", "exec")

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    })


add_md("""
# Milestone 11 — Laptop Review Usefulness Baseline

**Component:** Helpfulness-Guided Review Evidence Selection
with Minority Preservation

## Objective

Develop and evaluate a laptop review usefulness prediction
baseline using the cleaned Amazon Reviews 2023 laptop pilot.

## Experimental Protocol

1. Load the frozen Milestone 10 dataset and split.
2. Validate data and prevent target leakage.
3. Establish a DummyRegressor benchmark.
4. Train TF-IDF + Ridge models.
5. Compare text-only and text-plus-metadata features.
6. Select the configuration using validation NDCG@10.
7. Evaluate the chosen model on the untouched test set.
8. Generate product-level diagnostics.
9. Save model, predictions, metrics, and validation evidence.

The preliminary target is `log_helpful`.

No real C1 output is required for this milestone.

**Research limitation:** This is a controlled PP1 laptop
pilot. Full category-scale generalization and final
cross-category comparisons are outside this milestone.
""")


add_md("## 1. Imports and Project Paths")

add_code("""
from pathlib import Path
import json
import hashlib
import warnings

import joblib
import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    ndcg_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def find_project_root(start):
    for candidate in [start, *start.parents]:
        if (
            (candidate / "data" / "pilot").exists()
            and (candidate / "notebooks").exists()
        ):
            return candidate

    raise FileNotFoundError(
        "Component 2 project root was not found."
    )


ROOT = find_project_root(Path.cwd().resolve())

DATA_PATH = (
    ROOT / "data" / "pilot"
    / "laptop_pilot_cleaned.parquet"
)

SPLIT_PATH = (
    ROOT / "results" / "eda" / "laptop"
    / "laptop_chronological_split.csv"
)

RESULTS_DIR = (
    ROOT / "results" / "usefulness" / "laptop"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "log_helpful"

ALPHAS = [0.1, 1.0, 10.0, 100.0]
FEATURE_MODES = ["text_only", "text_metadata"]

print("Project root:", ROOT)
print("Laptop dataset:", DATA_PATH)
print("Split file:", SPLIT_PATH)
print("Results:", RESULTS_DIR)
""")


add_md("## 2. Load and Validate the Frozen Laptop Dataset")

add_code("""
assert DATA_PATH.exists(), (
    "Milestone 10 cleaned laptop dataset is missing."
)

assert SPLIT_PATH.exists(), (
    "Milestone 10 chronological split is missing."
)

reviews = pd.read_parquet(DATA_PATH)

assignments = pd.read_csv(
    SPLIT_PATH,
    dtype={"review_id": str, "parent_asin": str},
)

required = {
    "review_id",
    "parent_asin",
    "text",
    "rating",
    "verified_purchase",
    "review_length_words",
    "age_days",
    "review_date",
    "helpful_vote",
    TARGET,
}

assert required.issubset(reviews.columns)

reviews["review_id"] = reviews["review_id"].astype(str)
reviews["parent_asin"] = reviews["parent_asin"].astype(str)

assert reviews["review_id"].is_unique
assert assignments["review_id"].is_unique

assert set(reviews["review_id"]) == set(
    assignments["review_id"]
)

df = reviews.merge(
    assignments[["review_id", "split"]],
    on="review_id",
    how="left",
    validate="one_to_one",
)

assert not df["split"].isna().any()

assert set(df["split"]) == {
    "train", "validation", "test"
}

df["review_date"] = pd.to_datetime(
    df["review_date"],
    utc=True,
)

df["text"] = df["text"].fillna("").astype(str)
df["verified_purchase"] = (
    df["verified_purchase"].astype(int)
)

df["rating"] = pd.to_numeric(df["rating"])
df["age_days"] = pd.to_numeric(df["age_days"])
df["review_length_words"] = pd.to_numeric(
    df["review_length_words"]
)

assert df[TARGET].notna().all()
assert (df[TARGET] >= 0).all()

assert df["parent_asin"].nunique() == 10
assert len(df) == 997

train = df[df["split"] == "train"].copy()
validation = df[df["split"] == "validation"].copy()
test = df[df["split"] == "test"].copy()

assert len(train) == 697
assert len(validation) == 150
assert len(test) == 150

assert not (
    set(train["review_id"])
    & set(validation["review_id"])
)

assert not (
    set(train["review_id"])
    & set(test["review_id"])
)

assert not (
    set(validation["review_id"])
    & set(test["review_id"])
)

print("Cleaned laptop reviews:", len(df))
print("Products:", df["parent_asin"].nunique())
print("Train:", len(train))
print("Validation:", len(validation))
print("Test:", len(test))
""")


add_md("## 3. Chronological and Leakage Checks")

add_code("""
chronology_rows = []

for product, group in df.groupby("parent_asin"):
    tr = group.loc[
        group["split"] == "train",
        "review_date",
    ]

    va = group.loc[
        group["split"] == "validation",
        "review_date",
    ]

    te = group.loc[
        group["split"] == "test",
        "review_date",
    ]

    chronological = bool(
        not tr.empty
        and not va.empty
        and not te.empty
        and tr.max() <= va.min()
        and va.max() <= te.min()
    )

    chronology_rows.append({
        "parent_asin": product,
        "chronology_valid": chronological,
        "train_count": len(tr),
        "validation_count": len(va),
        "test_count": len(te),
    })

chronology_df = pd.DataFrame(chronology_rows)

assert chronology_df["chronology_valid"].all()

# Only inputs available without using the target outcome.
PREDICTOR_COLUMNS = [
    "text",
    "rating",
    "verified_purchase",
    "review_length_words",
    "age_days",
]

FORBIDDEN_PREDICTORS = {
    "helpful_vote",
    "log_helpful",
    "raw_helpful",
    "product_relative",
    "age_adjusted",
    "split",
    "review_id",
}

assert not (
    set(PREDICTOR_COLUMNS)
    & FORBIDDEN_PREDICTORS
)

assert not df[PREDICTOR_COLUMNS].isna().any().any()

display(chronology_df)

print("Chronological split verification: PASS")
print("Target-derived feature exclusion: PASS")
""")


add_md("## 4. Evaluation Functions")

add_code("""
def product_ranking_metrics(frame):
    rows = []

    for product, group in frame.groupby("parent_asin"):
        true = group["actual"].to_numpy(dtype=float)
        predicted = group["predicted"].to_numpy(
            dtype=float
        )

        # An all-zero relevant set provides no positive
        # relevance signal for NDCG ranking evaluation.
        eligible = (
            len(group) >= 2
            and np.any(true > 0)
        )

        if eligible:
            ndcg5 = ndcg_score(
                true.reshape(1, -1),
                predicted.reshape(1, -1),
                k=min(5, len(group)),
            )

            ndcg10 = ndcg_score(
                true.reshape(1, -1),
                predicted.reshape(1, -1),
                k=min(10, len(group)),
            )
        else:
            ndcg5 = np.nan
            ndcg10 = np.nan

        if (
            len(group) >= 3
            and np.unique(true).size > 1
            and np.unique(predicted).size > 1
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho = spearmanr(
                    true, predicted
                ).statistic
        else:
            rho = np.nan

        rows.append({
            "parent_asin": product,
            "review_count": len(group),
            "positive_relevance_count": int(
                np.sum(true > 0)
            ),
            "ndcg_eligible": bool(eligible),
            "ndcg_at_5": ndcg5,
            "ndcg_at_10": ndcg10,
            "spearman": rho,
            "mae": mean_absolute_error(
                true, predicted
            ),
            "rmse": np.sqrt(
                mean_squared_error(true, predicted)
            ),
        })

    return pd.DataFrame(rows)


def evaluate_predictions(frame, predicted, model_name):
    evaluated = frame[
        ["review_id", "parent_asin", "split"]
    ].copy()

    evaluated["actual"] = frame[
        TARGET
    ].to_numpy(dtype=float)

    evaluated["predicted"] = np.asarray(
        predicted, dtype=float
    )

    details = product_ranking_metrics(evaluated)

    true = evaluated["actual"].to_numpy()
    prediction = evaluated["predicted"].to_numpy()

    if (
        np.unique(true).size > 1
        and np.unique(prediction).size > 1
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho = spearmanr(
                true, prediction
            ).statistic
    else:
        rho = np.nan

    metrics = {
        "model": model_name,
        "review_count": len(evaluated),
        "product_count": details.shape[0],
        "ndcg_eligible_products": int(
            details["ndcg_eligible"].sum()
        ),
        "mae": float(
            mean_absolute_error(true, prediction)
        ),
        "rmse": float(np.sqrt(
            mean_squared_error(true, prediction)
        )),
        "spearman": float(rho),
        "ndcg_at_5": float(
            details["ndcg_at_5"].mean()
        ),
        "ndcg_at_10": float(
            details["ndcg_at_10"].mean()
        ),
    }

    return metrics, evaluated, details
""")


add_md("## 5. Leakage-Safe Feature Pipeline")

add_code("""
NUMERIC_COLUMNS = [
    "rating",
    "verified_purchase",
    "review_length_words",
    "age_days",
]


def make_model(mode, alpha):
    transformers = [
        (
            "tfidf",
            TfidfVectorizer(
                max_features=15000,
                min_df=2,
                ngram_range=(1, 2),
                sublinear_tf=True,
                strip_accents="unicode",
            ),
            "text",
        )
    ]

    if mode == "text_metadata":
        transformers.append(
            (
                "numeric",
                StandardScaler(with_mean=False),
                NUMERIC_COLUMNS,
            )
        )

    features = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
    )

    return Pipeline([
        ("features", features),
        (
            "regressor",
            Ridge(
                alpha=float(alpha),
                solver="lsqr",
            ),
        ),
    ])


X_train = train[PREDICTOR_COLUMNS]
X_validation = validation[PREDICTOR_COLUMNS]
X_test = test[PREDICTOR_COLUMNS]

y_train = train[TARGET].to_numpy(dtype=float)
y_validation = validation[TARGET].to_numpy(
    dtype=float
)
y_test = test[TARGET].to_numpy(dtype=float)

print("Feature columns:", PREDICTOR_COLUMNS)
print("Feature fitting will use training reviews only.")
""")


add_md("## 6. DummyRegressor Benchmark")

add_code("""
dummy = DummyRegressor(strategy="mean")

dummy.fit(
    np.zeros((len(train), 1)),
    y_train,
)

dummy_validation_predictions = dummy.predict(
    np.zeros((len(validation), 1))
)

dummy_validation_metrics, _, _ = evaluate_predictions(
    validation,
    dummy_validation_predictions,
    "DummyRegressor",
)

display(pd.DataFrame([dummy_validation_metrics]))
""")


add_md("## 7. Train Ridge Candidates and Evaluate Validation")

add_code("""
validation_results = []
trained_models = {}

for mode in FEATURE_MODES:
    for alpha in ALPHAS:
        name = f"{mode}_alpha_{alpha:g}"

        print("Training:", name)

        model = make_model(
            mode=mode,
            alpha=alpha,
        )

        # The entire feature pipeline is fitted on
        # training reviews only.
        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_validation
        )

        metrics, _, _ = evaluate_predictions(
            validation,
            predictions,
            name,
        )

        metrics["feature_mode"] = mode
        metrics["alpha"] = float(alpha)

        validation_results.append(metrics)
        trained_models[name] = model


validation_table = pd.DataFrame(validation_results)

# NDCG@10 is the primary model-selection metric.
# Spearman and MAE are secondary tie-breakers.
validation_table = validation_table.sort_values(
    by=[
        "ndcg_at_10",
        "spearman",
        "mae",
    ],
    ascending=[
        False,
        False,
        True,
    ],
    na_position="last",
).reset_index(drop=True)

assert len(validation_table) == (
    len(FEATURE_MODES) * len(ALPHAS)
)

selected_row = validation_table.iloc[0]
SELECTED_MODEL_NAME = str(selected_row["model"])

selected_model = trained_models[
    SELECTED_MODEL_NAME
]

display(validation_table)

print("Selected model:", SELECTED_MODEL_NAME)
print(
    "Validation NDCG@10:",
    selected_row["ndcg_at_10"],
)
""")


add_md("## 8. Final Held-Out Test Evaluation")

add_code("""
# Do not retune the model after inspecting these results.

test_predictions = selected_model.predict(X_test)

test_metrics, test_output, test_product_metrics = (
    evaluate_predictions(
        test,
        test_predictions,
        SELECTED_MODEL_NAME,
    )
)

dummy_test_predictions = dummy.predict(
    np.zeros((len(test), 1))
)

dummy_test_metrics, _, _ = evaluate_predictions(
    test,
    dummy_test_predictions,
    "DummyRegressor",
)

test_comparison = pd.DataFrame([
    dummy_test_metrics,
    test_metrics,
])

display(test_comparison)
display(test_product_metrics)

print("Final selected model:", SELECTED_MODEL_NAME)
print("Held-out evaluation complete.")
""")


add_md("## 9. Review-Level Scoring and Provenance")

add_code("""
# Scores for all reviews support later evidence selection.
# Training scores are in-sample and MUST NOT be reported
# as held-out performance.

all_predictions = selected_model.predict(
    df[PREDICTOR_COLUMNS]
)

all_review_scores = df[
    [
        "review_id",
        "parent_asin",
        "split",
    ]
].copy()

all_review_scores["predicted_log_helpful"] = (
    all_predictions
)

all_review_scores["predicted_usefulness"] = (
    np.maximum(all_predictions, 0.0)
)

provenance = {
    "train": "in_sample_training",
    "validation": "validation_used_for_model_selection",
    "test": "held_out_test",
}

all_review_scores["prediction_provenance"] = (
    all_review_scores["split"].map(provenance)
)

assert all_review_scores["review_id"].is_unique
assert len(all_review_scores) == len(df)

assert np.isfinite(
    all_review_scores["predicted_log_helpful"]
).all()

display(all_review_scores.head())
""")


add_md("## 10. Save Reproducible Evidence")

add_code("""
MODEL_PATH = RESULTS_DIR / "laptop_ridge_model.joblib"

VALIDATION_PATH = (
    RESULTS_DIR / "laptop_validation_model_selection.csv"
)

TEST_METRICS_PATH = (
    RESULTS_DIR / "laptop_test_metrics.csv"
)

PRODUCT_METRICS_PATH = (
    RESULTS_DIR / "laptop_product_test_metrics.csv"
)

TEST_PREDICTIONS_PATH = (
    RESULTS_DIR / "laptop_test_predictions.csv"
)

ALL_PREDICTIONS_PATH = (
    RESULTS_DIR / "laptop_all_review_predictions.csv"
)

CONFIG_PATH = RESULTS_DIR / "laptop_model_config.json"
AUDIT_PATH = RESULTS_DIR / "laptop_split_audit.csv"

joblib.dump(
    selected_model,
    MODEL_PATH,
)

validation_table.to_csv(
    VALIDATION_PATH,
    index=False,
)

test_comparison.to_csv(
    TEST_METRICS_PATH,
    index=False,
)

test_product_metrics.to_csv(
    PRODUCT_METRICS_PATH,
    index=False,
)

test_output.to_csv(
    TEST_PREDICTIONS_PATH,
    index=False,
)

all_review_scores.to_csv(
    ALL_PREDICTIONS_PATH,
    index=False,
)

chronology_df.to_csv(
    AUDIT_PATH,
    index=False,
)

config = {
    "milestone": 11,
    "category": "laptop",
    "dataset": str(DATA_PATH.relative_to(ROOT)),
    "split_file": str(SPLIT_PATH.relative_to(ROOT)),
    "target": TARGET,
    "candidate_alphas": ALPHAS,
    "feature_modes": FEATURE_MODES,
    "selected_model": SELECTED_MODEL_NAME,
    "selected_alpha": float(selected_row["alpha"]),
    "selected_feature_mode": str(
        selected_row["feature_mode"]
    ),
    "model_selection_metric": "validation_ndcg_at_10",
    "test_policy": "held_out_no_retuning",
    "numeric_features": NUMERIC_COLUMNS,
    "forbidden_predictors": sorted(FORBIDDEN_PREDICTORS),
    "all_review_prediction_warning": (
        "Training review predictions are in-sample; "
        "validation reviews were used for model selection. "
        "Only test predictions are held out."
    ),
    "ranking_metric_note": (
        "Per-product NDCG excludes products with "
        "no positive relevance labels and reports "
        "the eligible product count."
    ),
    "test_metrics": test_metrics,
    "dummy_test_metrics": dummy_test_metrics,
}

CONFIG_PATH.write_text(
    json.dumps(
        config,
        indent=2,
        allow_nan=True,
    ),
    encoding="utf-8",
)

print("Model saved:", MODEL_PATH)
print("Validation evidence:", VALIDATION_PATH)
print("Test evidence:", TEST_METRICS_PATH)
print("All review scores:", ALL_PREDICTIONS_PATH)
""")


add_md("## 11. Final Validation and Determinism")

add_code("""
# Load the serialized model and verify predictions.
reloaded_model = joblib.load(MODEL_PATH)

reloaded_predictions = reloaded_model.predict(
    X_test
)

np.testing.assert_allclose(
    test_predictions,
    reloaded_predictions,
    rtol=1e-9,
    atol=1e-9,
)

assert len(train) + len(validation) + len(test) == 997

assert df["parent_asin"].nunique() == 10

assert chronology_df["chronology_valid"].all()

assert len(validation_table) == 8

assert len(test_output) == 150

assert len(all_review_scores) == 997

assert all_review_scores["review_id"].is_unique

assert set(all_review_scores["split"]) == {
    "train", "validation", "test"
}

assert set(
    all_review_scores["prediction_provenance"]
) == {
    "in_sample_training",
    "validation_used_for_model_selection",
    "held_out_test",
}

assert np.isfinite(test_predictions).all()

assert all(path.exists() for path in [
    MODEL_PATH,
    VALIDATION_PATH,
    TEST_METRICS_PATH,
    PRODUCT_METRICS_PATH,
    TEST_PREDICTIONS_PATH,
    ALL_PREDICTIONS_PATH,
    CONFIG_PATH,
    AUDIT_PATH,
])

print()
print("Laptop Function 1 baseline validation passed.")
print()
print("Selected model:", SELECTED_MODEL_NAME)
print("Target:", TARGET)
print("Train reviews:", len(train))
print("Validation reviews:", len(validation))
print("Test reviews:", len(test))
print()
print("Test MAE:", round(test_metrics["mae"], 4))
print("Test RMSE:", round(test_metrics["rmse"], 4))
print(
    "Test Spearman:",
    round(test_metrics["spearman"], 4),
)
print(
    "Test NDCG@5:",
    round(test_metrics["ndcg_at_5"], 4),
)
print(
    "Test NDCG@10:",
    round(test_metrics["ndcg_at_10"], 4),
)
print(
    "NDCG-eligible products:",
    test_metrics["ndcg_eligible_products"],
)
""")


add_md("""
## 12. Research Limitations and Next Milestone

- The experiment uses 997 reviews from ten laptop products.
- `log_helpful` is a noisy observational proxy for usefulness.
- Zero helpful votes may reflect low exposure rather than poor content.
- Product popularity and temporal sampling may introduce bias.
- Held-out test metrics are distinct from all-review inference scores.
- The selected model is laptop-specific at this stage.
- A direct smartphone-versus-laptop comparison requires aligning
  evaluation policies across both categories.
- Training-review in-sample predictions should not be used as
  unbiased downstream selector-evaluation evidence.

**Next milestone:** Laptop Semantic Representation and Function 2
Evidence Selection.
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

OUTPUT.write_text(
    json.dumps(notebook, indent=1, ensure_ascii=False),
    encoding="utf-8",
)

print()
print("Notebook generated successfully:")
print(OUTPUT)
print("Code cells passed syntax validation.")
