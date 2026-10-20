
import pandas as pd

from component4.src.c1_adapter import adapt_c1_reviews
from component4.src.c2_adapter import adapt_c2_reviews
from component4.src.c3_adapter import adapt_c3_temporal


REVIEW_KEYS = ["review_id", "product_id", "aspect"]
ASPECT_KEYS = ["product_id", "aspect"]


class IntegrationError(ValueError):
    """Raised when upstream outputs cannot be safely aligned."""


def _check_table(df, name, columns, unique_keys):
    if not isinstance(df, pd.DataFrame):
        raise IntegrationError(
            f"{name} must be a DataFrame."
        )

    missing = set(columns) - set(df.columns)

    if missing:
        raise IntegrationError(
            f"{name} missing columns: {sorted(missing)}"
        )

    if df.empty:
        raise IntegrationError(
            f"{name} is empty."
        )

    if df[unique_keys].isna().any().any():
        raise IntegrationError(
            f"{name} contains missing join keys."
        )

    if df.duplicated(subset=unique_keys).any():
        raise IntegrationError(
            f"{name} contains duplicate join keys "
            f"{unique_keys}. Resolve them explicitly "
            "before integration."
        )


def integrate_standardized_outputs(c1, c2, c3):
    """
    Integrate already-adapted C1, C2 and C3 tables.

    Preserve unmatched records for auditing.
    Do not infer missing scores or create final
    ranking features at this stage.
    """

    _check_table(
        c1,
        "C1",
        REVIEW_KEYS + [
            "sentiment_score",
            "c1_confidence",
        ],
        REVIEW_KEYS,
    )

    _check_table(
        c2,
        "C2",
        REVIEW_KEYS + [
            "evidence_quality",
            "is_selected",
        ],
        REVIEW_KEYS,
    )

    _check_table(
        c3,
        "C3",
        ASPECT_KEYS + ["recent_quality"],
        ASPECT_KEYS,
    )

    # Join C1 and C2 at review-product-aspect level.
    # Outer join retains unmatched records for audit.
    review_join = c1.merge(
        c2,
        on=REVIEW_KEYS,
        how="outer",
        validate="one_to_one",
        indicator="review_match",
        suffixes=("_c1", "_c2"),
    )

    matched = review_join.loc[
        review_join["review_match"] == "both"
    ].drop(
        columns="review_match"
    ).copy()

    if matched.empty:
        raise IntegrationError(
            "No review-product-aspect matches "
            "between C1 and C2. Check product "
            "identifiers and aspect taxonomy."
        )

    # Count matched review evidence per product-aspect.
    coverage = (
        matched.groupby(
            ASPECT_KEYS,
            as_index=False,
        )
        .agg(
            matched_review_count=(
                "review_id",
                "nunique",
            ),
            selected_review_count=(
                "is_selected",
                "sum",
            ),
        )
    )

    # Align C3 temporal features by product-aspect.
    # Keep unmatched C3 aspects visible.
    product_aspects = coverage.merge(
        c3,
        on=ASPECT_KEYS,
        how="outer",
        validate="one_to_one",
        indicator="temporal_match",
    )

    # Integration audit.
    audit = {
        "c1_review_aspect_rows": int(len(c1)),
        "c2_review_aspect_rows": int(len(c2)),
        "matched_review_aspect_rows": int(
            len(matched)
        ),
        "c1_without_matching_c2": int(
            (
                review_join["review_match"]
                == "left_only"
            ).sum()
        ),
        "c2_without_matching_c1": int(
            (
                review_join["review_match"]
                == "right_only"
            ).sum()
        ),
        "matched_product_aspects_with_c3": int(
            (
                product_aspects["temporal_match"]
                == "both"
            ).sum()
        ),
        "product_aspects_without_c3": int(
            (
                product_aspects["temporal_match"]
                == "left_only"
            ).sum()
        ),
        "c3_aspects_without_joined_review_evidence": int(
            (
                product_aspects["temporal_match"]
                == "right_only"
            ).sum()
        ),
    }

    return {
        # Preserve all standardized source records.
        "c1_reviews": c1.copy(deep=True),
        "c2_evidence": c2.copy(deep=True),
        "c3_temporal": c3.copy(deep=True),

        # Matched review-level evidence.
        "review_evidence": matched.reset_index(
            drop=True
        ),

        # Temporal alignment and coverage.
        "product_aspects": (
            product_aspects.reset_index(drop=True)
        ),

        # Records requiring investigation.
        "unmatched_reviews": (
            review_join.loc[
                review_join["review_match"] != "both"
            ].reset_index(drop=True)
        ),

        "audit": audit,
    }


def integrate_actual_outputs(
    c1_raw,
    c2_raw,
    c3_raw,
):
    """
    Accept raw C1/C2/C3 output DataFrames.

    First run the three adapters, then integrate.
    """

    c1 = adapt_c1_reviews(c1_raw)
    c2 = adapt_c2_reviews(c2_raw)
    c3 = adapt_c3_temporal(c3_raw)

    return integrate_standardized_outputs(
        c1,
        c2,
        c3,
    )
