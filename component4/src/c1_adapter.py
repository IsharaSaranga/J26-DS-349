
import numpy as np
import pandas as pd


class C1AdapterError(ValueError):
    """Raised when the C1 output is invalid."""


REQUIRED_C1_COLUMNS = {
    "review_id",
    "parent_asin",
    "review_date",
    "text",
    "domain",
    "canonical_aspect",
    "semantic_group",
    "aspect_probability",
    "predicted_sentiment",
    "sentiment_score",
    "sentiment_confidence",
}


def adapt_c1_reviews(c1_output: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and standardize actual C1 review-level output.

    This function does not aggregate sentiment scores or
    convert them into product quality scores.
    """

    if not isinstance(c1_output, pd.DataFrame):
        raise C1AdapterError(
            "C1 output must be a pandas DataFrame."
        )

    missing = REQUIRED_C1_COLUMNS - set(c1_output.columns)

    if missing:
        raise C1AdapterError(
            f"Missing C1 columns: {sorted(missing)}"
        )

    if c1_output.empty:
        raise C1AdapterError("C1 output is empty.")

    df = c1_output.copy()

    # Validate essential identifiers.
    for column in [
        "review_id",
        "parent_asin",
        "canonical_aspect",
    ]:
        if (
            df[column].isna().any()
            or df[column].astype(str).str.strip().eq("").any()
        ):
            raise C1AdapterError(
                f"C1 contains missing or blank {column} values."
            )

    # Validate numerical fields without assuming
    # the scale of the sentiment score.
    for column in [
        "aspect_probability",
        "sentiment_score",
        "sentiment_confidence",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        if (
            df[column].isna().any()
            or not np.isfinite(df[column].to_numpy()).all()
        ):
            raise C1AdapterError(
                f"C1 {column} contains invalid numeric values."
            )

    # Prediction confidence values must be in [0, 1].
    for column in [
        "aspect_probability",
        "sentiment_confidence",
    ]:
        if not df[column].between(0, 1).all():
            raise C1AdapterError(
                f"C1 {column} must be between 0 and 1."
            )

    # Standardize identifiers for C4.
    df = df.rename(
        columns={
            "parent_asin": "product_id",
            "canonical_aspect": "aspect",
            "sentiment_confidence": "c1_confidence",
        }
    )

    df["review_id"] = (
        df["review_id"].astype(str).str.strip()
    )

    df["product_id"] = (
        df["product_id"].astype(str).str.strip()
    )

    df["aspect"] = (
        df["aspect"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Preserve review-level rows and supporting fields.
    output_columns = [
        "review_id",
        "product_id",
        "review_date",
        "text",
        "domain",
        "aspect",
        "semantic_group",
        "aspect_probability",
        "predicted_sentiment",
        "sentiment_score",
        "c1_confidence",
    ]

    return df[output_columns].copy()
