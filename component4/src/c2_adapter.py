
import ast

import numpy as np
import pandas as pd


class C2AdapterError(ValueError):
    """Raised when C2 output is invalid."""


REQUIRED_C2_COLUMNS = {
    "review_id",
    "parent_asin",
    "review_date",
    "text",
    "predicted_usefulness",
    "usefulness_norm",
    "representativeness",
    "representativeness_norm",
    "aspect_categories",
    "aspect_sentiment_groups",
    "supported_minority_groups",
    "supports_minority",
    "is_selected",
    "refined_selection_order",
    "minority_reserve_locked",
    "outcome_status",
}


def _parse_aspects(value):
    """
    Accept a list, a serialized list, or a single aspect.
    Also supports comma-separated and pipe-separated text.
    """

    if isinstance(value, (list, tuple, np.ndarray)):
        items = list(value)

    elif isinstance(value, str):
        value = value.strip()

        if not value:
            raise C2AdapterError(
                "C2 aspect_categories contains an empty value."
            )

        if value.startswith(("[", "(")):
            try:
                items = ast.literal_eval(value)
            except (ValueError, SyntaxError) as exc:
                raise C2AdapterError(
                    "Invalid serialized aspect_categories."
                ) from exc

            if not isinstance(items, (list, tuple)):
                raise C2AdapterError(
                    "Serialized aspect_categories must be a list."
                )

        elif "|" in value:
            items = value.split("|")

        elif "," in value:
            items = value.split(",")

        else:
            items = [value]

    else:
        raise C2AdapterError(
            "Unsupported aspect_categories format."
        )

    aspects = []

    for item in items:
        if not isinstance(item, str) or not item.strip():
            raise C2AdapterError(
                "Aspect names must be non-empty strings."
            )

        aspect = item.strip().lower()

        if aspect not in aspects:
            aspects.append(aspect)

    if not aspects:
        raise C2AdapterError(
            "C2 review has no valid aspects."
        )

    return aspects


def _parse_boolean(value):
    """Parse actual Boolean values or CSV Boolean strings."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, (int, float, np.integer, np.floating)):
        if pd.notna(value) and value in (0, 1):
            return bool(value)

    if isinstance(value, str):
        cleaned = value.strip().lower()

        if cleaned in ("true", "1"):
            return True

        if cleaned in ("false", "0"):
            return False

    raise C2AdapterError(
        f"Invalid is_selected value: {value!r}"
    )


def adapt_c2_reviews(c2_output: pd.DataFrame) -> pd.DataFrame:
    """
    Convert actual C2 output into C4 review-level evidence.

    The 50/50 evidence-quality weighting is provisional.
    No outcome_status filtering is applied at this stage.
    """

    if not isinstance(c2_output, pd.DataFrame):
        raise C2AdapterError(
            "C2 output must be a pandas DataFrame."
        )

    missing = REQUIRED_C2_COLUMNS - set(c2_output.columns)

    if missing:
        raise C2AdapterError(
            f"Missing C2 columns: {sorted(missing)}"
        )

    if c2_output.empty:
        raise C2AdapterError("C2 output is empty.")

    df = c2_output.copy(deep=True)

    for column in ["review_id", "parent_asin"]:
        if (
            df[column].isna().any()
            or df[column].astype(str).str.strip().eq("").any()
        ):
            raise C2AdapterError(
                f"C2 contains missing {column} values."
            )

        df[column] = df[column].astype(str).str.strip()

    for column in [
        "usefulness_norm",
        "representativeness_norm",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if (
            df[column].isna().any()
            or not np.isfinite(df[column].to_numpy()).all()
            or not df[column].between(0, 1).all()
        ):
            raise C2AdapterError(
                f"C2 {column} must contain values from 0 to 1."
            )

    df["is_selected"] = df["is_selected"].map(
        _parse_boolean
    )

    # Provisional evidence quality:
    # Equal weight for normalized usefulness and
    # representativeness. Validate this choice later.
    df["evidence_quality"] = (
        0.5 * df["usefulness_norm"]
        + 0.5 * df["representativeness_norm"]
    )

    # A review may discuss multiple aspects.
    df["aspect"] = df["aspect_categories"].apply(
        _parse_aspects
    )

    df = df.explode("aspect", ignore_index=True)

    df = df.rename(
        columns={
            "parent_asin": "product_id",
            "text": "representative_evidence",
        }
    )

    # Avoid counting duplicated review-aspect evidence twice.
    duplicates = df.duplicated(
        subset=["review_id", "product_id", "aspect"]
    )

    if duplicates.any():
        raise C2AdapterError(
            "Duplicate review-product-aspect evidence found."
        )

    # Selected reviews must have displayable text.
    selected = df[df["is_selected"]]

    if (
        selected["representative_evidence"].isna().any()
        or selected["representative_evidence"]
        .astype(str)
        .str.strip()
        .eq("")
        .any()
    ):
        raise C2AdapterError(
            "Selected C2 reviews contain missing text."
        )

    output_columns = [
        "review_id",
        "product_id",
        "aspect",
        "review_date",
        "representative_evidence",
        "evidence_quality",
        "usefulness_norm",
        "representativeness_norm",
        "predicted_usefulness",
        "representativeness",
        "aspect_sentiment_groups",
        "supported_minority_groups",
        "supports_minority",
        "is_selected",
        "refined_selection_order",
        "minority_reserve_locked",
        "outcome_status",
    ]

    return df[output_columns].copy()
