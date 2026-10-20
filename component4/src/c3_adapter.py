
import numpy as np
import pandas as pd


class C3AdapterError(ValueError):
    """Raised when the C3 output is invalid."""


REQUIRED_COLUMNS = {
    "parent_asin",
    "aspect",
    "quarter",
    "review_count",
    "review_perceived_quality_dev",
    "quality_ci_lower_dev",
    "quality_ci_upper_dev",
    "trend",
    "latest_change_point",
    "emerging_status",
    "emerging_label",
    "issue_risk",
    "issue_confidence_dev",
}


COLUMN_ALIASES = {
    "latest_quarter": "quarter",
    "change_point": "latest_change_point",
    "confidence_dev": "issue_confidence_dev",
}


def adapt_c3_temporal(c3_output: pd.DataFrame) -> pd.DataFrame:
    """
    Convert actual C3 aspect-level output into a
    standardized C4 temporal-feature table.

    Selects the latest quarter for each product-aspect pair.

    This stage preserves categorical trend and issue risk.
    It does not invent calibrated uncertainty scores.
    """

    if not isinstance(c3_output, pd.DataFrame):
        raise C3AdapterError(
            "C3 output must be a pandas DataFrame."
        )

    if c3_output.empty:
        raise C3AdapterError("C3 output is empty.")

    df = c3_output.copy(deep=True)

    # Support column names observed in C3 exports.
    for alias, canonical in COLUMN_ALIASES.items():
        if alias in df.columns:
            if canonical in df.columns:
                raise C3AdapterError(
                    f"Both {alias} and {canonical} are present."
                )

            df = df.rename(columns={alias: canonical})

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise C3AdapterError(
            f"Missing C3 columns: {sorted(missing)}"
        )

    # Product and aspect identifiers.
    for column in ["parent_asin", "aspect"]:
        if (
            df[column].isna().any()
            or df[column].astype(str).str.strip().eq("").any()
        ):
            raise C3AdapterError(
                f"C3 contains missing {column} values."
            )

    df["product_id"] = (
        df["parent_asin"].astype(str).str.strip()
    )

    df["aspect"] = (
        df["aspect"].astype(str).str.strip().str.lower()
    )

    # Validate quarterly periods.
    if df["quarter"].isna().any():
        raise C3AdapterError(
            "C3 contains missing quarter values."
        )

    df["quarter"] = (
        df["quarter"].astype(str).str.strip().str.upper()
    )

    if not df["quarter"].str.fullmatch(
        r"\d{4}Q[1-4]"
    ).all():
        raise C3AdapterError(
            "Quarter must use YYYYQ1 to YYYYQ4 format."
        )

    # Check numeric quality and count fields.
    numeric_columns = [
        "review_count",
        "review_perceived_quality_dev",
        "quality_ci_lower_dev",
        "quality_ci_upper_dev",
        "emerging_status",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

        if (
            df[column].isna().any()
            or not np.isfinite(df[column].to_numpy()).all()
        ):
            raise C3AdapterError(
                f"C3 {column} contains invalid values."
            )

    if (
        (df["review_count"] <= 0).any()
        or (df["review_count"] % 1 != 0).any()
    ):
        raise C3AdapterError(
            "C3 review_count must be a positive integer."
        )

    for column in [
        "review_perceived_quality_dev",
        "quality_ci_lower_dev",
        "quality_ci_upper_dev",
    ]:
        if not df[column].between(0, 100).all():
            raise C3AdapterError(
                f"{column} must be between 0 and 100."
            )

    if (
        df["quality_ci_lower_dev"]
        > df["quality_ci_upper_dev"]
    ).any():
        raise C3AdapterError(
            "Quality confidence interval bounds are reversed."
        )

    # Status codes observed in the current C3 output:
    # 0 = no flagged issue
    # 1 = existing issue
    # 2 = new or rapidly increasing issue
    if not df["emerging_status"].isin([0, 1, 2]).all():
        raise C3AdapterError(
            "Unsupported C3 emerging_status value."
        )

    # Issue confidence is optional for rows without issues.
    df["issue_confidence_dev"] = pd.to_numeric(
        df["issue_confidence_dev"],
        errors="coerce",
    )

    valid_confidence = (
        df["issue_confidence_dev"].isna()
        | df["issue_confidence_dev"].between(0, 1)
    )

    if not valid_confidence.all():
        raise C3AdapterError(
            "Issue confidence must be between 0 and 1."
        )

    # Preserve trend and risk categories.
    df["trend"] = (
        df["trend"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    if df["trend"].isna().any() or df["trend"].eq("").any():
        raise C3AdapterError(
            "C3 trend contains missing values."
        )

    df["issue_risk"] = (
        df["issue_risk"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    # Convert quality scores from 0-100 to 0-1.
    df["recent_quality"] = (
        df["review_perceived_quality_dev"] / 100.0
    )

    df["quality_ci_lower"] = (
        df["quality_ci_lower_dev"] / 100.0
    )

    df["quality_ci_upper"] = (
        df["quality_ci_upper_dev"] / 100.0
    )

    # Normalized CI width; not a calibrated
    # temporal uncertainty probability.
    df["quality_ci_width_normalized"] = (
        df["quality_ci_upper"]
        - df["quality_ci_lower"]
    )

    # Multiple quarterly rows may exist.
    # Duplicate rows for the SAME quarter are ambiguous.
    duplicate_keys = df.duplicated(
        subset=["product_id", "aspect", "quarter"]
    )

    if duplicate_keys.any():
        raise C3AdapterError(
            "Duplicate C3 product-aspect-quarter rows found."
        )

    # Keep the most recent quarter for each product-aspect.
    df = df.sort_values("quarter")

    df = df.drop_duplicates(
        subset=["product_id", "aspect"],
        keep="last",
    )

    df = df.rename(
        columns={"latest_change_point": "change_point"}
    )

    output_columns = [
        "product_id",
        "aspect",
        "quarter",
        "review_count",
        "recent_quality",
        "quality_ci_lower",
        "quality_ci_upper",
        "quality_ci_width_normalized",
        "trend",
        "change_point",
        "emerging_status",
        "emerging_label",
        "issue_risk",
        "issue_confidence_dev",
    ]

    return df[output_columns].reset_index(drop=True)
