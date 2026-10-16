import pandas as pd


REQUIRED_ASPECTS = {
    "battery",
    "camera",
    "performance",
    "display",
    "durability",
    "price",
}


class InputValidationError(ValueError):
    pass


def validate_preferences(preferences):
    required_columns = {
        "aspect",
        "importance",
        "confidence",
        "max_budget",
    }

    missing_columns = (
        required_columns
        - set(preferences.columns)
    )

    if missing_columns:
        raise InputValidationError(
            f"Missing preference columns: "
            f"{sorted(missing_columns)}"
        )

    if preferences.empty:
        raise InputValidationError(
            "Preference data is empty."
        )

    missing_aspects = (
        REQUIRED_ASPECTS
        - set(preferences["aspect"])
    )

    if missing_aspects:
        raise InputValidationError(
            f"Missing preference aspects: "
            f"{sorted(missing_aspects)}"
        )

    if not preferences[
        "importance"
    ].between(1, 5).all():
        raise InputValidationError(
            "Importance values must be between 1 and 5."
        )

    if not preferences[
        "confidence"
    ].between(0, 1).all():
        raise InputValidationError(
            "Confidence values must be between 0 and 1."
        )

    if (
        preferences["max_budget"]
        <= 0
    ).any():
        raise InputValidationError(
            "Maximum budget must be greater than zero."
        )

    return True


def validate_product_ids(
    products,
    *datasets
):
    valid_ids = set(
        products["product_id"]
    )

    for dataset_name, dataset in datasets:

        unknown_ids = (
            set(dataset["product_id"])
            - valid_ids
        )

        if unknown_ids:
            raise InputValidationError(
                f"{dataset_name} contains unknown "
                f"product IDs: {sorted(unknown_ids)}"
            )

    return True


def validate_aspect_coverage(
    dataset,
    dataset_name
):
    missing_aspects = (
        REQUIRED_ASPECTS
        - set(dataset["aspect"])
    )

    if missing_aspects:
        raise InputValidationError(
            f"{dataset_name} is missing aspects: "
            f"{sorted(missing_aspects)}"
        )

    return True


def validate_c1(c1):
    required_columns = {
        "product_id",
        "aspect",
        "aspect_quality",
        "sentiment_score",
        "c1_confidence",
    }

    missing = (
        required_columns
        - set(c1.columns)
    )

    if missing:
        raise InputValidationError(
            f"C1 missing columns: {sorted(missing)}"
        )

    if not c1[
        "aspect_quality"
    ].between(0, 1).all():
        raise InputValidationError(
            "C1 aspect_quality must be between 0 and 1."
        )

    if not c1[
        "sentiment_score"
    ].between(0, 1).all():
        raise InputValidationError(
            "C1 sentiment_score must be between 0 and 1."
        )

    if not c1[
        "c1_confidence"
    ].between(0, 1).all():
        raise InputValidationError(
            "C1 confidence must be between 0 and 1."
        )

    validate_aspect_coverage(
        c1,
        "C1"
    )

    return True


def validate_c2(
    c2,
    minimum_evidence_per_group=2
):
    required_columns = {
        "review_id",
        "product_id",
        "aspect",
        "evidence_quality",
        "representative_evidence",
    }

    missing = (
        required_columns
        - set(c2.columns)
    )

    if missing:
        raise InputValidationError(
            f"C2 missing columns: {sorted(missing)}"
        )

    if not c2[
        "evidence_quality"
    ].between(0, 1).all():
        raise InputValidationError(
            "C2 evidence_quality must be between 0 and 1."
        )

    validate_aspect_coverage(
        c2,
        "C2"
    )

    counts = (
        c2.groupby(
            ["product_id", "aspect"]
        )
        .size()
    )

    insufficient = counts[
        counts
        < minimum_evidence_per_group
    ]

    if not insufficient.empty:
        raise InputValidationError(
            "Insufficient C2 review evidence "
            "for one or more product-aspect groups."
        )

    if c2[
        "evidence_quality"
    ].sum() == 0:
        raise InputValidationError(
            "All C2 evidence quality values are zero."
        )

    return True


def validate_c3(c3):
    required_columns = {
        "product_id",
        "aspect",
        "recent_quality",
        "trend_score",
        "temporal_uncertainty",
        "emerging_issue_score",
    }

    missing = (
        required_columns
        - set(c3.columns)
    )

    if missing:
        raise InputValidationError(
            f"C3 missing columns: {sorted(missing)}"
        )

    if not c3[
        "recent_quality"
    ].between(0, 1).all():
        raise InputValidationError(
            "C3 recent_quality must be between 0 and 1."
        )

    if not c3[
        "temporal_uncertainty"
    ].between(0, 1).all():
        raise InputValidationError(
            "C3 temporal_uncertainty must be between 0 and 1."
        )

    if not c3[
        "emerging_issue_score"
    ].between(0, 1).all():
        raise InputValidationError(
            "C3 emerging_issue_score must be between 0 and 1."
        )

    validate_aspect_coverage(
        c3,
        "C3"
    )

    return True


def validate_budget_eligibility(
    products,
    preferences
):
    max_budget = (
        preferences[
            "max_budget"
        ].iloc[0]
    )

    eligible = products[
        products["price"]
        <= max_budget
    ]

    if eligible.empty:
        raise InputValidationError(
            "No products are within the user's budget."
        )

    return eligible


def validate_all_inputs(
    products,
    preferences,
    c1,
    c2,
    c3
):
    validate_preferences(
        preferences
    )

    validate_c1(c1)
    validate_c2(c2)
    validate_c3(c3)

    validate_product_ids(
        products,
        ("C1", c1),
        ("C2", c2),
        ("C3", c3),
    )

    eligible = (
        validate_budget_eligibility(
            products,
            preferences
        )
    )

    return eligible