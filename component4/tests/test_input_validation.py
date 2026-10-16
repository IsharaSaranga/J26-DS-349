import pandas as pd
import pytest
from component4.src.input_validation import (
    validate_c1,
    validate_c2,
    validate_c3,
)


def test_c2_insufficient_evidence_fails():
    c2 = pd.DataFrame({
        "review_id": ["R1", "R2"],
        "product_id": ["P001", "P001"],
        "aspect": ["battery", "camera"],
        "evidence_quality": [0.8, 0.9],
        "representative_evidence": ["A", "B"],
    })

    with pytest.raises(InputValidationError):
        validate_c2(
            c2,
            minimum_evidence_per_group=2
        )


def test_c2_all_zero_quality_fails():
    rows = []

    aspects = [
        "battery",
        "camera",
        "performance",
        "display",
        "durability",
        "price",
    ]

    for aspect in aspects:
        for i in range(2):
            rows.append({
                "review_id": f"{aspect}_{i}",
                "product_id": "P001",
                "aspect": aspect,
                "evidence_quality": 0.0,
                "representative_evidence": "test",
            })

    c2 = pd.DataFrame(rows)

    with pytest.raises(InputValidationError):
        validate_c2(c2)


def test_c1_missing_required_column_fails():
    c1 = pd.DataFrame({
        "product_id": ["P001"],
        "aspect": ["battery"],
        "aspect_quality": [0.8],
        "sentiment_score": [0.7],
        # c1_confidence missing
    })

    with pytest.raises(InputValidationError):
        validate_c1(c1)


def test_c3_missing_required_column_fails():
    c3 = pd.DataFrame({
        "product_id": ["P001"],
        "aspect": ["battery"],
        "recent_quality": [0.8],
        "trend_score": [0.1],
        "temporal_uncertainty": [0.2],
        # emerging_issue_score missing
    })

    with pytest.raises(InputValidationError):
        validate_c3(c3)


def test_c1_invalid_quality_range_fails():
    rows = []

    for aspect in [
        "battery",
        "camera",
        "performance",
        "display",
        "durability",
        "price",
    ]:
        rows.append({
            "product_id": "P001",
            "aspect": aspect,
            "aspect_quality": 0.8,
            "sentiment_score": 0.7,
            "c1_confidence": 0.9,
        })

    c1 = pd.DataFrame(rows)

    c1.loc[0, "aspect_quality"] = 1.5

    with pytest.raises(InputValidationError):
        validate_c1(c1)

from component4.src.input_validation import (
    InputValidationError,
    validate_preferences,
    validate_product_ids,
    validate_budget_eligibility,
)


def valid_preferences():
    return pd.DataFrame({
        "aspect": [
            "battery",
            "camera",
            "performance",
            "display",
            "durability",
            "price",
        ],
        "importance": [
            5, 3, 4, 3, 2, 4
        ],
        "confidence": [
            0.9, 0.7, 0.85,
            0.8, 0.75, 0.9
        ],
        "max_budget": [
            170000
        ] * 6
    })


def test_valid_preferences_pass():
    preferences = valid_preferences()

    assert (
        validate_preferences(
            preferences
        )
        is True
    )


def test_invalid_confidence_fails():
    preferences = valid_preferences()

    preferences.loc[
        0,
        "confidence"
    ] = 1.5

    with pytest.raises(
        InputValidationError
    ):
        validate_preferences(
            preferences
        )


def test_invalid_importance_fails():
    preferences = valid_preferences()

    preferences.loc[
        0,
        "importance"
    ] = 8

    with pytest.raises(
        InputValidationError
    ):
        validate_preferences(
            preferences
        )


def test_missing_aspect_fails():
    preferences = (
        valid_preferences()
        .query(
            "aspect != 'camera'"
        )
    )

    with pytest.raises(
        InputValidationError
    ):
        validate_preferences(
            preferences
        )


def test_unknown_product_id_fails():
    products = pd.DataFrame({
        "product_id": [
            "P001",
            "P002"
        ]
    })

    c1 = pd.DataFrame({
        "product_id": [
            "P001",
            "UNKNOWN"
        ]
    })

    with pytest.raises(
        InputValidationError
    ):
        validate_product_ids(
            products,
            ("C1", c1)
        )


def test_no_product_within_budget_fails():
    products = pd.DataFrame({
        "product_id": [
            "P001",
            "P002"
        ],
        "price": [
            200000,
            250000
        ]
    })

    preferences = (
        valid_preferences()
    )

    preferences[
        "max_budget"
    ] = 100000

    with pytest.raises(
        InputValidationError
    ):
        validate_budget_eligibility(
            products,
            preferences
        )


def test_one_product_within_budget_allowed():
    products = pd.DataFrame({
        "product_id": [
            "P001",
            "P002"
        ],
        "price": [
            100000,
            250000
        ]
    })

    preferences = (
        valid_preferences()
    )

    preferences[
        "max_budget"
    ] = 150000

    eligible = (
        validate_budget_eligibility(
            products,
            preferences
        )
    )

    assert len(eligible) == 1
    assert (
        eligible.iloc[0][
            "product_id"
        ]
        == "P001"
    )
    