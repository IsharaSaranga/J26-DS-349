
import pandas as pd
import pytest

from component4.src.c1_adapter import (
    adapt_c1_reviews,
    C1AdapterError,
)


def sample_c1():
    return pd.DataFrame({
        "review_id": ["R001", "R002", "R003"],
        "parent_asin": ["B001", "B001", "B002"],
        "review_date": [
            "2023-01-01",
            "2023-01-02",
            "2023-01-03",
        ],
        "text": [
            "Good battery",
            "Poor camera",
            "Fast performance",
        ],
        "domain": [
            "smartphone",
            "smartphone",
            "smartphone",
        ],
        "canonical_aspect": [
            "Battery",
            "Camera",
            "Performance",
        ],
        "semantic_group": [
            "battery_life",
            "photo_quality",
            "speed",
        ],
        "aspect_probability": [0.95, 0.90, 0.88],
        "predicted_sentiment": [
            "positive",
            "negative",
            "positive",
        ],
        "sentiment_score": [0.8, -0.7, 0.6],
        "sentiment_confidence": [0.92, 0.85, 0.91],
    })


def test_c1_adapter_creates_output():
    result = adapt_c1_reviews(sample_c1())

    assert len(result) == 3
    assert "product_id" in result.columns
    assert "aspect" in result.columns
    assert "c1_confidence" in result.columns


def test_c1_product_id_mapping():
    result = adapt_c1_reviews(sample_c1())

    assert result.iloc[0]["product_id"] == "B001"


def test_c1_aspect_standardization():
    result = adapt_c1_reviews(sample_c1())

    assert result["aspect"].tolist() == [
        "battery",
        "camera",
        "performance",
    ]


def test_c1_sentiment_values_preserved():
    result = adapt_c1_reviews(sample_c1())

    assert result.iloc[1]["sentiment_score"] == -0.7


def test_c1_missing_column_fails():
    data = sample_c1().drop(
        columns=["canonical_aspect"]
    )

    with pytest.raises(C1AdapterError):
        adapt_c1_reviews(data)


def test_c1_invalid_confidence_fails():
    data = sample_c1()
    data.loc[0, "sentiment_confidence"] = 1.5

    with pytest.raises(C1AdapterError):
        adapt_c1_reviews(data)


def test_c1_missing_product_id_fails():
    data = sample_c1()
    data.loc[0, "parent_asin"] = None

    with pytest.raises(C1AdapterError):
        adapt_c1_reviews(data)


def test_c1_input_not_modified():
    data = sample_c1()
    original = data.copy(deep=True)

    adapt_c1_reviews(data)

    pd.testing.assert_frame_equal(data, original)
