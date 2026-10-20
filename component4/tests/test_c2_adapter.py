
import pandas as pd
import pytest

from component4.src.c2_adapter import (
    adapt_c2_reviews,
    C2AdapterError,
)


def sample_c2():
    return pd.DataFrame({
        "review_id": ["R001", "R002", "R003"],
        "parent_asin": ["B001", "B001", "B002"],
        "review_date": [
            "2023-01-01",
            "2023-01-02",
            "2023-01-03",
        ],
        "text": [
            "Good battery and performance",
            "Camera is acceptable",
            "Excellent display",
        ],
        "predicted_usefulness": [0.8, 0.4, 0.5],
        "usefulness_norm": [0.8, 0.4, 0.5],
        "representativeness": [0.6, 0.9, 0.5],
        "representativeness_norm": [0.6, 0.9, 0.5],
        "aspect_categories": [
            ["battery", "performance"],
            "camera",
            "['display']",
        ],
        "aspect_sentiment_groups": [
            "positive",
            "neutral",
            "positive",
        ],
        "supported_minority_groups": [
            None, None, None
        ],
        "supports_minority": [False, True, False],
        "is_selected": [True, "False", 1],
        "refined_selection_order": [1, None, 2],
        "minority_reserve_locked": [False, False, False],
        "outcome_status": [
            "selected",
            "not_selected",
            "selected",
        ],
    })


def test_c2_adapter_creates_output():
    result = adapt_c2_reviews(sample_c2())

    assert "product_id" in result.columns
    assert "aspect" in result.columns
    assert "evidence_quality" in result.columns


def test_c2_product_id_mapping():
    result = adapt_c2_reviews(sample_c2())

    assert result.iloc[0]["product_id"] == "B001"


def test_c2_multiple_aspects_exploded():
    result = adapt_c2_reviews(sample_c2())

    assert len(result) == 4

    aspects = result[
        result["review_id"] == "R001"
    ]["aspect"].tolist()

    assert aspects == ["battery", "performance"]


def test_c2_evidence_quality_calculation():
    result = adapt_c2_reviews(sample_c2())

    value = result.iloc[0]["evidence_quality"]

    assert value == pytest.approx(0.7)


def test_c2_selection_flag_preserved():
    result = adapt_c2_reviews(sample_c2())

    selected = result[result["is_selected"]]

    assert len(selected) == 3
    assert not result[
        result["review_id"] == "R002"
    ]["is_selected"].iloc[0]


def test_c2_unselected_evidence_preserved():
    result = adapt_c2_reviews(sample_c2())

    assert "R002" in result["review_id"].values


def test_c2_missing_column_fails():
    data = sample_c2().drop(
        columns=["usefulness_norm"]
    )

    with pytest.raises(C2AdapterError):
        adapt_c2_reviews(data)


def test_c2_invalid_quality_range_fails():
    data = sample_c2()
    data.loc[0, "usefulness_norm"] = 1.5

    with pytest.raises(C2AdapterError):
        adapt_c2_reviews(data)


def test_c2_empty_aspect_fails():
    data = sample_c2()
    data.at[0, "aspect_categories"] = []

    with pytest.raises(C2AdapterError):
        adapt_c2_reviews(data)


def test_c2_duplicate_evidence_fails():
    data = sample_c2()

    data = pd.concat(
        [data, data.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(C2AdapterError):
        adapt_c2_reviews(data)


def test_c2_input_not_modified():
    data = sample_c2()
    original = data.copy(deep=True)

    adapt_c2_reviews(data)

    pd.testing.assert_frame_equal(data, original)
