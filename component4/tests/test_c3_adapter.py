
import pandas as pd
import pytest

from component4.src.c3_adapter import (
    adapt_c3_temporal,
    C3AdapterError,
)


def sample_c3():
    return pd.DataFrame({
        "parent_asin": ["B001", "B001", "B002"],
        "aspect": ["Battery", "Battery", "Thermal"],
        "quarter": ["2022Q4", "2023Q3", "2023Q2"],
        "review_count": [8, 12, 9],
        "review_perceived_quality_dev": [
            62.0, 36.5, 55.0
        ],
        "quality_ci_lower_dev": [
            58.0, 30.9, 50.0
        ],
        "quality_ci_upper_dev": [
            67.0, 42.4, 60.0
        ],
        "trend": [
            "Stable", "Declining", "Stable"
        ],
        "latest_change_point": [
            None, "2022Q4", None
        ],
        "emerging_status": [0, 2, 1],
        "emerging_label": [
            None,
            "New / rapidly increasing issue",
            "Existing issue",
        ],
        "issue_risk": [
            None, "High", "Medium"
        ],
        "issue_confidence_dev": [
            None, 0.787, 0.628
        ],
    })


def test_c3_adapter_creates_output():
    result = adapt_c3_temporal(sample_c3())

    assert len(result) == 2
    assert "product_id" in result.columns
    assert "recent_quality" in result.columns


def test_c3_quality_normalization():
    result = adapt_c3_temporal(sample_c3())

    battery = result[
        (result["product_id"] == "B001")
        & (result["aspect"] == "battery")
    ].iloc[0]

    assert battery["recent_quality"] == pytest.approx(
        0.365
    )

    assert battery["quality_ci_lower"] == pytest.approx(
        0.309
    )

    assert battery["quality_ci_upper"] == pytest.approx(
        0.424
    )


def test_c3_ci_width():
    result = adapt_c3_temporal(sample_c3())

    battery = result[
        result["aspect"] == "battery"
    ].iloc[0]

    assert battery[
        "quality_ci_width_normalized"
    ] == pytest.approx(0.115)


def test_c3_latest_quarter_selected():
    result = adapt_c3_temporal(sample_c3())

    battery = result[
        result["aspect"] == "battery"
    ].iloc[0]

    assert battery["quarter"] == "2023Q3"


def test_c3_thermal_aspect_preserved():
    result = adapt_c3_temporal(sample_c3())

    assert "thermal" in result["aspect"].values


def test_c3_column_aliases_supported():
    data = sample_c3().rename(
        columns={
            "quarter": "latest_quarter",
            "latest_change_point": "change_point",
            "issue_confidence_dev": "confidence_dev",
        }
    )

    result = adapt_c3_temporal(data)

    assert len(result) == 2


def test_c3_missing_column_fails():
    data = sample_c3().drop(
        columns=["review_perceived_quality_dev"]
    )

    with pytest.raises(C3AdapterError):
        adapt_c3_temporal(data)


def test_c3_invalid_quality_fails():
    data = sample_c3()
    data.loc[0, "review_perceived_quality_dev"] = 150

    with pytest.raises(C3AdapterError):
        adapt_c3_temporal(data)


def test_c3_invalid_ci_fails():
    data = sample_c3()
    data.loc[0, "quality_ci_lower_dev"] = 80
    data.loc[0, "quality_ci_upper_dev"] = 60

    with pytest.raises(C3AdapterError):
        adapt_c3_temporal(data)


def test_c3_invalid_quarter_fails():
    data = sample_c3()
    data.loc[0, "quarter"] = "2023Q5"

    with pytest.raises(C3AdapterError):
        adapt_c3_temporal(data)


def test_c3_duplicate_quarter_fails():
    data = sample_c3()

    data = pd.concat(
        [data, data.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(C3AdapterError):
        adapt_c3_temporal(data)


def test_c3_invalid_status_fails():
    data = sample_c3()
    data.loc[0, "emerging_status"] = 5

    with pytest.raises(C3AdapterError):
        adapt_c3_temporal(data)


def test_c3_input_not_modified():
    data = sample_c3()
    original = data.copy(deep=True)

    adapt_c3_temporal(data)

    pd.testing.assert_frame_equal(data, original)
