
import pandas as pd
import pytest

from component4.src.integration import (
    IntegrationError,
    integrate_standardized_outputs,
)


def sample_tables():
    c1 = pd.DataFrame({
        "review_id": ["R1", "R1", "R2", "R3"],
        "product_id": ["B1", "B1", "B1", "B2"],
        "aspect": [
            "battery",
            "performance",
            "camera",
            "display",
        ],
        "sentiment_score": [0.8, 0.6, -0.4, 0.7],
        "c1_confidence": [0.9, 0.8, 0.7, 0.8],
    })

    c2 = pd.DataFrame({
        "review_id": ["R1", "R1", "R2", "R4"],
        "product_id": ["B1", "B1", "B1", "B2"],
        "aspect": [
            "battery",
            "performance",
            "camera",
            "display",
        ],
        "evidence_quality": [0.8, 0.9, 0.7, 0.6],
        "is_selected": [True, False, True, True],
    })

    c3 = pd.DataFrame({
        "product_id": ["B1", "B1", "B1", "B2"],
        "aspect": [
            "battery",
            "performance",
            "camera",
            "thermal",
        ],
        "recent_quality": [0.6, 0.7, 0.8, 0.5],
    })

    return c1, c2, c3


def test_review_join_preserves_sentiment_and_evidence():
    result = integrate_standardized_outputs(
        *sample_tables()
    )

    evidence = result["review_evidence"]

    assert len(evidence) == 3

    battery = evidence.query(
        "review_id == 'R1' and aspect == 'battery'"
    ).iloc[0]

    assert battery["sentiment_score"] == pytest.approx(
        0.8
    )
    assert battery["evidence_quality"] == pytest.approx(
        0.8
    )


def test_audit_reports_unmatched_review_rows():
    result = integrate_standardized_outputs(
        *sample_tables()
    )

    audit = result["audit"]

    assert audit["c1_without_matching_c2"] == 1
    assert audit["c2_without_matching_c1"] == 1
    assert audit["matched_review_aspect_rows"] == 3


def test_temporal_aspect_alignment_audited():
    result = integrate_standardized_outputs(
        *sample_tables()
    )

    audit = result["audit"]

    assert audit[
        "matched_product_aspects_with_c3"
    ] == 3

    assert audit[
        "c3_aspects_without_joined_review_evidence"
    ] == 1

    assert "thermal" in result[
        "product_aspects"
    ]["aspect"].values


def test_duplicate_c1_key_fails():
    c1, c2, c3 = sample_tables()

    c1 = pd.concat(
        [c1, c1.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(
        IntegrationError,
        match="duplicate",
    ):
        integrate_standardized_outputs(
            c1, c2, c3
        )


def test_zero_review_overlap_fails():
    c1, c2, c3 = sample_tables()

    c2["review_id"] = [
        "X1",
        "X1",
        "X2",
        "X4",
    ]

    with pytest.raises(
        IntegrationError,
        match="No review",
    ):
        integrate_standardized_outputs(
            c1, c2, c3
        )


def test_inputs_not_modified():
    originals = sample_tables()

    copies = [
        df.copy(deep=True)
        for df in originals
    ]

    integrate_standardized_outputs(
        *originals
    )

    for original, copy in zip(
        originals,
        copies,
    ):
        pd.testing.assert_frame_equal(
            original,
            copy,
        )
