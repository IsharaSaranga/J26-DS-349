import numpy as np
import pandas as pd

from component4.src.evidence_uncertainty import (
    weighted_bootstrap_evidence,
    summarize_evidence_uncertainty,
)


def create_test_evidence():

    return pd.DataFrame({
        "review_id": [
            "R1", "R2", "R3",
            "R4", "R5", "R6"
        ],

        "product_id": [
            "P001", "P001", "P001",
            "P002", "P002", "P002"
        ],

        "aspect": [
            "battery", "battery", "battery",
            "battery", "battery", "battery"
        ],

        "evidence_quality": [
            0.8, 0.8, 0.8,
            0.3, 0.8, 1.0
        ]
    })


def test_bootstrap_row_count():

    evidence = create_test_evidence()

    results = weighted_bootstrap_evidence(
        evidence,
        n_bootstrap=100,
        random_state=42
    )

    # 2 product-aspect groups × 100 bootstrap runs
    assert len(results) == 200


def test_evidence_reliability_range():

    evidence = create_test_evidence()

    results = weighted_bootstrap_evidence(
        evidence,
        n_bootstrap=100,
        random_state=42
    )

    assert (
        results["evidence_reliability"]
        .between(0, 1)
        .all()
    )


def test_reproducibility():

    evidence = create_test_evidence()

    first = weighted_bootstrap_evidence(
        evidence,
        n_bootstrap=50,
        random_state=42
    )

    second = weighted_bootstrap_evidence(
        evidence,
        n_bootstrap=50,
        random_state=42
    )

    assert np.allclose(
        first["evidence_reliability"],
        second["evidence_reliability"]
    )


def test_variable_evidence_has_more_uncertainty():

    controlled_evidence = pd.DataFrame({
        "review_id": [
            "S1", "S2", "S3",
            "V1", "V2", "V3"
        ],

        "product_id": [
            "STABLE", "STABLE", "STABLE",
            "VARIABLE", "VARIABLE", "VARIABLE"
        ],

        "aspect": [
            "battery", "battery", "battery",
            "battery", "battery", "battery"
        ],

        "evidence_quality": [
            0.8, 0.8, 0.8,
            0.3, 0.8, 1.0
        ]
    })

    results = weighted_bootstrap_evidence(
        controlled_evidence,
        n_bootstrap=2000,
        random_state=42
    )

    uncertainty = (
        results
        .groupby("product_id")
        ["evidence_reliability"]
        .std()
    )

    assert (
        uncertainty["VARIABLE"]
        >
        uncertainty["STABLE"]
    )


def test_summary_is_created():

    evidence = create_test_evidence()

    results = weighted_bootstrap_evidence(
        evidence,
        n_bootstrap=100,
        random_state=42
    )

    summary = summarize_evidence_uncertainty(
        results
    )

    assert len(summary) == 2

    assert "mean_reliability" in summary.columns
    assert "evidence_uncertainty" in summary.columns
    assert "ci_lower" in summary.columns
    assert "ci_upper" in summary.columns