import numpy as np
import pandas as pd

from component4.src.stability import (
    top1_stability,
    mean_kendall_tau,
    calculate_rank_entropy,
    summarize_stability,
)


def create_test_rankings():

    return pd.DataFrame({
        "simulation_id": [
            0, 0, 0,
            1, 1, 1,
            2, 2, 2
        ],

        "product_id": [
            "P001", "P002", "P003",
            "P001", "P002", "P003",
            "P002", "P001", "P003"
        ],

        "rank": [
            1, 2, 3,
            1, 2, 3,
            1, 2, 3
        ]
    })


def test_top1_stability():

    rankings = create_test_rankings()

    stability = top1_stability(
        rankings,
        nominal_winner="P001"
    )

    assert np.isclose(
        stability,
        2 / 3
    )


def test_kendall_tau_range():

    rankings = create_test_rankings()

    nominal_rank_map = {
        "P001": 1,
        "P002": 2,
        "P003": 3
    }

    tau = mean_kendall_tau(
        rankings,
        nominal_rank_map
    )

    assert -1 <= tau <= 1


def test_rank_entropy_range():

    rankings = create_test_rankings()

    entropy = calculate_rank_entropy(
        rankings
    )

    assert len(entropy) == 3

    assert (
        entropy["rank_entropy"]
        .between(0, 1)
        .all()
    )


def test_stable_ranking_has_zero_entropy():

    rankings = pd.DataFrame({
        "simulation_id": [
            0, 0,
            1, 1,
            2, 2
        ],

        "product_id": [
            "P001", "P002",
            "P001", "P002",
            "P001", "P002"
        ],

        "rank": [
            1, 2,
            1, 2,
            1, 2
        ]
    })

    entropy = calculate_rank_entropy(
        rankings
    )

    assert np.allclose(
        entropy["rank_entropy"],
        0.0
    )


def test_stability_summary():

    rankings = create_test_rankings()

    nominal_rank_map = {
        "P001": 1,
        "P002": 2,
        "P003": 3
    }

    summary = summarize_stability(
        rankings,
        nominal_winner="P001",
        nominal_rank_map=
            nominal_rank_map
    )

    assert (
        0
        <= summary[
            "top1_stability"
        ]
        <= 1
    )

    assert (
        -1
        <= summary[
            "mean_kendall_tau"
        ]
        <= 1
    )

    assert (
        0
        <= summary[
            "mean_rank_entropy"
        ]
        <= 1
    )