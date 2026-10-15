import numpy as np
import pandas as pd

from component4.src.counterfactual import (
    rank_products,
    find_standard_counterfactual,
    evaluate_robust_validity,
    find_robust_counterfactual,
)


def create_test_data():

    X = pd.DataFrame(
        {
            "battery": [
                0.90,
                0.60,
                0.50
            ],

            "camera": [
                0.40,
                0.90,
                0.50
            ],

            "performance": [
                0.50,
                0.70,
                0.80
            ]
        },

        index=[
            "P001",
            "P002",
            "P003"
        ]
    )

    original_weights = np.array([
        0.70,
        0.20,
        0.10
    ])

    return X, original_weights


def create_scenarios(X):

    rows = []

    for simulation_id in range(20):

        for product_id in X.index:

            for aspect in X.columns:

                value = X.loc[
                    product_id,
                    aspect
                ]

                # Small deterministic variation
                variation = (
                    ((simulation_id % 5) - 2)
                    * 0.005
                )

                adjusted = np.clip(
                    value + variation,
                    0,
                    1
                )

                rows.append({
                    "simulation_id":
                        simulation_id,

                    "product_id":
                        product_id,

                    "aspect":
                        aspect,

                    "adjusted_score":
                        adjusted
                })

    return pd.DataFrame(rows)


def test_original_ranking_created():

    X, original_weights = (
        create_test_data()
    )

    ranking = rank_products(
        X,
        original_weights
    )

    assert len(ranking) == 3

    assert ranking.iloc[0][
        "product_id"
    ] == "P001"


def test_standard_counterfactual_valid():

    X, original_weights = (
        create_test_data()
    )

    result = (
        find_standard_counterfactual(
            X,
            original_weights,
            target_product="P002"
        )
    )

    assert result["success"]

    assert np.isclose(
        result["weights"].sum(),
        1.0
    )

    assert (
        result["weights"] >= 0
    ).all()

    assert (
        result["ranking"]
        .iloc[0]["product_id"]
        == "P002"
    )


def test_counterfactual_change_metrics():

    X, original_weights = (
        create_test_data()
    )

    result = (
        find_standard_counterfactual(
            X,
            original_weights,
            target_product="P002"
        )
    )

    assert result["l1_change"] >= 0
    assert result["l2_change"] >= 0


def test_robust_validity_range():

    X, original_weights = (
        create_test_data()
    )

    standard = (
        find_standard_counterfactual(
            X,
            original_weights,
            target_product="P002"
        )
    )

    scenarios = create_scenarios(X)

    validity = (
        evaluate_robust_validity(
            scenarios,
            standard["weights"],
            list(X.columns),
            "P002"
        )
    )

    assert 0 <= validity <= 1


def test_robust_counterfactual_valid():

    X, original_weights = (
        create_test_data()
    )

    standard = (
        find_standard_counterfactual(
            X,
            original_weights,
            target_product="P002"
        )
    )

    train_scenarios = (
        create_scenarios(X)
    )

    result = (
        find_robust_counterfactual(
            X=X,
            original_weights=
                original_weights,

            standard_cf_weights=
                standard["weights"],

            target_product="P002",

            aspect_order=
                list(X.columns),

            train_scenarios=
                train_scenarios,

            holdout_scenarios=
                train_scenarios,

            penalty_values=[
                1,
                10,
                100,
                1000
            ],

            robust_target=0.80
        )
    )

    assert np.isclose(
        result["weights"].sum(),
        1.0
    )

    assert (
        result["weights"] >= 0
    ).all()

    assert (
        result["ranking"]
        .iloc[0]["product_id"]
        == "P002"
    )

    assert (
        0
        <= result[
            "holdout_robust_validity"
        ]
        <= 1
    )