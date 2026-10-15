import numpy as np
import pandas as pd

from component4.src.explainability import (
    get_preference_weights,
    calculate_personalized_scores,
    calculate_shap_explanations,
    explain_product,
)


def create_test_data():

    X = pd.DataFrame(
        {
            "battery": [0.8, 0.9],
            "camera": [0.7, 0.6],
            "performance": [0.75, 0.85]
        },
        index=["P001", "P002"]
    )

    preferences = pd.DataFrame({
        "aspect": [
            "battery",
            "camera",
            "performance"
        ],
        "importance": [5, 3, 4]
    })

    return X, preferences


def test_preference_weights_sum_to_one():

    X, preferences = create_test_data()

    aspect_order = list(X.columns)

    weights = get_preference_weights(
        preferences,
        aspect_order
    )

    assert np.isclose(
        weights.sum(),
        1.0
    )


def test_personalized_scores_created():

    X, preferences = create_test_data()

    weights = get_preference_weights(
        preferences,
        list(X.columns)
    )

    scores = calculate_personalized_scores(
        X,
        weights
    )

    assert len(scores) == 2

    assert np.isfinite(scores).all()


def test_shap_consistency():

    X, preferences = create_test_data()

    aspect_order = list(X.columns)

    weights = get_preference_weights(
        preferences,
        aspect_order
    )

    scores = calculate_personalized_scores(
        X,
        weights
    )

    shap_values = (
        calculate_shap_explanations(
            X,
            weights
        )
    )

    reconstructed = (
        shap_values.base_values
        + shap_values.values.sum(axis=1)
    )

    assert np.allclose(
        scores,
        reconstructed,
        atol=1e-6
    )


def test_explanation_table_created():

    X, preferences = create_test_data()

    aspect_order = list(X.columns)

    weights = get_preference_weights(
        preferences,
        aspect_order
    )

    shap_values = (
        calculate_shap_explanations(
            X,
            weights
        )
    )

    explanation, message = explain_product(
        product_id="P002",
        product_name="Phone B",
        X=X,
        weights=weights,
        aspect_order=aspect_order,
        shap_values=shap_values
    )

    assert len(explanation) == 3

    assert "shap_value" in explanation.columns

    assert isinstance(
        message,
        str
    )

    assert len(message) > 0