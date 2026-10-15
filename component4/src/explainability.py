import numpy as np
import pandas as pd
import shap


def build_aspect_matrix(
    c1,
    c3,
    preferences,
    products
):
    features = c1.merge(
        c3,
        on=["product_id", "aspect"],
        how="inner"
    )

    features["nominal_aspect_score"] = (
        features["aspect_quality"]
        + features["recent_quality"]
    ) / 2

    aspect_order = preferences["aspect"].tolist()

    aspect_matrix = (
        features
        .pivot(
            index="product_id",
            columns="aspect",
            values="nominal_aspect_score"
        )
    )

    aspect_matrix = aspect_matrix[
        aspect_order
    ]

    max_budget = preferences[
        "max_budget"
    ].iloc[0]

    eligible_ids = products[
        products["price"] <= max_budget
    ]["product_id"]

    X = aspect_matrix.loc[
        aspect_matrix.index.isin(
            eligible_ids
        )
    ].copy()

    return X, aspect_order


def get_preference_weights(
    preferences,
    aspect_order
):
    pref = preferences[
        ["aspect", "importance"]
    ].copy()

    pref["importance_weight"] = (
        pref["importance"]
        / pref["importance"].sum()
    )

    weights = (
        pref
        .set_index("aspect")
        .loc[
            aspect_order,
            "importance_weight"
        ]
        .to_numpy()
    )

    return weights


def calculate_personalized_scores(
    X,
    weights
):
    return np.dot(
        np.asarray(X),
        weights
    )


def calculate_shap_explanations(
    X,
    weights
):
    def scoring_function(X_input):
        return np.dot(
            np.asarray(X_input),
            weights
        )

    background = (
        X.mean(axis=0)
        .to_frame()
        .T
    )

    explainer = shap.Explainer(
        scoring_function,
        background
    )

    shap_values = explainer(X)

    return shap_values


def explain_product(
    product_id,
    product_name,
    X,
    weights,
    aspect_order,
    shap_values,
    top_n=3
):
    product_index = (
        list(X.index)
        .index(product_id)
    )

    explanation = pd.DataFrame({
        "aspect": aspect_order,
        "aspect_score":
            X.loc[product_id].values,
        "importance_weight":
            weights,
        "shap_value":
            shap_values.values[
                product_index
            ]
    })

    explanation["absolute_shap"] = (
        explanation[
            "shap_value"
        ].abs()
    )

    explanation = (
        explanation
        .sort_values(
            "absolute_shap",
            ascending=False
        )
    )

    positive = (
        explanation[
            explanation["shap_value"] > 0
        ]
        .head(top_n)
    )

    top_aspects = (
        positive["aspect"].tolist()
    )

    if len(top_aspects) >= 2:
        message = (
            f"{product_name} ranks first mainly because "
            f"its {top_aspects[0]} and "
            f"{top_aspects[1]} features contribute "
            f"positively to the user's priorities."
        )

    elif len(top_aspects) == 1:
        message = (
            f"{product_name} ranks first mainly because "
            f"its {top_aspects[0]} feature contributes "
            f"positively to the user's priorities."
        )

    else:
        message = (
            f"{product_name} ranks first based on "
            f"its overall personalized score."
        )

    return explanation, message