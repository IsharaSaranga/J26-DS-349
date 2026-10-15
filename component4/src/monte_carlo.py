import numpy as np
import pandas as pd


def calculate_nominal_evidence(c2):
    """
    Calculate the nominal evidence reliability
    for each product-aspect pair.
    """

    rows = []

    for (product_id, aspect), group in c2.groupby(
        ["product_id", "aspect"]
    ):
        quality = (
            group["evidence_quality"]
            .to_numpy(dtype=float)
        )

        weights = np.clip(
            quality,
            0,
            None
        )

        if weights.sum() == 0:
            reliability = quality.mean()
        else:
            reliability = np.average(
                quality,
                weights=weights
            )

        rows.append({
            "product_id": product_id,
            "aspect": aspect,
            "nominal_evidence_reliability":
                reliability
        })

    return pd.DataFrame(rows)


def create_simulation_rankings(
    simulation_data,
    products,
    max_budget
):
    """
    Convert simulation-level aspect contributions
    into ranked products for each simulation.
    """

    scores = (
        simulation_data
        .groupby(
            ["simulation_id", "product_id"],
            as_index=False
        )
        ["weighted_contribution"]
        .sum()
        .rename(
            columns={
                "weighted_contribution":
                    "personalized_score"
            }
        )
    )

    scores = scores.merge(
        products,
        on="product_id",
        how="left"
    )

    scores = scores[
        scores["price"] <= max_budget
    ].copy()

    scores = scores.sort_values(
        [
            "simulation_id",
            "personalized_score"
        ],
        ascending=[
            True,
            False
        ]
    )

    scores["rank"] = (
        scores
        .groupby("simulation_id")
        .cumcount() + 1
    )

    return scores


def run_preference_only(
    features,
    preference_samples,
    products,
    max_budget
):
    """
    Run simulations where only user-preference
    weights are varied.
    """

    data = features.merge(
        preference_samples[
            [
                "simulation_id",
                "aspect",
                "sampled_weight"
            ]
        ],
        on="aspect",
        how="inner"
    )

    data["weighted_contribution"] = (
        data["nominal_aspect_score"]
        * data["sampled_weight"]
    )

    return create_simulation_rankings(
        data,
        products,
        max_budget
    )


def run_evidence_only(
    features,
    nominal_preferences,
    nominal_evidence,
    evidence_samples,
    products,
    max_budget
):
    """
    Run simulations where user preferences stay fixed
    and review evidence varies.
    """

    data = features.merge(
        nominal_preferences[
            ["aspect", "nominal_weight"]
        ],
        on="aspect",
        how="inner"
    )

    data = data.merge(
        nominal_evidence,
        on=[
            "product_id",
            "aspect"
        ],
        how="inner"
    )

    data = data.merge(
        evidence_samples,
        on=[
            "product_id",
            "aspect"
        ],
        how="inner"
    )

    denominator = (
        data[
            "nominal_evidence_reliability"
        ]
        .replace(0, np.nan)
    )

    data["evidence_factor"] = (
        data["evidence_reliability"]
        / denominator
    ).fillna(1.0)

    data["adjusted_aspect_score"] = np.clip(
        data["nominal_aspect_score"]
        * data["evidence_factor"],
        0,
        1
    )

    data["weighted_contribution"] = (
        data["adjusted_aspect_score"]
        * data["nominal_weight"]
    )

    return create_simulation_rankings(
        data,
        products,
        max_budget
    )


def run_joint(
    features,
    preference_samples,
    nominal_evidence,
    evidence_samples,
    products,
    max_budget
):
    """
    Run simulations where both preference uncertainty
    and evidence uncertainty are active.
    """

    data = features.merge(
        preference_samples[
            [
                "simulation_id",
                "aspect",
                "sampled_weight"
            ]
        ],
        on="aspect",
        how="inner"
    )

    data = data.merge(
        nominal_evidence,
        on=[
            "product_id",
            "aspect"
        ],
        how="inner"
    )

    data = data.merge(
        evidence_samples,
        on=[
            "simulation_id",
            "product_id",
            "aspect"
        ],
        how="inner"
    )

    denominator = (
        data[
            "nominal_evidence_reliability"
        ]
        .replace(0, np.nan)
    )

    data["evidence_factor"] = (
        data["evidence_reliability"]
        / denominator
    ).fillna(1.0)

    data["adjusted_aspect_score"] = np.clip(
        data["nominal_aspect_score"]
        * data["evidence_factor"],
        0,
        1
    )

    data["weighted_contribution"] = (
        data["adjusted_aspect_score"]
        * data["sampled_weight"]
    )

    return create_simulation_rankings(
        data,
        products,
        max_budget
    )