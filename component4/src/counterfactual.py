import numpy as np
import pandas as pd
from scipy.optimize import minimize


def rank_products(X, weights):
    scores = np.dot(
        X.to_numpy(),
        weights
    )

    ranking = pd.DataFrame({
        "product_id": X.index,
        "score": scores
    })

    ranking = (
        ranking
        .sort_values(
            "score",
            ascending=False
        )
        .reset_index(drop=True)
    )

    ranking["rank"] = np.arange(
        1,
        len(ranking) + 1
    )

    return ranking


def find_standard_counterfactual(
    X,
    original_weights,
    target_product,
    margin=0.0001
):
    target_vector = (
        X.loc[target_product]
        .to_numpy()
    )

    def objective(new_weights):
        return np.sum(
            (
                new_weights
                - original_weights
            ) ** 2
        )

    constraints = [
        {
            "type": "eq",
            "fun": lambda w:
                np.sum(w) - 1.0
        }
    ]

    for competitor_id in X.index:

        if competitor_id == target_product:
            continue

        competitor_vector = (
            X.loc[competitor_id]
            .to_numpy()
        )

        constraints.append({
            "type": "ineq",
            "fun": lambda w,
            t=target_vector,
            c=competitor_vector:
                np.dot(t - c, w) - margin
        })

    result = minimize(
        objective,
        x0=original_weights,
        method="SLSQP",
        bounds=[
            (0.0, 1.0)
            for _ in original_weights
        ],
        constraints=constraints,
        options={
            "maxiter": 2000,
            "ftol": 1e-12
        }
    )

    new_weights = result.x

    ranking = rank_products(
        X,
        new_weights
    )

    l1_change = np.sum(
        np.abs(
            new_weights
            - original_weights
        )
    )

    l2_change = np.linalg.norm(
        new_weights
        - original_weights
    )

    return {
        "success": result.success,
        "message": result.message,
        "weights": new_weights,
        "ranking": ranking,
        "l1_change": float(l1_change),
        "l2_change": float(l2_change)
    }


def scenario_target_margins(
    scenario_data,
    weights,
    aspect_order,
    target_product
):
    weight_map = dict(
        zip(
            aspect_order,
            weights
        )
    )

    data = scenario_data.copy()

    data["weight"] = (
        data["aspect"]
        .map(weight_map)
    )

    data["contribution"] = (
        data["adjusted_score"]
        * data["weight"]
    )

    scores = (
        data
        .groupby(
            [
                "simulation_id",
                "product_id"
            ],
            as_index=False
        )["contribution"]
        .sum()
    )

    target_scores = (
        scores[
            scores["product_id"]
            == target_product
        ]
        .set_index("simulation_id")[
            "contribution"
        ]
    )

    competitor_scores = (
        scores[
            scores["product_id"]
            != target_product
        ]
        .groupby("simulation_id")[
            "contribution"
        ]
        .max()
    )

    margins = (
        target_scores
        - competitor_scores
    )

    return margins.to_numpy()


def evaluate_robust_validity(
    scenario_data,
    weights,
    aspect_order,
    target_product
):
    margins = scenario_target_margins(
        scenario_data,
        weights,
        aspect_order,
        target_product
    )

    return float(
        np.mean(margins >= 0)
    )


def find_robust_counterfactual(
    X,
    original_weights,
    standard_cf_weights,
    target_product,
    aspect_order,
    train_scenarios,
    holdout_scenarios=None,
    penalty_values=None,
    robust_target=0.80,
    margin=0.0001
):
    if penalty_values is None:
        penalty_values = [
            1,
            10,
            100,
            1000,
            10000,
            100000
        ]

    target_vector = (
        X.loc[target_product]
        .to_numpy()
    )

    constraints = [
        {
            "type": "eq",
            "fun": lambda w:
                np.sum(w) - 1.0
        }
    ]

    for competitor_id in X.index:

        if competitor_id == target_product:
            continue

        competitor_vector = (
            X.loc[competitor_id]
            .to_numpy()
        )

        constraints.append({
            "type": "ineq",
            "fun": lambda w,
            t=target_vector,
            c=competitor_vector:
                np.dot(t - c, w) - margin
        })

    candidates = []

    for penalty in penalty_values:

        def robust_objective(weights):

            distance = np.sum(
                (
                    weights
                    - original_weights
                ) ** 2
            )

            margins = scenario_target_margins(
                train_scenarios,
                weights,
                aspect_order,
                target_product
            )

            violations = np.maximum(
                0,
                margin - margins
            )

            robustness_penalty = np.mean(
                violations ** 2
            )

            return (
                distance
                + penalty
                * robustness_penalty
            )

        result = minimize(
            robust_objective,
            x0=standard_cf_weights,
            method="SLSQP",
            bounds=[
                (0.0, 1.0)
                for _ in original_weights
            ],
            constraints=constraints,
            options={
                "maxiter": 3000,
                "ftol": 1e-12
            }
        )

        weights = result.x

        train_validity = (
            evaluate_robust_validity(
                train_scenarios,
                weights,
                aspect_order,
                target_product
            )
        )

        if holdout_scenarios is not None:

            holdout_validity = (
                evaluate_robust_validity(
                    holdout_scenarios,
                    weights,
                    aspect_order,
                    target_product
                )
            )

        else:
            holdout_validity = (
                train_validity
            )

        l2_change = np.linalg.norm(
            weights
            - original_weights
        )

        candidates.append({
            "penalty": penalty,
            "success": result.success,
            "weights": weights,
            "train_robust_validity":
                train_validity,
            "holdout_robust_validity":
                holdout_validity,
            "l2_change":
                float(l2_change)
        })

    valid_candidates = [
        candidate
        for candidate in candidates
        if (
            candidate["success"]
            and candidate[
                "holdout_robust_validity"
            ] >= robust_target
        )
    ]

    if valid_candidates:

        best = min(
            valid_candidates,
            key=lambda c:
                c["l2_change"]
        )

        target_reached = True

    else:

        best = max(
            candidates,
            key=lambda c: (
                c[
                    "holdout_robust_validity"
                ],
                -c["l2_change"]
            )
        )

        target_reached = False

    best["target_reached"] = (
        target_reached
    )

    best["ranking"] = rank_products(
        X,
        best["weights"]
    )

    best["candidates"] = candidates

    return best