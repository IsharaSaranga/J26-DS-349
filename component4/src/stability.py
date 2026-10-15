import numpy as np
import pandas as pd
from scipy.stats import kendalltau


def top1_stability(
    rankings,
    nominal_winner
):
    """
    Fraction of simulations where the nominal
    top-ranked product remains Rank 1.
    """

    top1 = rankings[
        rankings["rank"] == 1
    ]

    if len(top1) == 0:
        return 0.0

    retained = (
        top1["product_id"]
        == nominal_winner
    ).sum()

    return float(
        retained / len(top1)
    )


def mean_kendall_tau(
    rankings,
    nominal_rank_map
):
    """
    Mean Kendall's Tau between each simulated
    ranking and the nominal ranking.
    """

    product_ids = list(
        nominal_rank_map.keys()
    )

    nominal_ranks = [
        nominal_rank_map[p]
        for p in product_ids
    ]

    taus = []

    for _, group in rankings.groupby(
        "simulation_id"
    ):

        simulated_map = dict(
            zip(
                group["product_id"],
                group["rank"]
            )
        )

        # Require the same candidate set
        if not all(
            p in simulated_map
            for p in product_ids
        ):
            continue

        simulated_ranks = [
            simulated_map[p]
            for p in product_ids
        ]

        tau, _ = kendalltau(
            nominal_ranks,
            simulated_ranks
        )

        if not np.isnan(tau):
            taus.append(tau)

    if not taus:
        return 0.0

    return float(
        np.mean(taus)
    )


def calculate_rank_entropy(
    rankings
):
    """
    Calculate normalized rank entropy for
    each product.

    0 = highly stable rank
    1 = highly variable rank
    """

    n_products = (
        rankings["product_id"]
        .nunique()
    )

    results = []

    for product_id, group in rankings.groupby(
        "product_id"
    ):

        probabilities = (
            group["rank"]
            .value_counts(normalize=True)
            .values
        )

        entropy = -np.sum(
            probabilities
            * np.log(probabilities)
        )

        if n_products > 1:
            normalized_entropy = (
                entropy
                / np.log(n_products)
            )
        else:
            normalized_entropy = 0.0

        results.append({
            "product_id": product_id,
            "rank_entropy":
                float(normalized_entropy)
        })

    return pd.DataFrame(results)


def summarize_stability(
    rankings,
    nominal_winner,
    nominal_rank_map
):
    """
    Return the three main stability metrics.
    """

    top1 = top1_stability(
        rankings,
        nominal_winner
    )

    tau = mean_kendall_tau(
        rankings,
        nominal_rank_map
    )

    entropy_df = calculate_rank_entropy(
        rankings
    )

    mean_entropy = float(
        entropy_df[
            "rank_entropy"
        ].mean()
    )

    return {
        "top1_stability": top1,
        "mean_kendall_tau": tau,
        "mean_rank_entropy":
            mean_entropy
    }