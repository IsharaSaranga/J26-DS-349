import numpy as np
import pandas as pd


def weighted_bootstrap_evidence(
    evidence_data,
    n_bootstrap=500,
    random_state=42
):
    """
    Estimate review-evidence uncertainty using a weighted bootstrap.

    Reviews with higher evidence_quality receive higher
    resampling probability.
    """

    rng = np.random.default_rng(random_state)

    results = []

    grouped = evidence_data.groupby(
        ["product_id", "aspect"]
    )

    for (product_id, aspect), group in grouped:

        quality = (
            group["evidence_quality"]
            .to_numpy(dtype=float)
        )

        # Prevent negative sampling weights
        weights = np.clip(
            quality,
            0,
            None
        )

        # If all qualities are zero,
        # fall back to equal probabilities
        if weights.sum() == 0:
            probabilities = (
                np.ones(len(weights))
                / len(weights)
            )
        else:
            probabilities = (
                weights / weights.sum()
            )

        for bootstrap_id in range(n_bootstrap):

            sampled_quality = rng.choice(
                quality,
                size=len(quality),
                replace=True,
                p=probabilities
            )

            evidence_reliability = (
                sampled_quality.mean()
            )

            results.append({
                "bootstrap_id": bootstrap_id,
                "product_id": product_id,
                "aspect": aspect,
                "evidence_reliability":
                    evidence_reliability
            })

    return pd.DataFrame(results)


def summarize_evidence_uncertainty(
    bootstrap_results
):
    """
    Summarize bootstrap variability for each
    product-aspect pair.
    """

    summary = (
        bootstrap_results
        .groupby(["product_id", "aspect"])
        ["evidence_reliability"]
        .agg(
            mean_reliability="mean",
            evidence_uncertainty="std",
            min_reliability="min",
            max_reliability="max"
        )
        .reset_index()
    )

    confidence_intervals = (
        bootstrap_results
        .groupby(["product_id", "aspect"])
        ["evidence_reliability"]
        .quantile([0.025, 0.975])
        .unstack()
        .reset_index()
    )

    confidence_intervals.columns = [
        "product_id",
        "aspect",
        "ci_lower",
        "ci_upper"
    ]

    summary = summary.merge(
        confidence_intervals,
        on=["product_id", "aspect"],
        how="left"
    )

    return summary