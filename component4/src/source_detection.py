import numpy as np


def instability_score(
    top1_stability,
    mean_kendall_tau,
    mean_rank_entropy
):
    """
    Combine three ranking-stability metrics into
    a single instability score between 0 and 1.
    """

    top1_instability = 1 - top1_stability

    # Kendall Tau range: -1 to +1
    # Convert to instability range: 0 to 1
    kendall_instability = (
        1 - mean_kendall_tau
    ) / 2

    entropy_instability = mean_rank_entropy

    return float(
        np.mean([
            top1_instability,
            kendall_instability,
            entropy_instability
        ])
    )


def classify_uncertainty_source(
    preference_instability,
    evidence_instability,
    stable_threshold=0.20,
    source_margin=0.10
):
    """
    Classify the dominant source of ranking instability
    and choose the appropriate system response.
    """

    # Both sources have low impact
    if (
        preference_instability <= stable_threshold
        and evidence_instability <= stable_threshold
    ):
        return "Stable", "Recommend"

    difference = (
        preference_instability
        - evidence_instability
    )

    # Preference uncertainty dominates
    if difference > source_margin:
        return (
            "Preference-sensitive",
            "Clarify"
        )

    # Evidence uncertainty dominates
    if difference < -source_margin:
        return (
            "Evidence-sensitive",
            "Warn"
        )

    # Both sources have similar meaningful impact
    return (
        "Mixed",
        "Combine"
    )