import numpy as np
import pandas as pd


def sample_preferences(
    preferences,
    n_samples=500,
    uncertainty_scale=1.0,
    random_state=42
):
    rng = np.random.default_rng(random_state)

    base = preferences.copy()

    base["nominal_weight"] = (
        base["importance"] /
        base["importance"].sum()
    )

    samples = []

    for simulation_id in range(n_samples):

        sampled_raw = []

        for _, row in base.iterrows():

            sigma = (
                uncertainty_scale
                * (1 - row["confidence"])
            )

            noise = rng.normal(
                loc=0.0,
                scale=sigma
            )

            perturbed_weight = (
                row["nominal_weight"]
                * np.exp(noise)
            )

            sampled_raw.append(
                perturbed_weight
            )

        sampled_raw = np.array(sampled_raw)

        sampled_weights = (
            sampled_raw /
            sampled_raw.sum()
        )

        for i, (_, row) in enumerate(base.iterrows()):

            samples.append({
                "simulation_id": simulation_id,
                "aspect": row["aspect"],
                "importance": row["importance"],
                "confidence": row["confidence"],
                "nominal_weight": row["nominal_weight"],
                "sampled_weight": sampled_weights[i]
            })

    return pd.DataFrame(samples)