import numpy as np
import pandas as pd

from component4.src.preference_uncertainty import (
    sample_preferences,
)

def test_sample_count():

    preferences = pd.DataFrame({
        "aspect": ["battery", "camera", "performance"],
        "importance": [5, 3, 4],
        "confidence": [0.9, 0.7, 0.8]
    })

    samples = sample_preferences(
        preferences,
        n_samples=100,
        random_state=42
    )

    assert len(samples) == 300


def test_weights_sum_to_one():

    preferences = pd.DataFrame({
        "aspect": ["battery", "camera", "performance"],
        "importance": [5, 3, 4],
        "confidence": [0.9, 0.7, 0.8]
    })

    samples = sample_preferences(
        preferences,
        n_samples=100,
        random_state=42
    )

    sums = (
        samples
        .groupby("simulation_id")
        ["sampled_weight"]
        .sum()
    )

    assert np.allclose(
        sums.values,
        1.0
    )


def test_reproducibility():

    preferences = pd.DataFrame({
        "aspect": ["battery", "camera", "performance"],
        "importance": [5, 3, 4],
        "confidence": [0.9, 0.7, 0.8]
    })

    a = sample_preferences(
        preferences,
        n_samples=10,
        random_state=42
    )

    b = sample_preferences(
        preferences,
        n_samples=10,
        random_state=42
    )

    assert np.allclose(
        a["sampled_weight"],
        b["sampled_weight"]
    )


def test_lower_confidence_creates_more_variation():

    preferences = pd.DataFrame({
        "aspect": [
            "high_confidence",
            "low_confidence",
            "anchor"
        ],
        "importance": [3, 3, 3],
        "confidence": [0.95, 0.50, 1.00]
    })

    samples = sample_preferences(
        preferences,
        n_samples=2000,
        random_state=42
    )

    stds = (
        samples
        .groupby("aspect")
        ["sampled_weight"]
        .std()
    )

    assert (
        stds["low_confidence"]
        >
        stds["high_confidence"]
    )