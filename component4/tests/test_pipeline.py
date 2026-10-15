from component4.src.pipeline import (
    run_c4_pipeline,
)


def test_full_pipeline_runs():
    result = run_c4_pipeline(
        n_simulations=50,
        random_state=42,
        counterfactual_target="P002"
    )

    assert result is not None


def test_pipeline_ranking_created():
    result = run_c4_pipeline(
        n_simulations=50,
        random_state=42,
        counterfactual_target="P002"
    )

    ranking = result["ranking"]

    assert len(ranking) > 0

    assert "rank" in ranking.columns
    assert "product_id" in ranking.columns
    assert "personalized_score" in ranking.columns


def test_pipeline_detects_valid_state():
    result = run_c4_pipeline(
        n_simulations=50,
        random_state=42,
        counterfactual_target="P002"
    )

    valid_states = {
        "Stable",
        "Preference-sensitive",
        "Evidence-sensitive",
        "Mixed",
    }

    valid_actions = {
        "Recommend",
        "Clarify",
        "Warn",
        "Combine",
    }

    assert result["state"] in valid_states
    assert result["action"] in valid_actions


def test_pipeline_stability_metrics_valid():
    result = run_c4_pipeline(
        n_simulations=50,
        random_state=42,
        counterfactual_target="P002"
    )

    stability = result["stability"]

    assert (
        stability["top1_stability"]
        .between(0, 1)
        .all()
    )

    assert (
        stability["mean_kendall_tau"]
        .between(-1, 1)
        .all()
    )

    assert (
        stability["mean_rank_entropy"]
        .between(0, 1)
        .all()
    )


def test_pipeline_explanation_created():
    result = run_c4_pipeline(
        n_simulations=50,
        random_state=42,
        counterfactual_target="P002"
    )

    assert isinstance(
        result["explanation"],
        str
    )

    assert len(
        result["explanation"]
    ) > 0


def test_pipeline_counterfactual_created():
    result = run_c4_pipeline(
        n_simulations=50,
        random_state=42,
        counterfactual_target="P002"
    )

    counterfactual = result[
        "counterfactual"
    ]

    assert counterfactual is not None
    assert counterfactual["success"]

    assert (
        counterfactual[
            "ranking"
        ]
        .iloc[0]["product_id"]
        == "P002"
    )