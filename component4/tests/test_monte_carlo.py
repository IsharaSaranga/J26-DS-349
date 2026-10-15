import pandas as pd

from component4.src.monte_carlo import (
    calculate_nominal_evidence,
    run_preference_only,
    run_evidence_only,
    run_joint,
)


def create_test_data():

    products = pd.DataFrame({
        "product_id": ["P001", "P002", "P003"],
        "product_name": ["A", "B", "C"],
        "category": [
            "smartphone",
            "smartphone",
            "smartphone"
        ],
        "price": [
            100000,
            120000,
            200000
        ]
    })

    features = pd.DataFrame({
        "product_id": [
            "P001", "P001",
            "P002", "P002",
            "P003", "P003"
        ],
        "aspect": [
            "battery", "camera",
            "battery", "camera",
            "battery", "camera"
        ],
        "nominal_aspect_score": [
            0.9, 0.6,
            0.7, 0.9,
            0.8, 0.8
        ]
    })

    nominal_preferences = pd.DataFrame({
        "aspect": [
            "battery",
            "camera"
        ],
        "nominal_weight": [
            0.6,
            0.4
        ]
    })

    preference_samples = pd.DataFrame({
        "simulation_id": [
            0, 0,
            1, 1,
            2, 2
        ],
        "aspect": [
            "battery", "camera",
            "battery", "camera",
            "battery", "camera"
        ],
        "sampled_weight": [
            0.6, 0.4,
            0.5, 0.5,
            0.7, 0.3
        ]
    })

    c2 = pd.DataFrame({
        "product_id": [
            "P001", "P001",
            "P002", "P002",
            "P003", "P003"
        ],
        "aspect": [
            "battery", "camera",
            "battery", "camera",
            "battery", "camera"
        ],
        "evidence_quality": [
            0.8, 0.7,
            0.9, 0.8,
            0.7, 0.7
        ]
    })

    evidence_samples = pd.DataFrame({
        "simulation_id": [
            0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2, 2
        ],
        "product_id": [
            "P001", "P001",
            "P002", "P002",
            "P003", "P003"
        ] * 3,
        "aspect": [
            "battery", "camera",
            "battery", "camera",
            "battery", "camera"
        ] * 3,
        "evidence_reliability": [
            0.8, 0.7,
            0.9, 0.8,
            0.7, 0.7,

            0.75, 0.65,
            0.85, 0.85,
            0.7, 0.7,

            0.85, 0.75,
            0.95, 0.75,
            0.7, 0.7
        ]
    })

    return (
        products,
        features,
        nominal_preferences,
        preference_samples,
        c2,
        evidence_samples
    )


def test_nominal_evidence_created():

    (
        _,
        _,
        _,
        _,
        c2,
        _
    ) = create_test_data()

    result = calculate_nominal_evidence(c2)

    assert len(result) == 6

    assert (
        result[
            "nominal_evidence_reliability"
        ]
        .between(0, 1)
        .all()
    )


def test_preference_only_simulations():

    (
        products,
        features,
        _,
        preference_samples,
        _,
        _
    ) = create_test_data()

    rankings = run_preference_only(
        features,
        preference_samples,
        products,
        max_budget=150000
    )

    assert (
        rankings["simulation_id"]
        .nunique()
        == 3
    )

    # P003 is over budget
    assert (
        "P003"
        not in rankings["product_id"].values
    )


def test_evidence_only_simulations():

    (
        products,
        features,
        nominal_preferences,
        _,
        c2,
        evidence_samples
    ) = create_test_data()

    nominal_evidence = (
        calculate_nominal_evidence(c2)
    )

    rankings = run_evidence_only(
        features,
        nominal_preferences,
        nominal_evidence,
        evidence_samples,
        products,
        max_budget=150000
    )

    assert (
        rankings["simulation_id"]
        .nunique()
        == 3
    )


def test_joint_simulations():

    (
        products,
        features,
        _,
        preference_samples,
        c2,
        evidence_samples
    ) = create_test_data()

    nominal_evidence = (
        calculate_nominal_evidence(c2)
    )

    rankings = run_joint(
        features,
        preference_samples,
        nominal_evidence,
        evidence_samples,
        products,
        max_budget=150000
    )

    assert (
        rankings["simulation_id"]
        .nunique()
        == 3
    )

    assert (
        rankings.groupby(
            "simulation_id"
        )["rank"].min()
        == 1
    ).all()


def test_one_rank_one_per_simulation():

    (
        products,
        features,
        _,
        preference_samples,
        c2,
        evidence_samples
    ) = create_test_data()

    nominal_evidence = (
        calculate_nominal_evidence(c2)
    )

    rankings = run_joint(
        features,
        preference_samples,
        nominal_evidence,
        evidence_samples,
        products,
        max_budget=150000
    )

    top1 = rankings[
        rankings["rank"] == 1
    ]

    assert (
        top1.groupby(
            "simulation_id"
        ).size()
        == 1
    ).all()