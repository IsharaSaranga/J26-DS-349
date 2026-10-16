from pathlib import Path
import pandas as pd

from component4.src.baseline_ranker import (
    build_product_features,
    calculate_personalized_scores,
    apply_budget_filter,
    generate_ranking,
)

from component4.src.preference_uncertainty import (
    sample_preferences,
)

from component4.src.evidence_uncertainty import (
    weighted_bootstrap_evidence,
)

from component4.src.monte_carlo import (
    calculate_nominal_evidence,
    run_preference_only,
    run_evidence_only,
    run_joint,
)

from component4.src.stability import (
    summarize_stability,
)

from component4.src.source_detection import (
    instability_score,
    classify_uncertainty_source,
)

from component4.src.explainability import (
    build_aspect_matrix,
    get_preference_weights,
    calculate_personalized_scores as calculate_explanation_scores,
    calculate_shap_explanations,
    explain_product,
)

from component4.src.counterfactual import (
    find_standard_counterfactual,
)
from component4.src.input_validation import (
    validate_all_inputs,
)


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "mock"


def load_pipeline_inputs():
    products = pd.read_csv(
        DATA_DIR / "products.csv"
    )

    preferences = pd.read_csv(
        DATA_DIR / "user_preferences.csv"
    )

    c1 = pd.read_csv(
        DATA_DIR / "c1_product_aspects.csv"
    )

    c2 = pd.read_csv(
        DATA_DIR / "c2_review_evidence.csv"
    )

    c3 = pd.read_csv(
        DATA_DIR / "c3_temporal_features.csv"
    )

    return (
        products,
        preferences,
        c1,
        c2,
        c3
    )


def run_c4_pipeline(
    n_simulations=500,
    random_state=42,
    counterfactual_target=None
):
    # -------------------------------------------------
    # 1. Load inputs
    # -------------------------------------------------

    (
        products,
        preferences,
        c1,
        c2,
        c3
    ) = load_pipeline_inputs()
    validate_all_inputs(
    products,
    preferences,
    c1,
    c2,
    c3
)

    max_budget = (
        preferences["max_budget"]
        .iloc[0]
    )

    # -------------------------------------------------
    # 2. Baseline personalized ranking
    # -------------------------------------------------

    features = build_product_features(
        c1,
        c3
    )

    baseline_scores = (
        calculate_personalized_scores(
            features,
            preferences,
            products
        )
    )

    baseline_scores = (
        apply_budget_filter(
            baseline_scores,
            preferences
        )
    )

    ranking = generate_ranking(
        baseline_scores
    )

    nominal_winner = (
        ranking.iloc[0]["product_id"]
    )

    nominal_rank_map = dict(
        zip(
            ranking["product_id"],
            ranking["rank"]
        )
    )

    # -------------------------------------------------
    # 3. Preference uncertainty
    # -------------------------------------------------

    preference_samples = (
        sample_preferences(
            preferences,
            n_samples=n_simulations,
            random_state=random_state
        )
    )

    # -------------------------------------------------
    # 4. Evidence uncertainty
    # -------------------------------------------------

    evidence_samples = (
        weighted_bootstrap_evidence(
            c2,
            n_bootstrap=n_simulations,
            random_state=random_state
        )
    )

    evidence_samples = (
        evidence_samples.rename(
            columns={
                "bootstrap_id":
                    "simulation_id"
            }
        )
    )

    nominal_evidence = (
        calculate_nominal_evidence(
            c2
        )
    )

    # -------------------------------------------------
    # 5. Nominal preference weights
    # -------------------------------------------------

    nominal_preferences = (
        preferences[
            [
                "aspect",
                "importance"
            ]
        ]
        .copy()
    )

    nominal_preferences[
        "nominal_weight"
    ] = (
        nominal_preferences[
            "importance"
        ]
        /
        nominal_preferences[
            "importance"
        ].sum()
    )

    # -------------------------------------------------
    # 6. Monte Carlo modes
    # -------------------------------------------------

    preference_rankings = (
        run_preference_only(
            features,
            preference_samples,
            products,
            max_budget
        )
    )

    evidence_rankings = (
        run_evidence_only(
            features,
            nominal_preferences,
            nominal_evidence,
            evidence_samples,
            products,
            max_budget
        )
    )

    joint_rankings = (
        run_joint(
            features,
            preference_samples,
            nominal_evidence,
            evidence_samples,
            products,
            max_budget
        )
    )

    # -------------------------------------------------
    # 7. Ranking stability
    # -------------------------------------------------

    preference_stability = (
        summarize_stability(
            preference_rankings,
            nominal_winner,
            nominal_rank_map
        )
    )

    evidence_stability = (
        summarize_stability(
            evidence_rankings,
            nominal_winner,
            nominal_rank_map
        )
    )

    joint_stability = (
        summarize_stability(
            joint_rankings,
            nominal_winner,
            nominal_rank_map
        )
    )

    # -------------------------------------------------
    # 8. Instability scores
    # -------------------------------------------------

    preference_instability = (
        instability_score(
            preference_stability[
                "top1_stability"
            ],
            preference_stability[
                "mean_kendall_tau"
            ],
            preference_stability[
                "mean_rank_entropy"
            ]
        )
    )

    evidence_instability = (
        instability_score(
            evidence_stability[
                "top1_stability"
            ],
            evidence_stability[
                "mean_kendall_tau"
            ],
            evidence_stability[
                "mean_rank_entropy"
            ]
        )
    )

    joint_instability = (
        instability_score(
            joint_stability[
                "top1_stability"
            ],
            joint_stability[
                "mean_kendall_tau"
            ],
            joint_stability[
                "mean_rank_entropy"
            ]
        )
    )

    # -------------------------------------------------
    # 9. Source-aware decision
    # -------------------------------------------------

    state, action = (
        classify_uncertainty_source(
            preference_instability,
            evidence_instability
        )
    )

    # -------------------------------------------------
    # 10. SHAP explanation
    # -------------------------------------------------

    X, aspect_order = (
        build_aspect_matrix(
            c1,
            c3,
            preferences,
            products
        )
    )

    weights = (
        get_preference_weights(
            preferences,
            aspect_order
        )
    )

    explanation_scores = (
        calculate_explanation_scores(
            X,
            weights
        )
    )

    shap_values = (
        calculate_shap_explanations(
            X,
            weights
        )
    )

    top_product_name = (
        products.loc[
            products["product_id"]
            == nominal_winner,
            "product_name"
        ]
        .iloc[0]
    )

    shap_table, explanation_message = (
        explain_product(
            product_id=nominal_winner,
            product_name=
                top_product_name,
            X=X,
            weights=weights,
            aspect_order=aspect_order,
            shap_values=shap_values
        )
    )

    # -------------------------------------------------
    # 11. Counterfactual
    # -------------------------------------------------

    if counterfactual_target is None:

        if len(ranking) > 1:
            counterfactual_target = (
                ranking.iloc[1][
                    "product_id"
                ]
            )
        else:
            counterfactual_target = None

    counterfactual_result = None

    if (
        counterfactual_target is not None
        and
        counterfactual_target
        in X.index
        and
        counterfactual_target
        != nominal_winner
    ):

        counterfactual_result = (
            find_standard_counterfactual(
                X=X,
                original_weights=weights,
                target_product=
                    counterfactual_target
            )
        )

    # -------------------------------------------------
    # 12. Final C4 output
    # -------------------------------------------------

    stability_table = pd.DataFrame({
        "mode": [
            "Preference-only",
            "Evidence-only",
            "Joint"
        ],

        "top1_stability": [
            preference_stability[
                "top1_stability"
            ],
            evidence_stability[
                "top1_stability"
            ],
            joint_stability[
                "top1_stability"
            ]
        ],

        "mean_kendall_tau": [
            preference_stability[
                "mean_kendall_tau"
            ],
            evidence_stability[
                "mean_kendall_tau"
            ],
            joint_stability[
                "mean_kendall_tau"
            ]
        ],

        "mean_rank_entropy": [
            preference_stability[
                "mean_rank_entropy"
            ],
            evidence_stability[
                "mean_rank_entropy"
            ],
            joint_stability[
                "mean_rank_entropy"
            ]
        ]
    })

    return {
        "ranking": ranking,

        "nominal_winner":
            nominal_winner,

        "top_product_name":
            top_product_name,

        "preference_instability":
            preference_instability,

        "evidence_instability":
            evidence_instability,

        "joint_instability":
            joint_instability,

        "state":
            state,

        "action":
            action,

        "stability":
            stability_table,

        "shap_table":
            shap_table,

        "explanation":
            explanation_message,

        "counterfactual":
            counterfactual_result,

        "counterfactual_target":
            counterfactual_target,

        "n_simulations":
            n_simulations
    }


if __name__ == "__main__":

    result = run_c4_pipeline(
        n_simulations=500,
        random_state=42,
        counterfactual_target="P002"
    )

    print("\nFINAL PERSONALIZED RANKING")
    print(
        result["ranking"]
        .to_string(index=False)
    )

    print(
        "\nTop Product:",
        result["top_product_name"]
    )

    print(
        "Preference Instability:",
        round(
            result[
                "preference_instability"
            ],
            4
        )
    )

    print(
        "Evidence Instability:",
        round(
            result[
                "evidence_instability"
            ],
            4
        )
    )

    print(
        "Joint Instability:",
        round(
            result[
                "joint_instability"
            ],
            4
        )
    )

    print(
        "Detected State:",
        result["state"]
    )

    print(
        "Recommended Action:",
        result["action"]
    )

    print(
        "\nExplanation:",
        result["explanation"]
    )

    if result["counterfactual"] is not None:

        print(
            "\nCounterfactual Target:",
            result[
                "counterfactual_target"
            ]
        )

        print(
            "Counterfactual Success:",
            result[
                "counterfactual"
            ]["success"]
        )