from pathlib import Path
import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "mock"
OUTPUT_DIR = BASE_DIR / "outputs"


def load_inputs():
    products = pd.read_csv(DATA_DIR / "products.csv")
    preferences = pd.read_csv(DATA_DIR / "user_preferences.csv")
    c1 = pd.read_csv(DATA_DIR / "c1_product_aspects.csv")
    c3 = pd.read_csv(DATA_DIR / "c3_temporal_features.csv")

    return products, preferences, c1, c3


def build_product_features(c1, c3):
    features = c1.merge(
        c3,
        on=["product_id", "aspect"],
        how="inner"
    )

    features["nominal_aspect_score"] = (
        features["aspect_quality"]
        + features["recent_quality"]
    ) / 2

    return features


def calculate_personalized_scores(
    features,
    preferences,
    products
):
    pref = preferences[
        ["aspect", "importance", "confidence"]
    ].copy()

    pref["importance_weight"] = (
        pref["importance"] / pref["importance"].sum()
    )

    ranking_features = features.merge(
        pref,
        on="aspect",
        how="inner"
    )

    ranking_features["weighted_contribution"] = (
        ranking_features["nominal_aspect_score"]
        * ranking_features["importance_weight"]
    )

    product_scores = (
        ranking_features
        .groupby("product_id", as_index=False)
        ["weighted_contribution"]
        .sum()
        .rename(
            columns={
                "weighted_contribution": "personalized_score"
            }
        )
    )

    product_scores = product_scores.merge(
        products,
        on="product_id",
        how="left"
    )

    return product_scores


def apply_budget_filter(product_scores, preferences):
    max_budget = preferences["max_budget"].iloc[0]

    product_scores = product_scores.copy()

    product_scores["within_budget"] = (
        product_scores["price"] <= max_budget
    )

    return product_scores


def generate_ranking(product_scores):
    ranking = (
        product_scores[
            product_scores["within_budget"]
        ]
        .sort_values(
            "personalized_score",
            ascending=False
        )
        .reset_index(drop=True)
    )

    ranking["rank"] = np.arange(
        1,
        len(ranking) + 1
    )

    return ranking[
        [
            "rank",
            "product_id",
            "product_name",
            "price",
            "personalized_score"
        ]
    ]


def main():
    products, preferences, c1, c3 = load_inputs()

    features = build_product_features(c1, c3)

    scores = calculate_personalized_scores(
        features,
        preferences,
        products
    )

    scores = apply_budget_filter(
        scores,
        preferences
    )

    ranking = generate_ranking(scores)

    OUTPUT_DIR.mkdir(exist_ok=True)

    output_file = OUTPUT_DIR / "baseline_ranking.csv"

    ranking.to_csv(
        output_file,
        index=False
    )

    print("\nPERSONALIZED BASELINE RANKING")
    print(ranking.to_string(index=False))

    print(
        f"\nSaved to: {output_file}"
    )


if __name__ == "__main__":
    main()