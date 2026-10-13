from component4.src.baseline_ranker import (
    load_inputs,
    build_product_features,
    calculate_personalized_scores,
    apply_budget_filter,
    generate_ranking,
)


def create_ranking():
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

    return ranking, scores, preferences


def test_ranking_is_created():
    ranking, _, _ = create_ranking()

    assert len(ranking) > 0
    assert "personalized_score" in ranking.columns
    assert "rank" in ranking.columns


def test_ranking_sorted_high_to_low():
    ranking, _, _ = create_ranking()

    assert ranking["personalized_score"].is_monotonic_decreasing


def test_budget_filter():
    ranking, _, preferences = create_ranking()

    max_budget = preferences["max_budget"].iloc[0]

    assert (ranking["price"] <= max_budget).all()


def test_over_budget_phone_c_excluded():
    ranking, _, _ = create_ranking()

    assert "P003" not in ranking["product_id"].values


def test_expected_top_product():
    ranking, _, _ = create_ranking()

    assert ranking.iloc[0]["product_id"] == "P005"


def test_rank_numbers_are_correct():
    ranking, _, _ = create_ranking()

    expected_ranks = list(range(1, len(ranking) + 1))

    assert ranking["rank"].tolist() == expected_ranks