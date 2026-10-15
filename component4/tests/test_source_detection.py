from component4.src.source_detection import (
    instability_score,
    classify_uncertainty_source,
)


def test_instability_score_range():
    score = instability_score(
        top1_stability=0.8,
        mean_kendall_tau=0.7,
        mean_rank_entropy=0.3
    )

    assert 0 <= score <= 1


def test_stable_state():
    state, action = classify_uncertainty_source(
        preference_instability=0.10,
        evidence_instability=0.15
    )

    assert state == "Stable"
    assert action == "Recommend"


def test_preference_sensitive_state():
    state, action = classify_uncertainty_source(
        preference_instability=0.60,
        evidence_instability=0.20
    )

    assert state == "Preference-sensitive"
    assert action == "Clarify"


def test_evidence_sensitive_state():
    state, action = classify_uncertainty_source(
        preference_instability=0.17,
        evidence_instability=0.52
    )

    assert state == "Evidence-sensitive"
    assert action == "Warn"


def test_mixed_state():
    state, action = classify_uncertainty_source(
        preference_instability=0.55,
        evidence_instability=0.50
    )

    assert state == "Mixed"
    assert action == "Combine"


def test_current_mock_scenario():
    state, action = classify_uncertainty_source(
        preference_instability=0.1749,
        evidence_instability=0.5190
    )

    assert state == "Evidence-sensitive"
    assert action == "Warn"