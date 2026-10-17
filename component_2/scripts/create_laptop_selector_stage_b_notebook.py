
from pathlib import Path
import json
import textwrap

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = (
    ROOT / "notebooks"
    / "12b_laptop_selector_fidelity_evaluation.ipynb"
)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
cells = []


def md(source):
    source = textwrap.dedent(source).strip() + "\n"
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    })


def code(source):
    source = textwrap.dedent(source).strip() + "\n"
    compile(source, "<notebook-cell>", "exec")
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    })


md("""
# Milestone 12 — Stage B: Laptop Evidence Selection

## Objective

Apply the existing smartphone C2 selection methodology
to the laptop pilot using Stage A artifacts.

The following are retained from the smartphone implementation:

- Within-product usefulness and representativeness normalization.
- Fixed review budget of five.
- Two priority supported-minority groups.
- Two minority-reserved selection slots.
- Greedy usefulness, representation, coverage,
  minority-gain, and redundancy scoring.
- Observed Top-K, Predicted Top-K, MMR, Aspect Coverage,
  and No-Minority baselines.
- JSD-based opinion fidelity and bounded repair.
- Hard protection of reviews selected during minority reservation.

## Research boundary

Aspect–sentiment inputs are synthetic C1 fixtures,
not actual C1 predictions.

Training-review usefulness predictions are in-sample.
This notebook provides integration and preliminary
selector-behaviour evidence, not unbiased final
research performance estimates.
""")


md("## 1. Imports and Paths")

code("""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import time

import numpy as np
import pandas as pd

from scipy.spatial.distance import jensenshannon


def find_root(start):
    for candidate in [start, *start.parents]:
        if (
            (candidate / "data" / "pilot").exists()
            and (candidate / "notebooks").exists()
        ):
            return candidate

    raise FileNotFoundError(
        "Component 2 root not found."
    )


ROOT = find_root(Path.cwd().resolve())

SEMANTIC_DIR = ROOT / "results" / "semantic" / "laptop"
SELECTOR_DIR = ROOT / "results" / "selector" / "laptop"

OUTPUT_DIR = SELECTOR_DIR / "stage_b"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = (
    SELECTOR_DIR
    / "laptop_selector_integration_inputs.csv"
)

FIXTURE_PATH = (
    SELECTOR_DIR
    / "laptop_synthetic_c1_fixture.csv"
)

GROUP_PATH = (
    SELECTOR_DIR
    / "laptop_fixture_minority_groups.csv"
)

EMBEDDING_PATH = (
    SEMANTIC_DIR
    / "laptop_review_embeddings.npz"
)

K = 5
RESERVED_SLOTS = 2
PRIORITY_GROUP_COUNT = 2

MMR_LAMBDA = 0.70

WEIGHTS = {
    "usefulness": 0.30,
    "representativeness": 0.25,
    "coverage": 0.20,
    "minority": 0.25,
    "redundancy": 0.20,
}

FIDELITY_THRESHOLD = 0.35
MAX_REPAIR_SWAPS = 3
MAX_MEAN_UTILITY_DROP = 0.10

print("Stage B output directory:", OUTPUT_DIR)
""")


md("## 2. Load and Validate Stage A Artifacts")

code("""
for path in [
    INPUT_PATH,
    FIXTURE_PATH,
    GROUP_PATH,
    EMBEDDING_PATH,
]:
    assert path.exists(), f"Missing input: {path}"

df = pd.read_csv(
    INPUT_PATH,
    dtype={"review_id": str, "parent_asin": str},
)

fixture = pd.read_csv(
    FIXTURE_PATH,
    dtype={"review_id": str, "parent_asin": str},
)

groups = pd.read_csv(
    GROUP_PATH,
    dtype={"parent_asin": str},
)

saved = np.load(
    EMBEDDING_PATH,
    allow_pickle=False,
)

embeddings = saved["embeddings"]
embedding_ids = saved["review_ids"].astype(str)

assert len(df) == 997
assert df["parent_asin"].nunique() == 10
assert df["review_id"].is_unique
assert embeddings.shape == (997, 384)
assert np.isfinite(embeddings).all()

assert set(df["review_id"]) == set(embedding_ids)
assert set(fixture["review_id"]) == set(df["review_id"])

for row in df.itertuples(index=False):
    assert (
        embedding_ids[int(row.embedding_row)]
        == row.review_id
    )

assert fixture["source"].eq(
    "synthetic_c1_fixture"
).all()

assert set(df["prediction_provenance"]) == {
    "in_sample_training",
    "validation_used_for_model_selection",
    "held_out_test",
}

assert groups["qualified"].notna().all()

print("Reviews:", len(df))
print("Products:", df["parent_asin"].nunique())
print("Embeddings:", embeddings.shape)
print("Stage A joins and provenance: PASS")
""")


md("## 3. Build Selector-Ready Laptop Features")

code("""
def minmax(series):
    values = series.astype(float)
    low, high = values.min(), values.max()

    if np.isclose(low, high):
        return pd.Series(
            np.zeros(len(series)),
            index=series.index,
        )

    return (values - low) / (high - low)


df["usefulness_norm"] = (
    df.groupby("parent_asin")["predicted_usefulness"]
    .transform(minmax)
)

df["representativeness_norm"] = (
    df.groupby("parent_asin")[
        "semantic_representativeness"
    ].transform(minmax)
)

assert df["usefulness_norm"].between(0, 1).all()
assert df["representativeness_norm"].between(
    0, 1
).all()

# Create canonical aspect-sentiment keys.
fixture["group_key"] = (
    fixture["aspect"].astype(str)
    + "::"
    + fixture["sentiment"].astype(str)
)

qualified = groups[
    groups["qualified"].astype(str).str.lower()
    == "true"
].copy()

qualified["group_key"] = (
    qualified["aspect"].astype(str)
    + "::"
    + qualified["sentiment"].astype(str)
)

# Priority ordering follows the smartphone approach:
# lower prevalence, higher support, higher confidence.
qualified = qualified.sort_values(
    [
        "parent_asin",
        "within_aspect_share",
        "support",
        "mean_confidence",
        "group_key",
    ],
    ascending=[True, True, False, False, True],
)

priority_map = {}
supported_map = {}

for product, group in qualified.groupby("parent_asin"):
    supported_map[product] = set(group["group_key"])

    priority_map[product] = set(
        group["group_key"]
        .drop_duplicates()
        .head(PRIORITY_GROUP_COUNT)
    )

for product in df["parent_asin"].unique():
    priority_map.setdefault(product, set())
    supported_map.setdefault(product, set())

review_aspects = defaultdict(set)
review_groups = defaultdict(set)
review_supported = defaultdict(set)

for row in fixture.itertuples(index=False):
    review_aspects[row.review_id].add(row.aspect)
    review_groups[row.review_id].add(row.group_key)

    if row.group_key in supported_map[row.parent_asin]:
        review_supported[row.review_id].add(row.group_key)

product_aspects = {}

for product, group in df.groupby("parent_asin"):
    all_aspects = set()

    for rid in group["review_id"]:
        all_aspects |= review_aspects[rid]

    product_aspects[product] = all_aspects

df = df.sort_values(
    ["parent_asin", "review_id"]
).reset_index(drop=True)

print("Qualified fixture groups:", len(qualified))
print(
    "Products with priority minority groups:",
    sum(bool(v) for v in priority_map.values()),
)
""")


md("## 4. Common Selection Functions")

code("""
product_frames = {
    product: group.reset_index(drop=True)
    for product, group in df.groupby("parent_asin")
}

review_row_lookup = {
    row.review_id: row
    for row in df.itertuples(index=False)
}


def redundancy(candidate_id, selected_ids):
    if not selected_ids:
        return 0.0

    i = int(review_row_lookup[candidate_id].embedding_row)

    indices = [
        int(review_row_lookup[rid].embedding_row)
        for rid in selected_ids
    ]

    similarities = embeddings[i] @ embeddings[indices].T

    return max(0.0, float(similarities.max()))


def greedy_proposed(product, with_minority=True):
    frame = product_frames[product]
    candidates = frame["review_id"].tolist()

    selected = []
    protected = []

    priority = (
        priority_map[product] if with_minority else set()
    )

    weights = dict(WEIGHTS)

    if not with_minority:
        weights["minority"] = 0.0

    covered_aspects = set()
    covered_priority = set()

    all_aspects = product_aspects[product]

    def score(rid):
        row = review_row_lookup[rid]

        new_aspects = (
            review_aspects[rid] - covered_aspects
        )

        coverage_gain = (
            len(new_aspects) / max(len(all_aspects), 1)
        )

        new_priority = (
            review_supported[rid]
            & (priority - covered_priority)
        )

        minority_gain = (
            len(new_priority) / len(priority)
            if priority else 0.0
        )

        value = (
            weights["usefulness"]
            * float(row.usefulness_norm)
            + weights["representativeness"]
            * float(row.representativeness_norm)
            + weights["coverage"] * coverage_gain
            + weights["minority"] * minority_gain
            - weights["redundancy"]
            * redundancy(rid, selected)
        )

        return (
            float(value),
            float(minority_gain),
            float(coverage_gain),
            float(row.usefulness_norm),
        )

    # Phase 1: minority-reserved selection.
    if with_minority:
        for _ in range(min(RESERVED_SLOTS, K)):
            uncovered = priority - covered_priority

            if not uncovered:
                break

            eligible = [
                rid for rid in candidates
                if rid not in selected
                and review_supported[rid] & uncovered
            ]

            if not eligible:
                break

            # Minority gain has priority during reservation.
            chosen = max(
                eligible,
                key=lambda rid: (
                    score(rid)[1],
                    score(rid)[0],
                    score(rid)[2],
                    score(rid)[3],
                ),
            )

            selected.append(chosen)
            protected.append(chosen)

            covered_aspects |= review_aspects[chosen]
            covered_priority |= (
                review_supported[chosen] & priority
            )

    # Phase 2: general greedy evidence selection.
    while len(selected) < min(K, len(candidates)):
        remaining = [
            rid for rid in candidates
            if rid not in selected
        ]

        chosen = max(remaining, key=score)

        selected.append(chosen)
        covered_aspects |= review_aspects[chosen]
        covered_priority |= (
            review_supported[chosen] & priority
        )

    return selected, protected


def topk(product, column):
    frame = product_frames[product]

    return (
        frame.sort_values(
            [column, "review_id"],
            ascending=[False, True],
        )["review_id"]
        .head(K)
        .tolist()
    )


def select_mmr(product):
    frame = product_frames[product]
    candidates = frame["review_id"].tolist()

    selected = []

    while len(selected) < min(K, len(candidates)):
        remaining = [
            rid for rid in candidates
            if rid not in selected
        ]

        def mmr_score(rid):
            row = review_row_lookup[rid]

            return (
                MMR_LAMBDA * float(row.usefulness_norm)
                - (1 - MMR_LAMBDA)
                * redundancy(rid, selected),
                float(row.usefulness_norm),
            )

        selected.append(max(remaining, key=mmr_score))

    return selected


def select_aspect_coverage(product):
    frame = product_frames[product]
    candidates = frame["review_id"].tolist()

    selected = []
    covered = set()

    while len(selected) < min(K, len(candidates)):
        remaining = [
            rid for rid in candidates
            if rid not in selected
        ]

        def coverage_score(rid):
            return (
                len(review_aspects[rid] - covered),
                float(
                    review_row_lookup[rid].usefulness_norm
                ),
            )

        chosen = max(remaining, key=coverage_score)

        selected.append(chosen)
        covered |= review_aspects[chosen]

    return selected
""")


md("## 5. Run Baselines and Full Proposed Selector")

code("""
METHODS = [
    "ObservedHelpfulTopK",
    "PredictedUsefulnessTopK",
    "HelpfulnessMMR",
    "AspectCoverage",
    "ProposedNoMinority",
    "FullProposed",
]

selections = {}
protected_map = {}

start_time = time.perf_counter()

for product in sorted(product_frames):
    selections[(product, "ObservedHelpfulTopK")] = (
        topk(product, "helpful_vote")
    )

    selections[(product, "PredictedUsefulnessTopK")] = (
        topk(product, "predicted_usefulness")
    )

    selections[(product, "HelpfulnessMMR")] = (
        select_mmr(product)
    )

    selections[(product, "AspectCoverage")] = (
        select_aspect_coverage(product)
    )

    without_minority, _ = greedy_proposed(
        product, with_minority=False
    )

    full, protected = greedy_proposed(
        product, with_minority=True
    )

    selections[(product, "ProposedNoMinority")] = (
        without_minority
    )

    selections[(product, "FullProposed")] = full

    protected_map[product] = set(protected)

assert len(selections) == 10 * len(METHODS)

for (product, method), ids in selections.items():
    assert len(ids) == K
    assert len(set(ids)) == K
    assert set(ids).issubset(
        set(product_frames[product]["review_id"])
    )

print("Selector variants:", len(METHODS))
print("Product-level selections:", len(selections))
print(
    "Initial selection time (seconds):",
    round(time.perf_counter() - start_time, 2),
)
""")


md("## 6. Selection Evaluation and Opinion Fidelity")

code("""
# Cache the fixture opinion distribution for each product.
product_opinions = {}
review_opinions = defaultdict(Counter)

for row in fixture.itertuples(index=False):
    review_opinions[row.review_id][row.group_key] += 1

for product, group in fixture.groupby("parent_asin"):
    full_counts = Counter(group["group_key"])
    product_opinions[product] = full_counts


def opinion_jsd(product, selected_ids):
    full = product_opinions[product]
    selected = Counter()

    for rid in selected_ids:
        selected.update(review_opinions[rid])

    if not full or not selected:
        return np.nan

    keys = sorted(full)

    p = np.array(
        [full[key] for key in keys],
        dtype=float,
    )

    q = np.array(
        [selected.get(key, 0) for key in keys],
        dtype=float,
    )

    p /= p.sum()
    q /= q.sum()

    # Same convention as smartphone notebook 09:
    # squared Jensen-Shannon distance, base 2.
    return float(
        jensenshannon(p, q, base=2.0) ** 2
    )


def supported_recall(product, selected_ids):
    relevant = supported_map[product]

    if not relevant:
        return np.nan

    covered = set()

    for rid in selected_ids:
        covered |= review_supported[rid]

    return len(covered & relevant) / len(relevant)


def priority_coverage(product, selected_ids):
    priority = priority_map[product]

    if not priority:
        return np.nan

    covered = set()

    for rid in selected_ids:
        covered |= review_supported[rid]

    return len(covered & priority) / len(priority)


def pairwise_similarity(selected_ids):
    if len(selected_ids) < 2:
        return 0.0

    indices = [
        int(review_row_lookup[rid].embedding_row)
        for rid in selected_ids
    ]

    vectors = embeddings[indices]
    similarity = vectors @ vectors.T

    upper = similarity[
        np.triu_indices(len(indices), k=1)
    ]

    return float(upper.mean())


def mean_utility(selected_ids):
    return float(np.mean([
        0.55 * review_row_lookup[rid].usefulness_norm
        + 0.45 * review_row_lookup[
            rid
        ].representativeness_norm
        for rid in selected_ids
    ]))


def evaluation_row(product, method, ids):
    aspects = set()

    for rid in ids:
        aspects |= review_aspects[rid]

    all_aspects = product_aspects[product]

    return {
        "parent_asin": product,
        "selector": method,
        "selected_count": len(ids),
        "mean_usefulness_norm": float(np.mean([
            review_row_lookup[rid].usefulness_norm
            for rid in ids
        ])),
        "mean_representativeness_norm": float(np.mean([
            review_row_lookup[rid].representativeness_norm
            for rid in ids
        ])),
        "mean_observed_log_helpful": float(np.mean([
            np.log1p(review_row_lookup[rid].helpful_vote)
            for rid in ids
        ])),
        "aspect_coverage": (
            len(aspects) / max(len(all_aspects), 1)
        ),
        "supported_minority_recall": (
            supported_recall(product, ids)
        ),
        "priority_minority_coverage": (
            priority_coverage(product, ids)
        ),
        "mean_pairwise_similarity": (
            pairwise_similarity(ids)
        ),
        "opinion_js_divergence": (
            opinion_jsd(product, ids)
        ),
    }


baseline_metrics = pd.DataFrame([
    evaluation_row(product, method, ids)
    for (product, method), ids in selections.items()
])

display(
    baseline_metrics.groupby("selector").agg({
        "mean_usefulness_norm": "mean",
        "mean_representativeness_norm": "mean",
        "aspect_coverage": "mean",
        "supported_minority_recall": "mean",
        "priority_minority_coverage": "mean",
        "mean_pairwise_similarity": "mean",
        "opinion_js_divergence": "mean",
    }).round(4)
)
""")


md("## 7. Minority-Protected Fidelity Repair")

code("""
def refine_selection(product, baseline_ids):
    selected = list(baseline_ids)
    locked = protected_map[product]

    initial_jsd = opinion_jsd(product, selected)

    if not np.isfinite(initial_jsd):
        return {
            "selected_ids": selected,
            "status": "failed",
            "reason": "opinion_fidelity_not_computable",
            "initial_jsd": initial_jsd,
            "final_jsd": initial_jsd,
            "repair_swaps": 0,
            "trace": [],
        }

    if not locked.issubset(set(selected)):
        raise AssertionError(
            "Minority-reserved reviews missing before repair."
        )

    base_utility = mean_utility(selected)
    current_jsd = initial_jsd
    trace = []

    candidates = sorted(
        product_frames[product]["review_id"].tolist()
    )

    for swap_number in range(1, MAX_REPAIR_SWAPS + 1):
        if current_jsd <= FIDELITY_THRESHOLD:
            break

        best = None

        removable = [
            rid for rid in selected if rid not in locked
        ]

        available = sorted(set(candidates) - set(selected))

        for remove_id in removable:
            for add_id in available:
                trial = list(selected)

                trial[
                    trial.index(remove_id)
                ] = add_id

                if not locked.issubset(set(trial)):
                    continue

                jsd = opinion_jsd(product, trial)

                if (
                    not np.isfinite(jsd)
                    or jsd >= current_jsd - 1e-12
                ):
                    continue

                utility_drop = max(
                    0.0,
                    base_utility - mean_utility(trial),
                )

                if utility_drop > MAX_MEAN_UTILITY_DROP:
                    continue

                recall = supported_recall(product, trial)

                key = (
                    float(jsd),
                    float(utility_drop),
                    -float(recall)
                    if np.isfinite(recall) else 0.0,
                    add_id,
                    remove_id,
                )

                if best is None or key < best["key"]:
                    best = {
                        "key": key,
                        "selected": trial,
                        "jsd": float(jsd),
                        "removed": remove_id,
                        "added": add_id,
                        "utility_drop": float(utility_drop),
                    }

        if best is None:
            break

        trace.append({
            "swap_number": swap_number,
            "removed_review_id": best["removed"],
            "added_review_id": best["added"],
            "jsd_before": current_jsd,
            "jsd_after": best["jsd"],
            "utility_drop": best["utility_drop"],
        })

        selected = best["selected"]
        current_jsd = best["jsd"]

    final_jsd = opinion_jsd(product, selected)

    if not locked.issubset(set(selected)):
        status = "failed"
        reason = "minority_reserve_not_preserved"

    elif final_jsd <= FIDELITY_THRESHOLD:
        status = (
            "feasible"
            if priority_map[product]
            else "no_minority_required"
        )
        reason = "fidelity_satisfied"

    else:
        status = "relaxed"
        reason = "opinion_fidelity_threshold_not_met"

    return {
        "selected_ids": selected,
        "status": status,
        "reason": reason,
        "initial_jsd": float(initial_jsd),
        "final_jsd": float(final_jsd),
        "repair_swaps": len(trace),
        "trace": trace,
    }


repair_results = {
    product: refine_selection(
        product,
        selections[(product, "FullProposed")],
    )
    for product in sorted(product_frames)
}

repair_summary = pd.DataFrame([
    {
        "parent_asin": product,
        "status": result["status"],
        "reason": result["reason"],
        "initial_jsd": result["initial_jsd"],
        "final_jsd": result["final_jsd"],
        "repair_swaps": result["repair_swaps"],
        "locked_reviews": len(protected_map[product]),
        "locked_retention": (
            len(
                protected_map[product]
                & set(result["selected_ids"])
            )
            / len(protected_map[product])
            if protected_map[product]
            else np.nan
        ),
    }
    for product, result in repair_results.items()
])

assert repair_summary["status"].ne("failed").all()

for product, result in repair_results.items():
    selections[(product, "FidelityRefined")] = (
        result["selected_ids"]
    )

display(repair_summary)
""")


md("## 8. Reproducibility and Edge-Case Validation")

code("""
# Deterministic rerun of the core full selector.
for product in sorted(product_frames):
    repeated, repeated_protected = greedy_proposed(
        product, with_minority=True
    )

    assert repeated == selections[
        (product, "FullProposed")
    ]

    assert set(repeated_protected) == protected_map[
        product
    ]

# Deterministic rerun of fidelity repair.
for product in sorted(product_frames):
    repeated = refine_selection(
        product,
        selections[(product, "FullProposed")],
    )

    assert repeated["selected_ids"] == (
        repair_results[product]["selected_ids"]
    )

    assert repeated["status"] == (
        repair_results[product]["status"]
    )

# Explicitly verify set size and review-ID integrity.
for (product, method), ids in selections.items():
    assert len(ids) == K
    assert len(set(ids)) == K

    assert set(ids).issubset(
        set(product_frames[product]["review_id"])
    )

# No protected review may disappear during repair.
for product, result in repair_results.items():
    assert protected_map[product].issubset(
        set(result["selected_ids"])
    )

# If a product cannot meet the threshold, it must
# explicitly report a relaxed outcome.
for product, result in repair_results.items():
    if result["final_jsd"] > FIDELITY_THRESHOLD:
        assert result["status"] == "relaxed"

# Check no-minority scenario on a temporary test product.
test_product = sorted(product_frames)[0]
saved_priority = priority_map[test_product]
saved_supported = supported_map[test_product]

try:
    priority_map[test_product] = set()
    supported_map[test_product] = set()

    no_minority_ids, no_minority_locked = (
        greedy_proposed(
            test_product,
            with_minority=True,
        )
    )

    assert len(no_minority_ids) == K
    assert len(no_minority_locked) == 0
finally:
    priority_map[test_product] = saved_priority
    supported_map[test_product] = saved_supported

print("Deterministic full-selector rerun: PASS")
print("Deterministic fidelity rerun: PASS")
print("Minority-protected retention: PASS")
print("No-minority fixture test: PASS")
print("Explicit fidelity-outcome reporting: PASS")
""")


md("## 9. Save Stage B Evidence")

code("""
all_metrics = pd.concat([
    baseline_metrics,
    pd.DataFrame([
        evaluation_row(
            product,
            "FidelityRefined",
            result["selected_ids"],
        )
        for product, result in repair_results.items()
    ]),
], ignore_index=True)

summary_metrics = (
    all_metrics.groupby("selector", as_index=False)
    .agg(
        products=("parent_asin", "nunique"),
        mean_usefulness_norm=("mean_usefulness_norm", "mean"),
        mean_representativeness_norm=(
            "mean_representativeness_norm", "mean"
        ),
        mean_aspect_coverage=("aspect_coverage", "mean"),
        mean_supported_minority_recall=(
            "supported_minority_recall", "mean"
        ),
        mean_priority_minority_coverage=(
            "priority_minority_coverage", "mean"
        ),
        mean_pairwise_similarity=(
            "mean_pairwise_similarity", "mean"
        ),
        mean_opinion_js_divergence=(
            "opinion_js_divergence", "mean"
        ),
    )
)

selection_rows = []

for (product, method), ids in selections.items():
    for rank, rid in enumerate(ids, start=1):
        row = review_row_lookup[rid]

        selection_rows.append({
            "parent_asin": product,
            "selector": method,
            "rank": rank,
            "review_id": rid,
            "predicted_usefulness": float(
                row.predicted_usefulness
            ),
            "prediction_provenance": (
                row.prediction_provenance
            ),
            "minority_reserved_review": (
                rid in protected_map[product]
                if method in [
                    "FullProposed",
                    "FidelityRefined",
                ]
                else False
            ),
            "c1_source": "synthetic_c1_fixture",
        })

selection_output = pd.DataFrame(selection_rows)

selection_output.to_csv(
    OUTPUT_DIR / "laptop_selected_review_sets.csv",
    index=False,
)

all_metrics.to_csv(
    OUTPUT_DIR / "laptop_selector_product_metrics.csv",
    index=False,
)

summary_metrics.to_csv(
    OUTPUT_DIR / "laptop_selector_summary_metrics.csv",
    index=False,
)

repair_summary.to_csv(
    OUTPUT_DIR / "laptop_fidelity_repair_summary.csv",
    index=False,
)

repair_traces = []

for product, result in repair_results.items():
    for item in result["trace"]:
        repair_traces.append({
            "parent_asin": product,
            **item,
        })

pd.DataFrame(
    repair_traces,
    columns=[
        "parent_asin",
        "swap_number",
        "removed_review_id",
        "added_review_id",
        "jsd_before",
        "jsd_after",
        "utility_drop",
    ],
).to_csv(
    OUTPUT_DIR / "laptop_fidelity_repair_trace.csv",
    index=False,
)

config = {
    "milestone": 12,
    "stage": "B_laptop_selector",
    "products": 10,
    "reviews": 997,
    "budget_k": K,
    "priority_groups": PRIORITY_GROUP_COUNT,
    "reserved_minority_slots": RESERVED_SLOTS,
    "weights": WEIGHTS,
    "mmr_lambda": MMR_LAMBDA,
    "fidelity_threshold": FIDELITY_THRESHOLD,
    "max_repair_swaps": MAX_REPAIR_SWAPS,
    "max_mean_utility_drop": MAX_MEAN_UTILITY_DROP,
    "c1_source": "synthetic_c1_fixture",
    "real_c1_integrated": False,
    "evaluation_status": (
        "prototype_integration_not_independent_research_evidence"
    ),
    "selected_methods": sorted(
        selection_output["selector"].unique().tolist()
    ),
    "prediction_warning": (
        "Training-review scores are in-sample; "
        "validation scores were used in model selection."
    ),
}

(OUTPUT_DIR / "laptop_stage_b_config.json").write_text(
    json.dumps(config, indent=2),
    encoding="utf-8",
)

display(summary_metrics.round(4))

print("Stage B outputs saved:", OUTPUT_DIR)
""")


md("## 10. Final Stage B Validation")

code("""
assert len(selection_output) == 10 * 7 * K
assert selection_output["review_id"].notna().all()

assert selection_output.groupby(
    ["parent_asin", "selector"]
).size().eq(K).all()

assert all_metrics["parent_asin"].nunique() == 10
assert summary_metrics["selector"].nunique() == 7
assert repair_summary["status"].ne("failed").all()

assert all_metrics["opinion_js_divergence"].between(
    0, 1
).all()

assert all_metrics["aspect_coverage"].between(
    0, 1
).all()

assert all_metrics["mean_pairwise_similarity"].between(
    -1, 1
).all()

expected_files = [
    "laptop_selected_review_sets.csv",
    "laptop_selector_product_metrics.csv",
    "laptop_selector_summary_metrics.csv",
    "laptop_fidelity_repair_summary.csv",
    "laptop_fidelity_repair_trace.csv",
    "laptop_stage_b_config.json",
]

assert all(
    (OUTPUT_DIR / name).exists()
    for name in expected_files
)

print()
print("Milestone 12 Stage B validation passed.")
print("Laptop products:", 10)
print("Review candidates:", 997)
print("Selector variants:", 7)
print("Selected reviews per product:", K)
print(
    "Fidelity outcome counts:",
    repair_summary["status"].value_counts().to_dict(),
)
print("Real C1 integrated: False")
print("Evaluation type: Prototype integration")
""")


md("""
## 11. Next Research Steps

Stage B evaluates whether the existing C2 selector can
operate on laptop reviews with the same configuration.

The final evaluation must subsequently address:

- Actual C1 aspect–sentiment outputs.
- Out-of-fold or appropriately held-out usefulness scoring.
- Cross-category smartphone versus laptop comparison.
- Ablation, sensitivity and statistical robustness.
- Explicit feasibility and relaxation reporting.

Fixture-based improvements are not final research claims.
""")


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(
    json.dumps(notebook, indent=1),
    encoding="utf-8",
)

print("Generated:", OUTPUT)
print("Python cell syntax checks passed.")
