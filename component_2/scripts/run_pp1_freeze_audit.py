
from pathlib import Path
from collections import Counter
import hashlib
import json
import traceback

import numpy as np
import pandas as pd


# ============================================================
# MILESTONE 14 — PP1 IMPLEMENTATION FREEZE AUDIT
# Read-only source validation; writes audit reports only.
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

OUTPUT = ROOT / "results" / "pp1_freeze"
OUTPUT.mkdir(parents=True, exist_ok=True)

checks = []
inventory = []


def record(name, passed, details="", failure_level="FAIL"):
    status = "PASS" if passed else failure_level

    checks.append({
        "check": name,
        "status": status,
        "details": str(details),
    })

    print(f"[{status}] {name}")

    if details and not passed:
        print("   ", details)


def validate(name, fn):
    try:
        fn()
    except Exception as exc:
        record(
            name,
            False,
            f"{type(exc).__name__}: {exc}",
        )


def ensure(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def read_csv(path):
    return pd.read_csv(
        ROOT / path,
        dtype={
            "review_id": str,
            "parent_asin": str,
        },
    )


# ============================================================
# 1. REQUIRED IMPLEMENTATION FILES
# ============================================================

notebooks = [
    "01_dataset_inspection.ipynb",
    "02_eda.ipynb",
    "03_usefulness_baseline.ipynb",
    "04_c1_semantic_representation.ipynb",
    "05_supported_minority_selector.ipynb",
    "06_selector_baselines_ablation_sensitivity.ipynb",
    "07_statistical_validation_pp1_evidence.ipynb",
    "08_pp1_evidence_pack.ipynb",
    "09_selector_fidelity_constraint_handling.ipynb",
    "10_laptop_dataset_preparation_eda.ipynb",
    "11_laptop_usefulness_baseline.ipynb",
    "12_laptop_semantic_evidence_selection.ipynb",
    "12b_laptop_selector_fidelity_evaluation.ipynb",
    "13_smartphone_laptop_cross_category_evaluation.ipynb",
]

files = {
    f"notebooks/{name}": "notebook"
    for name in notebooks
}

files.update({
    "data/pilot/smartphone_pilot_cleaned.parquet": "dataset",
    "data/pilot/laptop_pilot_cleaned.parquet": "dataset",

    "results/eda/laptop/laptop_chronological_split.csv":
        "split",

    "results/usefulness/laptop/laptop_ridge_model.joblib":
        "model",
    "results/usefulness/laptop/laptop_model_config.json":
        "model_config",
    "results/usefulness/laptop/laptop_test_metrics.csv":
        "evaluation",
    "results/usefulness/laptop/laptop_all_review_predictions.csv":
        "predictions",

    "results/semantic/laptop/laptop_review_embeddings.npz":
        "embeddings",
    "results/semantic/laptop/laptop_embedding_index.csv":
        "embedding_index",

    "results/selector_evaluation/baseline_comparison_by_product.csv":
        "evaluation",
    "results/selector_refinement/fidelity_selector_comparison.csv":
        "evaluation",
    "results/selector_refinement/selector_refinement_outcomes.csv":
        "evaluation",

    "results/selector/laptop/stage_b/laptop_selected_review_sets.csv":
        "selection",
    "results/selector/laptop/stage_b/laptop_selector_product_metrics.csv":
        "evaluation",
    "results/selector/laptop/stage_b/laptop_selector_summary_metrics.csv":
        "evaluation",
    "results/selector/laptop/stage_b/laptop_fidelity_repair_summary.csv":
        "evaluation",

    "results/cross_category/comparison_tables/harmonized_selector_product_metrics.csv":
        "cross_category",
    "results/cross_category/comparison_tables/cross_category_selector_summary.csv":
        "cross_category",
    "results/cross_category/comparison_tables/cross_category_function1_results.csv":
        "cross_category",
    "results/cross_category/comparison_tables/cross_category_minority_ablation.csv":
        "cross_category",
    "results/cross_category/comparison_tables/cross_category_fidelity_comparison.csv":
        "cross_category",
    "results/cross_category/comparison_tables/cross_category_constraint_summary.csv":
        "cross_category",
    "results/cross_category/comparison_tables/cross_category_bootstrap_intervals.csv":
        "cross_category",

    "results/cross_category/validation/cross_category_evaluation_manifest.json":
        "manifest",
})


def audit_inventory():
    missing = []

    for relative, category in sorted(files.items()):
        path = ROOT / relative
        exists = path.is_file()

        if not exists:
            missing.append(relative)

        size = path.stat().st_size if exists else 0

        # Avoid unnecessarily hashing very large artifacts.
        digest = (
            sha256_file(path)
            if exists and size <= 100 * 1024 * 1024
            else ""
        )

        inventory.append({
            "relative_path": relative,
            "category": category,
            "exists": exists,
            "size_bytes": size,
            "sha256": digest,
        })

    ensure(
        not missing,
        "Missing required files: " + "; ".join(missing),
    )

    record(
        "Required artifact inventory",
        True,
        f"{len(files)} files available",
    )


validate("Required artifact inventory", audit_inventory)


# ============================================================
# 2. NOTEBOOK EXECUTION AUDIT
# ============================================================

def audit_notebooks():
    error_details = []

    for filename in notebooks:
        path = ROOT / "notebooks" / filename

        if not path.exists():
            continue

        notebook = json.loads(
            path.read_text(encoding="utf-8")
        )

        ensure(
            notebook.get("nbformat") == 4,
            f"Invalid notebook format: {filename}",
        )

        for index, cell in enumerate(
            notebook.get("cells", []),
            start=1,
        ):
            if cell.get("cell_type") != "code":
                continue

            for output in cell.get("outputs", []):
                if output.get("output_type") == "error":
                    error_details.append(
                        f"{filename}, cell {index}: "
                        f"{output.get('ename', 'Unknown error')}"
                    )

    if error_details:
        record(
            "Saved notebook error outputs",
            False,
            "; ".join(error_details),
            failure_level="WARN",
        )
    else:
        record(
            "Saved notebook error outputs",
            True,
            "No saved execution errors detected",
        )


validate("Notebook format and outputs", audit_notebooks)


# ============================================================
# 3. DATASET VALIDATION
# ============================================================

def audit_datasets():
    expected = {
        "smartphone": 896,
        "laptop": 997,
    }

    for category, expected_count in expected.items():
        path = (
            ROOT / "data" / "pilot"
            / f"{category}_pilot_cleaned.parquet"
        )

        frame = pd.read_parquet(path)

        ensure(
            len(frame) == expected_count,
            f"{category}: unexpected review count",
        )

        ensure(
            frame["parent_asin"].nunique() == 10,
            f"{category}: unexpected product count",
        )

        ensure(
            frame["review_id"].is_unique,
            f"{category}: duplicate review IDs",
        )

        ensure(
            frame["review_id"].notna().all(),
            f"{category}: missing review IDs",
        )

        ensure(
            np.isfinite(
                frame["log_helpful"].astype(float)
            ).all(),
            f"{category}: invalid usefulness target",
        )

        record(
            f"{category.title()} cleaned dataset",
            True,
            f"{len(frame)} reviews, 10 products",
        )


validate("Cleaned dataset integrity", audit_datasets)


# ============================================================
# 4. LAPTOP SPLIT AND FUNCTION 1
# ============================================================

def audit_laptop_function1():
    reviews = pd.read_parquet(
        ROOT / "data/pilot/laptop_pilot_cleaned.parquet"
    )

    split = read_csv(
        "results/eda/laptop/laptop_chronological_split.csv"
    )

    predictions = read_csv(
        "results/usefulness/laptop/"
        "laptop_all_review_predictions.csv"
    )

    reviews["review_id"] = reviews["review_id"].astype(str)

    ensure(
        set(split["review_id"]) == set(reviews["review_id"]),
        "Chronological split IDs do not match reviews",
    )

    ensure(
        split["review_id"].is_unique,
        "Duplicate IDs in chronological split",
    )

    ensure(
        set(predictions["review_id"]) ==
        set(reviews["review_id"]),
        "Prediction IDs do not match reviews",
    )

    ensure(
        predictions["review_id"].is_unique,
        "Duplicate prediction IDs",
    )

    counts = split["split"].value_counts().to_dict()

    ensure(
        counts.get("train", 0) == 697,
        "Unexpected training count",
    )

    ensure(
        counts.get("validation", 0) == 150,
        "Unexpected validation count",
    )

    ensure(
        counts.get("test", 0) == 150,
        "Unexpected test count",
    )

    ensure(
        np.isfinite(
            predictions["predicted_usefulness"]
            .astype(float)
        ).all(),
        "Non-finite usefulness predictions",
    )

    ensure(
        set(predictions["prediction_provenance"]) == {
            "in_sample_training",
            "validation_used_for_model_selection",
            "held_out_test",
        },
        "Incorrect prediction provenance labels",
    )

    joined = reviews[
        ["review_id", "parent_asin", "review_date"]
    ].copy()

    joined["parent_asin"] = (
        joined["parent_asin"].astype(str)
    )

    joined = joined.merge(
        split[["review_id", "split"]],
        on="review_id",
        how="left",
        validate="one_to_one",
    )

    joined["review_date"] = pd.to_datetime(
        joined["review_date"],
        utc=True,
    )

    for product, group in joined.groupby("parent_asin"):
        tr = group.loc[
            group["split"] == "train", "review_date"
        ]
        va = group.loc[
            group["split"] == "validation", "review_date"
        ]
        te = group.loc[
            group["split"] == "test", "review_date"
        ]

        ensure(
            len(tr) and len(va) and len(te),
            f"{product}: missing split",
        )

        ensure(
            tr.max() <= va.min() <= va.max() <= te.min(),
            f"{product}: chronology invalid",
        )

    record(
        "Laptop Function 1 consistency",
        True,
        "997 predictions; chronological split valid",
    )


validate("Laptop Function 1 integrity", audit_laptop_function1)


# ============================================================
# 5. SEMANTIC EMBEDDING VALIDATION
# ============================================================

def audit_embeddings():
    path = (
        ROOT / "results/semantic/laptop/"
        "laptop_review_embeddings.npz"
    )

    index = read_csv(
        "results/semantic/laptop/"
        "laptop_embedding_index.csv"
    )

    with np.load(path, allow_pickle=False) as saved:
        embeddings = saved["embeddings"]
        review_ids = saved["review_ids"].astype(str)

    ensure(
        embeddings.shape == (997, 384),
        "Unexpected embedding dimensions",
    )

    ensure(
        len(index) == 997,
        "Embedding index size mismatch",
    )

    ensure(
        np.isfinite(embeddings).all(),
        "Invalid embedding values",
    )

    ensure(
        index["review_id"].is_unique,
        "Duplicate embedding review IDs",
    )

    index = index.sort_values("embedding_row")

    ensure(
        np.array_equal(
            index["review_id"].astype(str).to_numpy(),
            review_ids,
        ),
        "Embedding row-to-review ID alignment invalid",
    )

    record(
        "Laptop semantic representation",
        True,
        "997 embeddings; 384 dimensions; IDs aligned",
    )


validate("Semantic embedding integrity", audit_embeddings)


# ============================================================
# 6. LAPTOP SELECTOR AND MINORITY PROTECTION
# ============================================================

def audit_laptop_selector():
    selections = read_csv(
        "results/selector/laptop/stage_b/"
        "laptop_selected_review_sets.csv"
    )

    repair = read_csv(
        "results/selector/laptop/stage_b/"
        "laptop_fidelity_repair_summary.csv"
    )

    ensure(
        len(selections) == 350,
        "Unexpected selected-review row count",
    )

    ensure(
        selections["selector"].nunique() == 7,
        "Unexpected selector variant count",
    )

    ensure(
        selections["parent_asin"].nunique() == 10,
        "Unexpected laptop product count",
    )

    sizes = selections.groupby(
        ["parent_asin", "selector"]
    ).size()

    ensure(
        len(sizes) == 70 and sizes.eq(5).all(),
        "Review-budget constraint failed",
    )

    duplicate_ids = selections.duplicated(
        ["parent_asin", "selector", "review_id"]
    )

    ensure(
        not duplicate_ids.any(),
        "Duplicate selected review IDs",
    )

    ensure(
        selections["c1_source"].eq(
            "synthetic_c1_fixture"
        ).all(),
        "Synthetic C1 provenance is missing",
    )

    ensure(
        len(repair) == 10,
        "Missing fidelity outcomes",
    )

    ensure(
        repair["locked_retention"].eq(1).all(),
        "Protected minority reviews were not retained",
    )

    ensure(
        repair["repair_swaps"].sum() == 3,
        "Unexpected fidelity repair-swap count",
    )

    ensure(
        repair["final_jsd"].le(0.35 + 1e-10).all(),
        "Fidelity threshold not satisfied",
    )

    # Verify actual protected review IDs survive refinement.
    for product, group in selections.groupby("parent_asin"):
        initial = group[
            group["selector"] == "FullProposed"
        ]

        refined = group[
            group["selector"] == "FidelityRefined"
        ]

        locked = set(
            initial.loc[
                initial["minority_reserved_review"]
                .astype(str).str.lower() == "true",
                "review_id",
            ]
        )

        refined_ids = set(refined["review_id"])

        ensure(
            locked.issubset(refined_ids),
            f"{product}: protected review removed",
        )

    record(
        "Laptop Function 2 evidence selection",
        True,
        "7 selectors; 350 records; minority locks retained",
    )


validate("Laptop Function 2 integrity", audit_laptop_selector)


# ============================================================
# 7. CROSS-CATEGORY METRIC RECONCILIATION
# ============================================================

def audit_cross_category():
    base = "results/cross_category/comparison_tables/"

    detail = read_csv(
        base + "harmonized_selector_product_metrics.csv"
    )

    summary = read_csv(
        base + "cross_category_selector_summary.csv"
    )

    outcomes = read_csv(
        base + "cross_category_constraint_summary.csv"
    )

    ensure(
        len(detail) == 140,
        "Expected 140 harmonized product metrics",
    )

    ensure(
        len(summary) == 14,
        "Expected 14 summary rows",
    )

    ensure(
        set(detail["category"]) == {
            "Smartphone", "Laptop"
        },
        "Unexpected categories",
    )

    ensure(
        detail.groupby(
            ["category", "selector"]
        ).size().eq(10).all(),
        "Incomplete product-selector coverage",
    )

    metric_pairs = {
        "mean_usefulness": "mean_usefulness_norm",
        "mean_representativeness":
            "mean_representativeness_norm",
        "mean_aspect_coverage": "aspect_coverage",
        "mean_supported_minority_recall":
            "supported_minority_recall",
        "mean_pairwise_similarity":
            "mean_pairwise_similarity",
        "mean_opinion_jsd":
            "opinion_js_divergence",
    }

    for row in summary.itertuples(index=False):
        selected = detail[
            (detail["category"] == row.category)
            & (detail["selector"] == row.selector)
        ]

        for summary_col, detail_col in metric_pairs.items():
            expected = selected[detail_col].mean()
            actual = getattr(row, summary_col)

            ensure(
                np.isclose(
                    expected,
                    actual,
                    atol=1e-9,
                    equal_nan=True,
                ),
                (
                    f"{row.category}/{row.selector}: "
                    f"{summary_col} mismatch"
                ),
            )

    ensure(
        len(outcomes) == 2,
        "Constraint comparison must include both categories",
    )

    ensure(
        outcomes["feasible_products"].eq(10).all(),
        "Not all products are feasible",
    )

    ensure(
        outcomes["mean_locked_retention"].eq(1).all(),
        "Protected-review retention mismatch",
    )

    record(
        "Cross-category evaluation consistency",
        True,
        "140 metrics and 14 summaries reconciled",
    )


validate(
    "Cross-category evaluation integrity",
    audit_cross_category,
)


# ============================================================
# 8. EVIDENCE MANIFEST VALIDATION
# ============================================================

def audit_manifest():
    manifest_path = (
        ROOT / "results/cross_category/validation/"
        "cross_category_evaluation_manifest.json"
    )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    source_files = manifest.get("source_files", {})

    ensure(
        bool(source_files),
        "Source evidence manifest is empty",
    )

    mismatches = []

    for name, item in source_files.items():
        relative = item["relative_path"]
        path = ROOT / relative

        if not path.exists():
            mismatches.append(f"{name}: missing")
            continue

        actual_hash = sha256_file(path)

        if actual_hash != item["sha256"]:
            mismatches.append(f"{name}: SHA-256 mismatch")

    ensure(
        not mismatches,
        "; ".join(mismatches),
    )

    record(
        "Cross-category evidence provenance",
        True,
        f"{len(source_files)} source hashes verified",
    )


validate("Source evidence integrity", audit_manifest)


# ============================================================
# 9. RESEARCH LIMITATIONS AND FREEZE REPORT
# ============================================================

limitations = [
    "Real Component 1 aspect-sentiment outputs "
    "have not yet been integrated.",

    "Synthetic C1 fixtures are integration-test inputs, "
    "not validated customer minority-opinion labels.",

    "Some Function 1 usefulness scores used for "
    "selector evaluation are in-sample.",

    "Smartphone and laptop Function 1 models use "
    "different feature configurations.",

    "The evaluation uses controlled ten-product pilots "
    "rather than representative category populations.",

    "Prototype fidelity and minority improvements "
    "do not establish statistical generalization.",

    "The audit checks saved artifacts and consistency. "
    "It does not retrain models or independently "
    "reproduce the original data extraction.",
]

limitations_text = (
    "# PP1 Research Limitations and Pending Work\n\n"
    + "\n".join(f"- {item}" for item in limitations)
    + "\n\n## Post-PP1 Work\n\n"
    + "- Integrate actual C1 outputs.\n"
    + "- Implement appropriately held-out or out-of-fold "
      "usefulness scoring for selector evaluation.\n"
    + "- Extend datasets and product-level evaluation.\n"
    + "- Conduct controlled cross-category experiments.\n"
    + "- Complete final statistical and robustness validation.\n"
)

(OUTPUT / "limitations_and_pending_work.md").write_text(
    limitations_text,
    encoding="utf-8",
)


# ============================================================
# 10. SAVE FINAL AUDIT EVIDENCE
# ============================================================

inventory_df = pd.DataFrame(inventory)
checks_df = pd.DataFrame(checks)

inventory_df.to_csv(
    OUTPUT / "artifact_inventory.csv",
    index=False,
)

checks_df.to_csv(
    OUTPUT / "audit_checks.csv",
    index=False,
)

status_counts = Counter(checks_df["status"])

fail_count = status_counts.get("FAIL", 0)
warn_count = status_counts.get("WARN", 0)

if fail_count:
    readiness = "NOT_READY"
elif warn_count:
    readiness = "READY_WITH_LIMITATIONS"
else:
    readiness = "READY_WITH_DOCUMENTED_RESEARCH_LIMITATIONS"

readiness_summary = {
    "milestone": 14,
    "scope": "Component 2 PP1 Implementation Freeze",
    "readiness": readiness,
    "total_checks": int(len(checks_df)),
    "passed": int(status_counts.get("PASS", 0)),
    "warnings": int(warn_count),
    "failed": int(fail_count),
    "research_boundary": (
        "Controlled PP1 prototype; real C1 integration "
        "and unbiased selector evaluation remain pending."
    ),
    "audit_method": (
        "Saved artifact and cross-file consistency audit"
    ),
}

(OUTPUT / "readiness_summary.json").write_text(
    json.dumps(readiness_summary, indent=2),
    encoding="utf-8",
)

report = f"""# Component 2 — PP1 Implementation Freeze Audit

## Readiness

**{readiness}**

## Audit Summary

- Passed checks: {status_counts.get("PASS", 0)}
- Warnings: {warn_count}
- Failed checks: {fail_count}

## Audit Checks

{checks_df.to_markdown(index=False) if False else checks_df.to_string(index=False)}

## Scope

This audit validates available saved implementation artifacts,
review identifiers, semantic representations, selection outputs,
minority-review protection, and cross-category metric consistency.

It does not establish final scientific validity or reproduce
training from scratch.

## Known Limitations

See `limitations_and_pending_work.md`.
"""

(OUTPUT / "pp1_freeze_report.md").write_text(
    report,
    encoding="utf-8",
)

print()
print("=" * 55)
print("MILESTONE 14 — PP1 FREEZE AUDIT")
print("=" * 55)
print("Readiness:", readiness)
print("Passed:", status_counts.get("PASS", 0))
print("Warnings:", warn_count)
print("Failed:", fail_count)
print("Reports:", OUTPUT)
print("=" * 55)

if fail_count:
    raise SystemExit(
        "Freeze audit identified issues requiring review."
    )

print("PP1 implementation freeze audit completed.")
