"""Validate final C1 model predictions and export a small C2/C3 shared view.

After successful `python src/inference.py --input data/c1_real_review_input_10.csv`, run:
  python src/export_c1_shared.py --input results/inference/c1_review_aspect_predictions.csv
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

SHARED_COLUMNS = [
    "review_id", "parent_asin", "review_date", "text", "domain",
    "canonical_aspect", "semantic_group", "aspect_probability",
    "predicted_sentiment", "sentiment_score", "sentiment_confidence",
]
REQUIRED = [
    "review_id", "product_id", "domain", "timestamp", "text",
    "canonical_aspect", "semantic_group", "aspect_probability",
    "predicted_sentiment", "sentiment_score", "sentiment_confidence",
    "p_negative", "p_neutral", "p_positive",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("results/inference/c1_shared_output.csv"))
    p.add_argument("--taxonomy-config", type=Path, default=Path("artifacts/aspect_b4/b4_inference_config.json"))
    args = p.parse_args()
    data = pd.read_csv(args.input, dtype={"review_id": "string", "product_id": "string"})
    missing = set(REQUIRED) - set(data.columns)
    if missing:
        raise ValueError(f"Missing inference columns: {sorted(missing)}")
    if data.empty:
        print("No detected aspects: shared output will contain headers only. Check inference status CSV.")
    else:
        assert not data.duplicated(["review_id", "canonical_aspect"]).any(), "Duplicated review-aspect rows"
        assert data["product_id"].notna().all(), "Missing parent ASIN"
        assert data["domain"].isin(["phone", "laptop"]).all(), "Invalid domain"
        assert data["predicted_sentiment"].isin(["negative", "neutral", "positive"]).all(), "Invalid sentiment"
        assert data["semantic_group"].isin(["COMMON", "PHONE_SPECIFIC", "LAPTOP_SPECIFIC"]).all(), "Invalid group"
        assert data["aspect_probability"].between(0, 1).all(), "Invalid aspect probability"
        assert data["sentiment_confidence"].between(0, 1).all(), "Invalid confidence"
        assert data["sentiment_score"].between(-1, 1).all(), "Invalid sentiment score"
        probs = data[["p_negative", "p_neutral", "p_positive"]]
        assert probs.notna().all().all(), "Missing probabilities"
        assert probs.ge(0).all().all() and probs.le(1).all().all(), "Invalid probability"
        assert ((probs.sum(axis=1) - 1.0).abs() < 1e-4).all(), "Probabilities don't sum to one"
        assert ((data["sentiment_score"] - data["p_positive"] + data["p_negative"]).abs() < 1e-4).all(), "Incorrect sentiment score"
        assert ((data["sentiment_confidence"] - probs.max(axis=1)).abs() < 1e-4).all(), "Incorrect sentiment confidence"
        if args.taxonomy_config.is_file():
            conf = json.loads(args.taxonomy_config.read_text(encoding="utf-8"))
            mapper = conf["semantic_group_by_aspect"]
            expected = data["canonical_aspect"].map(mapper)
            assert expected.notna().all(), "Unknown canonical aspect"
            assert expected.eq(data["semantic_group"]).all(), "Wrong semantic group"
            assert ~((data["domain"] == "phone") & (data["semantic_group"] == "LAPTOP_SPECIFIC")).any(), "Laptop aspects in phone"
            assert ~((data["domain"] == "laptop") & (data["semantic_group"] == "PHONE_SPECIFIC")).any(), "Phone aspects in laptop"
    # Input timestamp is Unix MILLISECONDS; C2/C3 review_date is normalized UTC.
    numeric_ms = pd.to_numeric(data["timestamp"], errors="raise")
    data["review_date"] = pd.to_datetime(numeric_ms, unit="ms", utc=True, errors="raise").dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    data = data.rename(columns={"product_id": "parent_asin"})
    shared = data.loc[:, SHARED_COLUMNS].copy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shared.to_csv(args.output, index=False)
    print(f"Shared output saved: {args.output}")
    print(f"Detected aspect rows: {len(shared)}")
    print(f"Unique reviews with aspects: {shared['review_id'].nunique()}")
    print("Validated schema, probabilities, sentiment scores, groups and UTC review dates.")
    print("NOTE: C2 may additionally need per-review aspect list aggregation; that depends on its interface.")

if __name__ == "__main__":
    main()
