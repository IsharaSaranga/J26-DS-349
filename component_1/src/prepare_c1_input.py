"""Prepare honest C1 inference inputs from C2 review CSV or Parquet files.

Run from component_1/ root:
  python src/prepare_c1_input.py --phone path/to/smartphone_pilot_cleaned.parquet \
      --laptop path/to/laptop_pilot_cleaned.parquet \
      --output data/c1_real_review_input_full.csv
CSV sources also work. Parquet requires pyarrow/fastparquet.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

NEEDED = ("review_id", "parent_asin", "timestamp", "text")
C1_COLS = ("review_id", "product_id", "domain", "timestamp", "text")


def read_source(path: Path, domain: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {domain} file: {path}")
    if path.suffix.lower() == ".parquet":
        try:
            df = pd.read_parquet(path)
        except ImportError as exc:
            raise RuntimeError("Parquet reading requires: pip install pyarrow") from exc
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path, dtype={"review_id": "string", "parent_asin": "string"})
    else:
        raise ValueError(f"Expected .csv or .parquet: {path}")
    missing = set(NEEDED) - set(df.columns)
    if missing:
        raise ValueError(f"{domain} file missing {sorted(missing)}")

    result = df.loc[:, NEEDED].rename(columns={"parent_asin": "product_id"}).copy()
    result["domain"] = domain
    for field in ("review_id", "product_id", "text"):
        result[field] = result[field].astype("string")
    result["timestamp"] = pd.to_numeric(result["timestamp"], errors="raise").astype("Int64")
    # C2 timestamp is Unix milliseconds, not seconds.
    if result["timestamp"].isna().any() or not result["timestamp"].between(1_000_000_000_000, 4_000_000_000_000).all():
        raise ValueError(f"{domain}: timestamp must be valid-looking Unix milliseconds")
    for field in ("review_id", "product_id", "text"):
        if result[field].isna().any() or result[field].str.strip().eq("").any():
            raise ValueError(f"{domain}: {field} has missing/blank values")
    return result.loc[:, C1_COLS]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phone", type=Path, required=True)
    p.add_argument("--laptop", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("data/c1_real_review_input_full.csv"))
    args = p.parse_args()
    phone = read_source(args.phone, "phone")
    laptop = read_source(args.laptop, "laptop")
    all_reviews = pd.concat([phone, laptop], ignore_index=True)
    if all_reviews.duplicated("review_id").any():
        raise ValueError("Duplicate review_id across/within domains; do not silently deduplicate")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_reviews.to_csv(args.output, index=False)
    print(f"Input saved: {args.output}")
    print(f"Phone={len(phone)}, Laptop={len(laptop)}, Total={len(all_reviews)}")
    print("Original text and Unix-millisecond timestamp preserved; no sentiment labels added.")


if __name__ == "__main__":
    main()
