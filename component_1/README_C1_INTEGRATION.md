# C1 → C2/C3/C4 real-review integration (10-row pilot)

## Copy these files into existing `component_1/`

- `data/c1_real_review_input_10.csv`: genuine source reviews: 5 phone, 5 laptop; NO model predictions or fake labels
- `src/prepare_c1_input.py`: optional conversion for full C2 phone and laptop `.csv` or `.parquet`
- `src/export_c1_shared.py`: optional model-output validation + simplified shared view

**Do not replace the existing** `src/inference.py` file. Do not replace trained artifacts.

## Prerequisites

Locally within `component_1/` the previously trained artifacts must exist:

- `artifacts/aspect_b4/b4_hgss_state_dict.pt`
- `artifacts/aspect_b4/b4_inference_config.json`
- `artifacts/aspect_b4/tokenizer/`
- `artifacts/sentiment/sentiment_model_state_dict.pt`
- `artifacts/sentiment/sentiment_inference_config.json`
- `artifacts/sentiment/tokenizer/`

Your existing model-loading code depends on installed `torch` and `transformers` and may download the public `roberta-base` architecture if it is not already cached.

## Run pilot in VS Code / PowerShell (from `component_1/`)

```powershell
python src\inference.py --input data\c1_real_review_input_10.csv
```

Expected output paths (not pre-generated):

- `results/inference/c1_review_aspect_predictions.csv` (full model outputs)
- `results/inference/c1_review_aspect_predictions_status.csv` (reviews with zero detected aspects)

No fixed row count for predictions: 10 reviews may have 0, 1, or several detected aspects each.

Then export a C2/C3-friendly view:

```powershell
python src\export_c1_shared.py --input results\inference\c1_review_aspect_predictions.csv
```

Expected output (only after successful inference):

- `results/inference/c1_shared_output.csv`: review_id, parent_asin, review_date, text, domain, canonical_aspect, semantic_group, aspect_probability, predicted_sentiment, sentiment_score, sentiment_confidence

The full inference CSV retains p_negative, p_neutral, p_positive, version/threshold details for research traceability. The shared CSV does not contain the full probability distributions. `review_date` is derived from original Unix-millisecond timestamps using UTC, not treated as Unix seconds.

## When ready for all C2 pilot reviews

Get the actual file paths used by C2. The following paths are **examples**; they are not assumed to exist here.

```powershell
python src\prepare_c1_input.py --phone "PATH_TO\smartphone_pilot_cleaned.parquet" --laptop "PATH_TO\laptop_pilot_cleaned.parquet" --output data\c1_real_review_input_full.csv
python src\inference.py --input data\c1_real_review_input_full.csv --output results\inference\c1_full_predictions.csv
python src\export_c1_shared.py --input results\inference\c1_full_predictions.csv --output results\inference\c1_full_shared_output.csv
```

For `.parquet`, ensure an appropriate reader is installed (`pip install pyarrow` where permitted). Confirm that the full C2/C3 datasets actually share review_id/parent_asin. Merely having the same schema does not prove a join works.

## Technical limitations

- The sample is *only 10 reviews*. Five laptop reviews relate to the same product, in 2012 Q4: this is not sufficient to validate C3 change points or trends.
- The longest laptop text is over 15,000 characters and the B4 encoder uses 192 tokens. Truncation may omit important aspects; be transparent about that.
- The review text is unchanged. HTML such as `<br />` in source reviews is still present; add a documented preprocessing rule only if you also verify consistency with the model's training preprocessing.
- B4 aspect sigmoid probabilities are **not claimed calibrated**. Only the sentiment probabilities were temperature-scaled (temperature ~1.5113).
- Successful CSV export establishes that the models can be invoked and joined by shared identifiers; it does **not** establish accuracy on the new Amazon-review distribution.
- The post-inference shared table contains one row per detected (review_id, canonical_aspect); C2 may additionally require review-level lists of aspects/sentiment groups. Map that separately against the actual C2 schema.
