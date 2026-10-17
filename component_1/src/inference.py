from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

FINAL_COLUMNS = [
    "review_id","product_id","domain","timestamp","text","canonical_aspect",
    "semantic_group","aspect_probability","aspect_threshold","predicted_sentiment",
    "p_negative","p_neutral","p_positive","sentiment_confidence","sentiment_score",
    "aspect_model_version","sentiment_model_version","taxonomy_version",
    "calibration_version","run_id"
]
INPUT_COLUMNS = ["review_id","product_id","domain","timestamp","text"]

class B4HierarchyGuided(nn.Module):
    def __init__(self, model_name, common_labels, phone_specific_labels,
                 laptop_specific_labels, projection_dim, dropout):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.shared_proj = nn.Sequential(nn.Linear(hidden, projection_dim), nn.Tanh(), nn.Dropout(dropout))
        self.phone_private = nn.Sequential(nn.Linear(hidden, projection_dim), nn.Tanh(), nn.Dropout(dropout))
        self.laptop_private = nn.Sequential(nn.Linear(hidden, projection_dim), nn.Tanh(), nn.Dropout(dropout))
        self.gate = nn.Linear(projection_dim * 2, projection_dim)
        self.common_head = nn.Linear(projection_dim, len(common_labels))
        self.phone_head = nn.Linear(projection_dim, len(phone_specific_labels))
        self.laptop_head = nn.Linear(projection_dim, len(laptop_specific_labels))

    def forward(self, input_ids, attention_mask, domain_id):
        pooled = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        shared = self.shared_proj(pooled)
        phone_private = self.phone_private(pooled)
        laptop_private = self.laptop_private(pooled)
        private = torch.where(domain_id.unsqueeze(1).eq(0), phone_private, laptop_private)
        gate = torch.sigmoid(self.gate(torch.cat([shared, private], dim=1)))
        specific_repr = gate * shared + (1.0 - gate) * private
        common_logits = self.common_head(shared)
        phone_logits = self.phone_head(specific_repr)
        laptop_logits = self.laptop_head(specific_repr)
        phone_logits = torch.where(domain_id.unsqueeze(1).eq(0), phone_logits, torch.full_like(phone_logits, -20.0))
        laptop_logits = torch.where(domain_id.unsqueeze(1).eq(1), laptop_logits, torch.full_like(laptop_logits, -20.0))
        return torch.cat([common_logits, phone_logits, laptop_logits], dim=1)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def stable_softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)

def validate_input(df):
    missing = set(INPUT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing columns: {sorted(missing)}")
    invalid_domains = sorted(set(df["domain"].dropna().astype(str)) - {"phone", "laptop"})
    if invalid_domains:
        raise ValueError(f"Invalid domain values: {invalid_domains}")
    if df["review_id"].isna().any():
        raise ValueError("review_id contains missing values.")
    if df["text"].isna().any():
        raise ValueError("text contains missing values.")

def build_domain_valid_mask(domain, config):
    labels = config["global_labels"]
    valid = np.zeros(len(labels), dtype=bool)
    common_count = len(config["common_labels"])
    phone_count = len(config["phone_specific_labels"])
    valid[:common_count] = True
    if domain == "phone":
        valid[common_count:common_count + phone_count] = True
    else:
        valid[common_count + phone_count:] = True
    return valid

def load_models(project_root, device):
    aspect_dir = project_root / "artifacts" / "aspect_b4"
    sentiment_dir = project_root / "artifacts" / "sentiment"
    aspect_config = load_json(aspect_dir / "b4_inference_config.json")
    sentiment_config = load_json(sentiment_dir / "sentiment_inference_config.json")
    aspect_tokenizer = AutoTokenizer.from_pretrained(aspect_dir / "tokenizer")
    sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_dir / "tokenizer")
    aspect_model = B4HierarchyGuided(
        aspect_config["model_name"], aspect_config["common_labels"],
        aspect_config["phone_specific_labels"], aspect_config["laptop_specific_labels"],
        int(aspect_config["projection_dim"]), float(aspect_config["dropout"])
    )
    aspect_state = torch.load(aspect_dir / "b4_hgss_state_dict.pt", map_location=device, weights_only=True)
    aspect_model.load_state_dict(aspect_state)
    aspect_model.to(device).eval()
    sentiment_to_id = {str(k): int(v) for k, v in sentiment_config["sentiment_to_id"].items()}
    id_to_sentiment = {int(k): v for k, v in sentiment_config["id_to_sentiment"].items()}
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(
        sentiment_config["model_name"], num_labels=3,
        label2id=sentiment_to_id, id2label=id_to_sentiment
    )
    sentiment_state = torch.load(sentiment_dir / "sentiment_model_state_dict.pt", map_location=device, weights_only=True)
    sentiment_model.load_state_dict(sentiment_state)
    sentiment_model.to(device).eval()
    return aspect_model, aspect_tokenizer, aspect_config, sentiment_model, sentiment_tokenizer, sentiment_config

def infer_aspects(row, model, tokenizer, config, device):
    encoded = tokenizer(str(row["text"]), truncation=True, padding=True,
                        max_length=int(config["max_length"]), return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}
    domain_id = torch.tensor([int(config["domain_to_id"][row["domain"]])], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(encoded["input_ids"], encoded["attention_mask"], domain_id)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    valid_mask = build_domain_valid_mask(row["domain"], config)
    threshold = float(config["threshold"])
    detected = []
    for idx, aspect in enumerate(config["global_labels"]):
        if valid_mask[idx] and float(probs[idx]) >= threshold:
            detected.append({
                "canonical_aspect": aspect,
                "semantic_group": config["semantic_group_by_aspect"][aspect],
                "aspect_probability": float(probs[idx]),
                "aspect_threshold": threshold,
            })
    return detected

def infer_sentiment(text, aspect, model, tokenizer, config, device):
    encoded = tokenizer(str(text), f"Aspect: {aspect}", truncation=True, padding=True,
                        max_length=int(config["max_length"]), return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits.cpu().numpy()
    probs = stable_softmax(logits / float(config["temperature"]))[0]
    pred_id = int(np.argmax(probs))
    id_to_sentiment = {int(k): v for k, v in config["id_to_sentiment"].items()}
    p_negative, p_neutral, p_positive = map(float, probs)
    return {
        "predicted_sentiment": id_to_sentiment[pred_id],
        "p_negative": p_negative,
        "p_neutral": p_neutral,
        "p_positive": p_positive,
        "sentiment_confidence": float(np.max(probs)),
        "sentiment_score": p_positive - p_negative,
    }

def validate_export(df):
    missing = set(FINAL_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Export is missing columns: {sorted(missing)}")
    dup = df.duplicated(["review_id", "canonical_aspect"]).sum()
    if dup:
        raise ValueError(f"Duplicate review-aspect rows found: {dup}")
    for col in ["aspect_probability","p_negative","p_neutral","p_positive","sentiment_confidence"]:
        if not df[col].between(0.0, 1.0, inclusive="both").all():
            raise ValueError(f"{col} contains values outside [0, 1].")
    if not df["sentiment_score"].between(-1.0, 1.0, inclusive="both").all():
        raise ValueError("sentiment_score contains values outside [-1, 1].")

def run_inference(input_path, output_path, project_root):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = pd.read_csv(input_path)
    validate_input(batch)
    aspect_model, aspect_tokenizer, aspect_config, sentiment_model, sentiment_tokenizer, sentiment_config = load_models(project_root, device)
    run_id = "c1_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_rows, status_rows = [], []
    for _, row in batch.iterrows():
        detected = infer_aspects(row, aspect_model, aspect_tokenizer, aspect_config, device)
        if not detected:
            status_rows.append({"review_id": row["review_id"], "product_id": row["product_id"],
                                "domain": row["domain"], "status": "no_aspect_detected", "run_id": run_id})
            continue
        for aspect_result in detected:
            sentiment_result = infer_sentiment(row["text"], aspect_result["canonical_aspect"], sentiment_model,
                                               sentiment_tokenizer, sentiment_config, device)
            output_rows.append({
                "review_id": row["review_id"], "product_id": row["product_id"], "domain": row["domain"],
                "timestamp": row["timestamp"], "text": row["text"], **aspect_result, **sentiment_result,
                "aspect_model_version": aspect_config["model_version"],
                "sentiment_model_version": sentiment_config["model_version"],
                "taxonomy_version": aspect_config["taxonomy_version"],
                "calibration_version": sentiment_config["calibration_version"], "run_id": run_id,
            })
    output_df = pd.DataFrame(output_rows, columns=FINAL_COLUMNS)
    if not output_df.empty:
        output_df = (output_df.sort_values(["review_id", "aspect_probability"], ascending=[True, False])
                     .drop_duplicates(["review_id", "canonical_aspect"], keep="first").reset_index(drop=True))
        validate_export(output_df)
    status_df = pd.DataFrame(status_rows, columns=["review_id","product_id","domain","status","run_id"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    status_path = output_path.parent / (output_path.stem + "_status.csv")
    status_df.to_csv(status_path, index=False)
    print(f"Device: {device}")
    print(f"Input reviews: {len(batch)}")
    print(f"Review-aspect predictions: {len(output_df)}")
    print(f"No-aspect reviews: {len(status_df)}")
    print(f"Prediction export: {output_path}")
    print(f"Status export: {status_path}")
    return output_df, status_df

def main():
    parser = argparse.ArgumentParser(description="C1 end-to-end inference and integration export")
    parser.add_argument("--input", required=True, help="CSV with review_id,product_id,domain,timestamp,text")
    parser.add_argument("--output", default="results/inference/c1_review_aspect_predictions.csv")
    parser.add_argument("--project-root", default=None)
    args = parser.parse_args()
    script_path = Path(__file__).resolve()
    project_root = Path(args.project_root).resolve() if args.project_root else script_path.parent.parent
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    run_inference(Path(args.input).resolve(), output_path, project_root)

if __name__ == "__main__":
    main()
