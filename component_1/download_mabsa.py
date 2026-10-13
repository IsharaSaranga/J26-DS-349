from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Multilingual-NLP/M-ABSA",
    repo_type="dataset",
    allow_patterns=[
        "phone/en/*",
        "laptop/en/*",
    ],
    local_dir="data/raw/mabsa",
)

print("M-ABSA English Phone and Laptop data downloaded successfully.")