"""Download the HRCS dataset files from Hugging Face into data/raw/ (never edited after download)."""
from huggingface_hub import hf_hub_download

from settings import DATASET, DATASET_REVISION  # revision pinned 2026-10-03, so reruns get identical files

FILES = ["hrcs_training_data.csv", "hrcs_test_data.csv", "README.md"]

for name in FILES:
    path = hf_hub_download(DATASET, name, repo_type="dataset", revision=DATASET_REVISION, local_dir="data/raw")
    print("saved", path)
