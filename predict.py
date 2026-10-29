"""Step 6b: score validation with a run's best model. Writes data/predictions/validation/<name>.csv.

Not Trainer.predict: with grouping by length on, the library also shuffles the order it scores grants in
(transformers 5.19 `_get_eval_sampler`), so its output does not line up with the grant IDs. This loop keeps the order.
Test predictions are added here at the end, once Step 8 fixes the threshold and output rule (decision 52).

Usage: predict.py run1
"""
import argparse
import json
from pathlib import Path

from load_model import DEVICE                            # first: settings sets the GPU fallback before torch loads
import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from score import pick_threshold, report, save_predictions
from settings import SPLITS_DIR, TOKENIZED_DIR

cli = argparse.ArgumentParser()
cli.add_argument("name")
cli = cli.parse_args()
run_dir = Path("runs") / cli.name

tok = AutoTokenizer.from_pretrained(run_dir / "best")
model = AutoModelForSequenceClassification.from_pretrained(run_dir / "best", dtype=torch.float32).to(DEVICE).eval()

val = load_from_disk(TOKENIZED_DIR)["validation"]
ids = list(load_from_disk(SPLITS_DIR)["validation"]["id"])   # same row order as data/tokenized (checked)
batches = DataLoader(val.select_columns(["input_ids", "attention_mask"]), batch_size=16, shuffle=False,
                     collate_fn=DataCollatorWithPadding(tok))
with torch.no_grad():
    probs = np.concatenate([torch.sigmoid(model(**{k: v.to(DEVICE) for k, v in b.items()}).logits).cpu().numpy()
                            for b in batches])

t = pick_threshold(probs, np.array(val["labels"], dtype=int))
m = report(save_predictions("validation", cli.name, ids, probs >= t, probs))
best = json.loads((run_dir / "run_record.json").read_text())["best_f1_at_target"]
print(f"\nthreshold {t:.3f} | micro-F1 {m['micro_f1']:.3f} here vs {best:.3f} at the training check (should match)")
