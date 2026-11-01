"""Step 6b: score validation with a run's best model. Writes data/predictions/validation/<name>.csv.

Not Trainer.predict: with grouping by length on, the library also shuffles the order it scores grants in
(transformers 5.19 `_get_eval_sampler`), so its output does not line up with the grant IDs. This loop keeps the order.
With --test (at the end only, decision 52), also writes data/predictions/test/<name>.csv, with no score printed. The
threshold is picked on validation, and both files keep every grant to 1–5 categories (decision 62).

Usage: predict.py run1 [--test]
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

from score import metrics, output_rule, pick_threshold, report, save_predictions
from settings import SPLITS_DIR, TOKENIZED_DIR

cli = argparse.ArgumentParser()
cli.add_argument("name")
cli.add_argument("--test", action="store_true")
cli = cli.parse_args()
run_dir = Path("runs") / cli.name

tok = AutoTokenizer.from_pretrained(run_dir / "best")
model = AutoModelForSequenceClassification.from_pretrained(run_dir / "best", dtype=torch.float32).to(DEVICE).eval()


def scores(split):
    """The model's 21 scores per grant for one split, in data order (same row order as data/splits, checked)."""
    batches = DataLoader(load_from_disk(TOKENIZED_DIR)[split].select_columns(["input_ids", "attention_mask"]),
                         batch_size=16, shuffle=False, collate_fn=DataCollatorWithPadding(tok))
    with torch.no_grad():
        return np.concatenate([torch.sigmoid(model(**{k: v.to(DEVICE) for k, v in b.items()}).logits).cpu().numpy()
                               for b in batches])


splits = load_from_disk(SPLITS_DIR)
probs, labels = scores("validation"), np.array(splits["validation"]["label"], dtype=int)
t = pick_threshold(probs, labels)
raw = metrics(probs >= t, labels)["micro_f1"]
m = report(save_predictions("validation", cli.name, splits["validation"]["id"], output_rule(probs, t), probs))
best = json.loads((run_dir / "run_record.json").read_text())["best_f1_at_target"]
print(f"\nthreshold {t:.3f} | micro-F1 {raw:.3f} before the 1–5 rule vs {best:.3f} at the training check (should match) "
      f"| {m['micro_f1']:.3f} with it")
if cli.test:                                    # at the end only (decision 52): saved for scoring together, no score here
    test_probs = scores("test")
    print(f"saved {save_predictions('test', cli.name, splits['test']['id'], output_rule(test_probs, t), test_probs)}")
