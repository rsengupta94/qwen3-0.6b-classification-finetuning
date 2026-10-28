"""Step 6: train the model. Writes runs/<name>/ and data/predictions/validation/<name>.csv.

Applies these decisions: whole model in float32 (41, 42); 4 x 4 = 16 grants per update (43); grants grouped by
length (44); no half-size math (45); 3 passes, learning rate 5e-5 with 10% warm-up, validation scored before training
and every 10% of the run, best and latest checkpoints kept (46); best checkpoint = micro-F1 at the catch-80%
threshold (48).

Usage:
  train.py run1            the real run
  train.py smoke --test    the 5-minute check: 8 updates, checks on 64 validation grants; every part runs once
  train.py run1 --resume   continue run1 from its latest checkpoint
"""
import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

from load_model import load_model                        # first: settings sets the GPU fallback before torch loads
import numpy as np
import torch
import transformers
from datasets import load_from_disk
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from score import metrics, pick_threshold, report, save_predictions
from settings import MAX_LEN, SEED, SPLITS_DIR, TOKENIZED_DIR

cli = argparse.ArgumentParser()
cli.add_argument("name")
cli.add_argument("--test", action="store_true", help="5-minute check of every part")
cli.add_argument("--resume", action="store_true", help="continue from the latest checkpoint")
cli = cli.parse_args()
run_dir = Path("runs") / cli.name
if run_dir.exists() and not cli.resume:
    raise SystemExit(f"{run_dir} already exists. Use --resume to continue it, or choose a new name.")
torch.mps.set_per_process_memory_fraction(1.0)           # stop cleanly at ~30 GB instead of swapping to disk
started = datetime.now()

# ─────────────────────────────────────────────────────────────
# 1. DATA AND MODEL
# ─────────────────────────────────────────────────────────────
data = load_from_disk(TOKENIZED_DIR)
val = data["validation"]
val_ids = list(load_from_disk(SPLITS_DIR)["validation"]["id"])  # same row order as data/tokenized (checked)
check_set = val.select(range(64)) if cli.test else val
tok, model = load_model()


def compute_metrics(pred):
    """At each check: the catch-80% threshold and the scores at it (decision 48); F1 at 0.5 for reference."""
    probs = torch.sigmoid(torch.tensor(pred.predictions)).numpy()
    labels = pred.label_ids.astype(int)
    t = pick_threshold(probs, labels)
    m = metrics(probs >= t, labels)
    return {"f1_at_target": m["micro_f1"], "precision": m["precision"], "recall": m["recall"],
            "macro_f1": m["macro_f1"], "threshold": t, "f1_at_half": metrics(probs >= 0.5, labels)["micro_f1"]}


# ─────────────────────────────────────────────────────────────
# 2. TRAIN
# ─────────────────────────────────────────────────────────────
check, log = (4, 1) if cli.test else (0.1, 0.01)        # check every 10% of the run; training loss every 1%
args = TrainingArguments(
    output_dir=str(run_dir),
    per_device_train_batch_size=4, gradient_accumulation_steps=4,    # 16 grants per update
    per_device_eval_batch_size=16,                                    # scoring only, so a bigger group fits
    num_train_epochs=3, max_steps=8 if cli.test else -1,
    learning_rate=5e-5, lr_scheduler_type="linear", warmup_steps=0.1,  # warm-up over the first 10%
    train_sampling_strategy="group_by_length",
    eval_strategy="steps", eval_steps=check, eval_on_start=True,
    save_strategy="steps", save_steps=check, save_total_limit=2,      # keeps the best and the latest
    load_best_model_at_end=True, metric_for_best_model="f1_at_target", greater_is_better=True,
    logging_steps=log, logging_first_step=True,
    seed=SEED, report_to="none", dataloader_pin_memory=False,
)
trainer = Trainer(model=model, args=args, train_dataset=data["train"], eval_dataset=check_set,
                  data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics, processing_class=tok)
trainer.train(resume_from_checkpoint=True if cli.resume else None)

# ─────────────────────────────────────────────────────────────
# 3. SAVE: the best model, its validation predictions, the history and the run record
# ─────────────────────────────────────────────────────────────
tok.model_max_length = MAX_LEN                           # must: the saved tokenizer cuts new text at 512 too
trainer.save_model(str(run_dir / "best"))                # best checkpoint (loaded at the end), config, tokenizer

probs = torch.sigmoid(torch.tensor(trainer.predict(val).predictions)).numpy()
t = pick_threshold(probs, np.array(val["labels"], dtype=int))
report(save_predictions("validation", cli.name, val_ids, probs >= t, probs))

(run_dir / "history.json").write_text(json.dumps(trainer.state.log_history, indent=1))
git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
record = {
    "name": cli.name, "test": cli.test, "resumed": cli.resume,
    "started": started.isoformat(timespec="seconds"), "finished": datetime.now().isoformat(timespec="seconds"),
    "hours": round((datetime.now() - started).total_seconds() / 3600, 2),
    "git_commit": git("rev-parse", "HEAD"), "uncommitted_changes": bool(git("status", "--porcelain")),
    "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
    "best_checkpoint": trainer.state.best_model_checkpoint, "best_f1_at_target": trainer.state.best_metric,
    "validation_threshold": t, "settings": args.to_dict(),
}
(run_dir / "run_record.json").write_text(json.dumps(record, indent=1, default=str))
print(f"\nbest checkpoint: {record['best_checkpoint']} | validation micro-F1 at target {record['best_f1_at_target']:.3f}"
      f" | threshold {t:.3f} | {record['hours']} h")
