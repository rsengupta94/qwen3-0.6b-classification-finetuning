"""The scoring rules every row shares, and a scorer for any predictions file.

  - pick_threshold: the strictest single threshold that still catches 80% of true categories (decisions 32-33, 48)
  - metrics: micro-F1 (the headline), precision, recall, macro-F1
  - output_rule: yes/no answers at the threshold, kept to 1–5 categories per grant (decision 62)
  - save_predictions: the one file format every row writes (test files only at the end, decision 52)

Usage: score.py data/predictions/<split>/<row>.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_from_disk
from sklearn.metrics import precision_recall_fscore_support

from settings import LABELS, MAX_LABELS, PREDICTIONS_DIR, SPLITS_DIR, TARGET_RECALL


def pick_threshold(scores, labels, target=TARGET_RECALL):
    """The strictest single threshold that still catches `target` of all true categories, counted together."""
    true = np.sort(scores[labels == 1])[::-1]                  # the score of every true category, highest first
    k = int(np.ceil(round(target * len(true), 6)))             # how many of them must be caught
    return float(true[k - 1])                                  # yes = score >= threshold


def metrics(pred, labels):
    """Headline and supporting scores for yes/no arrays of shape (grants, categories)."""
    p, r, f1, _ = precision_recall_fscore_support(labels, pred, average="micro", zero_division=0)
    macro_f1 = precision_recall_fscore_support(labels, pred, average="macro", zero_division=0)[2]
    return {"micro_f1": float(f1), "precision": float(p), "recall": float(r), "macro_f1": float(macro_f1)}


def output_rule(scores, threshold):
    """Yes/no answers at the threshold, kept to the 1–5 categories HRCS allows (decision 62): a grant with none gets its
    top-scoring category, and a grant with more than five keeps its top five."""
    pred = scores >= threshold
    for i in np.where(pred.sum(axis=1) == 0)[0]:
        pred[i, scores[i].argmax()] = True
    for i in np.where(pred.sum(axis=1) > MAX_LABELS)[0]:
        pred[i] = False
        pred[i, np.argsort(scores[i])[-MAX_LABELS:]] = True
    return pred


def save_predictions(split, row, ids, pred, scores=None):
    """Write one row's predictions: grant ID, a yes/no column per category, and a score column per category if any."""
    df = pd.DataFrame(np.asarray(pred, dtype=int), columns=LABELS)
    if scores is not None:
        df = pd.concat([df, pd.DataFrame(np.asarray(scores), columns=[f"{l} score" for l in LABELS])], axis=1)
    df.insert(0, "id", list(ids))
    path = Path(PREDICTIONS_DIR) / split / f"{row}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def report(path):
    """Score a predictions file against the true labels of its split (read from the folder name), and print it."""
    path = Path(path)
    split = path.parent.name
    truth = load_from_disk(SPLITS_DIR)[split].to_pandas().set_index("id")
    df = pd.read_csv(path)
    assert len(df) == len(truth) and set(df["id"]) == set(truth.index), f"{path}: must cover every {split} grant once"
    labels = np.array(truth.loc[df["id"], "label"].tolist(), dtype=int)
    pred = df[LABELS].to_numpy(dtype=int)

    m = metrics(pred, labels)
    print(f"\n{path} ({len(df)} grants)")
    print(f"micro-F1 {m['micro_f1']:.3f} | precision {m['precision']:.3f} | recall {m['recall']:.3f} "
          f"(target {TARGET_RECALL:.2f}) | macro-F1 {m['macro_f1']:.3f}")
    p, r, f1, n = precision_recall_fscore_support(labels, pred, average=None, zero_division=0)
    print(f"\n{'category':<38}{'true grants':>12}{'precision':>11}{'recall':>8}{'F1':>7}")
    for i in np.argsort(-n):
        print(f"{LABELS[i]:<38}{n[i]:>12}{p[i]:>11.2f}{r[i]:>8.2f}{f1[i]:>7.2f}")
    counts = pred.sum(axis=1)
    print(f"\ngrants given 0 categories: {(counts == 0).sum()} | more than {MAX_LABELS}: {(counts > MAX_LABELS).sum()}")
    return m


if __name__ == "__main__":
    report(sys.argv[1])
