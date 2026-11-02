"""Step 7: where the fine-tuned model goes wrong. Describes the frozen model (decision 61); changes nothing.

Part 1, test set: how many grants are fully right, which categories get mixed up, grants cut at MAX_LEN tokens
against the rest, results by funder, and F1 per category next to the other rows. Part 2, validation: a fixed random
sample of wrong predictions, printed to read and sort by kind of mistake (the sorting is done by hand).

Usage: error_analysis.py run1
"""
import argparse
import collections

import numpy as np
import pandas as pd
from datasets import load_from_disk
from sklearn.metrics import f1_score

from settings import LABELS, MAX_LEN, PREDICTIONS_DIR, SEED, SPLITS_DIR, TOKENIZED_DIR

# ── Settings used only by this step (shared ones live in settings.py) ──
OTHER_ROWS = ["claude-sonnet-fixed", "claude-haiku-fixed", "keyword"]   # compared per category
MIN_FUNDER_GRANTS = 40                          # smaller funders are left out of the funder table
MIN_CATEGORY_GRANTS = 9                         # no per-category claims at 8 or fewer test grants (section 6)
N_READ = 25                                     # wrong validation predictions to read (playbook: 20–30)

cli = argparse.ArgumentParser()
cli.add_argument("name")
cli = cli.parse_args()
splits = load_from_disk(SPLITS_DIR)


def predictions(split, row, scores=False):
    """One row's yes/no answers (or its scores) for a split, in data order."""
    df = pd.read_csv(f"{PREDICTIONS_DIR}/{split}/{row}.csv").set_index("id").loc[splits[split]["id"]]
    return df[[f"{l} score" for l in LABELS] if scores else LABELS].to_numpy(float if scores else int)


# ─────────────────────────────────────────────────────────────
# 1. PATTERNS ON THE TEST SET
# ─────────────────────────────────────────────────────────────
test = splits["test"].to_pandas()
y, p = np.array(test["label"].tolist(), dtype=int), predictions("test", cli.name)
missed, added = (y == 1) & (p == 0), (p == 1) & (y == 0)

right = ~(missed | added).any(axis=1)
print(f"grants fully right: {right.sum()} of {len(y)} ({right.mean():.0%})")
for name, rows in [("1 true category", y.sum(1) == 1), ("2+ true categories", y.sum(1) >= 2)]:
    print(f"  {name}: {rows.sum()} grants, fully right {right[rows].mean():.0%}, "
          f"micro-F1 {f1_score(y[rows], p[rows], average='micro'):.3f}")
print(f"wrong grants: missed only {(missed.any(1) & ~added.any(1)).sum()} | added only "
      f"{(added.any(1) & ~missed.any(1)).sum()} | missed one and added another {(missed.any(1) & added.any(1)).sum()}")

mixups = collections.Counter((LABELS[a], LABELS[b]) for i in range(len(y))
                             for a in np.where(missed[i])[0] for b in np.where(added[i])[0])
print("\nmix-ups on the same grant (true category missed -> category given instead):")
for (a, b), n in mixups.most_common(12):
    print(f"  {n:>3}  {a} -> {b}")
print("most missed:", [(LABELS[j], int(n)) for j, n in sorted(enumerate(missed.sum(0)), key=lambda x: -x[1])[:6]])
print("most often added wrongly:", [(LABELS[j], int(n)) for j, n in sorted(enumerate(added.sum(0)), key=lambda x: -x[1])[:6]])

cut = np.array([len(t) for t in load_from_disk(TOKENIZED_DIR)["test"]["input_ids"]]) >= MAX_LEN
print(f"\ncut at {MAX_LEN} tokens: {cut.sum()} grants ({cut.mean():.1%}) | micro-F1 {f1_score(y[cut], p[cut], average='micro'):.3f}"
      f" vs {f1_score(y[~cut], p[~cut], average='micro'):.3f} for the rest")

print(f"\nby funder ({MIN_FUNDER_GRANTS}+ test grants):")
for funder, rows in test.groupby("funder").indices.items():
    if len(rows) >= MIN_FUNDER_GRANTS:
        print(f"  {funder[:45]:<45} {len(rows):>4} grants | micro-F1 {f1_score(y[rows], p[rows], average='micro'):.3f}")

others = {r: predictions("test", r) for r in OTHER_ROWS}
print(f"\nF1 per category ({MIN_CATEGORY_GRANTS}+ test grants)\n{'category':<36}{'grants':>7}{cli.name:>9}"
      + "".join(f"{r.replace('claude-', ''):>15}" for r in OTHER_ROWS))
for j in np.argsort(-y.sum(0)):
    if y[:, j].sum() >= MIN_CATEGORY_GRANTS:
        print(f"{LABELS[j]:<36}{y[:, j].sum():>7}{f1_score(y[:, j], p[:, j], zero_division=0):>9.2f}"
              + "".join(f"{f1_score(y[:, j], o[:, j], zero_division=0):>15.2f}" for o in others.values()))

# ─────────────────────────────────────────────────────────────
# 2. WRONG VALIDATION PREDICTIONS TO READ (playbook: sort each by kind of mistake)
# ─────────────────────────────────────────────────────────────
val = splits["validation"]
yv, pv, sv = np.array(val["label"], dtype=int), predictions("validation", cli.name), predictions("validation", cli.name, True)
wrong = np.where((pv != yv).any(axis=1))[0]
print(f"\nwrong validation grants: {len(wrong)} of {len(yv)}; reading {N_READ} picked at random (seed {SEED})")
for n, i in enumerate(np.random.default_rng(SEED).choice(wrong, N_READ, replace=False), 1):
    title, _, abstract = val[int(i)]["text"].partition("\n")
    print(f"\n#{n} {val[int(i)]['id']} | {title[:110]}\n   abstract: {abstract[:420]}")
    print(f"   TRUE: {[LABELS[j] for j in np.where(yv[i])[0]]}\n   MODEL: {[LABELS[j] for j in np.where(pv[i])[0]]}"
          " | top scores: " + ", ".join(f"{LABELS[j]} {sv[i, j]:.2f}" for j in np.argsort(-sv[i])[:4]))
