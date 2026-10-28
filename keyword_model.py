"""Keyword model: the floor row (decisions 7, 49). Word weights plus one logistic regression per category.

Applies these decisions:
  - reads the full title and abstract, not cut at 512 tokens (decision 50)
  - no class weights (decision 22); three strengths fixed in advance, picked on validation by decision 48's score (decision 51)
  - validation predictions only; test predictions are made later, for every row together (decision 52)
"""
from pathlib import Path

import joblib
import numpy as np
from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from score import metrics, pick_threshold, report, save_predictions
from settings import SEED, SPLITS_DIR

# ── Settings used only by this step (shared ones live in settings.py) ──
STRENGTHS = [2, 8, 32, 128]                     # C: higher = trusts individual words more (decision 51; 128 added, decision 53)
MODEL_FILE = "models/keyword_model.joblib"      # kept for the test predictions at the end (decision 52)

splits = load_from_disk(SPLITS_DIR)
train, val = splits["train"], splits["validation"]
y_train, y_val = np.array(train["label"], dtype=int), np.array(val["label"], dtype=int)

# Word and word-pair weights: rare, telling words count more; anything seen in fewer than 2 grants is ignored
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, dtype=np.float32)
X_train, X_val = vec.fit_transform(list(train["text"])), vec.transform(list(val["text"]))
print(f"{X_train.shape[1]:,} words and word pairs")

best = None
for c in STRENGTHS:
    clf = OneVsRestClassifier(LogisticRegression(C=c, solver="liblinear", random_state=SEED))
    scores = clf.fit(X_train, y_train).predict_proba(X_val)
    t = pick_threshold(scores, y_val)
    m = metrics(scores >= t, y_val)
    print(f"C={c:<3} threshold {t:.3f} | micro-F1 {m['micro_f1']:.3f} | precision {m['precision']:.3f} "
          f"| recall {m['recall']:.3f} | macro-F1 {m['macro_f1']:.3f}")
    if best is None or m["micro_f1"] > best["micro_f1"]:
        best = {"C": c, "model": clf, "scores": scores, "threshold": t, "micro_f1": m["micro_f1"]}

print(f"\nkept C={best['C']} (threshold {best['threshold']:.3f})")
Path(MODEL_FILE).parent.mkdir(exist_ok=True)
joblib.dump({"vectorizer": vec, "model": best["model"], "C": best["C"], "threshold": best["threshold"]}, MODEL_FILE)
report(save_predictions("validation", "keyword", val["id"], best["scores"] >= best["threshold"], best["scores"]))
