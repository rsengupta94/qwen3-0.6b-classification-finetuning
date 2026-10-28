"""Shared settings for every step. Change a value here, never in a script."""
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Mac GPU: operations it can't run fall back to the CPU; set before torch loads

# ── Task ─────────────────────────────────────────────────────
TASK_TYPE = "multilabel"                        # a grant can have several categories
LABELS = ["blood", "cancer and neoplasms", "cardiovascular", "congenital disorders", "disputed aetiology and other",
          "ear", "eye", "generic health relevance", "infection", "inflammatory and immune system",
          "injuries and accidents", "mental health", "metabolic and endocrine", "musculoskeletal", "neurological",
          "oral and gastrointestinal", "renal and urogenital", "reproductive health and childbirth", "respiratory",
          "skin", "stroke"]                     # lowercase; list order = label IDs (decision 21)
LABEL_SEP = ";"                                 # separator in the raw files, e.g. "BLOOD;CANCER AND NEOPLASMS"
MAX_LABELS = 5                                  # HRCS allows at most five categories per grant

# ── Data ─────────────────────────────────────────────────────
DATASET, DATASET_REVISION = "NIHRDataInsights/HRCSData", "bd6adc7dbf9db61e4f20e31344772bf3e97135c0"  # pinned 2026-10-03
RAW_TRAIN, RAW_TEST = "data/raw/hrcs_training_data.csv", "data/raw/hrcs_test_data.csv"
SPLITS_DIR = "data/splits"
TOKENIZED_DIR = "data/tokenized"
PREDICTIONS_DIR = "data/predictions"            # one CSV per row per split, in the shared format (score.py)

# ── Model ────────────────────────────────────────────────────
MODEL_NAME, MODEL_REVISION = "Qwen/Qwen3-0.6B-Base", "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"  # decision 31; pinned
MAX_LEN = 512                                   # tokens; longer texts are cut (decision 30)

# ── Decided rules ────────────────────────────────────────────
IMBALANCE = "none"                              # no class weights at first; recall comes from the threshold (decision 22)
TARGET_RECALL = 0.80                            # one threshold for all categories: the strictest that catches 80% (decisions 32–33)
SEED = 123

# ── Derived values: don't edit ───────────────────────────────
id2label = {i: name for i, name in enumerate(LABELS)}
label2id = {name: i for i, name in enumerate(LABELS)}
PROBLEM_TYPE = "multi_label_classification" if TASK_TYPE == "multilabel" else "single_label_classification"
assert all(l == l.lower() for l in LABELS), "LABELS must be lowercase (Step 1 lowercases the data)"
