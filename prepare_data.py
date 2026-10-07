"""Step 1: prepare the HRCS data. Reads data/raw/, writes data/splits/ and data/prepare_report.json.

Applies these data decisions:
  - clean formatting in every set (web-page tags, HTML symbol codes, garbled characters)
  - strip the "not available in GtR" placeholder abstract, so those grants are judged on their title
  - drop broken training rows; never drop test rows (the script stops if a test row is broken)
  - remove training grants whose abstract near-copies a test grant (leakage into the final exam)
  - keep near-copy groups within training, each group in one split
  - validation = about 2,000 of the 2022 training grants, stratified so each category keeps its share
"""
import html
import json
import re

import ftfy
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Settings used by this step (they move to the shared settings file at Setup) ──
LABELS = ["blood", "cancer and neoplasms", "cardiovascular", "congenital disorders", "disputed aetiology and other",
          "ear", "eye", "generic health relevance", "infection", "inflammatory and immune system",
          "injuries and accidents", "mental health", "metabolic and endocrine", "musculoskeletal", "neurological",
          "oral and gastrointestinal", "renal and urogenital", "reproductive health and childbirth", "respiratory",
          "skin", "stroke"]                     # lowercase; list order = label IDs
LABEL_SEP = ";"
MAX_LABELS = 5                                  # HRCS allows at most five categories per grant
SEED = 123
RAW_TRAIN, RAW_TEST = "data/raw/hrcs_training_data.csv", "data/raw/hrcs_test_data.csv"
SPLITS_DIR = "data/splits"
VAL_SIZE = 2000                                 # from 2022 training grants, to match the 2022 test set
NEAR_COPY = 0.90                                # abstract similarity at or above this counts as a near-copy
GTR_PLACEHOLDER = "Abstracts are not currently available in GtR"
SPREADSHEET_ERROR = re.compile(r"#(?:NAME\?|VALUE!|REF!|N/A|DIV/0!|NULL!)")

report = {}

# ─────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────
train = pd.read_csv(RAW_TRAIN).rename(columns={"ID": "id"})
test = pd.read_csv(RAW_TEST).rename(columns={"HRCS2022_ID": "id"})
report["loaded"] = {"train": len(train), "test": len(test)}

# ─────────────────────────────────────────────────────────────
# 2. CLEAN FORMATTING (every set, and new grants later, get the same preparation)
# ─────────────────────────────────────────────────────────────
TAG_BREAK = re.compile(r"<\s*(?:br|/?p)\s*/?\s*>", re.I)                       # <br/>, <p>, </p> → line break
TAG_OTHER = re.compile(r"<\s*/?\s*[a-zA-Z][a-zA-Z0-9]{0,8}(?:\s[^<>]{0,80})?\s*/?>")  # <em>, <sub>, ... → removed


def clean_text(s):
    if not isinstance(s, str):
        return ""
    s = TAG_BREAK.sub("\n", s)
    s = TAG_OTHER.sub("", s)
    s = html.unescape(s)                 # &pound; → £, &ndash; → –, &nbsp; → no-break space
    s = ftfy.fix_encoding(s)             # BarrettÃªs → Barrettês (repairs the encoding mix-up only)
    s = s.replace("\xa0", " ")
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


for name, df in (("train", train), ("test", test)):
    df["title"] = df["AwardTitle"].map(clean_text)
    df["abstract"] = df["AwardAbstract"].map(clean_text)
    placeholder = df["abstract"].str.startswith(GTR_PLACEHOLDER)
    df.loc[placeholder, "abstract"] = ""
    df["text"] = [("\n".join(p for p in (t, a) if p)) for t, a in zip(df["title"], df["abstract"])]
    raw_text = df["AwardTitle"].fillna("") + "\n" + df["AwardAbstract"].fillna("")
    report[f"cleaning_{name}"] = {"rows_with_text_changed": int((df["text"] != raw_text.str.strip()).sum()),
                                  "placeholder_abstracts_stripped": int(placeholder.sum())}

# ─────────────────────────────────────────────────────────────
# 3. LABELS: parse, validate (test: format check only, nothing counted)
# ─────────────────────────────────────────────────────────────
def parse(s):
    return sorted({l.strip().lower() for l in str(s).split(LABEL_SEP) if l.strip()})


for df in (train, test):
    df["labels"] = df["HCs"].map(parse)
    df["broken"] = (df["labels"].map(lambda ls: not ls or len(ls) > MAX_LABELS or any(l not in LABELS for l in ls))
                    | df["abstract"].str.fullmatch(SPREADSHEET_ERROR.pattern)
                    | (df["text"] == ""))

if test["broken"].any():
    raise SystemExit(f"Stop: {int(test['broken'].sum())} broken test rows. Test rows are never dropped; fix the rule.")
report["broken_train_rows_dropped"] = train.loc[train["broken"], "id"].tolist()
train = train[~train["broken"]].copy()

# ─────────────────────────────────────────────────────────────
# 4. EXACT DUPLICATES (training): drop every copy of conflicting texts, then keep one of each repeat
# ─────────────────────────────────────────────────────────────
key = train["labels"].map(tuple)
conflicting = key.groupby(train["text"]).nunique()
conflicting = conflicting[conflicting > 1].index
train = train[~train["text"].isin(conflicting)]
before = len(train)
train = train.drop_duplicates(subset="text", keep="first").copy()
report["exact_duplicates"] = {"conflicting_texts_dropped": len(conflicting), "repeats_dropped": before - len(train)}

# ─────────────────────────────────────────────────────────────
# 5. NEAR-COPIES by abstract similarity (word TF-IDF, cosine)
# ─────────────────────────────────────────────────────────────
vec = TfidfVectorizer(stop_words="english", min_df=2, max_df=0.5, sublinear_tf=True, dtype=np.float32)


def pairs_at_or_above(A, B, same):
    """Row/column index pairs where similarity >= NEAR_COPY. Empty abstracts have no words, so never match."""
    rows, cols = [], []
    for s in range(0, A.shape[0], 1000):
        S = (A[s:s + 1000] @ B.T).tocoo()
        keep = S.data >= NEAR_COPY
        r, c = S.row[keep] + s, S.col[keep]
        if same:
            r, c = r[r != c], c[r != c]
        rows.append(r)
        cols.append(c)
    return np.concatenate(rows), np.concatenate(cols)


# 5a. Training grants that near-copy a test grant: removed from TRAINING (test stays whole)
Xtr = vec.fit_transform(train["abstract"])
Xte = vec.transform(test["abstract"])
r, c = pairs_at_or_above(Xtr, Xte, same=False)
leak_rows = sorted(set(r.tolist()))
report["train_near_copies_of_test_removed"] = [
    {"train_id": train.iloc[i]["id"], "train_title": train.iloc[i]["title"][:80],
     "test_title": test.iloc[int(c[r == i][0])]["title"][:80]} for i in leak_rows]
train = train.drop(train.index[leak_rows]).reset_index(drop=True)

# 5b. Near-copy groups within training: each group must land in one split
Xtr = vec.transform(train["abstract"])
r, c = pairs_at_or_above(Xtr, Xtr, same=True)
graph = coo_matrix((np.ones(len(r)), (r, c)), shape=(len(train), len(train)))
_, train["group"] = connected_components(graph, directed=False)
sizes = train["group"].value_counts()
report["near_copy_groups"] = {"groups_with_2_or_more": int((sizes > 1).sum()),
                              "rows_in_those_groups": int(sizes[sizes > 1].sum()),
                              "largest_group": int(sizes.max())}

# ─────────────────────────────────────────────────────────────
# 6. YEAR from the ID format (2014 grants with funder IDs show as "unknown")
# ─────────────────────────────────────────────────────────────
def year_of(i):
    m = re.match(r"HRCS(?:20)?(\d\d)_", str(i))
    return f"20{m.group(1)}" if m else "unknown"


train["year"] = train["id"].map(year_of)
test["year"] = test["id"].map(year_of)

# ─────────────────────────────────────────────────────────────
# 7. MULTI-HOT LABELS: one 0/1 float per category (the multi-label loss needs floats)
# ─────────────────────────────────────────────────────────────
for df in (train, test):
    df["label"] = df["labels"].map(lambda ls: [1.0 if l in ls else 0.0 for l in LABELS])

# ─────────────────────────────────────────────────────────────
# 8. VALIDATION: ~VAL_SIZE grants from 2022, stratified by category, whole near-copy groups only
# ─────────────────────────────────────────────────────────────
group_all_2022 = train.groupby("group")["year"].agg(lambda ys: (ys == "2022").all())
pool = train[train["group"].map(group_all_2022)]                       # 2022 grants whose group is all 2022
unit_ids, Y = [], []                                                  # one unit per group; labels = union of members'
for g, members in pool.groupby("group"):
    unit_ids.append(g)
    Y.append(np.max(np.array(members["label"].tolist()), axis=0))
Y = np.array(Y)
splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=VAL_SIZE / len(pool), random_state=SEED)
_, val_units = next(splitter.split(np.zeros(len(Y)), Y))
val_groups = {unit_ids[i] for i in val_units}
is_val = train["group"].isin(val_groups)
val_df, train_df = train[is_val], train[~is_val]

# ─────────────────────────────────────────────────────────────
# 9. VERIFY (must)
# ─────────────────────────────────────────────────────────────
tr, va, te = set(train_df["text"]), set(val_df["text"]), set(test["text"])
assert not (tr & va) and not (tr & te) and not (va & te), "Leak: text shared between splits"
assert not set(train_df["group"]) & set(val_df["group"]), "Leak: a near-copy group spans training and validation"
assert len(test) == report["loaded"]["test"], "Test rows were dropped"
assert all(sum(l) >= 1 for d in (train_df, val_df, test) for l in d["label"]), "An example has no label"
assert (val_df["year"] == "2022").all(), "Validation must be 2022 grants only"

# ─────────────────────────────────────────────────────────────
# 10. SAVE + REPORT
# ─────────────────────────────────────────────────────────────
cols = ["text", "label", "id", "year", "FundingOrganisation"]
parts = {"train": train_df, "validation": val_df, "test": test}
splits = DatasetDict({k: Dataset.from_pandas(d[cols].rename(columns={"FundingOrganisation": "funder"}),
                                             preserve_index=False) for k, d in parts.items()})
splits.save_to_disk(SPLITS_DIR)

report["final"] = {k: len(d) for k, d in parts.items()}
report["train_years"] = train_df["year"].value_counts().to_dict()
for k in ("train", "validation"):                                      # test counts stay unseen
    counts = np.array(parts[k]["label"].tolist()).sum(axis=0).astype(int)
    report[f"{k}_category_counts"] = dict(sorted(zip(LABELS, counts.tolist()), key=lambda kv: kv[1]))
with open("data/prepare_report.json", "w") as f:
    json.dump(report, f, indent=2, default=str)
print(json.dumps(report, indent=2, default=str))
