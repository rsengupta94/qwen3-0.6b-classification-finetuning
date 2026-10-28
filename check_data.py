"""Data checks for HRCS, run before any cleaning or splitting. Prints a report; changes no files.

Training file: full checks. Test file: format and overlap with training only (its labels are not counted).
"""
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer

from settings import LABEL_SEP, LABELS, MODEL_NAME, MODEL_REVISION, RAW_TEST, RAW_TRAIN

train = pd.read_csv(RAW_TRAIN)
test = pd.read_csv(RAW_TEST)
for df in (train, test):
    df["text"] = df["AwardTitle"].fillna("") + "\n" + df["AwardAbstract"].fillna("")
    df["labels"] = df["HCs"].astype(str).apply(lambda s: [l.strip().lower() for l in s.split(LABEL_SEP) if l.strip()])
    df["labelset"] = df["labels"].apply(lambda ls: tuple(sorted(set(ls))))


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def pct(a, b):
    return f"{100 * a / b:.1f}%"


# ---------------------------------------------------------------------------
section("1. LABELS (training file)")
raw_tokens = Counter(t for s in train["HCs"].astype(str) for t in s.split(LABEL_SEP))
print("distinct raw label strings:", len(raw_tokens))
variants = [t for t in raw_tokens if t != t.strip() or t.strip().lower() not in LABELS]
print("raw strings that are not one of the 21 (after trim + lowercase):", variants or "none")
messy = [t for t in raw_tokens if t != t.strip()]
print("raw strings with extra spaces:", messy or "none")
unknown_test = sorted({l for ls in test["labels"] for l in ls if l not in LABELS})
print("TEST format check, labels not in the 21:", unknown_test or "none")

n = len(train)
per_grant = train["labels"].apply(len)
dupe_within = (train["labels"].apply(len) != train["labelset"].apply(len)).sum()
print(f"\ncategories per grant: {dict(sorted(Counter(per_grant).items()))}")
print(f"grants with more than 5 categories (HRCS allows at most 5): {(per_grant > 5).sum()}")
print(f"grants listing the same category twice: {dupe_within}")
multi = train[per_grant > 1]
alpha = (multi["labels"].apply(lambda ls: ls == sorted(ls))).mean()
print(f"multi-category grants whose labels are already in alphabetical order: {pct(alpha * len(multi), len(multi))} "
      "(far below 100% means the order may carry meaning, e.g. main category first)")

total = Counter(l for ls in train["labelset"] for l in ls)
single = Counter(ls[0] for ls in train["labelset"] if len(ls) == 1)
print(f"\n{'category':36s}{'grants (any)':>14s}{'only label':>12s}{'share':>8s}")
for lab in sorted(LABELS, key=lambda l: total[l]):
    print(f"{lab:36s}{total[lab]:>14,}{single[lab]:>12,}{pct(total[lab], n):>8s}")
print(f"largest / smallest (grants with the category): {max(total.values())} / {min(total.values())} "
      f"= {max(total.values()) / min(total.values()):.0f} to 1")
ghr_mixed = train["labelset"].apply(lambda ls: "generic health relevance" in ls and len(ls) > 1).sum()
print(f"grants with 'generic health relevance' plus other categories: {ghr_mixed:,}")
print("expected validation count per category if validation = random 10% of training:",
      {l: round(total[l] * 0.1) for l in sorted(LABELS, key=lambda l: total[l])[:6]})

# ---------------------------------------------------------------------------
section("2. ROWS TO DROP")
for name, df in (("train", train), ("test", test)):
    print(f"{name}: missing title {df['AwardTitle'].isna().sum()}, missing abstract {df['AwardAbstract'].isna().sum()}, "
          f"blank abstract {(df['AwardAbstract'].fillna('').str.strip() == '').sum()}")
abs_len = train["AwardAbstract"].fillna("").str.len()
print(f"\ntrain abstract length in characters: min {abs_len.min()}, median {int(abs_len.median())}, max {abs_len.max()}")
print("10 shortest training abstracts:")
for i in abs_len.nsmallest(10).index:
    print(f"  [{abs_len[i]:>4} chars] {train.at[i, 'AwardAbstract'][:110]!r}")
print("\nmost repeated training abstracts (placeholders and templates repeat):")
for txt, c in train["AwardAbstract"].value_counts().head(5).items():
    print(f"  x{c}: {txt[:110]!r}")
print("most repeated training titles:")
for txt, c in train["AwardTitle"].value_counts().head(5).items():
    print(f"  x{c}: {txt[:90]!r}")

PATTERNS = {
    "HTML tags (<br>, <p>, ...)": r"<\s*/?\s*[a-zA-Z][a-zA-Z0-9]{0,8}\s*/?>",
    "HTML entities (&amp; &#39; ...)": r"&(?:[a-zA-Z]{2,8}|#\d{2,5});",
    "broken characters (â€™, Ã©, Â)": r"â€|Ã[\x80-\xbf©¨¼½¾]|Â[\xa0-\xbf ]",
    "replacement character (�)": "�",
    "placeholder phrases": r"(?i)\b(?:abstract (?:not available|to follow|unavailable)|no abstract|not applicable|to be confirmed|tbc)\b",
}
print(f"\n{'odd content':40s}{'train rows':>12s}{'test rows':>11s}")
for label, pat in PATTERNS.items():
    tr = train["text"].str.contains(pat, regex=True).sum()
    te = test["text"].str.contains(pat, regex=True).sum()
    print(f"{label:40s}{tr:>12,}{te:>11,}")
ex = train[train["text"].str.contains(PATTERNS["broken characters (â€™, Ã©, Â)"], regex=True)]
if len(ex):
    m = re.search(PATTERNS["broken characters (â€™, Ã©, Â)"], ex.iloc[0]["text"])
    s = max(0, m.start() - 50)
    print("  example of broken characters:", repr(ex.iloc[0]["text"][s:s + 110]))

# ---------------------------------------------------------------------------
section("3. DUPLICATES")
def dup_report(key, label):
    g = train.groupby(key)
    sizes = g.size()
    groups = sizes[sizes > 1]
    conflicts = g["labelset"].nunique()
    conflicts = conflicts[conflicts > 1]
    print(f"{label}: {len(groups):,} repeated values, {int((groups - 1).sum()):,} extra rows; "
          f"with conflicting labels: {len(conflicts):,} groups ({int(sizes[conflicts.index].sum()):,} rows)")
    return conflicts

dup_report("text", "exact same title + abstract")
conf = dup_report("AwardAbstract", "exact same abstract (any title)")
for a in list(conf.index)[:2]:
    rows = train[train["AwardAbstract"] == a]
    print("  conflict example:", repr(a[:80]))
    for _, r in rows.iterrows():
        print(f"     {r['ID']:<18} title={r['AwardTitle'][:45]!r:50} labels={list(r['labelset'])}")

tr_abs, tr_title = set(train["AwardAbstract"]), set(train["AwardTitle"].dropna().str.lower())
print(f"\nTRAIN vs TEST exact overlap: same abstract {test['AwardAbstract'].isin(tr_abs).sum()}, "
      f"same title (case-insensitive) {test['AwardTitle'].str.lower().isin(tr_title).sum()}")

# Near-copies: word TF-IDF on abstracts, cosine similarity to the single most similar other abstract
vec = TfidfVectorizer(stop_words="english", min_df=2, max_df=0.5, sublinear_tf=True, dtype=np.float32)
Xtr = vec.fit_transform(train["AwardAbstract"])
Xte = vec.transform(test["AwardAbstract"])

def nearest(A, B, same):
    best, arg = np.zeros(A.shape[0], np.float32), np.zeros(A.shape[0], int)
    for s in range(0, A.shape[0], 1000):
        sims = (A[s:s + 1000] @ B.T).toarray()
        if same:
            sims[np.arange(sims.shape[0]), np.arange(s, s + sims.shape[0])] = -1
        arg[s:s + 1000], best[s:s + 1000] = sims.argmax(1), sims.max(1)
    return best, arg

bt, at = nearest(Xtr, Xtr, True)
be, ae = nearest(Xte, Xtr, False)
print(f"\nnear-copies by abstract similarity (0 = no shared words, 1 = identical)")
print(f"{'similarity to closest other abstract':42s}{'train rows':>11s}{'test rows (vs train)':>22s}")
for t in (0.99, 0.95, 0.90, 0.85, 0.80):
    print(f"   >= {t:.2f}{'':34s}{(bt >= t).sum():>11,}{(be >= t).sum():>22,}")
pairs = [(i, at[i]) for i in np.where((bt >= 0.85) & (bt < 0.99))[0] if i < at[i]]
diff = sum(train.at[i, "labelset"] != train.at[j, "labelset"] for i, j in pairs)
print(f"train near-copy pairs at 0.85-0.99: {len(pairs):,}; with different labels: {diff:,} ({pct(diff, max(len(pairs), 1))})")
for i, j in pairs[:2]:
    print(f"  pair at {bt[i]:.2f}:")
    for k in (i, j):
        print(f"     {train.at[k, 'ID']:<18} {train.at[k, 'AwardTitle'][:60]!r:64} {list(train.at[k, 'labelset'])}")
for i in np.argsort(-be)[:2]:
    print(f"  closest TEST grant to training, {be[i]:.2f}: test {test.at[i, 'AwardTitle'][:50]!r} ~ train {train.at[ae[i], 'AwardTitle'][:50]!r}")

# ---------------------------------------------------------------------------
section("4. LENGTH (Qwen3-0.6B-Base tokenizer, pinned revision)")
tok = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
for name, df in (("train", train), ("test", test)):
    L = np.array([len(x) for x in tok(df["text"].tolist(), add_special_tokens=True)["input_ids"]])
    df["tokens"] = L
    print(f"{name}: median {int(np.median(L))}, 90th {int(np.percentile(L, 90))}, 95th {int(np.percentile(L, 95))}, "
          f"99th {int(np.percentile(L, 99))}, max {L.max()}")
    print(f"       over 256: {pct((L > 256).sum(), len(L))}   over 512: {pct((L > 512).sum(), len(L))}   "
          f"over 1024: {pct((L > 1024).sum(), len(L))}")
for cap in (256, 512):
    print(f"training tokens per pass at a {cap} cap: {np.minimum(train['tokens'], cap).sum():,}")

# ---------------------------------------------------------------------------
section("5. SHORTCUTS: funder names (training file)")
def names_funder(r):
    t = r["text"].lower()
    full = str(r["FundingOrganisation"]).lower()
    acr = str(r["FunderAcronym"]) if pd.notna(r["FunderAcronym"]) else ""
    return full in t or (len(acr) >= 3 and re.search(rf"\b{re.escape(acr)}\b", r["text"]) is not None)

train["names_funder"] = train.apply(names_funder, axis=1)
print(f"grants whose own funder is named in the title or abstract: {train['names_funder'].sum():,} "
      f"({pct(train['names_funder'].sum(), n)})")
rows = []
for f, g in train.groupby("FundingOrganisation"):
    if len(g) < 50:
        continue
    top, c = Counter(l for ls in g["labelset"] for l in ls).most_common(1)[0]
    rows.append((f, len(g), top, c / len(g), g["names_funder"].mean()))
rows.sort(key=lambda r: -r[3])
print(f"\nfunders with 50+ grants, by how strongly they point to one category:")
print(f"{'funder':44s}{'grants':>7s}  {'most common category':30s}{'share':>7s}{'named in text':>15s}")
for f, k, top, share, named in rows[:12]:
    print(f"{f[:43]:44s}{k:>7,}  {top[:29]:30s}{share * 100:>6.0f}%{named * 100:>14.0f}%")
spec = [r for r in rows if r[3] >= 0.8]
spec_grants = sum(r[1] for r in spec)
spec_named = sum(r[1] * r[4] for r in spec)
print(f"specialist funders (80%+ of grants in one category): {len(spec)}, covering {spec_grants:,} grants "
      f"({pct(spec_grants, n)}); of those grants, {pct(spec_named, spec_grants)} name the funder in the text")

# ---------------------------------------------------------------------------
section("6. YEARS (from the ID format)")
def year_of(i):
    m = re.match(r"HRCS(?:20)?(\d\d)_", str(i))
    return f"20{m.group(1)}" if m else "no year in ID"
train["year"] = train["ID"].apply(year_of)
print("training grants by year:", dict(Counter(train["year"]).most_common()))
print("test ID formats:", dict(Counter(test["HRCS2022_ID"].str.replace(r"\d+$", "#", regex=True))))
noyear = train[train["year"] == "no year in ID"]
print("examples of IDs with no year:", noyear["ID"].sample(6, random_state=1).tolist())
