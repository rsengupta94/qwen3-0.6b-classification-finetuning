"""Step 2: turn each grant's text into token IDs. Reads data/splits/, writes data/tokenized/.

Applies these decisions:
  - cut grants longer than MAX_LEN tokens, keeping the start: title + start of abstract (decision 40)
  - no padding here: during training each group is padded to its own longest grant (decision 39)
  - keep only what the model reads: token IDs, attention mask, labels (the collator fails on text columns)
"""
import numpy as np
from datasets import List, Value, load_from_disk
from transformers import AutoTokenizer, DataCollatorWithPadding

from settings import LABELS, MAX_LEN, MODEL_NAME, MODEL_REVISION, SPLITS_DIR, TOKENIZED_DIR

# ─────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────
tok = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
assert tok.pad_token_id is not None, "Tokenizer has no pad token; Step 3 needs one for the model config"
assert tok.truncation_side == "right", "Cutting must remove the end, so the title and start are kept (decision 40)"
splits = load_from_disk(SPLITS_DIR)

# ─────────────────────────────────────────────────────────────
# 2. TOKENIZE: cut at MAX_LEN, no padding (the collator pads each group during training)
# ─────────────────────────────────────────────────────────────
tokenized = splits.map(lambda b: tok(b["text"], truncation=True, max_length=MAX_LEN),
                       batched=True, remove_columns=["text", "id", "year", "funder"])
tokenized = tokenized.rename_column("label", "labels")                     # the name the model's loss looks for
tokenized = tokenized.cast_column("labels", List(Value("float32")))       # the Mac GPU can't hold float64

# ─────────────────────────────────────────────────────────────
# 3. VERIFY (must)
# ─────────────────────────────────────────────────────────────
for k in splits:
    texts = list(splits[k]["text"])
    full = tok(texts)["input_ids"]                                         # uncut, to measure what the cap removes
    ids = list(tokenized[k]["input_ids"])
    print(f"{k:<10} {len(full):>6} grants, {np.mean([len(f) > MAX_LEN for f in full]):.1%} cut at {MAX_LEN} tokens")
    assert max(len(i) for i in ids) <= MAX_LEN, f"{k}: a grant is longer than MAX_LEN"
    assert not any(tok.pad_token_id in i for i in ids), \
        f"{k}: real text contains the pad token, so the model would skip it when finding the last token"

    # Decoded text must match the original: whole for a short grant, the start for a cut one
    short = next(n for n, f in enumerate(full) if len(f) <= MAX_LEN)
    cut = next(n for n, f in enumerate(full) if len(f) > MAX_LEN)
    assert tok.decode(ids[short]) == texts[short], f"{k}: short grant changed"
    kept = tok.decode(ids[cut])
    assert texts[cut].startswith(kept), f"{k}: cut grant lost its start"
    if k == "train":
        print(f"cut grant, kept start: {kept[:120]!r}\ncut grant, kept end:   {kept[-120:]!r}")

# One group of 4, padded the way training will pad it
rows = tokenized["train"].select(range(4))
batch = DataCollatorWithPadding(tok)(list(rows))
lengths = [len(ids) for ids in rows["input_ids"]]
print("group of 4:", {name: (tuple(t.shape), str(t.dtype)) for name, t in batch.items()}, "lengths", lengths)
assert batch["labels"].shape == (4, len(LABELS)), "Each grant must carry one label value per category"
assert batch["input_ids"].shape[1] == max(lengths), "The group must be padded to its own longest grant"
for i, n in enumerate(lengths):
    assert batch["attention_mask"][i, :n].all() and not batch["attention_mask"][i, n:].any(), "Mask must be 0 exactly at padding"
    assert (batch["input_ids"][i, n:] == tok.pad_token_id).all(), "Padding must use the pad token"

# ─────────────────────────────────────────────────────────────
# 4. SAVE
# ─────────────────────────────────────────────────────────────
tokenized.save_to_disk(TOKENIZED_DIR)
print("saved", TOKENIZED_DIR, {k: len(d) for k, d in tokenized.items()})
