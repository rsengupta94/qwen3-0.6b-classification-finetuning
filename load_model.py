"""Step 3: load the base model with a new output layer, one output per category. The timing run and training import it.

Applies these decisions:
  - load the weights in full size, float32 (decision 41); the library's default is now the file's own half size

Run directly to check it: size figures, the new layer, the pad token, and one real group of grants on the Mac GPU.
"""
from settings import (LABELS, MODEL_NAME, MODEL_REVISION, PROBLEM_TYPE, SEED, TOKENIZED_DIR,  # first: sets the
                      id2label, label2id)                                                  # GPU fallback before torch
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

assert torch.backends.mps.is_available(), "Mac GPU not available; never train silently on the CPU"
DEVICE = torch.device("mps")


def load_model():
    """The tokenizer, and the base model with a new random output layer, on the Mac GPU."""
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    set_seed(SEED)                                                     # right before loading: same starting layer every run
    model, info = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        num_labels=len(LABELS),                                        # one output per category
        id2label=id2label, label2id=label2id,                          # saved in the config, so predictions come back as names
        problem_type=PROBLEM_TYPE,                                     # multi-label: one yes/no score per category
        dtype=torch.float32,                                           # decision 41
        output_loading_info=True,
    )
    # Only the new output layer may be missing from the download; anything else means a broken load
    assert info["missing_keys"] == {"score.weight"}, f"Newly created weights beyond the output layer: {info['missing_keys']}"
    assert not info["unexpected_keys"] and not info["mismatched_keys"], f"Weights didn't fit the model: {info}"
    model.config.pad_token_id = tok.pad_token_id                       # must: how the model finds the last real token
    return tok, model.to(DEVICE)


if __name__ == "__main__":
    from datasets import load_from_disk
    from transformers import DataCollatorWithPadding

    tok, model = load_model()
    model.eval()                                                       # checking only, not training
    c = model.config

    # 1. Size figures match the model card (README at the pinned revision)
    print("width:", c.hidden_size, "| layers:", c.num_hidden_layers, "| heads:", c.num_attention_heads,
          "+", c.num_key_value_heads, "| context:", c.max_position_embeddings)
    assert (c.num_hidden_layers, c.num_attention_heads, c.num_key_value_heads, c.max_position_embeddings) == (28, 16, 8, 32768)

    # 2. The new layer, the pad token, full-size numbers
    print(model.score)
    assert (model.score.in_features, model.score.out_features) == (c.hidden_size, len(LABELS))
    assert c.pad_token_id == tok.pad_token_id == 151643, "Pad token not copied into the model"
    assert all(p.dtype == torch.float32 for p in model.parameters()), "Some weights aren't full size (decision 41)"
    new, total = sum(p.numel() for p in model.score.parameters()), sum(p.numel() for p in model.parameters())
    print(f"new (random): {new:,} | total: {total:,}")

    # 3. One real group of 4 on the GPU, with labels, so the loss is computed too
    rows = load_from_disk(TOKENIZED_DIR)["train"].select(range(4))
    batch = DataCollatorWithPadding(tok)(list(rows)).to(DEVICE)
    with torch.no_grad():
        out = model(**batch)
    print("logits:", tuple(out.logits.shape), "| loss:", round(out.loss.item(), 4), "(meaningless until trained)")
    assert out.logits.shape == (4, len(LABELS)) and torch.isfinite(out.loss), "Forward pass or loss failed"

    # 4. Padding doesn't change the answer: the shortest grant alone must score the same as inside the padded group
    i = min(range(4), key=lambda n: len(rows[n]["input_ids"]))
    with torch.no_grad():
        alone = model(input_ids=torch.tensor([rows[i]["input_ids"]], device=DEVICE)).logits[0]
    gap = (alone - out.logits[i]).abs().max().item()
    print(f"shortest grant alone vs in the group: largest difference {gap:.2e}")
    assert gap < 1e-3, "The model reads the wrong position for padded grants"
