"""Claude rows: Haiku or Sonnet through Claude Code's headless mode, one call per grant (decisions 36, 54–60).

Rows: "fixed" = guide + 5 fixed examples (the headline; frozen, decision 60); "zeroshot" = guide only (test set
only, at the end; here just the script test). Each call replaces Claude Code's own instructions with our prompt and
has no tools (rule 7). The answer is forced into a list of the 21 names (--json-schema), and a grant the safety
filter refuses counts as wrong (rule 5). Answers are saved as they arrive, so a stopped run carries on where it left
off. Stops before all Claude spending together passes the budget (rule 10). Test-set runs are added at the end
(decision 52).

Usage: claude_baseline.py haiku fixed smoke      the script test: 20 validation grants
       claude_baseline.py haiku fixed early      Haiku's early check: 300 validation grants
       claude_baseline.py haiku zeroshot smoke   the zero-shot prompt on the same 20 grants
"""
import argparse
import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from tqdm import tqdm

from score import metrics
from settings import GUIDE_FILE, LABELS, SEED, SPLITS_DIR

# ── Settings used only by this step (shared ones live in settings.py) ──
MODEL_IDS = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5"}
EFFORT = "medium"                               # every Claude call (decision 55)
PROMPT_VERSION = "v3"                           # frozen (decision 60); bump if the prompt text ever changes
# v1 → v2: the answer instruction is repeated after the grant (6 of 19 v1 answers broke the format)
# v2 → v3: answers forced into a list of the 21 names (57 of 298 v2 answers broke the format)
SCHEMA = {"type": "object", "required": ["categories"],  # the answer form every call must fill (--json-schema)
          "properties": {"categories": {"type": "array", "items": {"type": "string", "enum": LABELS}}}}
N_EXAMPLES = 5                                  # fixed examples, picked at random (decision 56)
SETS = {"smoke": (0, 20), "early": (0, 300)}   # slices of one shuffle
WORKERS = 4                                     # calls at a time
CALL_TIMEOUT = 120                              # seconds; a normal call takes 2–5
BUDGET_USD = 120                                # all Claude runs together (rule 10)
CWD = Path("/tmp/hrcs-claude-empty")            # same empty folder every call: its path is part of what Claude sees
RUNS = Path("runs/claude")

cli = argparse.ArgumentParser()
cli.add_argument("model", choices=MODEL_IDS)
cli.add_argument("row", choices=["fixed", "zeroshot"])
cli.add_argument("set", choices=SETS)
cli = cli.parse_args()
if cli.row == "zeroshot" and cli.set != "smoke":
    raise SystemExit("zero-shot gets no tuning: it runs on the test set only, at the end (decision 58)")

# ─────────────────────────────────────────────────────────────
# 1. THE PROMPT: repeated parts first, so they are cached (rule 8)
# ─────────────────────────────────────────────────────────────
splits = load_from_disk(SPLITS_DIR)
train, val = splits["train"], splits["validation"]
cats = lambda row: "; ".join(l for l, y in zip(LABELS, row["label"]) if y)

examples = ""
if cli.row == "fixed":
    picked = [train[int(i)] for i in np.random.default_rng(SEED).choice(len(train), N_EXAMPLES, replace=False)]
    examples = ("\n<examples>\nFive grants coded by HRCS coders:\n"
                + "".join(f"<example>\n<grant>\n{r['text']}\n</grant>\n<categories>{cats(r)}</categories>\n</example>\n"
                          for r in picked) + "</examples>\n")
system = ("You assign HRCS Health Categories to UK health research grants. "
          "The official HRCS guidance is below; follow it.\n\n"
          f"<hrcs_guidance>\n{Path(GUIDE_FILE).read_text()}</hrcs_guidance>\n{examples}\n"
          "For the grant you are given, give the health categories that apply, using these exact names. "
          "Do not give percentages.\n" + "\n".join(LABELS))

MESSAGE = "<grant>\n{text}\n</grant>\n\nGive the health categories that apply, using only the exact names from the list."
run_dir = RUNS / f"{cli.model}-{cli.row}-{PROMPT_VERSION}"
run_dir.mkdir(parents=True, exist_ok=True)
prompt = (f"{system}\n\n===== user message ({{text}} = the grant) =====\n{MESSAGE}"
          f"\n\n===== answer schema =====\n{json.dumps(SCHEMA)}")
prompt_file = run_dir / "prompt.txt"
if prompt_file.exists():
    assert prompt_file.read_text() == prompt, f"prompt changed but PROMPT_VERSION is still {PROMPT_VERSION}"
else:
    prompt_file.write_text(prompt)
if cli.row == "fixed":
    (run_dir / "fixed_examples.json").write_text(json.dumps([r["id"] for r in picked]))

# ─────────────────────────────────────────────────────────────
# 2. ONE CALL PER GRANT
# ─────────────────────────────────────────────────────────────
CWD.mkdir(exist_ok=True)
assert not any(CWD.iterdir()), f"{CWD} must stay empty"
env = {k: v for k, v in os.environ.items() if not k.startswith(("CLAUDE", "ANTHROPIC"))}  # nothing inherited
cmd = ["claude", "-p", "--model", MODEL_IDS[cli.model], "--effort", EFFORT, "--system-prompt", system,
       "--tools", "", "--strict-mcp-config", "--setting-sources", "", "--disable-slash-commands",
       "--no-session-persistence", "--max-budget-usd", "1", "--output-format", "json",
       "--json-schema", json.dumps(SCHEMA)]


def ask(text):
    """Claude Code's JSON result for one grant (a refusal included), or None after 3 failed tries (each is logged)."""
    for attempt in range(3):
        try:
            r = subprocess.run(cmd, input=MESSAGE.format(text=text), capture_output=True, text=True, cwd=CWD,
                               env=env, timeout=CALL_TIMEOUT)
            out = json.loads(r.stdout)
            if not out.get("is_error") or out.get("stop_reason") == "refusal":  # a refusal repeats on every try
                return out
            why = f"is_error: {str(out.get('result'))[:300]}"
        except subprocess.TimeoutExpired:
            why = f"no answer within {CALL_TIMEOUT} s"
        except json.JSONDecodeError:
            why = f"exit {r.returncode}, not JSON: {r.stdout[:200]!r} {r.stderr[:300]!r}"
        with lock, (run_dir / "errors.log").open("a") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')} try {attempt + 1}: {why}\n")
    return None


def parse(out):
    """The categories in the forced list, in label order; None if it is missing, empty or off the 21 names (rule 5)."""
    items = (out.get("structured_output") or {}).get("categories") or []
    if not items or any(i not in LABELS for i in items):
        return None
    return [l for l in LABELS if l in items]


start, end = SETS[cli.set]
order = np.random.default_rng(SEED).permutation(len(val))[start:end]  # one fixed shuffle of validation
out_file = run_dir / f"{cli.set}.jsonl"
done = {json.loads(l)["id"] for l in out_file.open()} if out_file.exists() else set()
todo = [int(i) for i in order if val[int(i)]["id"] not in done]
spent = sum(json.loads(l)["cost_usd"] for f in RUNS.glob("*/*.jsonl") for l in f.open())
lock, state, failed = threading.Lock(), {"spent": spent, "largest": 0.05, "stopped": False}, []


def work(i):
    with lock:                                  # keep room for the calls already running
        if state["spent"] + WORKERS * 2 * state["largest"] > BUDGET_USD:
            state["stopped"] = True
            return
    out = ask(val[i]["text"])
    if out is None:
        failed.append(val[i]["id"])
        return
    rec = {"id": val[i]["id"], "answer": out["result"], "categories": parse(out),
           "refused": out.get("stop_reason") == "refusal",
           "cost_usd": out["total_cost_usd"], "usage": out["usage"], "models": list(out["modelUsage"]),
           "thinking_tokens": sum(m.get("thinkingTokens", 0) for m in out["modelUsage"].values()),
           "seconds": out["duration_ms"] / 1000, "at": datetime.now().isoformat(timespec="seconds")}
    with lock:
        state["spent"] += rec["cost_usd"]
        state["largest"] = max(state["largest"], rec["cost_usd"])
        with out_file.open("a") as f:
            f.write(json.dumps(rec) + "\n")


print(f"{cli.model} {cli.row} {PROMPT_VERSION} | {cli.set}: {len(order)} grants, {len(todo)} to do | "
      f"spent so far ${spent:.2f} of ${BUDGET_USD}")
with ThreadPoolExecutor(WORKERS) as pool:
    for _ in tqdm(as_completed([pool.submit(work, i) for i in todo]), total=len(todo)):
        pass
if state["stopped"]:
    print(f"STOPPED: the next calls could take Claude spending past ${BUDGET_USD} (spent ${state['spent']:.2f}).")
if failed:
    print(f"{len(failed)} calls failed 3 times; run the same command again to retry them.")

# ─────────────────────────────────────────────────────────────
# 3. SCORE THE SET (validation slices only; test-set rows are written at the end in the shared format)
# ─────────────────────────────────────────────────────────────
recs = {r["id"]: r for r in map(json.loads, out_file.open())}
rows = [int(i) for i in order if val[int(i)]["id"] in recs]
labels = np.array([val[i]["label"] for i in rows], dtype=int)
pred = np.array([[l in (recs[val[i]["id"]]["categories"] or []) for l in LABELS] for i in rows], dtype=int)
m = metrics(pred, labels)
r = list(recs.values())
print(f"\n{len(rows)} of {len(order)} grants answered | micro-F1 {m['micro_f1']:.3f} | precision {m['precision']:.3f} | "
      f"recall {m['recall']:.3f} | macro-F1 {m['macro_f1']:.3f}")
print(f"invalid answers: {sum(x['categories'] is None for x in r)} (refused: {sum(x['refused'] for x in r)}) | "
      f"models: {sorted({n for x in r for n in x['models']})} | "
      f"cost ${sum(x['cost_usd'] for x in r):.4f} | thinking tokens per grant {np.mean([x['thinking_tokens'] for x in r]):.0f} | "
      f"seconds per call {np.mean([x['seconds'] for x in r]):.1f}")
print(f"all Claude spending so far: ${state['spent']:.2f} of ${BUDGET_USD}")

if cli.set == "smoke":                          # read by eye: what Claude was shown, and a few of its answers
    if cli.row == "fixed":
        print("\nthe 5 fixed examples:")
        for x in picked:
            print(f"  {x['text'].splitlines()[0][:90]}  [{cats(x)}]")
    print("\nfirst 3 answers:")
    for i in rows[:3]:
        print(f"  {val[i]['text'].splitlines()[0][:80]}\n    true: {cats(val[i])}\n    Claude: {recs[val[i]['id']]['answer']!r}")
