"""Model card charts: test micro-F1 per row with the fine-tuned model's lead, and F1 per category (light + dark).

Writes transparent PNGs for the Hugging Face page, one per theme (shown with #hf-light-mode-only /
#hf-dark-mode-only). Leads come with 95% ranges from 2,000 paired resamples of the test grants (decision 64).
Categories with 8 or fewer test grants are left out of the per-category chart (section 6).

Usage: make_charts.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                           # draw to files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_from_disk
from matplotlib import font_manager
from matplotlib.lines import Line2D
from sklearn.metrics import f1_score

from settings import LABELS, PREDICTIONS_DIR, SEED, SPLITS_DIR

# ── Settings used only by this step (shared ones live in settings.py) ──
OUT_DIR = Path("model_card/assets")
ROWS = [("run1", "Fine-tuned Qwen3-0.6B (this model)"),
        ("keyword", "Keyword model (TF-IDF + logistic regression)"),
        ("claude-sonnet-fixed", "Claude Sonnet 5.5 · guide + 5 examples"),
        ("claude-haiku-fixed", "Claude Haiku 5.5 · guide + 5 examples"),
        ("claude-haiku-zeroshot", "Claude Haiku 5.5 · guide only")]
NIHR_F1 = 0.81                                  # quoted from NIHR's model page; its test set is not stated
N_RESAMPLES = 2000
MIN_CATEGORY_GRANTS = 9                         # no per-category claims at 8 or fewer test grants (section 6)
THEMES = {"light": dict(ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", base="#c3c2b7",
                        s1="#2a78d6", s2="#eb6834", ring="#fcfcfb", page="#ffffff"),
          "dark": dict(ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", base="#383835",
                       s1="#3987e5", s2="#d95926", ring="#1a1a19", page="#0b0f19")}   # colours checked for colour blindness
installed = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = next(f for f in ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"] if f in installed)

test = load_from_disk(SPLITS_DIR)["test"]
y = np.array(test["label"], dtype=int)
pred = {r: pd.read_csv(f"{PREDICTIONS_DIR}/test/{r}.csv").set_index("id").loc[test["id"], LABELS].to_numpy(int)
        for r, _ in ROWS}
micro = lambda yy, pp: f1_score(yy, pp, average="micro")
score = {r: micro(y, p) for r, p in pred.items()}
rng = np.random.default_rng(SEED)
draws = [rng.integers(0, len(y), len(y)) for _ in range(N_RESAMPLES)]
lead = {r: 100 * (score["run1"] - score[r]) for r, _ in ROWS[1:]}
lead_range = {r: np.percentile([100 * (micro(y[i], pred["run1"][i]) - micro(y[i], pred[r][i])) for i in draws], [2.5, 97.5])
              for r, _ in ROWS[1:]}
n = y.sum(axis=0)
shown = [j for j in np.argsort(-n, kind="stable") if n[j] >= MIN_CATEGORY_GRANTS]
left_out = [LABELS[j] for j in np.argsort(-n, kind="stable") if n[j] < MIN_CATEGORY_GRANTS]


def plain(ax, c):
    """Recessive axes: hairline x grid, no box, muted ticks."""
    ax.set_facecolor("none")
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(c["base"])
    ax.tick_params(colors=c["muted"], labelsize=9, length=0)
    ax.grid(axis="x", color=c["grid"], linewidth=1)
    ax.set_axisbelow(True)


OUT_DIR.mkdir(parents=True, exist_ok=True)
for theme, c in THEMES.items():
    # ── 1. Scores (left) and the fine-tuned model's lead over each row (right) ──
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9, 3.4), dpi=200, sharey=True, gridspec_kw={"width_ratios": [1.15, 1]})
    plain(ax, c)
    plain(bx, c)
    ys = np.arange(len(ROWS))[::-1]
    for (r, _), row_y in zip(ROWS, ys):
        ours = r == "run1"
        ax.scatter([score[r]], [row_y], s=70, color=c["s1"] if ours else c["muted"], edgecolors="none", zorder=3)
        ax.text(score[r] + 0.004, row_y, f"{score[r]:.3f}", va="center", fontsize=9.5, color=c["ink"],
                fontweight="bold" if ours else "normal")
        if ours:
            bx.text(0.15, row_y, "reference row", va="center", fontsize=8.5, color=c["muted"])
            continue
        bx.plot(lead_range[r], [row_y, row_y], color=c["muted"], linewidth=2, solid_capstyle="round", alpha=0.6)
        bx.scatter([lead[r]], [row_y], s=55, color=c["muted"], edgecolors="none", zorder=3)
        bx.text(lead_range[r][1] + 0.25, row_y, f"+{lead[r]:.1f}", va="center", fontsize=9.5, color=c["ink"])
    ax.set_yticks(ys, [label for _, label in ROWS], fontsize=9.5)
    for tick, (r, _) in zip(ax.get_yticklabels(), ROWS):
        tick.set_color(c["ink"] if r == "run1" else c["ink2"])
        tick.set_fontweight("bold" if r == "run1" else "normal")
    ax.plot([NIHR_F1, NIHR_F1], [-0.6, len(ROWS) - 0.55], color=c["ink2"], linewidth=1, zorder=1)
    ax.text(NIHR_F1, len(ROWS) - 0.45, "NIHR's model: 0.81\n(quoted; its test set not stated)", ha="center",
            va="bottom", fontsize=7.5, color=c["ink2"], linespacing=1.1)
    ax.set_xlim(0.74, 0.84)
    ax.set_ylim(-0.6, len(ROWS) + 0.45)
    ax.set_xlabel("Micro-F1, official HRCS test set (972 grants)", fontsize=9, color=c["ink2"])
    bx.axvline(0, color=c["ink2"], linewidth=1, zorder=1)
    bx.set_xlim(-2, 9.5)
    bx.set_xlabel("Fine-tuned model's lead, in points (line: 95% range)", fontsize=9, color=c["ink2"])
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"results-{theme}.png", transparent=True)
    plt.close(fig)

    # ── 2. F1 per category: fine-tuned model (filled dot) against Sonnet (ring) ──
    fig, ax = plt.subplots(figsize=(8, 6.4), dpi=200)
    plain(ax, c)
    ys = np.arange(len(shown))[::-1]
    for j, row_y in zip(shown, ys):
        ours, sonnet = (f1_score(y[:, j], pred[r][:, j], zero_division=0) for r in ["run1", "claude-sonnet-fixed"])
        ax.plot([ours, sonnet], [row_y, row_y], color=c["base"], linewidth=2, zorder=1)
        ax.scatter([sonnet], [row_y], s=60, facecolors=c["page"], edgecolors=c["s2"], linewidths=2, zorder=2)
        ax.scatter([ours], [row_y], s=40, color=c["s1"], edgecolors=c["ring"], linewidths=1, zorder=3)
    ax.set_yticks(ys, [f"{LABELS[j]} ({n[j]})" for j in shown], fontsize=9)
    for tick in ax.get_yticklabels():
        tick.set_color(c["ink"])
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(-0.7, len(shown) - 0.3)
    ax.set_xlabel("F1 per category, official HRCS test set (test grants in brackets)", fontsize=9, color=c["ink2"])
    key = [Line2D([], [], marker="o", linestyle="none", markersize=7, color=c["s1"], label="Fine-tuned Qwen3-0.6B"),
           Line2D([], [], marker="o", linestyle="none", markersize=8, markerfacecolor="none", markeredgecolor=c["s2"],
                  markeredgewidth=2, label="Claude Sonnet 5.5 · guide + 5 examples")]
    fig.legend(handles=key, loc="upper left", ncol=2, frameon=False, fontsize=9, labelcolor=c["ink"],
               bbox_to_anchor=(0.01, 0.995), handletextpad=0.4, columnspacing=1.6)
    fig.text(0.01, 0.01, f"Not shown ({MIN_CATEGORY_GRANTS - 1} or fewer test grants, too few to judge): "
             + ", ".join(left_out) + ".", fontsize=8, color=c["muted"])
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(OUT_DIR / f"per-category-{theme}.png", transparent=True)
    plt.close(fig)

print(f"saved 4 charts to {OUT_DIR}/")
