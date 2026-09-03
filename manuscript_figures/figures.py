"""Manuscript and supplement figures.

    python manuscript_figures/figures.py
    python manuscript_figures/figures.py --only figure1 figure2
    python manuscript_figures/figures.py --out /path/to/dir
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sources import (  # noqa: E402
    CORPUS_FLOW,
    DEFAULT_OUTPUT,
    FOLD_NAMES,
    FOLDS,
    LABELS,
    OVERALL_TO_CANNABIS,
    TASKS,
    load_selected_targets,
    load_split,
    load_val_curves,
    CONDITIONS,
    CONDITION_NAMES,
    CONTRAST_NAMES,
    SEEDS,
    load_contrasts,
    load_gemma,
    load_test_runs,
    load_test_summary,
    load_training_composition,
)
from style import (  # noqa: E402
    BLUE,
    INK,
    TERM_BG,
    TERM_INK,
    SENTIMENT,
    SKY,
    VERMILLION,
    apply_style,
    arrow,
    bar_grid,
    inch_axes,
    line,
    node,
    panel_label,
    plain,
    save,
)

BOX_W, BOX_H = 3.3, 0.46


def _flow_stages() -> list[tuple[str, str]]:
    """(unit-bearing count, descriptor) per stage, worded as the Methods words it."""
    f = CORPUS_FLOW
    return [
        (
            f"{f['retrieved_posts']:,} posts",
            f"collected from {f['communities']} communities",
        ),
        (f"{f['matched_posts']:,} posts", "containing a lexicon match"),
        (f"{f['matched_sentences']:,} matched sentences", "within those posts"),
        (f"{f['sampled']:,} matched sentences", "randomly sampled, stratified by subreddit"),
        (
            f"{f['annotated_rows']} annotated examples",
            f"from {f['source_posts']} posts",
        ),
    ]


SIDE_W, SIDE_H = 2.25, 0.62


def _draw_flow(ax, cx: float, ys: tuple[float, ...]):
    """Shared vertical corpus flow with the exclusion branch. Returns bottom y."""
    stages = _flow_stages()
    for i, (y, (count, label)) in enumerate(zip(ys, stages)):
        last = i == len(stages) - 1
        node(
            ax,
            cx,
            y,
            BOX_W,
            BOX_H,
            [(count, 10.5, "bold"), (label, 7.5, "normal")],
            kind="end" if last else "main",
        )
        if not last:
            arrow(ax, cx, y - BOX_H / 2, cx, ys[i + 1] + BOX_H / 2)

    branch_y = (ys[-2] + ys[-1]) / 2
    side_x = cx + BOX_W / 2 + 0.85 + SIDE_W / 2
    arrow(ax, cx, branch_y, side_x - SIDE_W / 2, branch_y)
    node(
        ax,
        side_x,
        branch_y,
        SIDE_W,
        SIDE_H,
        [
            (f"{CORPUS_FLOW['excluded']} excluded on manual review", 8.0, "bold"),
            ("matched term not cannabis-related", 7.0, "normal"),
            ("in context", 7.0, "normal"),
        ],
        kind="side",
    )
    return ys[-1] - BOX_H / 2


# Constructed for this paper; NOT source text. Never replace these with real posts
# -- the manuscript states that no verbatim Reddit text is reproduced.
EXAMPLES = (
    (
        [("My health has been awful, although ", False), ("cannabis", True),
         (" has made the nausea easier.", False)],
        "Negative",
        "Positive",
    ),
    (
        [("Asking for a friend \u2014 she says ", False), ("CBD oil", True),
         (" is the only thing that helps.", False)],
        "Neutral",
        "Positive",
    ),
    (
        [("Another flare and I am exhausted. My doctor asked whether I use ", False),
         ("cannabis", True), (".", False)],
        "Negative",
        "Neutral",
    ),
)


def _rich_text(fig, ax, x, y, segments, size=7.8):
    """Left-aligned run of (text, is_term) segments; terms are highlighted."""
    fig.canvas.draw()  # a renderer is needed to measure each segment
    renderer = fig.canvas.get_renderer()
    cursor = x
    for text, is_term in segments:
        artist = ax.text(
            cursor,
            y,
            text,
            ha="left",
            va="center",
            fontsize=size,
            fontweight="bold" if is_term else "normal",
            color=TERM_INK if is_term else INK,
            zorder=4,
        )
        width = artist.get_window_extent(renderer).width / fig.dpi
        if is_term:
            ax.add_patch(
                Rectangle(
                    (cursor - 0.025, y - 0.078),
                    width + 0.05,
                    0.156,
                    facecolor=TERM_BG,
                    edgecolor="none",
                    zorder=3,
                )
            )
        cursor += width
    return cursor


def figure_s_label_targets(out: str) -> list[str]:
    """Constructed examples showing that the two sentiment targets can differ."""
    apply_style()
    fig = plt.figure(figsize=(6.7, 2.85))
    ax, _, _ = inch_axes(fig)

    panel_cx, panel_w = 2.15, 3.95
    cols = (4.85, 6.05)
    chip_w, chip_h = 1.10, 0.30
    header_y = 2.50
    ys = (2.02, 1.28, 0.54)

    for x, title in zip(cols, ("Traditional", "Aspect-based")):
        ax.text(x, header_y, title, ha="center", va="center", fontsize=7.5,
                fontweight="bold", color=INK)

    for y, (segments, overall, specific) in zip(ys, EXAMPLES):
        node(ax, panel_cx, y, panel_w, 0.44, [], kind="main")
        _rich_text(fig, ax, panel_cx - panel_w / 2 + 0.13, y, segments)
        for x, label in zip(cols, (overall, specific)):
            node(ax, x, y, chip_w, chip_h, [(label, 7.8, "bold")],
                 kind=(SENTIMENT[label], SENTIMENT[label]), textcolor="#FFFFFF")

    ax.text(panel_cx - panel_w / 2, 0.14, "Constructed examples; no source text is reproduced.",
            ha="left", va="center", fontsize=6.8, fontstyle="italic", color="#4A4A4A")
    return save(fig, os.path.join(out, "figure_s_label_targets"))


def figure_s1(out: str) -> list[str]:
    """Corpus flow forked into the public release and the local modeling file."""
    apply_style()
    fig = plt.figure(figsize=(6.3, 5.7))
    ax, _, _ = inch_axes(fig)
    bottom = _draw_flow(ax, cx=2.2, ys=(5.35, 4.49, 3.63, 2.77, 1.91))

    rows = CORPUS_FLOW["annotated_rows"]
    fork_y, fork_h, left_x, right_x = 0.50, 0.68, 1.35, 4.35
    line(ax, 2.2, bottom, 2.2, 1.15)
    line(ax, left_x, 1.15, right_x, 1.15)
    arrow(ax, left_x, 1.15, left_x, fork_y + fork_h / 2)
    arrow(ax, right_x, 1.15, right_x, fork_y + fork_h / 2)
    node(
        ax,
        left_x,
        fork_y,
        2.40,
        fork_h,
        [
            ("Public Zenodo release", 8.5, "bold"),
            (f"{rows} rows", 7.5, "normal"),
            ("post IDs, cannabis terms, labels", 7.0, "normal"),
        ],
        kind="note",
    )
    node(
        ax,
        right_x,
        fork_y,
        2.40,
        fork_h,
        [
            ("Study file", 8.5, "bold"),
            (f"{rows} rows", 7.5, "normal"),
            ("text and labels; used to train", 7.0, "normal"),
        ],
    )
    return save(fig, os.path.join(out, "figure_s1_corpus_flow"))


def figure_s2(out: str) -> list[str]:
    apply_style()
    split = load_split()
    x = np.arange(len(FOLDS))
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), constrained_layout=True)

    ax = axes[0]
    posts = [split["folds"][f]["n_posts"] for f in FOLDS]
    rows = [split["folds"][f]["n_rows"] for f in FOLDS]
    for offset, values, color, label in (
        (-0.18, posts, BLUE, "Posts"),
        (0.18, rows, SKY, "Rows"),
    ):
        ax.bar(x + offset, values, 0.36, color=color, label=label)
        for i, value in enumerate(values):
            ax.text(i + offset, value + 8, str(value), ha="center", va="bottom", fontsize=6.5)
    ax.set_ylim(0, max(posts + rows) * 1.2)
    ax.set_ylabel("Count")
    ax.legend(frameon=False, loc="upper right")

    for ax, key in ((axes[1], "absa"), (axes[2], "trad")):
        top = 0
        for offset, label, index in zip((-0.24, 0.0, 0.24), LABELS, range(3)):
            values = [split["folds"][f][key][index] for f in FOLDS]
            top = max(top, max(values))
            ax.bar(x + offset, values, 0.24, color=SENTIMENT[label], label=label)
            for i, value in enumerate(values):
                ax.text(i + offset, value + 4, str(value), ha="center", va="bottom", fontsize=6.5)
        ax.set_ylim(0, top * 1.3)
        ax.set_ylabel("Rows")
        ax.set_title(TASKS[key], fontsize=8.5)
        ax.legend(frameon=False, ncol=3, loc="upper right", columnspacing=0.7, handlelength=1.0)

    for ax in axes:
        ax.set_xticks(x, [FOLD_NAMES[f] for f in FOLDS])
        bar_grid(ax)
    return save(fig, os.path.join(out, "figure_s2_split"))


def figure_s3(out: str) -> list[str]:
    """Validation macro-F1 by augmentation target, with the selected target starred."""
    apply_style()
    curves = load_val_curves()
    selected = load_selected_targets()
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8), sharey=True, constrained_layout=True)

    for ax, task, color in ((axes[0], "absa", BLUE), (axes[1], "trad", VERMILLION)):
        subset = curves[curves["task"] == task]
        targets = subset["target"].to_numpy()
        means = subset["f1_mean"].to_numpy()
        ax.errorbar(
            targets,
            means,
            yerr=subset["f1_sd"].to_numpy(),
            fmt="o-",
            color=color,
            elinewidth=1.0,
            capsize=3.0,
            markersize=5.5,
            linewidth=1.3,
            markerfacecolor="white",
            markeredgewidth=1.2,
        )
        chosen = selected[task]
        ax.scatter(
            [chosen],
            [float(subset.loc[subset["target"] == chosen, "f1_mean"].iloc[0])],
            s=95,
            marker="*",
            color=color,
            zorder=5,
            label=f"Selected ({chosen})",
        )
        ax.set_title(TASKS[task], fontsize=8.5)
        ax.set_xlabel("Per-class training-size target")
        ax.set_xticks(list(targets))
        ax.set_xlim(60, 540)
        ax.legend(frameon=False, loc="lower right")
        bar_grid(ax)

    axes[0].set_ylabel("Mean validation macro-F1")
    return save(fig, os.path.join(out, "figure_s3_validation"))


def figure_s_dataflow(out: str) -> list[str]:
    """Privacy-safe data flow from collection to aggregate reporting."""
    apply_style()
    fig = plt.figure(figsize=(7.6, 4.1))
    ax, width, _ = inch_axes(fig)
    h2, h3 = 0.52, 0.74  # two- and three-line node heights

    def stage(x, y, w, h, title, *rest, kind="main"):
        node(ax, x, y, w, h, [(title, 8.2, "bold")] + [(t, 7.0, "normal") for t in rest], kind=kind)

    top_y, w, h = 3.55, 1.70, h2
    xs = (1.00, 2.95, 4.90, 6.85)
    stage(xs[0], top_y, w, h, "Collection", f"{CORPUS_FLOW['communities']} ARD communities")
    stage(xs[1], top_y, w, h, "Matching", "cannabis lexicon")
    stage(xs[2], top_y, w, h, "Sampling", f"{CORPUS_FLOW['sampled']} matched sentences")
    stage(xs[3], top_y, w, h, "Annotation", "paired labels")
    for a, b in zip(xs, xs[1:]):
        arrow(ax, a + w / 2, top_y, b - w / 2, top_y)

    mid_y, left_x, right_x = 2.20, 2.10, 5.30
    stage(left_x, mid_y, 2.30, h3, "Public Zenodo release", "post IDs and terms", "labels; no post text",
          kind="note")
    stage(right_x, mid_y, 2.30, h3, "Study file", "text and labels",
          f"{CORPUS_FLOW['annotated_rows']} rows, all kept")
    fork_y = (top_y - h / 2 + mid_y + h3 / 2) / 2
    line(ax, xs[3], top_y - h / 2, xs[3], fork_y)
    line(ax, left_x, fork_y, xs[3], fork_y)
    arrow(ax, left_x, fork_y, left_x, mid_y + h3 / 2)
    arrow(ax, right_x, fork_y, right_x, mid_y + h3 / 2)

    low_y, split_x, gen_x, clf_x = 0.55, 1.35, 3.85, 6.35
    stage(split_x, low_y, 2.15, h3, "Post-grouped split", "train / val / test",
          "val and test real only")
    stage(gen_x, low_y, 2.15, h3, "LLM generation", "train examples only",
          "provider transfer", kind="side")
    stage(clf_x, low_y, 2.15, h3, "DeBERTa classifiers", "aggregate metrics",
          "no verbatim quotations", kind="end")
    rail_y = (mid_y - h3 / 2 + low_y + h3 / 2) / 2
    line(ax, right_x, mid_y - h3 / 2, right_x, rail_y)
    line(ax, split_x, rail_y, right_x, rail_y)
    arrow(ax, split_x, rail_y, split_x, low_y + h3 / 2)
    arrow(ax, split_x + 1.08, low_y, gen_x - 1.08, low_y)
    arrow(ax, gen_x + 1.08, low_y, clf_x - 1.08, low_y)

    for y, label in ((3.95, "Corpus construction"), (2.75, "Two analysis files"),
                     (1.15, "Predictive study")):
        ax.text(0.02, y, label, fontsize=7.5, color="#4A4A4A", fontstyle="italic", va="center")
    ax.set_xlim(0, width)
    return save(fig, os.path.join(out, "figure_s_dataflow"))



# --- Main-text figures --------------------------------------------------------
# Stock matplotlib defaults, one chart type per figure, no text inside the axes
# beyond tick labels and cell values. Captions carry the numbers.


def _annotate_cells(ax, values, fmt, vmax):
    for i in range(3):
        for j in range(3):
            ax.text(j, i, fmt.format(values[i, j]), ha="center", va="center",
                    color="white" if values[i, j] > 0.6 * vmax else "black")


def figure1(out: str) -> list[str]:
    """Label counts per fold for both tasks (A), and the selected training sets
    after augmentation (B). Uniform small multiples."""
    plain()
    split = load_split()
    train = load_training_composition()
    x = np.arange(3)
    fig, axes = plt.subplots(2, 4, figsize=(10, 4.8), constrained_layout=True)
    for r, task in enumerate(("absa", "trad")):
        for c, fold in enumerate(FOLDS):
            axes[r, c].bar(x, split["folds"][fold][task], width=0.6, color="tab:blue")
        comp = train[task]
        axes[r, 3].bar(x, comp["real"], width=0.6, color="tab:blue", label="original")
        axes[r, 3].bar(x, comp["synthetic"], bottom=comp["real"], width=0.6, color="tab:orange", label="generated")
        axes[r, 3].set_ylim(0, 1.45 * max(a + b for a, b in zip(comp["real"], comp["synthetic"])))
        axes[r, 0].set_ylabel(f"{TASKS[task]}\nexamples")
    for c, fold in enumerate(FOLDS):
        axes[0, c].set_title(FOLD_NAMES[fold])
    axes[0, 3].set_title("Train + generated")
    axes[0, 3].legend(frameon=False, loc="upper left")
    for ax in axes.ravel():
        ax.set_xticks(x, LABELS)
    panel_label(axes[0, 0], "A", x=-0.3)
    panel_label(axes[0, 3], "B")
    return save(fig, os.path.join(out, "figure1_split_and_augmentation"))


def figure2(out: str) -> list[str]:
    """Traditional (rows) versus aspect-based (columns) labels, 479 examples."""
    plain()
    counts = OVERALL_TO_CANNABIS
    fig, ax = plt.subplots(figsize=(4.8, 4.0), constrained_layout=True)
    image = ax.imshow(counts, cmap="Blues")
    _annotate_cells(ax, counts, "{:d}", counts.max())
    ax.set_xticks(range(3), LABELS)
    ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("Aspect-based sentiment label")
    ax.set_ylabel("Traditional sentiment label")
    fig.colorbar(image, ax=ax, label="Examples")
    return save(fig, os.path.join(out, "figure2_paired_labels"))


def figure3(out: str) -> list[str]:
    """(A) test macro-F1 per condition, five seeds, baselines and prompted Gemma (five sampling runs);
    (B) the four prespecified paired contrasts with bootstrap CIs."""
    plain()
    runs = load_test_runs()
    summary = load_test_summary()
    contrasts = load_contrasts()
    gemma = load_gemma()

    fig = plt.figure(figsize=(11, 7.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=(1.0, 0.8))
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1], sharey=ax_left)
    ax_b = fig.add_subplot(gs[1, :])

    for ax, task in ((ax_left, "absa"), (ax_right, "trad")):
        for i, cond in enumerate(CONDITIONS):
            f1 = runs[(runs["task"] == task) & (runs["condition"] == cond)]["macro_f1"]
            ax.plot([i] * len(f1), f1, "o", color="tab:blue", alpha=0.3, markersize=4)
            ax.errorbar(i, f1.mean(), yerr=f1.std(ddof=1), fmt="o", color="tab:blue", capsize=4,
                        label="DeBERTa, mean ± SD (5 seeds)" if i == 0 else None)
        g = gemma[task]
        ax.plot([4] * len(g["runs"]), g["runs"], "o", color="tab:orange", alpha=0.3, markersize=4)
        ax.errorbar(4, g["macro_f1"], yerr=g["sd"], fmt="D", color="tab:orange", capsize=4,
                    label="Gemma 4, mean ± SD (5 sampling runs)")
        base = summary.loc[task]
        tfidf = max(base.loc[c, "f1_mean"] for c in base.index if c.startswith("baseline::tfidf"))
        ax.axhline(tfidf, color="grey", linestyle="-", linewidth=1, label="best TF-IDF + LR")
        ax.axhline(base.loc["baseline::stratified_random", "f1_mean"], color="grey", linestyle="--", linewidth=1, label="stratified random")
        ax.axhline(base.loc["baseline::always_neutral", "f1_mean"], color="grey", linestyle=":", linewidth=1, label="always neutral")
        ax.set_xticks(range(5), [CONDITION_NAMES[c] for c in CONDITIONS] + ["Gemma 4\nprompted"], fontsize=9)
        ax.set_title(TASKS[task])
        ax.set_ylim(0.2, 0.9)
    ax_left.set_ylabel("Test macro-F1")
    ax_left.legend(frameon=False, loc="upper left", fontsize=8)
    plt.setp(ax_right.get_yticklabels(), visible=False)

    keys = ("C1", "C2", "C3", "C4")
    ys = np.arange(len(keys))[::-1]
    for task, offset, color in (("absa", 0.15, "tab:blue"), ("trad", -0.15, "tab:green")):
        rows = [contrasts.loc[(task, k)] for k in keys]
        ax_b.errorbar([r["delta"] for r in rows], ys + offset,
                      xerr=[[r["delta"] - r["lo"] for r in rows], [r["hi"] - r["delta"] for r in rows]],
                      fmt="o", color=color, capsize=3, label=TASKS[task])
    ax_b.axvline(0, color="grey", linewidth=1)
    ax_b.set_yticks(ys, [f"{k}: {CONTRAST_NAMES[k]}" for k in keys])
    ax_b.set_xlabel("Difference in test macro-F1 (mean of 5 paired seeds, 95% bootstrap CI)")
    ax_b.legend(frameon=False, loc="lower right")
    panel_label(ax_left, "A")
    panel_label(ax_b, "B", x=-0.25)
    return save(fig, os.path.join(out, "figure3_performance_contrasts"))


def figure4(out: str) -> list[str]:
    """Confusion matrices on the 96 test examples: best DeBERTa condition (mean
    over 5 seeds) beside prompted Gemma (mean over 5 sampling runs), one row per task."""
    plain()
    runs = load_test_runs()
    gemma = load_gemma()
    vmax = 63  # largest test support (traditional neutral)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4), constrained_layout=True)
    for r, task in enumerate(("absa", "trad")):
        sub = runs[runs["task"] == task]
        best = sub.groupby("condition")["macro_f1"].mean().idxmax()
        pooled = np.mean(np.stack(sub[sub["condition"] == best]["confusion"].to_list()), axis=0)
        for c, cm in enumerate((pooled, gemma[task]["confusion"])):
            ax = axes[r, c]
            image = ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)
            _annotate_cells(ax, cm, "{:.1f}", vmax)
            ax.set_xticks(range(3), LABELS)
            ax.set_yticks(range(3), LABELS)
        axes[r, 0].set_ylabel(f"{TASKS[task]}\nreference label")
    axes[0, 0].set_title("DeBERTa")
    axes[0, 1].set_title("Gemma 4")
    for ax in axes[1]:
        ax.set_xlabel("Predicted label")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8, label="Examples")
    for ax, letter in zip(axes.ravel(), "ABCD"):
        panel_label(ax, letter, x=-0.3, y=1.02)
    return save(fig, os.path.join(out, "figure4_confusion_grid"))


FIGURES = {
    "figure1": figure1,
    "figure2": figure2,
    "figure3": figure3,
    "figure4": figure4,
    "figure_s_label_targets": figure_s_label_targets,
    "figure_s1": figure_s1,
    "figure_s2": figure_s2,
    "figure_s3": figure_s3,
    "figure_s_dataflow": figure_s_dataflow,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=list(FIGURES))
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name in args.only or list(FIGURES):
        for path in FIGURES[name](args.out):
            print(path)


if __name__ == "__main__":
    main()
