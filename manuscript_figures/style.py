"""Shared publication style. Colorblind-safe (Okabe-Ito), embedded fonts.

Figures carry no titles or footnotes: the manuscript captions do that work.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

DPI = 300
INK = "#1A1A1A"
RULE = "#4A4A4A"

SENTIMENT = {"Neutral": "#7A7A7A", "Positive": "#009E73", "Negative": "#D55E00"}
BLUE = "#0072B2"
SKY = "#56B4E9"
VERMILLION = "#D55E00"
TERM_BG = "#FFF0C2"  # cannabis-term highlight
TERM_INK = "#7A5B00"

BOX = {
    "main": ("#F4F8FB", BLUE),
    "end": (BLUE, BLUE),
    "side": ("#FBF4EE", VERMILLION),
    "note": ("#F5F5F5", "#7A7A7A"),
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": DPI,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )


def plain() -> None:
    """Stock matplotlib look for the main-text figures: DejaVu Sans, default sizes,
    spines on, tab colours. Only font embedding and DPI are set."""
    plt.rcdefaults()
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": DPI})


def save(fig, stem: str) -> list[str]:
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    paths = [f"{stem}.{ext}" for ext in ("png", "pdf")]
    for path in paths:
        # No timestamps in the PDF so reruns are byte-identical.
        meta = {"CreationDate": None, "ModDate": None} if path.endswith(".pdf") else None
        fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.06, facecolor="white", metadata=meta)
    plt.close(fig)
    return paths


def inch_axes(fig):
    """Blank axes whose coordinates are figure inches, for hand-laid diagrams."""
    ax = fig.add_axes([0, 0, 1, 1])
    width, height = fig.get_size_inches()
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_axis_off()
    return ax, width, height


def node(ax, x, y, w, h, lines, kind="main", textcolor=None):
    """Rounded box centred at (x, y) holding ``lines`` of (text, size, weight).

    ``kind`` is a BOX key or an explicit (facecolor, edgecolor) pair.
    """
    fc, ec = BOX[kind] if isinstance(kind, str) else kind
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.07",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.1,
            mutation_aspect=1,
            clip_on=False,
            zorder=2,
        )
    )
    color = textcolor or ("#FFFFFF" if kind == "end" else INK)
    gap = 0.155
    top = (len(lines) - 1) * gap / 2
    for i, (text, size, weight) in enumerate(lines):
        ax.text(
            x,
            y + top - i * gap,
            text,
            ha="center",
            va="center",
            fontsize=size,
            fontweight=weight,
            color=color,
            zorder=3,
        )


def line(ax, x1, y1, x2, y2):
    ax.plot([x1, x2], [y1, y2], color=RULE, linewidth=1.0, zorder=1, clip_on=False)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=1.0,
            color=RULE,
            shrinkA=0,
            shrinkB=0,
            clip_on=False,
            zorder=1,
        )
    )


def panel_label(ax, letter: str, x: float = -0.15, y: float = 1.04) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", ha="left", va="bottom")


def bar_grid(ax) -> None:
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.35, color="#888888")
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.6)
