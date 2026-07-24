"""
Manuscript figures for v12 -- matched to the revised_v6 figure plan
===================================================================
Regenerates Figure 1 (4 panels) and Figure 2 (2 panels) EXACTLY as captioned in
sivroni2025_temporal_decoding_revised_v6.html, in the author's report39 colour
scheme (violet #8324c4 = local/regional rate, crimson-pink #c2185b = inter-area)
and the sivroni2023 (Fig-5) house style.

Figure 1  (2x2):
  A  Choice decoding across time (matched-dimensional inter vs regional rate)
  B  Stimulus-side decoding on unequal-contrast trials
  C  Choice decoding restricted to equal-contrast trials
  D  Stimulus decoder trained on correct, tested on error (generalization)
Figure 2  (1x2):
  A  Choice-decoding balanced accuracy by absolute contrast difference (difficulty)
  B  Choice-decoding balanced accuracy on correct vs error trials

All values are exact re-renders of PDFs/reanalysis_results.json (mouse-level,
balanced accuracy, bootstrap 95% CIs, exact paired sign-flip, Holm-corrected p).
No decoding is recomputed.

Output: FIGURES/manuscript_v12/{fig1_main,fig2_controls}.{png,pdf}
Palette override: FIGPAL=matlab for the sivroni2023 purple/magenta pair.
"""

import os, json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _lighten(hex_color, amount):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    return "#%02x%02x%02x" % (int(r+(255-r)*amount), int(g+(255-g)*amount), int(b+(255-b)*amount))

# Each figure uses a DIFFERENT report39 colour family (author request).
FAMILIES = {
    "fig1": {"local": "#8324c4", "inter": "#c2185b"},   # purple / pink
    "fig2": {"local": "#1a7a33", "inter": "#00838f"},   # green / turquoise
}
LAB = {"local": "Regional firing rate", "inter": "Inter-area correlation"}
GREY = "#8A8A8A"

# module-level colour state; set per figure via apply_family()
LOCAL = INTER = LOCAL_L = INTER_L = None
COL = COL_L = None

def apply_family(name):
    global LOCAL, INTER, LOCAL_L, INTER_L, COL, COL_L
    fam = FAMILIES[name]
    LOCAL, INTER = fam["local"], fam["inter"]
    LOCAL_L, INTER_L = _lighten(LOCAL, 0.80), _lighten(INTER, 0.80)
    COL = {"local": LOCAL, "inter": INTER}
    COL_L = {"local": LOCAL_L, "inter": INTER_L}

apply_family("fig1")

plt.rcParams.update({
    "font.family": "Arial", "font.size": 9, "axes.linewidth": 0.9,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "xtick.major.width": 0.9, "ytick.major.width": 0.9,
    "legend.frameon": False, "figure.dpi": 150,
})

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUTDIR = os.path.join(ROOT, "FIGURES", "manuscript_v12")
os.makedirs(OUTDIR, exist_ok=True)

with open(os.path.join(ROOT, "PDFs", "reanalysis_results.json"), encoding="utf-8") as f:
    D = json.load(f)
ROWS = D["session_rows"]

WIN = ["pre_early", "pre_late", "post_early", "post_late"]
WIN_C = {"pre_early": -375, "pre_late": -125, "post_early": 125, "post_late": 375}
WIN_TXT = {"pre_early": "−500 to\n−250", "pre_late": "−250 to\n0",
           "post_early": "0 to\n+250", "post_late": "+250 to\n+500"}


def per_mouse(outcome, feat):
    b = defaultdict(list)
    for r in ROWS:
        if r["outcome"] == outcome and r["feature"] == feat and r["value"] is not None:
            b[r["mouse"]].append(r["value"])
    return np.array([np.mean(b[m]) for m in sorted(b)])


def summ(section, outcome):
    for e in D[section]:
        if e["outcome"] == outcome:
            return e
    return None


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "n.s."


def style_ax(ax):
    ax.spines["left"].set_position(("outward", 3))
    ax.spines["bottom"].set_position(("outward", 3))


def time_panel(ax, section, outcome_fmt, title, ylim, star=True, show_yl=True):
    """Line of local & inter across 4 windows with bootstrap 95% CI bands."""
    x = [WIN_C[w] for w in WIN]
    for feat in ["local", "inter"]:
        m, lo, hi = [], [], []
        ok = True
        for w in WIN:
            s = summ(section, outcome_fmt.format(w=w))
            if s is None:
                ok = False; break
            m.append(s[feat + "_mean"]); lo.append(s[feat + "_ci"][0]); hi.append(s[feat + "_ci"][1])
        if not ok:
            continue
        m, lo, hi = map(np.array, (m, lo, hi))
        ax.fill_between(x, lo, hi, color=COL_L[feat], alpha=0.85, lw=0, zorder=1)
        ax.plot(x, m, "-o", color=COL[feat], lw=1.7, ms=5, zorder=3,
                markeredgecolor="white", markeredgewidth=0.6, label=LAB[feat])
    ax.axhline(0.5, ls="--", lw=1.0, color="k", zorder=0)
    ax.axvline(0, ls=":", lw=1.0, color=GREY, zorder=0)
    ax.set_xticks(x); ax.set_xticklabels([WIN_TXT[w] for w in WIN], fontsize=7.2)
    ax.set_ylim(*ylim)
    if show_yl:
        ax.set_ylabel("Balanced accuracy")
    ax.set_title(title, fontsize=8.6)
    if star:
        for w in ["post_early", "post_late"]:
            s = summ(section, outcome_fmt.format(w=w))
            if s and "paired_holm_p" in s:
                ax.text(WIN_C[w], s["local_ci"][1] + 0.008, stars(s["paired_holm_p"]),
                        ha="center", va="bottom", fontsize=9)
    style_ax(ax)


def group_panel(ax, section, outcomes, xlabels, title, ylim, show_yl=True):
    """Grouped bars (mean fill + 95% CI whisker + per-mouse dots), local vs inter."""
    width = 0.34
    xb = np.arange(len(outcomes))
    for k, feat in enumerate(["inter", "local"]):
        off = (-0.5 + k) * width
        for j, oc in enumerate(outcomes):
            s = summ(section, oc)
            if s is None:
                continue
            m = s[feat + "_mean"]; lo, hi = s[feat + "_ci"]
            xx = xb[j] + off
            # light bar (tint) + strong dots/whisker, mirroring the strong-line/light-band of Fig 1
            ax.bar(xx, m - 0.5, width * 0.92, bottom=0.5, color=COL_L[feat],
                   edgecolor=COL[feat], linewidth=0.9, zorder=1)
            ax.plot([xx, xx], [lo, hi], color=COL[feat], lw=1.7, zorder=3)
            vals = per_mouse(oc, feat)
            jit = (np.random.RandomState(j * 7 + k).rand(len(vals)) - 0.5) * width * 0.5
            ax.scatter(xx + jit, vals, s=12, color=COL[feat], alpha=0.95, zorder=4,
                       edgecolor="white", linewidth=0.4)
            if feat == "local" and "paired_holm_p" in s:
                ax.text(xb[j], max(hi, s["inter_ci"][1]) + 0.01, stars(s["paired_holm_p"]),
                        ha="center", va="bottom", fontsize=9)
    ax.axhline(0.5, ls="--", lw=1.0, color="k", zorder=0)
    ax.set_xticks(xb); ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylim(*ylim)
    if show_yl:
        ax.set_ylabel("Balanced accuracy")
    ax.set_title(title, fontsize=8.6)
    style_ax(ax)


def letters(fig, axes, labs):
    for ax, lab in zip(axes, labs):
        ax.text(-0.17, 1.08, lab, transform=ax.transAxes, fontsize=14,
                fontweight="bold", va="top", ha="right")


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTDIR, name + "." + ext), dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


def figure1():
    fig = plt.figure(figsize=(8.6, 6.4))
    gs = fig.add_gridspec(2, 2, wspace=0.34, hspace=0.5, left=0.09, right=0.985,
                          top=0.90, bottom=0.09)
    axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0]); axD = fig.add_subplot(gs[1, 1])

    time_panel(axA, "primary", "choice_{w}_all",
               "Choice decoding across time", (0.46, 0.72))
    time_panel(axB, "stimulus", "stimulus_{w}_unequal",
               "Stimulus-side decoding (unequal contrast)", (0.46, 0.74), show_yl=False)
    time_panel(axC, "equal_contrast", "choice_{w}_equal_contrast",
               "Choice decoding (equal contrast)", (0.44, 0.68))
    time_panel(axD, "correct_to_error", "stimulus_generalization_{w}_correct_to_error",
               "Stimulus generalization (correct→error)", (0.40, 0.66), star=False, show_yl=False)

    handles = [Line2D([0], [0], color=COL[f], lw=2.4, marker="o", ms=5,
                      markeredgecolor="white") for f in ["local", "inter"]]
    fig.legend(handles, [LAB["local"], LAB["inter"]], loc="upper center", ncol=2,
               fontsize=8.5, handlelength=1.5, columnspacing=1.8, bbox_to_anchor=(0.5, 0.99))
    letters(fig, [axA, axB, axC, axD], ["A", "B", "C", "D"])
    # shared x note
    fig.text(0.5, 0.02, "Time from stimulus onset (ms)", ha="center", fontsize=8.5)
    save(fig, "fig1_main")


def figure2():
    fig = plt.figure(figsize=(8.4, 3.4))
    gs = fig.add_gridspec(1, 2, wspace=0.30, left=0.085, right=0.985, top=0.80, bottom=0.17)
    axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])

    # A: difficulty (post-early, 0-250 ms), hard/medium/easy
    group_panel(axA, "difficulty",
                ["choice_post_early_hard", "choice_post_early_medium", "choice_post_early_easy"],
                ["Hard", "Medium", "Easy"],
                "Choice by contrast difference (0–250 ms)", (0.42, 0.78))
    # B: correct vs error (post-early, 0-250 ms)
    group_panel(axB, "correctness",
                ["choice_post_early_correct", "choice_post_early_error"],
                ["Correct", "Error"],
                "Choice on correct vs error trials (0–250 ms)", (0.44, 0.74), show_yl=False)

    handles = [Line2D([0], [0], color=COL[f], lw=3) for f in ["local", "inter"]]
    fig.legend(handles, [LAB["local"], LAB["inter"]], loc="upper center", ncol=2,
               fontsize=8.5, handlelength=1.4, columnspacing=1.8, bbox_to_anchor=(0.5, 1.0))
    letters(fig, [axA, axB], ["A", "B"])
    save(fig, "fig2_controls")


if __name__ == "__main__":
    apply_family("fig1")
    figure1()
    apply_family("fig2")
    figure2()
    print("done ->", OUTDIR)
