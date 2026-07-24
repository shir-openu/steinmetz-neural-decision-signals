"""
Publication figures in the Sivroni (2023) house style
=====================================================
Re-plots the trustworthy, mouse-level reanalysis (balanced accuracy, sign-flip
paired tests, 95% CIs) from PDFs/reanalysis_results.json into eNeuro-ready
figures matching the colour scheme and layout of sivroni2023 (Fig. 5):
  * purple / magenta two-colour scheme
  * shaded 95% CI bands
  * per-mouse paired dots with connecting lines
  * clean left/bottom axes, outward ticks, dashed chance line
  * bold panel letters, in-panel p-values

No decoding is recomputed here -- values come straight from the saved results,
so figures are exact re-renders of the audited numbers.

Output: FIGURES/eNeuro_v12/  (fig1_temporal_dissociation.png/.pdf, fig2_controls.png/.pdf)

Author: Shir Sivroni / regenerated for eNeuro submission
"""

import os, json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------
# House style
# ----------------------------------------------------------------------
def _lighten(hex_color, amount):
    """Mix a hex colour toward white. amount=0 -> same, 1 -> white."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"

# Palettes. "report39" = colours the author selected from
# META-TAGGING-PROJECT/REPORTS/report39.html (Cross-referencing section onward):
#   violet #8324c4 (local) + crimson-pink #c2185b (inter-area).
# "matlab" = the original sivroni2023 Fig-5 purple/magenta pair.
PALETTES = {
    "report39": {"local": "#8324c4", "inter": "#c2185b"},
    "matlab":   {"local": "#7E2F8E", "inter": "#D81B9A"},
}
PAL_NAME = os.environ.get("FIGPAL", "report39")
_PAL = PALETTES[PAL_NAME]

PURPLE    = _PAL["local"]              # local activity
PURPLE_L  = _lighten(PURPLE, 0.78)     # light CI band
MAGENTA   = _PAL["inter"]              # inter-area correlation
MAGENTA_L = _lighten(MAGENTA, 0.78)    # light CI band
CHANCE    = "#000000"
GREY      = "#8A8A8A"

# colour by feature
COL   = {"local": PURPLE,  "inter": MAGENTA}
COL_L = {"local": PURPLE_L, "inter": MAGENTA_L}
LAB   = {"local": "Local activity", "inter": "Inter-area correlation"}

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "legend.frameon": False,
    "figure.dpi": 150,
})

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
_SUFFIX = {"matlab": "eNeuro_v12", "report39": "eNeuro_v13"}
OUTDIR = os.path.join(ROOT, "FIGURES", _SUFFIX.get(PAL_NAME, f"eNeuro_{PAL_NAME}"))
os.makedirs(OUTDIR, exist_ok=True)

# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
with open(os.path.join(ROOT, "PDFs", "reanalysis_results.json"), encoding="utf-8") as f:
    D = json.load(f)
ROWS = D["session_rows"]

WIN_CENTER = {"pre_early": -375, "pre_late": -125, "post_early": 125, "post_late": 375}
WIN_ORDER  = ["pre_early", "pre_late", "post_early", "post_late"]
WIN_LABEL  = {"pre_early": "−500 to\n−250", "pre_late": "−250 to\n0",
              "post_early": "0 to\n+250", "post_late": "+250 to\n+500"}


def per_mouse(outcome, feature):
    """Mean value per mouse (averaging that mouse's sessions)."""
    b = defaultdict(list)
    for r in ROWS:
        if r["outcome"] == outcome and r["feature"] == feature and r["value"] is not None:
            b[r["mouse"]].append(r["value"])
    mice = sorted(b)
    return mice, np.array([np.mean(b[m]) for m in mice])


def summ(section, outcome):
    for e in D[section]:
        if e["outcome"] == outcome:
            return e
    raise KeyError(outcome)


def p_stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "n.s."


def style_ax(ax):
    ax.spines["left"].set_position(("outward", 3))
    ax.spines["bottom"].set_position(("outward", 3))


# ----------------------------------------------------------------------
# Paired-dot panel (sivroni2023 Fig 5F style)
# ----------------------------------------------------------------------
def paired_panel(ax, outcome, section, title, ylim=(0.35, 0.80), show_ylabel=True):
    mice, loc = per_mouse(outcome, "local")
    _,    itr = per_mouse(outcome, "inter")
    s = summ(section, outcome)
    p = s["paired_holm_p"]

    x_i, x_l = 0, 1
    # connecting lines per mouse
    for a, b in zip(itr, loc):
        ax.plot([x_i, x_l], [a, b], color=GREY, lw=0.7, alpha=0.55, zorder=1)
    # dots
    ax.scatter(np.full_like(itr, x_i), itr, s=26, color=MAGENTA, zorder=3,
               edgecolor="white", linewidth=0.5)
    ax.scatter(np.full_like(loc, x_l), loc, s=26, color=PURPLE, zorder=3,
               edgecolor="white", linewidth=0.5)
    # group means (thick horizontal ticks)
    for x, vals, c in [(x_i, itr, MAGENTA), (x_l, loc, PURPLE)]:
        m = vals.mean()
        ax.plot([x - 0.22, x + 0.22], [m, m], color=c, lw=2.4, zorder=4,
                solid_capstyle="round")

    ax.axhline(0.5, ls="--", lw=1.0, color=CHANCE, zorder=0)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(*ylim)
    ax.set_xticks([x_i, x_l])
    ax.set_xticklabels(["Inter-area\ncorrelation", "Local\nactivity"])
    if show_ylabel:
        ax.set_ylabel("Balanced accuracy")
    ax.set_title(title, fontsize=9)
    # significance bracket
    top = max(itr.max(), loc.max())
    y = min(ylim[1] - 0.03, top + 0.03)
    ax.plot([x_i, x_l], [y, y], color="k", lw=0.9)
    ax.text(0.5, y + 0.004, p_stars(p), ha="center", va="bottom", fontsize=11)
    ax.text(0.5, ylim[0] + 0.015, f"p = {p:.3f}" if p >= 1e-3 else "p < 0.001",
            ha="center", va="bottom", fontsize=8, color="#333333")
    style_ax(ax)


# ----------------------------------------------------------------------
# FIGURE 1  -- main temporal dissociation (choice decoding)
# ----------------------------------------------------------------------
def figure1():
    fig = plt.figure(figsize=(9.0, 3.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1], wspace=0.42,
                          left=0.075, right=0.985, top=0.82, bottom=0.20)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])

    # --- Panel A: accuracy across time, shaded 95% CI ---
    x = [WIN_CENTER[w] for w in WIN_ORDER]
    for feat in ["local", "inter"]:
        means, lo, hi = [], [], []
        for w in WIN_ORDER:
            s = summ("primary", f"choice_{w}_all")
            means.append(s[f"{feat}_mean"])
            lo.append(s[f"{feat}_ci"][0]); hi.append(s[f"{feat}_ci"][1])
        means, lo, hi = map(np.array, (means, lo, hi))
        axA.fill_between(x, lo, hi, color=COL_L[feat], alpha=0.8, zorder=1, lw=0)
        axA.plot(x, means, "-o", color=COL[feat], lw=1.8, ms=5.5, zorder=3,
                 markeredgecolor="white", markeredgewidth=0.6, label=LAB[feat])

    axA.axhline(0.5, ls="--", lw=1.0, color=CHANCE, zorder=0)
    axA.axvline(0, ls=":", lw=1.1, color=GREY, zorder=0)
    axA.set_xticks(x)
    axA.set_xticklabels([WIN_LABEL[w] for w in WIN_ORDER], fontsize=7.7)
    axA.set_xlabel("Time from stimulus onset (ms)", fontsize=8.5)
    axA.set_ylabel("Balanced accuracy")
    axA.set_ylim(0.46, 0.72)
    axA.set_title("Choice decoding across time", fontsize=9)
    axA.legend(loc="upper left", fontsize=7.6, handlelength=1.4, borderpad=0.2)
    # significance stars for post windows (paired local vs inter)
    for w in ["post_early", "post_late"]:
        s = summ("primary", f"choice_{w}_all")
        xx = WIN_CENTER[w]
        yy = s["local_ci"][1] + 0.012
        axA.text(xx, yy, p_stars(s["paired_holm_p"]), ha="center", va="bottom", fontsize=10)
    axA.annotate("stimulus\nonset", xy=(0, 0.47), fontsize=6.8, color=GREY,
                 ha="center", va="bottom")
    style_ax(axA)

    # --- Panel B: pre-stimulus paired ---
    paired_panel(axB, "choice_pre_late_all", "primary",
                 "Pre-stimulus (−250 to 0 ms)", ylim=(0.40, 0.66))
    # --- Panel C: post-stimulus paired ---
    paired_panel(axC, "choice_post_early_all", "primary",
                 "Post-stimulus (0 to +250 ms)", ylim=(0.44, 0.76), show_ylabel=False)

    # panel letters
    for ax, lab in [(axA, "A"), (axB, "B"), (axC, "C")]:
        ax.text(-0.16, 1.06, lab, transform=ax.transAxes, fontsize=15,
                fontweight="bold", va="top", ha="right")

    _save(fig, "fig1_temporal_dissociation")


# ----------------------------------------------------------------------
# FIGURE 2 -- controls addressing reviewer confounds
# ----------------------------------------------------------------------
def _grouped_windows(ax, section, outcome_fmt, windows, wlabels, title,
                     ylim=(0.42, 0.78), show_ylabel=True, legend=False):
    """Mean +/- 95% CI bars with per-mouse dots, local vs inter, over windows."""
    width = 0.34
    xbase = np.arange(len(windows))
    for k, feat in enumerate(["inter", "local"]):
        off = (-0.5 + k) * width
        for j, w in enumerate(windows):
            oc = outcome_fmt.format(w=w)
            s = summ(section, oc)
            m = s[f"{feat}_mean"]
            lo, hi = s[f"{feat}_ci"]
            x = xbase[j] + off
            ax.bar(x, m - 0.5, width*0.92, bottom=0.5, color=COL[feat],
                   alpha=0.28, zorder=1, edgecolor=COL[feat], linewidth=0.8)
            ax.plot([x, x], [lo, hi], color=COL[feat], lw=1.4, zorder=3)
            _, vals = per_mouse(oc, feat)
            jit = (np.random.RandomState(j*10+k).rand(len(vals)) - 0.5) * width*0.55
            ax.scatter(x + jit, vals, s=9, color=COL[feat], alpha=0.75,
                       zorder=4, edgecolor="white", linewidth=0.3)
            # star vs inter (paired) only once, above the pair
            if feat == "local":
                p = s["paired_holm_p"]
                ytop = max(hi, s["inter_ci"][1]) + 0.012
                ax.text(xbase[j], ytop, p_stars(p), ha="center", va="bottom", fontsize=9)
    ax.axhline(0.5, ls="--", lw=1.0, color=CHANCE, zorder=0)
    ax.set_xticks(xbase)
    ax.set_xticklabels(wlabels, fontsize=8)
    ax.set_ylim(*ylim)
    if show_ylabel:
        ax.set_ylabel("Balanced accuracy")
    ax.set_title(title, fontsize=9)
    style_ax(ax)


def figure2():
    fig = plt.figure(figsize=(9.2, 3.5))
    gs = fig.add_gridspec(1, 3, wspace=0.42, left=0.07, right=0.985,
                          top=0.78, bottom=0.19)
    axA, axB, axC = [fig.add_subplot(gs[0, i]) for i in range(3)]

    # A: equal-contrast (zero sensory evidence for L/R) -- stimulus-confound control
    _grouped_windows(axA, "equal_contrast", "choice_{w}_equal_contrast",
                     ["post_early", "post_late"], ["0 to +250", "+250 to +500"],
                     "Equal-contrast trials", ylim=(0.42, 0.70))

    # B: difficulty gradient (post-early) -- graded with evidence strength
    width = 0.34
    diff_order = ["hard", "medium", "easy"]
    xbase = np.arange(3)
    for k, feat in enumerate(["inter", "local"]):
        off = (-0.5 + k) * width
        for j, dd in enumerate(diff_order):
            s = summ("difficulty", f"choice_post_early_{dd}")
            m = s[f"{feat}_mean"]; lo, hi = s[f"{feat}_ci"]
            x = xbase[j] + off
            axB.bar(x, m - 0.5, width*0.92, bottom=0.5, color=COL[feat], alpha=0.28,
                    edgecolor=COL[feat], linewidth=0.8, zorder=1)
            axB.plot([x, x], [lo, hi], color=COL[feat], lw=1.4, zorder=3)
            _, vals = per_mouse(f"choice_post_early_{dd}", feat)
            jit = (np.random.RandomState(j*10+k).rand(len(vals)) - 0.5) * width*0.55
            axB.scatter(x + jit, vals, s=9, color=COL[feat], alpha=0.75, zorder=4,
                        edgecolor="white", linewidth=0.3)
            if feat == "local":
                axB.text(xbase[j], max(hi, s["inter_ci"][1]) + 0.012,
                         p_stars(s["paired_holm_p"]), ha="center", va="bottom", fontsize=9)
    axB.axhline(0.5, ls="--", lw=1.0, color=CHANCE, zorder=0)
    axB.set_xticks(xbase); axB.set_xticklabels(["Hard", "Medium", "Easy"], fontsize=8)
    axB.set_ylim(0.42, 0.78)
    axB.set_title("Post-stimulus by difficulty (0–250 ms)", fontsize=9)
    style_ax(axB)

    # C: pre-movement trials -- motor-confound control
    _grouped_windows(axC, "pre_movement", "choice_{w}_pre_movement",
                     ["post_early", "post_late"], ["0 to +250", "+250 to +500"],
                     "Pre-movement trials", ylim=(0.44, 0.68), show_ylabel=False)

    for ax, lab in [(axA, "A"), (axB, "B"), (axC, "C")]:
        ax.text(-0.16, 1.10, lab, transform=ax.transAxes, fontsize=15,
                fontweight="bold", va="top", ha="right")

    handles = [Line2D([0], [0], color=COL[f], lw=3) for f in ["inter", "local"]]
    fig.legend(handles, [LAB["inter"], LAB["local"]], loc="upper center",
               ncol=2, fontsize=8.5, handlelength=1.4, columnspacing=1.6,
               bbox_to_anchor=(0.5, 1.005))

    _save(fig, "fig2_controls")


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTDIR, f"{name}.{ext}"),
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


if __name__ == "__main__":
    figure1()
    figure2()
    print("Figures written to", OUTDIR)
