"""
Fast re-plot of the supplementary robustness figure (Figure S1) from the cached
numbers in FIGURES/manuscript_v12/fig3_robustness_data.json — NO decoding rerun.

Use this to recolour/tweak Figure S1 instantly. The slow decode
(make_robustness_fig_report39.py) only needs to run once to produce the cache.

Palette (report39 "bordeaux" family, matching the manuscript):
  strong=#880e4f (bordeaux), red=#c62828, neutral=#9e9e9e, wine=#ad1457
Output: FIGURES/manuscript_v12/fig3_robustness_report39.png (+ .pdf)
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BORDEAUX, RED, GREY, WINE = "#880e4f", "#c62828", "#9e9e9e", "#ad1457"

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "FIGURES", "manuscript_v12", "fig3_robustness_data.json")
OUT = os.path.join(HERE, "..", "FIGURES", "manuscript_v12", "fig3_robustness_report39.png")

with open(DATA, encoding="utf-8") as f:
    D = json.load(f)

plt.rcParams.update({"font.family": "Arial", "axes.linewidth": 0.9})
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# A. Confound control
ax = axes[0, 0]
cc = D["confound_control"]
names = ["Corr only", "Confounds\nonly", "Full model"]
means = [cc["corr_only"]["mean"], cc["confounds_only"]["mean"], cc["full_model"]["mean"]]
errs = [cc["corr_only"]["std"], cc["confounds_only"]["std"], cc["full_model"]["std"]]
bars = ax.bar(names, means, yerr=errs, capsize=5, color=[RED, GREY, BORDEAUX])
ax.axhline(50, color="gray", linestyle="--")
ax.set_ylabel("Accuracy (%)"); ax.set_title("A. Confound Control Analysis"); ax.set_ylim(45, 65)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
for b, a in zip(bars, means):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5, f"{a:.1f}%", ha="center", va="bottom", fontsize=10)

# B. Cross-mouse
ax = axes[0, 1]
cm = D["cross_mouse"]
ax.bar([n[:4] for n in cm["names"]], cm["accuracy_pct"], color=BORDEAUX)
ax.axhline(50, color="gray", linestyle="--", label="Chance")
ax.axhline(cm["mean_pct"], color=RED, linestyle="-", label=f"Mean: {cm['mean_pct']:.0f}%")
ax.set_ylabel("Accuracy (%)"); ax.set_title("B. Cross-Mouse Validation"); ax.set_ylim(45, 65)
ax.legend(loc="upper right", fontsize=8); ax.tick_params(axis="x", rotation=45)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# C. Sliding window
ax = axes[0, 2]
sw = D["sliding_window"]
xp = range(len(sw))
ax.errorbar(xp, [r["accuracy_pct"] for r in sw], yerr=[r["std_pct"] for r in sw],
            marker="o", capsize=5, color=BORDEAUX, linewidth=2, markersize=8)
ax.axhline(50, color="gray", linestyle="--")
ax.set_xticks(list(xp)); ax.set_xticklabels([str(r["start_ms"]) for r in sw], rotation=45)
ax.set_xlabel("Window start (ms from stim)"); ax.set_ylabel("Accuracy (%)"); ax.set_title("C. Sliding Window Analysis")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# D. Model comparison
ax = axes[1, 0]
mc = D["model_comparison"]
mn = list(mc.keys())
ax.barh(mn, [mc[m]["mean_pct"] for m in mn], xerr=[mc[m]["std_pct"] for m in mn], capsize=5, color=WINE)
ax.axvline(50, color="gray", linestyle="--")
ax.set_xlabel("Accuracy (%)"); ax.set_title("D. Model Comparison"); ax.set_xlim(45, 65)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# E. Feature importance
ax = axes[1, 1]
fi = D["feature_importance"]
colors = [BORDEAUX if c > 0 else GREY for c in fi["signed_coef"]]
ax.barh(fi["features"], fi["abs_coef"], color=colors)
ax.set_xlabel("|Coefficient|"); ax.set_title("E. Feature Importance (Full Model)")
ax.axvline(0, color="gray", linestyle="-")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

axes[1, 2].axis("off")
plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight")
plt.savefig(OUT.replace(".png", ".pdf"), bbox_inches="tight")
plt.close()
print("Figure S1 re-plotted from cache ->", OUT)
