"""
Figure 2: item-level negativity across platforms, on the full corpus.

(a) the distribution of mean negative probability per item, by platform;
(b) platform means with 95% bootstrap confidence intervals over items.

Both panels use the length-invariant measure of Section 4.4, so the 3.4-point
spread shown here is the corrected one, not the 21-point spread the argmax rules
produce.

Input:  data/results/items_en_meta.csv
Outputs (data/results/figures/):
  fig_item_negativity.png / .pdf
  fig_item_negativity_values.csv
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fig_config import FIGURES_DIR, PLATFORM_COLORS, PLATFORM_LABELS  # noqa: E402

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_RESULTS  # noqa: E402

META = os.path.join(DATA_RESULTS, "items_en_meta.csv")
OUT_PNG = os.path.join(FIGURES_DIR, "fig_item_negativity.png")
OUT_PDF = os.path.join(FIGURES_DIR, "fig_item_negativity.pdf")
OUT_VALS = os.path.join(FIGURES_DIR, "fig_item_negativity_values.csv")

ORDER = ["Guardian", "Reddit_Commento", "Telegram"]
MARKERS = {"Guardian": "o", "Reddit_Commento": "s", "Telegram": "^"}
B, SEED = 1000, 42


def boot_ci(x, rng, b=B):
    x = np.asarray(x, float)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(b)]
    return np.percentile(m, [2.5, 97.5])


def main():
    df = pd.read_csv(META, low_memory=False)
    df = df[df["fonte"].isin(ORDER)].dropna(subset=["neg_prob"]).copy()
    df["neg_pct"] = df["neg_prob"] * 100
    rng = np.random.default_rng(SEED)

    data = [df.loc[df["fonte"] == s, "neg_pct"].to_numpy() for s in ORDER]
    rows = []
    for s, v in zip(ORDER, data):
        lo, hi = boot_ci(v, rng)
        rows.append({"source": PLATFORM_LABELS[s], "n_items": len(v),
                     "mean_negprob_pct": round(v.mean(), 2),
                     "ci_lo": round(lo, 2), "ci_hi": round(hi, 2)})
    stats = pd.DataFrame(rows)
    print(stats.to_string(index=False))
    print(f"spread fra la media piu' alta e la piu' bassa: "
          f"{stats.mean_negprob_pct.max() - stats.mean_negprob_pct.min():.1f} punti")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # (a) distribution: violin for shape, boxplot for the quartiles
    parts = ax1.violinplot(data, positions=range(len(ORDER)), widths=0.78,
                           showextrema=False, showmedians=False)
    for body, s in zip(parts["bodies"], ORDER):
        body.set_facecolor(PLATFORM_COLORS[s])
        body.set_alpha(0.28)
        body.set_edgecolor(PLATFORM_COLORS[s])
        body.set_linewidth(1.1)
    bp = ax1.boxplot(data, positions=range(len(ORDER)), widths=0.20, showfliers=False,
                     patch_artist=True, medianprops={"color": "white", "linewidth": 1.6})
    for patch, s in zip(bp["boxes"], ORDER):
        patch.set_facecolor(PLATFORM_COLORS[s])
        patch.set_edgecolor(PLATFORM_COLORS[s])
    ax1.set_xticks(range(len(ORDER)))
    ax1.set_xticklabels([PLATFORM_LABELS[s] for s in ORDER])
    ax1.set_ylabel("Mean negative probability per item (%)")
    ax1.set_ylim(0, 100)
    ax1.set_title("(a) Item-level negativity distribution", fontsize=11.5, loc="left")

    # (b) means with bootstrap CIs, on a scale tight enough to show the spread
    for i, (s, r) in enumerate(zip(ORDER, rows)):
        ax2.errorbar(r["mean_negprob_pct"], i,
                     xerr=[[r["mean_negprob_pct"] - r["ci_lo"]],
                           [r["ci_hi"] - r["mean_negprob_pct"]]],
                     fmt=MARKERS[s], color=PLATFORM_COLORS[s], markersize=8,
                     markeredgecolor="white", markeredgewidth=1.0,
                     capsize=4, elinewidth=1.8)
        ax2.annotate(f"{r['mean_negprob_pct']:.1f}%",
                     (r["mean_negprob_pct"], i), textcoords="offset points",
                     xytext=(0, -17), ha="center", fontsize=9.5,
                     color=PLATFORM_COLORS[s], fontweight="bold")
    ax2.set_yticks(range(len(ORDER)))
    ax2.set_yticklabels([PLATFORM_LABELS[s] for s in ORDER])
    ax2.set_ylim(-0.6, len(ORDER) - 0.4)
    ax2.invert_yaxis()
    ax2.set_xlabel("Mean negative probability (%)")
    ax2.set_title("(b) Platform mean, 95% bootstrap CI", fontsize=11.5, loc="left")

    fig.suptitle("Item-level negativity across platforms", fontsize=13, y=1.02)
    fig.tight_layout()
    for p in (OUT_PNG, OUT_PDF):
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    stats.to_csv(OUT_VALS, index=False)
    print(f"\nscritto: {OUT_PNG}\n         {OUT_PDF}\n         {OUT_VALS}")


if __name__ == "__main__":
    main()
