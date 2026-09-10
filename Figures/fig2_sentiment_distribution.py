"""
Figure 2 – Item-level Negative-Probability Distribution Across Platforms.

Rebuilt (2026-07) on the item-level MEAN NEGATIVE PROBABILITY, not argmax class
shares: the argmax item share is confounded with document length (averaging many
sentences collapses long-document items onto the negative class), so it measured
length rather than sentiment. The mean per-item negative probability is
length-robust; this figure shows the three platforms are near-indistinguishable
in aggregate polarity.

(a) violin + box of per-item mean negative probability by platform;
(b) platform mean with 95% bootstrap CI, making the overlap explicit.

Depends on: items_en_meta.csv (from Topic_Modeling/04_item_level_ksweep.py).
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_RESULTS
from Figures.fig_config import FIGURES_DIR, PLATFORM_COLORS, setup_style

INPUT_FILE = os.path.join(DATA_RESULTS, "items_en_meta.csv")
OUTPUT_FILE = os.path.join(FIGURES_DIR, "fig2_sentiment_distribution.png")
OUTPUT_PDF = os.path.join(FIGURES_DIR, "fig2_sentiment_distribution.pdf")

PLATFORMS = ["Guardian", "Reddit_Commento", "Telegram"]
LABELS = {"Guardian": "Guardian", "Reddit_Commento": "Reddit", "Telegram": "Telegram"}
COLORS = {"Guardian": PLATFORM_COLORS["News_Media"],
          "Reddit_Commento": PLATFORM_COLORS["Reddit_Commento"],
          "Telegram": PLATFORM_COLORS["Telegram"]}
SEED = 42
B = 1000


def main():
    setup_style()
    print("Figure 2: Item-level negative-probability distribution across platforms")
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: {INPUT_FILE} not found. Run 04_item_level_ksweep.py first.")
        return

    df = pd.read_csv(INPUT_FILE)
    df = df[df['fonte'].isin(PLATFORMS)].copy()
    df['neg_pct'] = df['neg_prob'] * 100

    names = [LABELS[p] for p in PLATFORMS]
    data = [df[df['fonte'] == p]['neg_pct'].values for p in PLATFORMS]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [3, 2]})

    # (a) violin + box
    parts = ax1.violinplot(data, positions=range(len(PLATFORMS)),
                           showmeans=False, showmedians=False, showextrema=False)
    for body, p in zip(parts['bodies'], PLATFORMS):
        body.set_facecolor(COLORS[p]); body.set_alpha(0.3); body.set_edgecolor(COLORS[p])
    bp = ax1.boxplot(data, positions=range(len(PLATFORMS)), widths=0.15,
                     patch_artist=True, showfliers=False)
    for box, p in zip(bp['boxes'], PLATFORMS):
        box.set_facecolor(COLORS[p]); box.set_alpha(0.7)
    for el in ['whiskers', 'caps', 'medians']:
        for line in bp[el]:
            line.set_color('#2c3e50'); line.set_linewidth(1.2)
    ax1.set_xticks(range(len(PLATFORMS))); ax1.set_xticklabels(names, fontsize=11)
    ax1.set_ylabel('Mean negative probability per item (%)', fontsize=11)
    ax1.set_title('(a) Item-level negativity distribution', fontsize=13, fontweight='bold', pad=12)
    ax1.axhline(y=50, color='#95a5a6', linestyle='--', linewidth=0.8, alpha=0.7)
    ax1.set_ylim(0, 100)

    # (b) mean with bootstrap 95% CI
    rng = np.random.default_rng(SEED)
    means, los, his = [], [], []
    for p in PLATFORMS:
        x = df[df['fonte'] == p]['neg_pct'].values
        means.append(x.mean())
        bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
        los.append(np.percentile(bs, 2.5)); his.append(np.percentile(bs, 97.5))
    y = range(len(PLATFORMS))
    for i, p in enumerate(PLATFORMS):
        ax2.errorbar(means[i], i, xerr=[[means[i]-los[i]], [his[i]-means[i]]],
                     fmt='o', color=COLORS[p], markersize=10, capsize=6,
                     capthick=2, elinewidth=2)
        ax2.text(means[i], i + 0.12, f'{means[i]:.1f}%', ha='center', fontsize=10, fontweight='bold')
    ax2.set_yticks(list(y)); ax2.set_yticklabels(names, fontsize=11)
    ax2.set_xlabel('Mean negative probability (%)', fontsize=11)
    ax2.set_title('(b) Platform mean ± 95% CI', fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlim(45, 65); ax2.invert_yaxis()
    ax2.grid(True, axis='x', alpha=0.3, linestyle='--')

    plt.suptitle('Item-level Negativity Across Platforms', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_PDF, bbox_inches='tight', facecolor='white')
    print(f"Figure 2 saved to: {OUTPUT_FILE}")
    plt.close()


if __name__ == "__main__":
    main()
