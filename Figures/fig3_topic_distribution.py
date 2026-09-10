"""
Figure 3 – Topic Distribution Across Platforms (item-level, k=10).

Rebuilt on the item-level k=10 model (the canonical unit of analysis; the
sentence-level k=7 version was length-confounded / pseudo-replicated).
Single panel: English corpus only (the Italian sub-corpus, ~284 items and
mono-platform, is reported as a limitation, not a result). BBC (9 items) is
excluded from the platform comparison.

Depends on: document_topics_items_k10.csv (from 04_item_level_ksweep.py).
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_RESULTS
from Figures.fig_config import (
    FIGURES_DIR, PLATFORM_LABELS, ITEM_PLATFORM_ORDER, TOPIC_LABELS_ITEM_K10, setup_style
)

INPUT_FILE = os.path.join(DATA_RESULTS, "document_topics_items_k10.csv")
OUTPUT_FILE = os.path.join(FIGURES_DIR, "fig3_topic_distribution.png")
OUTPUT_FILE_PDF = os.path.join(FIGURES_DIR, "fig3_topic_distribution.pdf")


def main():
    setup_style()
    print("Figure 3: Topic distribution by platform (item-level, k=10)")
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: {INPUT_FILE} not found. Run 04_item_level_ksweep.py first.")
        return

    df = pd.read_csv(INPUT_FILE)
    df = df[df['fonte'].isin(ITEM_PLATFORM_ORDER)].copy()
    df['topic_label'] = df['topic_k10'].astype(int).map(TOPIC_LABELS_ITEM_K10)

    pct = pd.crosstab(df['fonte'], df['topic_label'], normalize='index') * 100
    pct = pct.reindex(ITEM_PLATFORM_ORDER)
    pct.index = [PLATFORM_LABELS[p] for p in pct.index]
    # order topics by overall prevalence
    pct = pct[pct.sum().sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(15, 4.2))
    sns.heatmap(pct, annot=True, fmt='.1f', cmap='YlOrRd', linewidths=0.5,
                linecolor='white', ax=ax, cbar_kws={'label': '% of items', 'shrink': 0.8},
                vmin=0, vmax=pct.values.max() * 1.1, annot_kws={'size': 8})
    ax.set_title('Topic Distribution by Platform — item level (%)',
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel(''); ax.set_ylabel('')
    ax.tick_params(axis='x', rotation=40, labelsize=8)
    ax.tick_params(axis='y', rotation=0, labelsize=11)
    for lbl in ax.get_xticklabels():
        lbl.set_ha('right')

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_FILE_PDF, bbox_inches='tight', facecolor='white')
    print(f"Figure 3 saved to: {OUTPUT_FILE}")
    plt.close()


if __name__ == "__main__":
    main()
