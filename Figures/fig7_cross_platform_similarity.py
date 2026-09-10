"""
Figure 7 – Cross-Platform Narrative Similarity Matrix
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_RESULTS
from Figures.fig_config import (FIGURES_DIR, PLATFORM_LABELS, ITEM_PLATFORM_ORDER, setup_style)

# Item-level k=10 model (canonical). Consistent with the k-sweep robustness
# check (item_ksweep_similarity.csv); this shows the chosen k=10 matrix.
INPUT_FILE = os.path.join(DATA_RESULTS, "document_topics_items_k10.csv")
OUTPUT_FILE = os.path.join(FIGURES_DIR, "fig7_cross_platform_similarity.png")
OUTPUT_PDF = os.path.join(FIGURES_DIR, "fig7_cross_platform_similarity.pdf")

def main():
    setup_style()
    print("Figure 7: Cross-Platform Similarity Matrix (item-level, k=10)")
    df = pd.read_csv(INPUT_FILE)
    df = df[df['fonte'].isin(ITEM_PLATFORM_ORDER)].copy()
    topic_col = 'topic_k10'
    df.dropna(subset=[topic_col], inplace=True)
    df[topic_col] = df[topic_col].astype(int)
    all_topics = sorted(df[topic_col].unique())
    vectors = {}
    for platform in df['fonte'].unique():
        counts = df[df['fonte']==platform][topic_col].value_counts()
        vec = np.array([counts.get(t,0) for t in all_topics], dtype=float)
        vectors[platform] = vec / vec.sum() if vec.sum()>0 else vec
    platforms = [p for p in ITEM_PLATFORM_ORDER if p in vectors]
    names = [PLATFORM_LABELS[p] for p in platforms]
    matrix = np.array([vectors[p] for p in platforms])
    cos_sim = cosine_similarity(matrix)
    n = len(platforms)
    js_sim = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            js_sim[i,j] = 1 - jensenshannon(matrix[i], matrix[j])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.heatmap(cos_sim, annot=True, fmt='.3f', cmap='YlGnBu', xticklabels=names, yticklabels=names,
                ax=ax1, vmin=0.4, vmax=1.0, linewidths=1, linecolor='white',
                cbar_kws={'label':'Cosine Similarity','shrink':0.8}, annot_kws={'size':12,'fontweight':'bold'})
    ax1.set_title('(a) Cosine Similarity', fontsize=13, fontweight='bold', pad=12)
    ax1.tick_params(axis='x', rotation=30); ax1.tick_params(axis='y', rotation=0)
    sns.heatmap(js_sim, annot=True, fmt='.3f', cmap='YlGnBu', xticklabels=names, yticklabels=names,
                ax=ax2, vmin=0.4, vmax=1.0, linewidths=1, linecolor='white',
                cbar_kws={'label':'JS Similarity (1-JSD)','shrink':0.8}, annot_kws={'size':12,'fontweight':'bold'})
    ax2.set_title('(b) Jensen-Shannon Similarity', fontsize=13, fontweight='bold', pad=12)
    ax2.tick_params(axis='x', rotation=30); ax2.tick_params(axis='y', rotation=0)
    plt.suptitle('Cross-Platform Narrative Similarity (item level, k=10)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_PDF, bbox_inches='tight', facecolor='white')
    print(f"Figure 7 saved to: {OUTPUT_FILE}")
    plt.close()

if __name__ == "__main__":
    main()
