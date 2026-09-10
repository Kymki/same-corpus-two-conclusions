"""
Figure 6 – Topic–Platform Network Graph
Bipartite network linking platforms to dominant narrative clusters.
Uses a deterministic bipartite layout for clarity.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_RESULTS
from Figures.fig_config import (
    FIGURES_DIR, PLATFORM_COLORS, PLATFORM_LABELS, ITEM_PLATFORM_ORDER,
    TOPIC_LABELS_ITEM_K10, setup_style
)

# Item-level k=10 model (canonical). BBC excluded (n=9 items).
INPUT_FILE = os.path.join(DATA_RESULTS, "document_topics_items_k10.csv")
OUTPUT_FILE = os.path.join(FIGURES_DIR, "fig6_topic_platform_network.png")
OUTPUT_PDF = os.path.join(FIGURES_DIR, "fig6_topic_platform_network.pdf")
EDGE_THRESHOLD = 3.0

def main():
    setup_style()
    print("Figure 6: Topic-Platform Network Graph (item-level, k=10)")

    df = pd.read_csv(INPUT_FILE)
    df = df[df['fonte'].isin(ITEM_PLATFORM_ORDER)].copy()
    topic_col = 'topic_k10'
    df.dropna(subset=[topic_col], inplace=True)
    df[topic_col] = df[topic_col].astype(int)
    df['topic_label'] = df[topic_col].map(lambda t: TOPIC_LABELS_ITEM_K10.get(t, f"Topic {t}"))

    pct = pd.crosstab(df['fonte'], df['topic_label'], normalize='index') * 100
    total_prev = pd.crosstab(df['fonte'], df['topic_label']).sum() / len(df) * 100

    # Build graph
    G = nx.Graph()
    platforms = [p for p in ITEM_PLATFORM_ORDER if p in pct.index]
    topics_with_edges = set()

    for p in platforms:
        G.add_node(p, node_type='platform')

    for t in pct.columns:
        G.add_node(t, node_type='topic', prevalence=total_prev.get(t, 1))

    for p in platforms:
        for t in pct.columns:
            w = pct.loc[p, t]
            if w > EDGE_THRESHOLD:
                G.add_edge(p, t, weight=w)
                topics_with_edges.add(t)

    # Remove orphan topic nodes
    orphans = [t for t in pct.columns if t not in topics_with_edges]
    G.remove_nodes_from(orphans)
    active_topics = sorted(topics_with_edges)

    # --- Deterministic bipartite layout ---
    pos = {}
    for i, p in enumerate(platforms):
        pos[p] = np.array([-3.0, (len(platforms)-1-i) * 2.5])

    for i, t in enumerate(active_topics):
        y_span = (len(platforms)-1) * 2.5
        y = y_span * (1 - i / max(len(active_topics)-1, 1))
        pos[t] = np.array([3.0, y])

    # --- Drawing ---
    fig, ax = plt.subplots(figsize=(16, 10))

    # Draw edges
    edges = list(G.edges(data=True))
    if edges:
        max_w = max(d['weight'] for _,_,d in edges)
        for (u, v, d) in edges:
            w = d['weight']
            alpha = 0.1 + 0.6 * (w / max_w)
            width = 0.5 + 5.0 * (w / max_w)
            color = PLATFORM_COLORS.get(u, PLATFORM_COLORS.get(v, '#aaa'))
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color=color, alpha=alpha, linewidth=width, zorder=1, solid_capstyle='round')
            # Edge label for strong connections
            if w > 10:
                mid_x = (pos[u][0] + pos[v][0]) / 2
                mid_y = (pos[u][1] + pos[v][1]) / 2
                ax.text(mid_x, mid_y, f'{w:.0f}%', fontsize=6, color=color,
                        alpha=0.7, ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.7, edgecolor='none'))

    # Draw platform nodes (left, square)
    for p in platforms:
        x, y = pos[p]
        ax.add_patch(plt.Rectangle((x-0.6, y-0.35), 1.2, 0.7, linewidth=2,
                                    edgecolor='white', facecolor=PLATFORM_COLORS[p],
                                    zorder=4, clip_on=False))
        ax.text(x, y, PLATFORM_LABELS[p], ha='center', va='center',
                fontsize=10, fontweight='bold', color='white', zorder=5)

    # Draw topic nodes (right, circles)
    for t in active_topics:
        if t in pos:
            prev = G.nodes[t].get('prevalence', 3)
            size = 300 + prev * 80
            x, y = pos[t]
            ax.scatter(x, y, s=size, c='#f8f9fa', edgecolors='#2c3e50',
                       linewidths=1.5, zorder=3, alpha=0.95)
            ax.text(x + 0.15, y, t, fontsize=8, ha='left', va='center', zorder=5,
                    fontstyle='italic', color='#2c3e50')

    # Legend
    handles = [mpatches.Patch(color=PLATFORM_COLORS[p], label=PLATFORM_LABELS[p]) for p in platforms]
    ax.legend(handles=handles, loc='lower left', framealpha=0.9, title='Platforms', title_fontsize=10)
    ax.set_title('Topic–Platform Network (item level, k=10)', fontsize=15, fontweight='bold', pad=15)
    ax.set_xlim(-5, 7.5)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_PDF, bbox_inches='tight', facecolor='white')
    print(f"Figure 6 saved to: {OUTPUT_FILE}")
    plt.close()

if __name__ == "__main__":
    main()
