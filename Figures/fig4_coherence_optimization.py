"""
Figure 4 – Topic Coherence Optimization (item-level).

Rebuilt on the item-level k-sweep (the canonical analysis). Shows C_v coherence
over k in {5,7,10,15,21}; the operational model uses k=10, the coherence
optimum. Single panel (English item corpus); the Italian panel is dropped.

Depends on: item_coherence_values.csv (from 04_item_level_ksweep.py).
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_RESULTS
from Figures.fig_config import FIGURES_DIR, setup_style

INPUT_CSV = os.path.join(DATA_RESULTS, "item_coherence_values.csv")
OUTPUT_FILE = os.path.join(FIGURES_DIR, "fig4_coherence_optimization.png")
OUTPUT_FILE_PDF = os.path.join(FIGURES_DIR, "fig4_coherence_optimization.pdf")


def main():
    setup_style()
    print("Figure 4: Topic coherence optimization (item-level)")
    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: {INPUT_CSV} not found. Run 04_item_level_ksweep.py first.")
        return

    df = pd.read_csv(INPUT_CSV).sort_values('k')
    k = df['k'].values
    cv = df['coherence_cv'].values
    best_i = int(np.argmax(cv))
    best_k, best_cv = k[best_i], cv[best_i]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k, cv, 'o-', color='#2c3e50', linewidth=2, markersize=8,
            markerfacecolor='#3498db', markeredgecolor='#2c3e50', markeredgewidth=1.2, zorder=3)
    ax.plot(best_k, best_cv, '*', color='#e74c3c', markersize=20, markeredgecolor='#c0392b',
            markeredgewidth=1, zorder=5, label=f'Selected = coherence optimum (k={best_k})')
    ax.annotate(f'$C_v$ = {best_cv:.3f}', xy=(best_k, best_cv),
                xytext=(best_k + 1.5, best_cv - 0.006), fontsize=10, color='#c0392b',
                fontweight='bold', arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.2))
    ax.set_xlabel('Number of Topics ($k$)', fontsize=12)
    ax.set_ylabel('Coherence Score ($C_v$)', fontsize=12)
    ax.set_title('Topic Coherence Optimization — item level ($C_v$)',
                 fontsize=13, fontweight='bold', pad=10)
    ax.set_xticks(k)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_FILE_PDF, bbox_inches='tight', facecolor='white')
    print(f"Figure 4 saved to: {OUTPUT_FILE}")
    plt.close()


if __name__ == "__main__":
    main()
