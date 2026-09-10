"""
Shared configuration for all publication-quality figures.
Ensures visual coherence across the entire paper.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import sys

# Add root directory to sys.path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_RESULTS

# --- OUTPUT DIRECTORY ---
FIGURES_DIR = os.path.join(DATA_RESULTS, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# --- PLATFORM COLOR PALETTE ---
# Consistent colors for each data source across all figures
PLATFORM_COLORS = {
    "News_Media":        "#1a5276",   # Dark institutional blue (legacy sentence-level figs)
    "Guardian":          "#1a5276",   # The Guardian keeps the institutional blue
    "Reddit_Commento":   "#e74c3c",   # Reddit orange-red
    "Telegram":          "#0088cc",   # Telegram blue
}

# Short display names for figures
PLATFORM_LABELS = {
    "News_Media":        "News Media",
    "Guardian":          "The Guardian",
    "Reddit_Commento":   "Reddit",
    "Telegram":          "Telegram",
}

# Platform order for consistent axis ordering
PLATFORM_ORDER = ["News_Media", "Reddit_Commento", "Telegram"]

# Item-level analysis: Guardian is its own platform (News Media was ~99.7%
# Guardian once Kyiv Independent was dropped and BBC = 9 items). BBC is excluded
# from the platform comparisons (n=9 items).
ITEM_PLATFORM_ORDER = ["Guardian", "Reddit_Commento", "Telegram"]

def unify_platforms(df):
    """Group institutional news sources into a single 'News_Media' ecosystem."""
    if 'fonte' in df.columns:
        df['fonte'] = df['fonte'].replace({
            'BBC_News': 'News_Media',
            'Kyiv_Independent': 'News_Media',
            'Guardian': 'News_Media'
        })
    return df

# --- SENTIMENT COLORS ---
SENTIMENT_COLORS = {
    "Positive": "#27ae60",
    "Neutral":  "#7f8c8d",
    "Negative": "#c0392b",
}

# --- TOPIC LABELS ---
# Derived from the top-10 words of the canonical run (see data/results/lda_topics_en.txt
# and lda_topics_it.txt). EN: k=7, IT: k=21 (coherence-optimal, optimal_k_*.json).
# Interpretive labels, assigned by inspection of the top-word lists (Section 3.5).
TOPIC_LABELS_EN = {
    0:  "War & International Order",
    1:  "Drone & Missile Strikes",
    2:  "Western Leaders & Diplomacy",
    3:  "Society & Human Stories",
    4:  "Peace Talks (Trump-Putin-Zelenskyy)",
    5:  "Energy, Oil & Sanctions",
    6:  "Colloquial Opinion Discourse",
}

TOPIC_LABELS_IT = {
    0:  "Missile Attacks & Negotiations",
    1:  "War Damage & Oil Trade",
    2:  "Israel-Lebanon & N. Korea Troops",
    3:  "Strait of Hormuz & Naval Traffic",
    4:  "Frontline Positions & Brigades",
    5:  "Strikes on Russian Targets",
    6:  "US Politics & War Officials",
    7:  "Trump, Pentagon & War Response",
    8:  "Air Bases & Military Logistics",
    9:  "Ceasefire & Peace Declarations",
    10: "FPV Drones & Interceptors",
    11: "Ukrainian Armed Forces Attacks",
    12: "Military Funding & Missile Systems",
    13: "Iran Command & Moscow Ties",
    14: "Public Decisions & Putin",
    15: "Diplomatic Pressure & Civilians",
    16: "Missile & Air Defense Operations",
    17: "Iran, Nuclear & US Weapons",
    18: "Russian Units & Satellite Ops",
    19: "Journalists & War Coverage",
    20: "Defense Ministry Statements",
}

# --- ITEM-LEVEL TOPIC LABELS (canonical analysis, k=10) ---
# Interpretive labels from the top words of the item-level k=10 model, the
# operational model of Section 3.5 (data/results/item_topics_k10.txt).
TOPIC_LABELS_ITEM_K10 = {
    0:  "Hungary & Orbán politics",
    1:  "Energy & prices",
    2:  "General war commentary",
    3:  "Drone & missile strikes",
    4:  "Misc international news",
    5:  "Ukraine–Russia military",
    6:  "Trump & geopolitics",
    7:  "Iran & nuclear",
    8:  "Peace talks (Putin/Zelensky/Trump)",
    9:  "Europe & NATO defence",
}

# --- MATPLOTLIB GLOBAL STYLE ---
def setup_style():
    """Apply publication-quality matplotlib style globally."""
    plt.rcParams.update({
        # Font settings (serif for LaTeX compatibility)
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif'],
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        
        # Figure quality
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        
        # Axes style
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        
        # Legend
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
        
        # LaTeX text rendering (if available)
        'text.usetex': False,  # Set True only if LaTeX is installed
        'mathtext.fontset': 'dejavuserif',
    })

# Auto-apply style when imported
setup_style()
