"""
Figure 5 – Temporal Evolution of Sentiment
Time-series showing how sentiment evolves over time across platforms.
Depends on: sentiment_results_multilingua.csv (from 01_sent.py)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_RESULTS
from Figures.fig_config import (
    FIGURES_DIR, PLATFORM_COLORS, PLATFORM_LABELS, ITEM_PLATFORM_ORDER, setup_style
)

INPUT_FILE = os.path.join(DATA_RESULTS, "sentiment_results_multilingua.csv")
OUTPUT_FILE = os.path.join(FIGURES_DIR, "fig5_sentiment_temporal.png")
OUTPUT_PDF = os.path.join(FIGURES_DIR, "fig5_sentiment_temporal.pdf")

def main():
    setup_style()
    print("=" * 60)
    print("Figure 5: Temporal Evolution of Sentiment (common window)")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Sentiment file not found: {INPUT_FILE}")
        print("Run Sentiment_analysis/01_sent.py first.")
        return

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    # Guardian as its own platform; BBC excluded (see fig_config).
    df = df[df['fonte'].isin(ITEM_PLATFORM_ORDER)].copy()
    df.dropna(subset=['data_originale_str', 'sentiment_label', 'fonte'], inplace=True)

    # Parse dates
    df['data'] = pd.to_datetime(df['data_originale_str'], format='mixed', errors='coerce', utc=True)
    df.dropna(subset=['data'], inplace=True)

    # Restrict to the COMMON window: the interval in which ALL platforms have
    # data (start = latest per-platform first date). Prevents comparing
    # platforms over near-disjoint periods.
    platforms = [p for p in ITEM_PLATFORM_ORDER if p in df['fonte'].unique()]
    common_start = max(df[df['fonte'] == p]['data'].min() for p in platforms)
    common_end = min(df[df['fonte'] == p]['data'].max() for p in platforms)
    df = df[(df['data'] >= common_start) & (df['data'] <= common_end)]
    print(f"Common window: {common_start.date()} -> {common_end.date()}")

    # Compute compound score
    if 'pos_prob' in df.columns and 'neg_prob' in df.columns:
        df['compound_score'] = df['pos_prob'] - df['neg_prob']
    else:
        df['compound_score'] = 0

    platforms = [p for p in ITEM_PLATFORM_ORDER if p in df['fonte'].unique()]

    if not platforms:
        print("ERROR: No platforms with valid temporal data.")
        return

    # --- Multi-panel time series ---
    fig, axes = plt.subplots(len(platforms), 1, figsize=(14, 3.5 * len(platforms)), sharex=True)
    if len(platforms) == 1:
        axes = [axes]

    for ax, platform in zip(axes, platforms):
        df_p = df[df['fonte'] == platform].copy()
        df_p.set_index('data', inplace=True)

        # Weekly aggregation
        weekly = df_p.resample('W').agg(
            mean_positive=('pos_prob', 'mean') if 'pos_prob' in df_p.columns else ('compound_score', 'mean'),
            mean_negative=('neg_prob', 'mean') if 'neg_prob' in df_p.columns else ('compound_score', 'mean'),
            mean_compound=('compound_score', 'mean'),
            volume=('compound_score', 'count'),
        ).dropna()

        if weekly.empty:
            ax.text(0.5, 0.5, f'No temporal data for {PLATFORM_LABELS[platform]}',
                    ha='center', va='center', transform=ax.transAxes)
            continue

        color = PLATFORM_COLORS[platform]

        # Plot compound sentiment with rolling average
        ax.plot(weekly.index, weekly['mean_compound'],
                color=color, alpha=0.3, linewidth=0.8)

        # 3-week rolling mean for smoothing
        if len(weekly) >= 3:
            rolling = weekly['mean_compound'].rolling(window=3, center=True).mean()
            ax.plot(weekly.index, rolling, color=color, linewidth=2.5,
                    label=f'Compound Score (3-week avg)')

        # Fill between positive and negative regions
        ax.fill_between(weekly.index, 0, weekly['mean_compound'],
                        where=weekly['mean_compound'] >= 0,
                        alpha=0.1, color='#27ae60', interpolate=True)
        ax.fill_between(weekly.index, 0, weekly['mean_compound'],
                        where=weekly['mean_compound'] < 0,
                        alpha=0.1, color='#c0392b', interpolate=True)

        # Volume overlay (secondary axis)
        ax2 = ax.twinx()
        ax2.bar(weekly.index, weekly['volume'], width=5, alpha=0.08, color=color)
        ax2.set_ylabel('Volume', fontsize=8, color='#bdc3c7')
        ax2.tick_params(axis='y', labelsize=7, colors='#bdc3c7')
        ax2.set_ylim(0, weekly['volume'].max() * 4)

        # Zero line
        ax.axhline(y=0, color='#95a5a6', linestyle='--', linewidth=0.6, alpha=0.5)

        ax.set_ylabel('Compound Score', fontsize=10)
        ax.set_title(PLATFORM_LABELS[platform], fontsize=12, fontweight='bold',
                     color=color, loc='left', pad=8)
        ax.set_ylim(-0.6, 0.6)
        
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc='upper right', fontsize=8, framealpha=0.8)

    # Format x-axis
    axes[-1].set_xlabel('Date', fontsize=11)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30, ha='right')

    plt.suptitle('Temporal Evolution of Sentiment Across Platforms',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_PDF, bbox_inches='tight', facecolor='white')
    print(f"Figure 5 saved to: {OUTPUT_FILE}")
    plt.close()

if __name__ == "__main__":
    main()
