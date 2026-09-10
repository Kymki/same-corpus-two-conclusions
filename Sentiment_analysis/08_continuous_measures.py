"""
Three continuous sentiment measures, per source and per topic x platform (RP request).

The paper reports mean negative probability. Section 3.4 argues that the
artefact argument rules out argmax class shares but does not by itself select
among continuous summaries, so the alternatives must be reported as a
robustness check:

  (1) mean negative probability      p_neg
  (2) signed valence                 p_pos - p_neg     (direction of affect)
  (3) affective charge               1 - p_neu         (intensity irrespective
                                                        of direction)

Note for Section 5.1: the claim that Telegram reports diplomacy in a "flat
register" is properly a claim about affective charge (1 - p_neu), not about
negativity, so measure (3) is the one that should carry it.

All quantities are item-level with 95% bootstrap CIs over items.

Outputs (data/results/):
  continuous_measures_by_source.csv
  continuous_measures_by_topic_platform.csv
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_RESULTS

ITEMS = os.path.join(DATA_RESULTS, "items_en_meta.csv")
TOPICS_K10 = os.path.join(DATA_RESULTS, "document_topics_items_k10.csv")
OUT_SRC = os.path.join(DATA_RESULTS, "continuous_measures_by_source.csv")
OUT_CELL = os.path.join(DATA_RESULTS, "continuous_measures_by_topic_platform.csv")

PLATFORMS = ["Guardian", "Reddit_Commento", "Telegram"]
NAMES = {"Guardian": "Guardian", "Reddit_Commento": "Reddit", "Telegram": "Telegram"}
MIN_CELL = 30
B = 1000
SEED = 42

TOPIC_LABELS_K10 = {
    0: "Hungary & Orban politics", 1: "Energy & prices", 2: "General war commentary",
    3: "Drone & missile strikes", 4: "Misc international news", 5: "Ukraine-Russia military",
    6: "Trump & geopolitics", 7: "Iran & nuclear", 8: "Peace talks (Putin/Zelensky/Trump)",
    9: "Europe & NATO defence",
}

MEASURES = {
    "neg_prob": "mean negative probability",
    "valence": "signed valence (pos - neg)",
    "charge": "affective charge (1 - neu)",
}


def boot_ci(x, rng, b=B):
    x = np.asarray(x, float)
    means = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(b)]
    return np.percentile(means, [2.5, 97.5])


def add_measures(df):
    df = df.copy()
    df['valence'] = df['pos_prob'] - df['neg_prob']
    df['charge'] = 1 - df['neu_prob']
    return df


def main():
    items = add_measures(pd.read_csv(ITEMS))
    items = items[items['fonte'].isin(PLATFORMS)]
    rng = np.random.default_rng(SEED)

    # --- per source ---
    rows = []
    for src in PLATFORMS:
        d = items[items['fonte'] == src]
        row = {'source': NAMES[src], 'n_items': len(d)}
        for m in MEASURES:
            lo, hi = boot_ci(d[m].values, rng)
            row[f'{m}_mean'] = round(d[m].mean() * 100, 2)
            row[f'{m}_lo'] = round(lo * 100, 2)
            row[f'{m}_hi'] = round(hi * 100, 2)
        rows.append(row)
    by_src = pd.DataFrame(rows)
    by_src.to_csv(OUT_SRC, index=False)

    print("=== CONTINUOUS MEASURES BY SOURCE (item level, %; 95% bootstrap CI) ===")
    for m, desc in MEASURES.items():
        print(f"\n{desc}:")
        for _, r in by_src.iterrows():
            print(f"  {r['source']:<10} {r[f'{m}_mean']:>7.2f}  [{r[f'{m}_lo']:.2f}, {r[f'{m}_hi']:.2f}]")
    # does the source ordering change across measures?
    print("\nOrdering by measure (most -> least):")
    for m, desc in MEASURES.items():
        order = by_src.sort_values(f'{m}_mean', ascending=False)['source'].tolist()
        print(f"  {desc:<32} {' > '.join(order)}")
    print(f"\nSaved: {OUT_SRC}")

    # --- per topic x platform ---
    topics = pd.read_csv(TOPICS_K10)
    merged = items.merge(topics[['parent', 'topic_k10']], on='parent', how='inner')
    cell_rows = []
    for t in sorted(merged['topic_k10'].unique()):
        for src in PLATFORMS:
            d = merged[(merged['topic_k10'] == t) & (merged['fonte'] == src)]
            if len(d) < MIN_CELL:
                continue
            row = {'topic': int(t), 'topic_label': TOPIC_LABELS_K10.get(int(t), f"Topic {t}"),
                   'platform': NAMES[src], 'n_items': len(d)}
            for m in MEASURES:
                lo, hi = boot_ci(d[m].values, rng)
                row[f'{m}_mean'] = round(d[m].mean() * 100, 2)
                row[f'{m}_lo'] = round(lo * 100, 2)
                row[f'{m}_hi'] = round(hi * 100, 2)
            cell_rows.append(row)
    by_cell = pd.DataFrame(cell_rows)
    by_cell.to_csv(OUT_CELL, index=False)
    print(f"Saved: {OUT_CELL}  ({len(by_cell)} cells with n>={MIN_CELL})")

    # --- the Section 5.1 claim: Telegram's flat register on diplomacy ---
    print("\n=== Section 5.1 check: 'Telegram flat register on Peace talks' ===")
    pt = by_cell[by_cell['topic_label'].str.startswith('Peace talks')]
    print("  (affective charge is the measure that properly carries this claim)")
    for _, r in pt.iterrows():
        print(f"  {r['platform']:<10} neg={r['neg_prob_mean']:>6.2f}  "
              f"valence={r['valence_mean']:>7.2f}  charge={r['charge_mean']:>6.2f} "
              f"[{r['charge_lo']:.2f}, {r['charge_hi']:.2f}]  n={r['n_items']}")

    # how often does the platform ordering differ between measures, per topic?
    print("\n=== Do topic-level platform orderings agree across measures? ===")
    flips = 0
    for t in sorted(by_cell['topic'].unique()):
        sub = by_cell[by_cell['topic'] == t]
        if len(sub) < 2:
            continue
        o_neg = sub.sort_values('neg_prob_mean', ascending=False)['platform'].tolist()
        o_val = sub.sort_values('valence_mean')['platform'].tolist()      # more negative = lower valence
        o_chg = sub.sort_values('charge_mean', ascending=False)['platform'].tolist()
        agree_val = (o_neg == o_val)
        agree_chg = (o_neg == o_chg)
        if not (agree_val and agree_chg):
            flips += 1
            print(f"  [T{t} {sub.iloc[0]['topic_label'][:32]:<32}] "
                  f"neg: {'>'.join(o_neg)} | valence: {'>'.join(o_val)} | charge: {'>'.join(o_chg)}")
    print(f"\n  topics where an ordering differs across measures: {flips}/{by_cell['topic'].nunique()}")


if __name__ == "__main__":
    main()
