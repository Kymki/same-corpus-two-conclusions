"""
Topic x platform sentiment interaction, item-level (RP request, 2026-07).

Aggregate polarity is flat across platforms (~53-57% mean neg prob), so the
only place a sentiment result can still emerge is the interaction: do platforms
frame the SAME topic with different negativity? This tests it on the k=10 model.

For each (topic, platform) cell it reports the mean per-item negative
probability with a bootstrap 95% CI (resampling items within the cell, same
scheme as the k-sweep), plus the cell size. Cells with n < MIN_CELL are dropped.
BBC is excluded (n=9 items).

Output: data/results/item_topic_sentiment_cells.csv
Console: for each topic present in >=2 platforms, whether any platform pair has
non-overlapping CIs (a candidate framing difference).
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_RESULTS

ITEMS = os.path.join(DATA_RESULTS, "items_en_meta.csv")
TOPICS_K10 = os.path.join(DATA_RESULTS, "document_topics_items_k10.csv")
OUT = os.path.join(DATA_RESULTS, "item_topic_sentiment_cells.csv")

PLATFORMS = ["Guardian", "Reddit_Commento", "Telegram"]
NAMES = {"Guardian": "Guardian", "Reddit_Commento": "Reddit", "Telegram": "Telegram"}
MIN_CELL = 30
B = 1000
SEED = 42

# Draft interpretive labels for the k=10 item-level model (from top words;
# see item_topics_k10.txt). To be reviewed before the manuscript.
TOPIC_LABELS_K10 = {
    0: "Hungary & Orban politics",
    1: "Energy & prices",
    2: "General war commentary",
    3: "Drone & missile strikes",
    4: "Misc international news",
    5: "Ukraine-Russia military",
    6: "Trump & geopolitics",
    7: "Iran & nuclear",
    8: "Peace talks (Putin/Zelensky/Trump)",
    9: "Europe & NATO defence",
}


def boot_ci(x, rng, b=B):
    x = np.asarray(x, dtype=float)
    means = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(b)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    items = pd.read_csv(ITEMS)
    topics = pd.read_csv(TOPICS_K10)
    df = items.merge(topics[['parent', 'topic_k10']], on='parent', how='inner')
    df = df[df['fonte'].isin(PLATFORMS)].copy()
    rng = np.random.default_rng(SEED)

    rows = []
    for t in sorted(df['topic_k10'].unique()):
        for pl in PLATFORMS:
            cell = df[(df['topic_k10'] == t) & (df['fonte'] == pl)]['neg_prob']
            if len(cell) < MIN_CELL:
                continue
            lo, hi = boot_ci(cell.values, rng)
            rows.append({
                'topic': int(t),
                'topic_label': TOPIC_LABELS_K10.get(int(t), f"Topic {t}"),
                'platform': NAMES[pl],
                'n_items': len(cell),
                'mean_neg_prob': round(cell.mean() * 100, 1),
                'ci_lo': round(lo * 100, 1),
                'ci_hi': round(hi * 100, 1),
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"Saved: {OUT}  ({len(out)} cells with n>={MIN_CELL})\n")

    # Which topics show a platform framing difference (non-overlapping CIs)?
    print("=== Topic x platform: dove le piattaforme inquadrano lo stesso tema diversamente? ===")
    print("(coppie con CI 95% NON sovrapposti = differenza credibile)\n")
    any_diff = False
    for t in sorted(out['topic'].unique()):
        sub = out[out['topic'] == t]
        if len(sub) < 2:
            continue
        diffs = []
        recs = sub.to_dict('records')
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                a, b = recs[i], recs[j]
                if a['ci_hi'] < b['ci_lo'] or b['ci_hi'] < a['ci_lo']:
                    hi_neg = a if a['mean_neg_prob'] > b['mean_neg_prob'] else b
                    lo_neg = b if hi_neg is a else a
                    diffs.append(f"{hi_neg['platform']}({hi_neg['mean_neg_prob']}) > "
                                 f"{lo_neg['platform']}({lo_neg['mean_neg_prob']})")
        label = sub.iloc[0]['topic_label']
        cells_str = ', '.join(f"{r['platform']} {r['mean_neg_prob']}% (n={r['n_items']})" for r in recs)
        if diffs:
            any_diff = True
            print(f"  [T{t} {label}] DIFFERENZA: {'; '.join(diffs)}")
            print(f"       ({cells_str})")
        else:
            print(f"  [T{t} {label}] nessuna differenza credibile — {cells_str}")
    print()
    if any_diff:
        print(">>> C'e' almeno un tema inquadrato diversamente tra piattaforme: "
              "l'interazione topic x piattaforma regge come risultato sul sentiment.")
    else:
        print(">>> Nessuna differenza credibile in nessun tema: anche l'interazione "
              "e' piatta; il sentiment non produce un risultato nemmeno qui.")


if __name__ == "__main__":
    main()
