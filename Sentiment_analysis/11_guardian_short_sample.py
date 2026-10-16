"""
Empirical length check for Section 4.3 (triage 2.11; reviewer 2).

The paper attributes the Guardian's 90.7% argmax-negative share to document length:
averaging ~46 sentence-level probability vectors drives the argmax towards the modal
class. The simulation in Figure 1 shows the mechanism; this script tests it on the real
articles. Each Guardian article is cut down to a few of its own sentences, sampled at
random, and the item-level measures are recomputed exactly as in the paper
(mean-probability aggregation, then argmax; and the mean negative probability):

  n = 2, n = 3        a fixed number of sentences per article
  reddit-matched      the number of sentences drawn from Reddit's item-length distribution
  telegram-matched    the same for Telegram

Articles with fewer sentences than drawn keep all of theirs. Each variant is repeated
R times with different samples; the median and 2.5-97.5 percentiles across repetitions
are reported, beside the full-length Guardian values and the observed Reddit and Telegram
values. If length drives the argmax share, it should fall towards the short-form
platforms' level while the mean negative probability stays where it was.

Output: data/results/guardian_short_sample.csv
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_RESULTS

SENTENCES = os.path.join(DATA_RESULTS, "sentiment_results_multilingua.csv")
ITEMS_META = os.path.join(DATA_RESULTS, "items_en_meta.csv")
OUT = os.path.join(DATA_RESULTS, "guardian_short_sample.csv")

R, SEED = 200, 42
PROBS = ['neg_prob', 'neu_prob', 'pos_prob']


def main():
    rng = np.random.default_rng(SEED)
    s = pd.read_csv(SENTENCES, low_memory=False, usecols=['id_originale', 'fonte', 'lingua_rilevata'] + PROBS)
    s = s[(s['fonte'] == 'Guardian') & (s['lingua_rilevata'] == 'en')].copy()
    s['parent'] = s['id_originale'].astype(str).str.replace(r'_s\d+$', '', regex=True)
    arts = [g[PROBS].to_numpy() for _, g in s.groupby('parent', sort=True)]
    meta = pd.read_csv(ITEMS_META)
    lengths = {p: meta.loc[meta['fonte'] == p, 'n_sentences'].to_numpy() for p in ('Reddit_Commento', 'Telegram')}
    print(f"Guardian articles: {len(arts)}; sentences: {sum(len(a) for a in arts)}")

    rows = []
    def add(label, share, pneg, spread=None):
        r = dict(condition=label, argmax_negative_share=round(share * 100, 2), mean_neg_prob=round(pneg * 100, 2))
        if spread:
            r.update(share_lo=round(spread[0] * 100, 2), share_hi=round(spread[1] * 100, 2),
                     pneg_lo=round(spread[2] * 100, 2), pneg_hi=round(spread[3] * 100, 2))
        rows.append(r)

    full = np.vstack([a.mean(axis=0) for a in arts])
    add("Guardian, full articles", (full.argmax(1) == 0).mean(), full[:, 0].mean())
    sent = np.vstack(arts)
    add("Guardian, single sentences", (sent.argmax(1) == 0).mean(), sent[:, 0].mean())

    variants = {"Guardian, 2 sentences per article": lambda: 2,
                "Guardian, 3 sentences per article": lambda: 3,
                "Guardian, Reddit-matched length": lambda: int(rng.choice(lengths['Reddit_Commento'])),
                "Guardian, Telegram-matched length": lambda: int(rng.choice(lengths['Telegram']))}
    for label, draw_n in variants.items():
        shares, pnegs = [], []
        for _ in range(R):
            agg = np.empty((len(arts), 3))
            for i, a in enumerate(arts):
                n = min(draw_n(), len(a))
                agg[i] = a[rng.choice(len(a), n, replace=False)].mean(axis=0)
            shares.append((agg.argmax(1) == 0).mean())
            pnegs.append(agg[:, 0].mean())
        sp = (*np.percentile(shares, [2.5, 97.5]), *np.percentile(pnegs, [2.5, 97.5]))
        add(label, np.median(shares), np.median(pnegs), sp)
        print(f"  {label}: done", flush=True)

    for p, name in (('Reddit_Commento', 'Reddit'), ('Telegram', 'Telegram')):
        d = meta[meta['fonte'] == p]
        add(f"{name}, observed (mean {d['n_sentences'].mean():.1f} sentences per item)",
            (d['sentiment_item_meanprob'] == 'Negative').mean(), d['neg_prob'].mean())

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print("\n" + out.to_string(index=False))
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
