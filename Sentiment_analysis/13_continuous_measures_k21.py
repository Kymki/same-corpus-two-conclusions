"""
The three continuous measures by topic x platform at k = 21 (reviewer 2, third round).

Script 08 compares mean negative probability, signed valence and affective charge by topic
and platform on the item-level model with k = 10, and counts the topics in which the
ordering of the platforms changes from one measure to another. Coherence on the symmetric
subset peaks at k = 21, so this script repeats the same comparison on the item-level model
with k = 21 (document_topics_items_k21.csv, script 04), with script 08's functions,
thresholds and bootstrap: cells with at least 30 items, 95% percentile intervals over
items, 1,000 resamples. Topics are labelled by their four leading terms
(item_topics_k21.txt); no interpretive labels are assigned.

For every topic in which an ordering changes, it lists the pairs of platforms that swap
and whether their intervals overlap under each measure.

Outputs (data/results/):
  continuous_measures_by_topic_platform_k21.csv   one row per topic x platform cell
  continuous_measures_inversions_k21.csv          the swapping pairs, with interval overlap
"""
import os
import re
import sys
import itertools
import importlib.util

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_RESULTS  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "s08", os.path.join(os.path.dirname(os.path.abspath(__file__)), "08_continuous_measures.py"))
s08 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s08)

K = 21
TOPICS = os.path.join(DATA_RESULTS, f"document_topics_items_k{K}.csv")
TERMS = os.path.join(DATA_RESULTS, f"item_topics_k{K}.txt")
OUT_CELL = os.path.join(DATA_RESULTS, f"continuous_measures_by_topic_platform_k{K}.csv")
OUT_INV = os.path.join(DATA_RESULTS, f"continuous_measures_inversions_k{K}.csv")


def labels():
    out = {}
    for line in open(TERMS, encoding="utf-8"):
        m = re.match(r"Topic #(\d+):(.*)", line)
        if m:
            out[int(m.group(1))] = ", ".join(re.findall(r'"([^"]+)"', m.group(2))[:4])
    return out


def main():
    items = s08.add_measures(pd.read_csv(s08.ITEMS))
    items = items[items['fonte'].isin(s08.PLATFORMS)]
    topics = pd.read_csv(TOPICS)
    col = f'topic_k{K}'
    merged = items.merge(topics[['parent', col]], on='parent', how='inner')
    lab = labels()
    rng = np.random.default_rng(s08.SEED)
    rows = []
    for t in sorted(merged[col].unique()):
        for src in s08.PLATFORMS:
            d = merged[(merged[col] == t) & (merged['fonte'] == src)]
            if len(d) < s08.MIN_CELL:
                continue
            row = {'topic': int(t), 'top_terms': lab.get(int(t), ''), 'platform': s08.NAMES[src],
                   'n_items': len(d)}
            for m in s08.MEASURES:
                lo, hi = s08.boot_ci(d[m].values, rng)
                row[f'{m}_mean'] = round(d[m].mean() * 100, 2)
                row[f'{m}_lo'] = round(lo * 100, 2)
                row[f'{m}_hi'] = round(hi * 100, 2)
            rows.append(row)
    cells = pd.DataFrame(rows)
    cells.to_csv(OUT_CELL, index=False)

    # same definition as script 08: an ordering by valence runs from most negative
    inv, flips, multi = [], 0, 0
    for t, sub in cells.groupby('topic'):
        if len(sub) < 2:
            continue
        multi += 1
        o_neg = sub.sort_values('neg_prob_mean', ascending=False)['platform'].tolist()
        o_val = sub.sort_values('valence_mean')['platform'].tolist()
        o_chg = sub.sort_values('charge_mean', ascending=False)['platform'].tolist()
        if o_neg == o_val == o_chg:
            continue
        flips += 1
        g = sub.set_index('platform')
        for a, b in itertools.combinations(g.index, 2):
            higher = {'neg_prob': g.loc[a, 'neg_prob_mean'] > g.loc[b, 'neg_prob_mean'],
                      'valence': g.loc[a, 'valence_mean'] < g.loc[b, 'valence_mean'],
                      'charge': g.loc[a, 'charge_mean'] > g.loc[b, 'charge_mean']}
            if len(set(higher.values())) == 1:
                continue
            overlap = {m: not (g.loc[a, f'{m}_lo'] > g.loc[b, f'{m}_hi'] or
                               g.loc[b, f'{m}_lo'] > g.loc[a, f'{m}_hi']) for m in s08.MEASURES}
            inv.append({'topic': t, 'top_terms': g.iloc[0]['top_terms'], 'pair': f'{a}-{b}',
                        'order_neg_prob': '>'.join(o_neg), 'order_valence': '>'.join(o_val),
                        'order_charge': '>'.join(o_chg),
                        **{f'overlap_{m}': v for m, v in overlap.items()}})
    pd.DataFrame(inv).to_csv(OUT_INV, index=False)
    print(f"k={K}: {len(cells)} cells with n>={s08.MIN_CELL}; {multi} topics with at least two "
          f"platforms; orderings differ across measures in {flips} of them")
    if inv:
        print(pd.DataFrame(inv).drop(columns='top_terms').to_string(index=False))
    print(f"Saved: {OUT_CELL}\n       {OUT_INV}")


if __name__ == "__main__":
    main()
