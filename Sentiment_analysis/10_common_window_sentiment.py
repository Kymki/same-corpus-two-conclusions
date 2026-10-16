"""
Sentiment within the common collection window (triage 2.10; reviewers 3 and 4).

The three sources cover different periods (Guardian ~237 days, Reddit ~87, Telegram ~41),
so platform differences in sentiment could reflect the events each period covers rather
than the platforms. Script 09 re-ran the topic analysis on the common window; this script
does the same for the sentiment results of Sections 4.3-4.4 and 4.6.

The window is defined exactly as in script 09: the intersection of the three sources'
date ranges on the symmetric subset. For the full English corpus and for the symmetric
subset, in the full and in the common window, it reports per platform:
  - mean negative probability, signed valence and affective charge (Sections 4.4, 4.6),
    with 95% intervals from the item bootstrap and from the cluster bootstrap
    (weekly clusters, data/results/item_clusters.csv; see script Topic_Modeling/18);
  - the pairwise differences in mean negative probability;
  - the argmax-negative share under mean-probability aggregation (Section 4.3), to see
    whether the length artefact persists within the window.

Output: data/results/common_window_sentiment.csv
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_PROCESSED, DATA_RESULTS

CORPUS = os.path.join(DATA_PROCESSED, "dati_testuali_preproc_consolidati.csv")
ITEMS_META = os.path.join(DATA_RESULTS, "items_en_meta.csv")
CLUSTERS = os.path.join(DATA_RESULTS, "item_clusters.csv")
OUT = os.path.join(DATA_RESULTS, "common_window_sentiment.csv")

PLATFORMS = ["Guardian", "Reddit_Commento", "Telegram"]
NAMES = {"Guardian": "Guardian", "Reddit_Commento": "Reddit", "Telegram": "Telegram"}
WAR_KW = ['ukrain', 'russia', 'putin', 'zelensk', 'kyiv', 'kremlin', 'moscow',
          'donbas', 'crimea', 'kharkiv', 'kherson', 'mariupol', 'bakhmut', 'nato']
B, BOOT_SEED = 1000, 42


def load_items():
    df = pd.read_csv(CORPUS, low_memory=False)
    df = df[(df['lingua_rilevata'] == 'en') & (df['fonte'].isin(PLATFORMS))]
    df = df.dropna(subset=['testo_lemmatizzato']).copy()
    df['parent'] = df['id_originale'].astype(str).str.replace(r'_s\d+$', '', regex=True)
    df['t'] = pd.to_datetime(df['data_originale_str'], format='mixed', errors='coerce', utc=True)
    items = df.groupby(['parent', 'fonte']).agg(
        clean=('testo_pulito_base', lambda s: ' '.join(map(str, s))), t=('t', 'first')).reset_index()
    items['relevant'] = items['clean'].str.lower().apply(lambda x: any(k in x for k in WAR_KW))
    meta = pd.read_csv(ITEMS_META, usecols=['parent', 'neg_prob', 'neu_prob', 'pos_prob',
                                            'sentiment_item_meanprob'])
    items = items.merge(meta, on='parent', how='left').merge(
        pd.read_csv(CLUSTERS, usecols=['parent', 'fonte', 'cluster']), on=['parent', 'fonte'], how='left')
    assert items['neg_prob'].notna().all() and items['cluster'].notna().all()
    items['valence'] = items['pos_prob'] - items['neg_prob']
    items['charge'] = 1 - items['neu_prob']
    items['argmax_negative'] = (items['sentiment_item_meanprob'] == 'Negative').astype(float)
    return items


def boot(x, groups, rng):
    if groups is None:
        return np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)])
    codes, uniq = pd.factorize(groups)
    members = [np.where(codes == c)[0] for c in range(len(uniq))]
    out = []
    for _ in range(B):
        pick = rng.integers(0, len(members), len(members))
        out.append(x[np.concatenate([members[c] for c in pick])].mean())
    return np.array(out)


def main():
    rng = np.random.default_rng(BOOT_SEED)
    items = load_items().dropna(subset=['t'])
    sym = items[items['relevant']]
    start = max(sym.loc[sym['fonte'] == p, 't'].min() for p in PLATFORMS)
    end = min(sym.loc[sym['fonte'] == p, 't'].max() for p in PLATFORMS)
    print(f"Common window (as script 09): {start.date()} -> {end.date()} ({(end - start).days} days)")

    rows = []
    for subset, d0 in [("full", items), ("symmetric", sym)]:
        for window, d in [("full", d0), ("common", d0[(d0['t'] >= start) & (d0['t'] <= end)])]:
            boots = {}
            for p in PLATFORMS:
                dp = d[d['fonte'] == p]
                for m in ('neg_prob', 'valence', 'charge', 'argmax_negative'):
                    x = dp[m].to_numpy()
                    for scheme, groups in [("item", None), ("cluster_week", dp['cluster'].to_numpy())]:
                        bt = boot(x, groups, rng)
                        boots[(p, m, scheme)] = bt
                        lo, hi = np.percentile(bt, [2.5, 97.5])
                        rows.append(dict(subset=subset, window=window, quantity=f"mean_{m}",
                                         platform=NAMES[p], scheme=scheme, n_items=len(dp),
                                         estimate=round(x.mean() * 100, 2),
                                         ci_lo=round(lo * 100, 2), ci_hi=round(hi * 100, 2)))
            for a, b in [("Guardian", "Reddit_Commento"), ("Guardian", "Telegram"), ("Reddit_Commento", "Telegram")]:
                est = d.loc[d['fonte'] == a, 'neg_prob'].mean() - d.loc[d['fonte'] == b, 'neg_prob'].mean()
                for scheme in ("item", "cluster_week"):
                    diff = boots[(a, 'neg_prob', scheme)] - boots[(b, 'neg_prob', scheme)]
                    lo, hi = np.percentile(diff, [2.5, 97.5])
                    rows.append(dict(subset=subset, window=window, quantity="diff_neg_prob",
                                     platform=f"{NAMES[a]} - {NAMES[b]}", scheme=scheme, n_items=None,
                                     estimate=round(est * 100, 2), ci_lo=round(lo * 100, 2), ci_hi=round(hi * 100, 2)))
            print(f"  {subset} corpus, {window} window: done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    show = out[(out['scheme'] == 'cluster_week') & out['quantity'].isin(['mean_neg_prob', 'diff_neg_prob', 'mean_argmax_negative'])]
    print(show.pivot_table(index=['subset', 'quantity', 'platform'], columns='window',
                           values=['estimate', 'ci_lo', 'ci_hi'], aggfunc='first').round(1).to_string())
    print(out[out['quantity'] == 'mean_neg_prob'].pivot_table(index=['subset', 'platform'], columns='window',
                                                              values='n_items', aggfunc='first').to_string())
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
