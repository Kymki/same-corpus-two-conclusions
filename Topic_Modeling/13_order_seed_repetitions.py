"""
Solution variability of the LDA similarity results: document order x random seed.

LDA as implemented in gensim updates the model online, chunk by chunk, so its output
depends on the order in which documents are passed to it as well as on the seed
(Agrawal, Fu & Menzies 2018, "order effects"). Every published LDA result is a single
draw: one order (data/results/item_order_manifest.csv) and one seed (100). This script
refits the two principal designs under many orders and seeds, so that the variability
of the solution can be reported next to the sampling variability the item bootstrap
already captures (reviewer request, triage 2.6).

Designs (each identical to its source script except for order and seed):
  symmetric   full-length items, symmetric relevance filter        (as script 05)
  chunked     equal-length chunks, equal training chunks per source (as script 10),
              C in {10, 20}

Repetitions: the canonical order plus N_PERMUTATIONS random permutations of the items,
each fitted with every seed in SEEDS. For the symmetric design the k grid is extended
beyond 21 (triage 2.7), and platform profiles are computed in two ways (triage 2.2):
  argmax  share of items whose dominant topic is t (as in the paper)
  mean    mean of the items' topic distributions
Point estimates only: the bootstrap over items is reported for the canonical run by
scripts 05 and 10; here the quantity of interest is the spread across repetitions.
Order 0 with seed 100 of the symmetric design reproduces script 05 exactly. In the
chunked design the seed sets both the LDA random state and the subsample of training
chunks, so no repetition coincides with script 10 (which samples with seed 42).

Usage (the two designs can run in parallel; the summary needs both):
  python 13_order_seed_repetitions.py symmetric
  python 13_order_seed_repetitions.py chunked
  python 13_order_seed_repetitions.py summary

Outputs (data/results/):
  repetitions_symmetric.csv, repetitions_chunked.csv
                              one row per order, seed, (chunk size,) k, profile, pair
  repetitions_summary.csv     per design, chunk size, k, profile: how often each pair
                              leads, and quantiles of the leader's margin
"""
import os
import sys
import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models import LdaMulticore
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_PROCESSED, DATA_RESULTS
from item_order import canonical_order

INPUT_CSV = os.path.join(DATA_PROCESSED, "dati_testuali_preproc_consolidati.csv")
OUT = {d: os.path.join(DATA_RESULTS, f"repetitions_{d}.csv") for d in ("symmetric", "chunked")}
OUT_SUMMARY = os.path.join(DATA_RESULTS, "repetitions_summary.csv")

PLATFORMS = ["Guardian", "Reddit_Commento", "Telegram"]
PAIRS = {(0, 1): "Guardian-Reddit", (0, 2): "Guardian-Telegram", (1, 2): "Reddit-Telegram"}
WAR_KW = ['ukrain', 'russia', 'putin', 'zelensk', 'kyiv', 'kremlin', 'moscow',
          'donbas', 'crimea', 'kharkiv', 'kherson', 'mariupol', 'bakhmut', 'nato']

K_SYMMETRIC = [5, 7, 10, 15, 21, 25, 30]     # 25 and 30 extend the grid (triage 2.7)
K_CHUNKED = [5, 7, 10, 15, 21]
CHUNK_SIZES = [10, 20]
N_PERMUTATIONS = 10                         # plus the canonical order
SEEDS = [100, 200]                          # 100 is the seed of every published result
LDA_PASSES = 20
LDA_WORKERS = 7


def pair_sims(mat):
    cos = cosine_similarity(mat)
    return {PAIRS[(i, j)]: (cos[i, j], 1 - jensenshannon(mat[i], mat[j], base=2))
            for i in range(3) for j in range(i + 1, 3)}


def orders(n):
    """Order 0 is the canonical manifest order; order p>0 is permutation seed p."""
    yield 0, np.arange(n)
    for p in range(1, N_PERMUTATIONS + 1):
        yield p, np.random.default_rng(p).permutation(n)


def load_items():
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    df = df[(df['lingua_rilevata'] == 'en') & (df['fonte'].isin(PLATFORMS))]
    df = df.dropna(subset=['testo_lemmatizzato']).copy()
    df['parent'] = df['id_originale'].astype(str).str.replace(r'_s\d+$', '', regex=True)
    items = df.groupby(['parent', 'fonte']).agg(
        lem=('testo_lemmatizzato', ' '.join),
        clean=('testo_pulito_base', lambda s: ' '.join(map(str, s))),
    ).reset_index()
    items = canonical_order(items)
    items = items[items['clean'].str.lower().apply(lambda t: any(k in t for k in WAR_KW))]
    items = items.reset_index(drop=True)
    items['tokens'] = items['lem'].str.split()
    return items


def symmetric(items, rows):
    for o, perm in orders(len(items)):
        d = items.iloc[perm].reset_index(drop=True)
        dic = corpora.Dictionary(d['tokens'])
        dic.filter_extremes(no_below=5, no_above=0.5)
        bow = [dic.doc2bow(t) for t in d['tokens']]
        src = d['fonte'].to_numpy()
        for seed in SEEDS:
            for k in K_SYMMETRIC:
                m = LdaMulticore(corpus=bow, id2word=dic, num_topics=k, random_state=seed,
                                 passes=LDA_PASSES, workers=LDA_WORKERS)
                theta = np.zeros((len(bow), k))
                for i, b in enumerate(bow):
                    for t, p in m.get_document_topics(b, minimum_probability=0.0):
                        theta[i, t] = p
                # dominant topic; with k <= 30 the maximum always exceeds gensim's 0.01
                # threshold, so this equals the argmax used by script 05
                dom = theta.argmax(axis=1)
                profiles = {
                    'argmax': np.vstack([np.bincount(dom[src == p], minlength=k) / (src == p).sum()
                                         for p in PLATFORMS]),
                    'mean': np.vstack([theta[src == p].mean(axis=0) for p in PLATFORMS]),
                }
                for prof, mat in profiles.items():
                    for pair, (c, j) in pair_sims(mat).items():
                        rows.append(dict(design='symmetric', chunk_size=None, order=o, seed=seed,
                                         k=k, profile=prof, pair=pair,
                                         cosine=round(c, 4), js=round(j, 4)))
                print(f"  symmetric order={o:>2} seed={seed} k={k:>2} done", flush=True)


def chunked(items, rows):
    for C in CHUNK_SIZES:
        base = items[items['tokens'].map(len) >= C].reset_index(drop=True)
        for o, perm in orders(len(base)):
            d = base.iloc[perm].reset_index(drop=True)
            chunks = d['tokens'].map(lambda t: [t[i:i + C] for i in range(0, len(t) - C + 1, C)])
            flat, owner = [], []
            for i, ch in enumerate(chunks):
                for c in ch:
                    flat.append(c)
                    owner.append(i)
            owner = np.asarray(owner)
            src = d['fonte'].to_numpy()
            pools = {p: np.where(src[owner] == p)[0] for p in PLATFORMS}
            cap = min(len(v) for v in pools.values())
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                train_idx = np.concatenate([rng.choice(pools[p], cap, replace=False)
                                            for p in PLATFORMS])
                train = [flat[i] for i in train_idx]
                dic = corpora.Dictionary(train)
                dic.filter_extremes(no_below=5, no_above=0.5)
                train_bow = [dic.doc2bow(t) for t in train]
                all_bow = [dic.doc2bow(t) for t in flat]
                for k in K_CHUNKED:
                    m = LdaMulticore(corpus=train_bow, id2word=dic, num_topics=k,
                                     random_state=seed, passes=LDA_PASSES, workers=LDA_WORKERS)
                    dist = np.zeros((len(all_bow), k))
                    for i, b in enumerate(all_bow):
                        for t, p in m.get_document_topics(b, minimum_probability=0.0):
                            dist[i, t] = p
                    item = np.zeros((len(d), k))
                    np.add.at(item, owner, dist)
                    item /= np.bincount(owner, minlength=len(d)).reshape(-1, 1)
                    mat = np.vstack([item[src == p].mean(axis=0) for p in PLATFORMS])
                    for pair, (c, j) in pair_sims(mat).items():
                        rows.append(dict(design='chunked', chunk_size=C, order=o, seed=seed,
                                         k=k, profile='mean', pair=pair,
                                         cosine=round(c, 4), js=round(j, 4)))
                    print(f"  chunked C={C} order={o:>2} seed={seed} k={k:>2} done", flush=True)


def summarise(rep):
    out = []
    keys = ['design', 'chunk_size', 'k', 'profile']
    for key, g in rep.groupby(keys, dropna=False):
        runs = g.groupby(['order', 'seed'])
        lead = runs.apply(lambda r: r.sort_values('cosine', ascending=False).iloc[0]['pair'])
        margin = runs.apply(lambda r: r['cosine'].nlargest(2).diff().abs().iloc[-1])
        row = dict(zip(keys, key), runs=len(lead))
        for p in PAIRS.values():
            row[f'leads_{p}'] = int((lead == p).sum())
        q = margin.quantile([0, .25, .5, .75, 1]).round(3).tolist()
        row.update(margin_min=q[0], margin_q25=q[1], margin_median=q[2], margin_q75=q[3], margin_max=q[4])
        out.append(row)
    return pd.DataFrame(out)


def main(what):
    if what in OUT:
        items = load_items()
        print(f"Symmetric subset: {len(items)} items; "
              f"{N_PERMUTATIONS + 1} orders x {len(SEEDS)} seeds", flush=True)
        rows = []
        (symmetric if what == "symmetric" else chunked)(items, rows)
        pd.DataFrame(rows).to_csv(OUT[what], index=False)
        print(f"Saved: {OUT[what]}")
        return
    rep = pd.concat([pd.read_csv(OUT[d]) for d in OUT], ignore_index=True)
    summary = summarise(rep)
    summary.to_csv(OUT_SUMMARY, index=False)
    print("\n" + summary.to_string())
    print(f"\nSaved: {OUT_SUMMARY}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "summary")
