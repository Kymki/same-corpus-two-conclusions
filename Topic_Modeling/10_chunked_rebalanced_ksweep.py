"""
Chunked cross-platform similarity with genuinely equal training chunks per platform.

Script 08 caps LDA training at 4,000 chunks per platform, but not every platform
has 4,000 chunks to give, so its training set is not balanced:

    C=10: Guardian 4,000 / Reddit 4,000 / Telegram 2,806
    C=20: Guardian 4,000 / Reddit 1,671 / Telegram 1,107

At C=20 The Guardian therefore supplies more than twice the training chunks of
either other source -- the token dominance the design was meant to remove.

This script repeats 08 with a single substantive change: the per-platform
training cap is set to the smallest platform's chunk count at each chunk size,
so every platform contributes exactly the same number of training chunks
(C=10: 2,806 each; C=20: 1,107 each). Everything else is identical to 08:

  - symmetric relevance criterion (>=1 war keyword) applied to all sources;
  - items split into consecutive chunks of exactly C lemmatised tokens;
  - dictionary policy filter_extremes(no_below=5, no_above=0.5) on training chunks;
  - LDA k in {5,7,10,15,21}, 20 passes, random_state=100;
  - item profile = mean of its chunks' SOFT topic distributions;
    platform profile = mean over its items (each item counts once);
  - cosine and Jensen-Shannon similarity, 1,000-replicate item bootstrap, seed 42.

Additions with respect to 08 (reporting only, no change to the estimates):
bootstrap CIs are also stored for Jensen-Shannon similarity, and the number
of training chunks per platform is written to every output row.

Output: data/results/chunked_rebalanced_ksweep_similarity.csv
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
OUT = os.path.join(DATA_RESULTS, "chunked_rebalanced_ksweep_similarity.csv")

PLATFORMS = ["Guardian", "Reddit_Commento", "Telegram"]
NAMES = {"Guardian": "Guardian", "Reddit_Commento": "Reddit", "Telegram": "Telegram"}
PAIRS = {(0, 1): "Guardian-Reddit", (0, 2): "Guardian-Telegram", (1, 2): "Reddit-Telegram"}
WAR_KW = ['ukrain', 'russia', 'putin', 'zelensk', 'kyiv', 'kremlin', 'moscow',
          'donbas', 'crimea', 'kharkiv', 'kherson', 'mariupol', 'bakhmut', 'nato']

K_SWEEP = [5, 7, 10, 15, 21]
CHUNK_SIZES = [10, 20]
LDA_PASSES = 20
LDA_WORKERS = 7
RANDOM_STATE = 100
N_BOOT = 1000
SEED = 42


def pair_sims(mat):
    out = {}
    cos = cosine_similarity(mat)
    for i in range(mat.shape[0]):
        for j in range(i + 1, mat.shape[0]):
            out[(i, j)] = (cos[i, j], 1 - jensenshannon(mat[i], mat[j], base=2))
    return out


def main():
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    df = df[(df['lingua_rilevata'] == 'en') & (df['fonte'].isin(PLATFORMS))]
    df = df.dropna(subset=['testo_lemmatizzato']).copy()
    df['parent'] = df['id_originale'].astype(str).str.replace(r'_s\d+$', '', regex=True)
    items = df.groupby(['parent', 'fonte']).agg(
        lem=('testo_lemmatizzato', ' '.join),
        clean=('testo_pulito_base', lambda s: ' '.join(map(str, s))),
    ).reset_index()
    items = canonical_order(items)
    items = items[items['clean'].str.lower().apply(lambda t: any(k in t for k in WAR_KW))].copy()
    items['tokens'] = items['lem'].str.split()

    rng = np.random.default_rng(SEED)
    rows = []

    for C in CHUNK_SIZES:
        d = items[items['tokens'].map(len) >= C].copy()
        d['chunks'] = d['tokens'].map(
            lambda t: [t[i:i + C] for i in range(0, len(t) - C + 1, C)])
        d = d[d['chunks'].map(len) > 0]

        print(f"\n########## CHUNK SIZE C={C} ##########")
        prof = d.groupby('fonte').agg(items=('parent', 'size'),
                                      chunks=('chunks', lambda s: sum(len(x) for x in s)))
        prof.index = [NAMES[i] for i in prof.index]
        print(prof.to_string())

        # flat chunk list with item index
        flat, owner = [], []
        for i, ch in enumerate(d['chunks'].to_numpy()):
            for c in ch:
                flat.append(c)
                owner.append(i)
        owner = np.asarray(owner)
        src_of_item = d['fonte'].to_numpy()

        # balanced training subsample: the cap is the smallest platform's chunk
        # count, so every platform contributes exactly the same number of chunks
        pools = {p: np.where(src_of_item[owner] == p)[0] for p in PLATFORMS}
        cap = min(len(pool) for pool in pools.values())
        train_idx = np.concatenate([rng.choice(pools[p], cap, replace=False) for p in PLATFORMS])
        print(f"  LDA training chunks: {len(train_idx)} ({cap} per platform)")

        train_docs = [flat[i] for i in train_idx]
        dic = corpora.Dictionary(train_docs)
        dic.filter_extremes(no_below=5, no_above=0.5)
        train_bow = [dic.doc2bow(t) for t in train_docs]
        all_bow = [dic.doc2bow(t) for t in flat]

        for k in K_SWEEP:
            model = LdaMulticore(corpus=train_bow, id2word=dic, num_topics=k,
                                 random_state=RANDOM_STATE, passes=LDA_PASSES,
                                 workers=LDA_WORKERS)
            # soft topic distribution per chunk
            dist = np.zeros((len(all_bow), k), dtype=float)
            for i, b in enumerate(all_bow):
                for t, p in model.get_document_topics(b, minimum_probability=0.0):
                    dist[i, t] = p
            # item profile = mean over its chunks
            item_dist = np.zeros((len(d), k))
            np.add.at(item_dist, owner, dist)
            counts = np.bincount(owner, minlength=len(d)).reshape(-1, 1)
            item_dist = item_dist / np.maximum(counts, 1)

            plat_idx = {p: np.where(src_of_item == p)[0] for p in PLATFORMS}
            mat = np.vstack([item_dist[plat_idx[p]].mean(axis=0) for p in PLATFORMS])
            point = pair_sims(mat)

            boot = {pk: ([], []) for pk in point}
            for _ in range(N_BOOT):
                rr = []
                for p in PLATFORMS:
                    ix = plat_idx[p]
                    rr.append(item_dist[ix[rng.integers(0, len(ix), len(ix))]].mean(axis=0))
                for pk, (cv, jv) in pair_sims(np.vstack(rr)).items():
                    boot[pk][0].append(cv)
                    boot[pk][1].append(jv)

            line = []
            for pk, (c, j) in point.items():
                lo, hi = np.percentile(boot[pk][0], [2.5, 97.5])
                jlo, jhi = np.percentile(boot[pk][1], [2.5, 97.5])
                rows.append({'chunk_size': C, 'train_chunks_per_platform': cap,
                             'k': k, 'pair': PAIRS[pk],
                             'cosine': round(c, 4), 'cos_ci_lo': round(lo, 4),
                             'cos_ci_hi': round(hi, 4), 'js': round(j, 4),
                             'js_ci_lo': round(jlo, 4), 'js_ci_hi': round(jhi, 4)})
                line.append(f"{PAIRS[pk]}={c:.3f}[{lo:.2f},{hi:.2f}]")
            best = max(point.items(), key=lambda kv: kv[1][0])
            print(f"  k={k:>2}: " + "  ".join(line) + f"  -> top: {PAIRS[best[0]]}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)

    print("\n================ SUMMARY: top pair per k ================")
    for C in CHUNK_SIZES:
        sub = out[out['chunk_size'] == C]
        tops = []
        for k in K_SWEEP:
            s = sub[sub['k'] == k].sort_values('cosine', ascending=False)
            top, second = s.iloc[0], s.iloc[1]
            tie = top['cos_ci_lo'] <= second['cos_ci_hi']
            short = top['pair'].replace('Guardian', 'G').replace('Reddit', 'R').replace('Telegram', 'T')
            tops.append(f"k={k}:{short}" + ("(tie)" if tie else ""))
        print(f"  chunk={C:>2}: " + "  ".join(tops))
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
