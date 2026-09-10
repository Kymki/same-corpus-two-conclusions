"""
Chunked, balanced cross-platform similarity — the best-designed available test.

Previous attempts and why each fails:
  - full length (03/04): the LDA topic space is estimated on The Guardian (87.6%
    of token mass) and a length-defined topic absorbs >50% of both short-form
    platforms (script 06);
  - symmetric relevance filter (05): removes the filtering asymmetry but leaves
    the length confound, and reverses the result for that reason;
  - truncation to a common length (07): removes the length confound but destroys
    information — topic distributions flatten toward uniform (entropy 3.14-3.24
    of a maximum 3.32) and every pair ties. Underpowered by construction.

This script controls all three problems at once:
  1. LENGTH: every item is split into consecutive chunks of exactly C tokens,
     so each unit seen by LDA has identical length. Long items keep all their
     content (they simply yield many chunks) instead of being truncated.
  2. TOKEN DOMINANCE: LDA is trained on an equal number of chunks per platform,
     so no source shapes the topic space more than another.
  3. FILTERING: the symmetric relevance criterion is applied to all sources.

An item's topic distribution is the mean of its chunks' SOFT topic
distributions (no per-chunk argmax, avoiding a second hard-assignment
artefact); a platform's profile is the mean over its items, so every item
counts once regardless of length.

Reported for k in {5,7,10,15,21}: cosine and Jensen-Shannon similarity with
95% bootstrap CIs over items. All configurations tried are reported in the
paper, not only this one.

Output: data/results/chunked_balanced_ksweep_similarity.csv
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

INPUT_CSV = os.path.join(DATA_PROCESSED, "dati_testuali_preproc_consolidati.csv")
OUT = os.path.join(DATA_RESULTS, "chunked_balanced_ksweep_similarity.csv")

PLATFORMS = ["Guardian", "Reddit_Commento", "Telegram"]
NAMES = {"Guardian": "Guardian", "Reddit_Commento": "Reddit", "Telegram": "Telegram"}
PAIRS = {(0, 1): "Guardian-Reddit", (0, 2): "Guardian-Telegram", (1, 2): "Reddit-Telegram"}
WAR_KW = ['ukrain', 'russia', 'putin', 'zelensk', 'kyiv', 'kremlin', 'moscow',
          'donbas', 'crimea', 'kharkiv', 'kherson', 'mariupol', 'bakhmut', 'nato']

K_SWEEP = [5, 7, 10, 15, 21]
CHUNK_SIZES = [10, 20]
TRAIN_CHUNKS_PER_PLATFORM = 4000
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

        # balanced training subsample: equal chunks per platform
        train_idx = []
        for p in PLATFORMS:
            pool = np.where(src_of_item[owner] == p)[0]
            n = min(TRAIN_CHUNKS_PER_PLATFORM, len(pool))
            train_idx.append(rng.choice(pool, n, replace=False))
        train_idx = np.concatenate(train_idx)
        print(f"  LDA training chunks: {len(train_idx)} "
              f"({', '.join(f'{NAMES[p]}={min(TRAIN_CHUNKS_PER_PLATFORM, (src_of_item[owner]==p).sum())}' for p in PLATFORMS)})")

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

            boot = {pk: [] for pk in point}
            for _ in range(N_BOOT):
                rr = []
                for p in PLATFORMS:
                    ix = plat_idx[p]
                    rr.append(item_dist[ix[rng.integers(0, len(ix), len(ix))]].mean(axis=0))
                for pk, (cv, _j) in pair_sims(np.vstack(rr)).items():
                    boot[pk].append(cv)

            line = []
            for pk, (c, j) in point.items():
                lo, hi = np.percentile(boot[pk], [2.5, 97.5])
                rows.append({'chunk_size': C, 'k': k, 'pair': PAIRS[pk],
                             'cosine': round(c, 4), 'cos_ci_lo': round(lo, 4),
                             'cos_ci_hi': round(hi, 4), 'js': round(j, 4)})
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
