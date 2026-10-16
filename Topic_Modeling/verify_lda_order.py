"""
Reproducibility checks for the LDA results: canonical order, run-to-run, document order.

Three checks on the k=10 model of script 05 (symmetric subset, 6,292 items, seed 100):

  1. canonical order   items in the order of data/results/item_order_manifest.csv; the
                       cross-platform similarities are compared with the published
                       data/results/symfilter_ksweep_similarity.csv.
  2. run-to-run        the canonical fit repeated: gensim's LdaMulticore merges the
                       workers' results in the order they arrive, so identical runs can
                       differ slightly in floating point.
  3. document order    the same items, the same seed, the same k, passed to the model in
                       three random orders (permutation seeds 1, 2, 3). This is the
                       "order effect" of Agrawal, Fu & Menzies (2018), and the reason the
                       manifest exists.

Script 13 extends check 3 to 10 permutations x 2 seeds over the whole k grid.

Output: data/results/verify_lda_order.csv (one row per check, run and pair)
"""
import os
import sys
import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models import LdaMulticore
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_PROCESSED, DATA_RESULTS
from item_order import canonical_order

INPUT_CSV = os.path.join(DATA_PROCESSED, "dati_testuali_preproc_consolidati.csv")
PUBLISHED = os.path.join(DATA_RESULTS, "symfilter_ksweep_similarity.csv")
OUT = os.path.join(DATA_RESULTS, "verify_lda_order.csv")

PLATFORMS = ["Guardian", "Reddit_Commento", "Telegram"]
PAIRS = {(0, 1): "Guardian-Reddit", (0, 2): "Guardian-Telegram", (1, 2): "Reddit-Telegram"}
WAR_KW = ['ukrain', 'russia', 'putin', 'zelensk', 'kyiv', 'kremlin', 'moscow',
          'donbas', 'crimea', 'kharkiv', 'kherson', 'mariupol', 'bakhmut', 'nato']
K, SEED, PASSES, WORKERS = 10, 100, 20, 7
N_REPEATS, PERMUTATIONS = 2, [1, 2, 3]


def load_items():
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    df = df[(df['lingua_rilevata'] == 'en') & (df['fonte'].isin(PLATFORMS))].copy()
    df = df.dropna(subset=['testo_lemmatizzato'])
    df['parent'] = df['id_originale'].astype(str).str.replace(r'_s\d+$', '', regex=True)
    items = df.groupby(['parent', 'fonte']).agg(
        testo_lemmatizzato=('testo_lemmatizzato', ' '.join),
        testo_pulito=('testo_pulito_base', lambda s: ' '.join(map(str, s))),
    ).reset_index()
    items = canonical_order(items)
    keep = items['testo_pulito'].str.lower().apply(lambda t: any(k in t for k in WAR_KW))
    return items[keep].reset_index(drop=True)


def fit(items):
    """Script 05's k=10 fit and dominant-topic profiles for the given item order."""
    tok = [t.split() for t in items['testo_lemmatizzato']]
    dic = corpora.Dictionary(tok)
    dic.filter_extremes(no_below=5, no_above=0.5)
    bow = [dic.doc2bow(t) for t in tok]
    m = LdaMulticore(corpus=bow, id2word=dic, num_topics=K, random_state=SEED,
                     passes=PASSES, workers=WORKERS)
    dom = np.array([max(r, key=lambda x: x[1])[0] if r else -1
                    for r in (m.get_document_topics(b) for b in bow)])
    src = items['fonte'].to_numpy()
    mat = np.vstack([np.bincount(dom[(src == p) & (dom >= 0)], minlength=K)
                     / max(((src == p) & (dom >= 0)).sum(), 1) for p in PLATFORMS])
    cos = cosine_similarity(mat)
    return {name: round(float(cos[i, j]), 4) for (i, j), name in PAIRS.items()}


def main():
    items = load_items()
    pub = pd.read_csv(PUBLISHED)
    pub = pub[pub['k'] == K].set_index('pair')['cosine'].to_dict()
    rows = [dict(check='published', run=0, pair=p, cosine=c) for p, c in pub.items()]

    for r in range(1, N_REPEATS + 1):
        for p, c in fit(items).items():
            rows.append(dict(check='canonical order', run=r, pair=p, cosine=c))
    for s in PERMUTATIONS:
        shuffled = items.sample(frac=1, random_state=s).reset_index(drop=True)
        for p, c in fit(shuffled).items():
            rows.append(dict(check='document order', run=s, pair=p, cosine=c))

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out.pivot_table(index=['check', 'run'], columns='pair', values='cosine').to_string())
    canon = out[out.check == 'canonical order']
    d_pub = max(abs(c - pub[p]) for p, c in zip(canon.pair, canon.cosine))
    d_run = canon.groupby('pair').cosine.agg(lambda s: s.max() - s.min()).max()
    perm = out[out.check == 'document order']
    d_ord = perm.groupby('pair').cosine.agg(lambda s: s.max() - s.min()).max()
    print(f"\n1. canonical order vs published:  max |diff| = {d_pub:.4f}")
    print(f"2. run-to-run (canonical order):   max range  = {d_run:.4f}")
    print(f"3. across document orders:         max range  = {d_ord:.4f}")
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
