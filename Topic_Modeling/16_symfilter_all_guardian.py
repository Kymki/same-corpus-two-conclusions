"""
A genuinely symmetric relevance filter: the one-keyword rule on every retrieved Guardian
article (triage 2.3; reviewers 2 and 4).

The "symmetric" subset of script 05 applies the one-keyword rule to all three sources,
but its Guardian input is already the 1,773 articles that passed the stricter
Guardian-only rule (title match, or >= 3 body occurrences outside live blogs). Here the
Guardian-only rule is never applied: the corpus comes from

    python Lemmatization/01_pre-processing_1.1.py --all-guardian

which runs every other preprocessing step unchanged and writes
data/processed/dati_testuali_preproc_allguardian.csv. The one-keyword rule
(>= 1 war keyword in the cleaned text) is then applied to all sources alike.

Reported:
  1. items per source before and after the rule;
  2. the share of items the rule retains, by item length (lemmatised tokens) and source:
     the rule is itself length-dependent, since a longer text is more likely to
     contain a keyword;
  3. cross-platform similarity on the resulting subset, as in script 13: 11 document
     orders x 2 seeds, dominant-topic ('argmax') and mean-distribution ('mean') profiles.

Item order. The canonical manifest (item_order_manifest.csv) covers the published
corpus only, and is written in pseudonymised Reddit ids. The order used here therefore
does not depend on identifiers at all: items are sorted by the SHA-256 of their source
and lemmatised text, which anyone re-running the pipeline obtains identically, with
original or pseudonymised ids. Permutations 1-10 are applied to that order.

Outputs (data/results/):
  symfilter_all_guardian_profile.csv              items per source, before and after the rule
  symfilter_all_guardian_retention_by_length.csv  share retained by length band and source
  symfilter_all_guardian_repetitions.csv          cosine / JS per order, seed, k, profile, pair
  symfilter_all_guardian_summary.csv              per k and profile: leading pair counts, margins
"""
import os
import sys
import hashlib
import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models import LdaMulticore
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_PROCESSED, DATA_RAW, DATA_RESULTS

INPUT_CSV = os.path.join(DATA_PROCESSED, "dati_testuali_preproc_allguardian.csv")
GUARDIAN_META = os.path.join(DATA_RAW, "Guardian_articles_metadata.csv")
OUT_PROFILE = os.path.join(DATA_RESULTS, "symfilter_all_guardian_profile.csv")
OUT_RETENTION = os.path.join(DATA_RESULTS, "symfilter_all_guardian_retention_by_length.csv")
OUT_REP = os.path.join(DATA_RESULTS, "symfilter_all_guardian_repetitions.csv")
OUT_SUMMARY = os.path.join(DATA_RESULTS, "symfilter_all_guardian_summary.csv")

PLATFORMS = ["Guardian", "Reddit_Commento", "Telegram"]
NAMES = {"Guardian": "Guardian", "Reddit_Commento": "Reddit", "Telegram": "Telegram"}
PAIRS = {(0, 1): "Guardian-Reddit", (0, 2): "Guardian-Telegram", (1, 2): "Reddit-Telegram"}
WAR_KW = ['ukrain', 'russia', 'putin', 'zelensk', 'kyiv', 'kremlin', 'moscow',
          'donbas', 'crimea', 'kharkiv', 'kherson', 'mariupol', 'bakhmut', 'nato']
BANDS = [0, 10, 25, 50, 100, 250, 500, 10**9]
BAND_LABELS = ["<10", "10-24", "25-49", "50-99", "100-249", "250-499", ">=500"]
K_GRID = [5, 7, 10, 15, 21]
N_PERMUTATIONS, SEEDS = 10, [100, 200]
LDA_PASSES, LDA_WORKERS = 20, 7


def pair_sims(mat):
    cos = cosine_similarity(mat)
    return {PAIRS[(i, j)]: (cos[i, j], 1 - jensenshannon(mat[i], mat[j], base=2))
            for i in range(3) for j in range(i + 1, 3)}


def load_items():
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    df = df[(df['lingua_rilevata'] == 'en') & (df['fonte'].isin(PLATFORMS))]
    df = df.dropna(subset=['testo_lemmatizzato']).copy()
    df['parent'] = df['id_originale'].astype(str).str.replace(r'_s\d+$', '', regex=True)
    items = df.groupby(['parent', 'fonte']).agg(
        lem=('testo_lemmatizzato', ' '.join),
        clean=('testo_pulito_base', lambda s: ' '.join(map(str, s))),
    ).reset_index()
    items['relevant'] = items['clean'].str.lower().apply(lambda t: any(k in t for k in WAR_KW))
    items['n_tokens'] = items['lem'].str.split().map(len)
    # identifier-free canonical order: SHA-256 of source + lemmatised text
    items['order_key'] = [hashlib.sha256(f"{f}\x1f{l}".encode("utf-8")).hexdigest()
                          for f, l in zip(items['fonte'], items['lem'])]
    return items.sort_values('order_key', kind='mergesort').reset_index(drop=True)


def describe(items):
    prof = items.groupby('fonte').agg(items=('relevant', 'size'), kept=('relevant', 'sum'))
    prof['kept_pct'] = (100 * prof['kept'] / prof['items']).round(1)
    prof.index = [NAMES[i] for i in prof.index]
    if os.path.exists(GUARDIAN_META):
        prof.loc['Guardian', 'retrieved'] = len(pd.read_csv(GUARDIAN_META))
    prof.to_csv(OUT_PROFILE)
    print("Items per source (English, after common preprocessing):\n" + prof.to_string())

    items = items.assign(band=pd.cut(items['n_tokens'], BANDS, right=False, labels=BAND_LABELS))
    ret = (items.groupby(['fonte', 'band'], observed=False)
           .agg(items=('relevant', 'size'), kept=('relevant', 'sum')).reset_index())
    ret['kept_pct'] = (100 * ret['kept'] / ret['items'].where(ret['items'] > 0)).round(1)
    ret['fonte'] = ret['fonte'].map(NAMES)
    ret.to_csv(OUT_RETENTION, index=False)
    print("\nShare retained by length band (%):\n"
          + ret.pivot(index='band', columns='fonte', values='kept_pct').to_string())


def repetitions(sub):
    rows = []
    tokens = sub['lem'].str.split()
    for o in range(N_PERMUTATIONS + 1):
        perm = np.arange(len(sub)) if o == 0 else np.random.default_rng(o).permutation(len(sub))
        tok, src = tokens.iloc[perm].tolist(), sub['fonte'].to_numpy()[perm]
        dic = corpora.Dictionary(tok)
        dic.filter_extremes(no_below=5, no_above=0.5)
        bow = [dic.doc2bow(t) for t in tok]
        for seed in SEEDS:
            for k in K_GRID:
                m = LdaMulticore(corpus=bow, id2word=dic, num_topics=k, random_state=seed,
                                 passes=LDA_PASSES, workers=LDA_WORKERS)
                theta = np.zeros((len(bow), k))
                for i, b in enumerate(bow):
                    for t, p in m.get_document_topics(b, minimum_probability=0.0):
                        theta[i, t] = p
                dom = theta.argmax(axis=1)
                profiles = {
                    'argmax': np.vstack([np.bincount(dom[src == p], minlength=k) / (src == p).sum()
                                         for p in PLATFORMS]),
                    'mean': np.vstack([theta[src == p].mean(axis=0) for p in PLATFORMS]),
                }
                for prof, mat in profiles.items():
                    for pair, (c, j) in pair_sims(mat).items():
                        rows.append(dict(order=o, seed=seed, k=k, profile=prof, pair=pair,
                                         cosine=round(c, 4), js=round(j, 4)))
                print(f"  order={o:>2} seed={seed} k={k:>2} done", flush=True)
    return pd.DataFrame(rows)


def summarise(rep):
    out = []
    for (prof, k), g in rep.groupby(['profile', 'k']):
        runs = g.groupby(['order', 'seed'])
        lead = runs.apply(lambda r: r.sort_values('cosine', ascending=False).iloc[0]['pair'])
        margin = runs.apply(lambda r: r['cosine'].nlargest(2).diff().abs().iloc[-1])
        row = dict(profile=prof, k=k, runs=len(lead))
        for p in PAIRS.values():
            row[f'leads_{p}'] = int((lead == p).sum())
        row.update(margin_median=round(margin.median(), 3), margin_min=round(margin.min(), 3),
                   margin_max=round(margin.max(), 3))
        out.append(row)
    return pd.DataFrame(out)


def main():
    items = load_items()
    describe(items)
    sub = items[items['relevant']].reset_index(drop=True)
    print(f"\nSymmetric subset: {len(sub)} items; "
          f"{N_PERMUTATIONS + 1} orders x {len(SEEDS)} seeds x {len(K_GRID)} k", flush=True)
    rep = repetitions(sub)
    rep.to_csv(OUT_REP, index=False)
    summary = summarise(rep)
    summary.to_csv(OUT_SUMMARY, index=False)
    print("\n" + summary.to_string(index=False))
    print(f"\nSaved: {OUT_PROFILE}\n       {OUT_RETENTION}\n       {OUT_REP}\n       {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
