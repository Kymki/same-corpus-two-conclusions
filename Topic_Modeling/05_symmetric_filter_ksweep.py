"""
Robustness check: symmetric topical filtering + k-sweep (RP request).

The relevance filter was applied to The Guardian only, so 100% of Guardian
items refer to the conflict against 51% of Reddit and 69% of Telegram items.
This offers an alternative explanation for the main similarity result: Reddit
may differ simply because it retained off-topic material, not because of any
difference in discursive function.

This script applies the SAME relevance criterion to all three sources — an item
is kept iff its text contains at least one war keyword — and re-runs the
item-level k-sweep (LDA over k in {5,7,10,15,21}, cosine and Jensen-Shannon
similarity, bootstrap CIs over items) on the symmetrically filtered subcorpora.

If Guardian-Telegram remains the most similar pair, the objection is answered.

Outputs (data/results/):
  symfilter_corpus_profile.csv      items kept per source
  symfilter_coherence_values.csv    C_v per k on the filtered corpus
  symfilter_ksweep_similarity.csv   cosine/JS + bootstrap CIs per k and pair
"""
import os
import re
import sys
import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_PROCESSED, DATA_RESULTS

INPUT_CSV = os.path.join(DATA_PROCESSED, "dati_testuali_preproc_consolidati.csv")
OUT_PROFILE = os.path.join(DATA_RESULTS, "symfilter_corpus_profile.csv")
OUT_COH = os.path.join(DATA_RESULTS, "symfilter_coherence_values.csv")
OUT_SIM = os.path.join(DATA_RESULTS, "symfilter_ksweep_similarity.csv")

PLATFORMS = ["Guardian", "Reddit_Commento", "Telegram"]
NAMES = {"Guardian": "Guardian", "Reddit_Commento": "Reddit", "Telegram": "Telegram"}
PAIR_NAMES = {(0, 1): "Guardian-Reddit", (0, 2): "Guardian-Telegram", (1, 2): "Reddit-Telegram"}

WAR_KW = ['ukrain', 'russia', 'putin', 'zelensk', 'kyiv', 'kremlin', 'moscow',
          'donbas', 'crimea', 'kharkiv', 'kherson', 'mariupol', 'bakhmut', 'nato']

K_SWEEP = [5, 7, 10, 15, 21]
LDA_PASSES = 20
LDA_WORKERS = 7
RANDOM_STATE = 100
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42


def pair_similarities(mat):
    out = {}
    cos = cosine_similarity(mat)
    for i in range(mat.shape[0]):
        for j in range(i + 1, mat.shape[0]):
            out[(i, j)] = (cos[i, j], 1 - jensenshannon(mat[i], mat[j], base=2))
    return out


def main():
    print("Loading corpus...")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    df = df[(df['lingua_rilevata'] == 'en') & (df['fonte'].isin(PLATFORMS))].copy()
    df = df.dropna(subset=['testo_lemmatizzato'])
    df['parent'] = df['id_originale'].astype(str).str.replace(r'_s\d+$', '', regex=True)

    items = df.groupby(['parent', 'fonte']).agg(
        testo_lemmatizzato=('testo_lemmatizzato', ' '.join),
        testo_pulito=('testo_pulito_base', lambda s: ' '.join(map(str, s))),
    ).reset_index()

    # --- symmetric relevance filter: >=1 war keyword, applied to ALL sources ---
    low = items['testo_pulito'].str.lower()
    items['relevant'] = low.apply(lambda t: any(k in t for k in WAR_KW))
    before = items.groupby('fonte').size()
    kept = items[items['relevant']].copy()
    after = kept.groupby('fonte').size()
    profile = pd.DataFrame({'items_before': before, 'items_after': after})
    profile['retained_pct'] = (profile['items_after'] / profile['items_before'] * 100).round(1)
    profile.index = [NAMES[i] for i in profile.index]
    profile.to_csv(OUT_PROFILE)
    print("\n=== Symmetric relevance filter (>=1 war keyword, all sources) ===")
    print(profile.to_string())
    print(f"Total items after filter: {len(kept)}")

    tokenized = [t.split() for t in kept['testo_lemmatizzato']]
    print("\nBuilding dictionary (same policy: filter_extremes no_below=5, no_above=0.5)...")
    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus_bow = [dictionary.doc2bow(t) for t in tokenized]
    print(f"  Dictionary: {len(dictionary)} tokens, corpus: {len(corpus_bow)} items")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    coh_rows, sim_rows = [], []

    for k in K_SWEEP:
        print(f"\n=== k={k}: training LDA ===", flush=True)
        model = LdaMulticore(corpus=corpus_bow, id2word=dictionary, num_topics=k,
                             random_state=RANDOM_STATE, passes=LDA_PASSES, workers=LDA_WORKERS)
        cv = CoherenceModel(model=model, texts=tokenized, dictionary=dictionary,
                            coherence='c_v').get_coherence()
        coh_rows.append({'k': k, 'coherence_cv': cv})
        print(f"  C_v = {cv:.4f}")

        dom = []
        for bow in corpus_bow:
            row = model.get_document_topics(bow)
            dom.append(max(row, key=lambda x: x[1])[0] if row else -1)
        kept[f'topic_k{k}'] = dom

        grouped = {p: kept.loc[kept['fonte'] == p, f'topic_k{k}'].to_numpy() for p in PLATFORMS}
        mat = np.vstack([np.bincount(g[g >= 0], minlength=k) / max((g >= 0).sum(), 1)
                         for g in grouped.values()])
        point = pair_similarities(mat)

        print(f"  Bootstrapping (B={N_BOOTSTRAP})...", flush=True)
        boot = {pk: {'cos': [], 'js': []} for pk in point}
        for _ in range(N_BOOTSTRAP):
            rows_ = []
            for p in PLATFORMS:
                arr = grouped[p]
                res = arr[rng.integers(0, len(arr), len(arr))]
                c = np.bincount(res[res >= 0], minlength=k).astype(float)
                rows_.append(c / c.sum())
            bs = pair_similarities(np.vstack(rows_))
            for pk, (c, j) in bs.items():
                boot[pk]['cos'].append(c)
                boot[pk]['js'].append(j)

        for pk, (c, j) in point.items():
            clo, chi = np.percentile(boot[pk]['cos'], [2.5, 97.5])
            jlo, jhi = np.percentile(boot[pk]['js'], [2.5, 97.5])
            sim_rows.append({'k': k, 'pair': PAIR_NAMES[pk],
                             'cosine': round(c, 4), 'cos_ci_lo': round(clo, 4), 'cos_ci_hi': round(chi, 4),
                             'js': round(j, 4), 'js_ci_lo': round(jlo, 4), 'js_ci_hi': round(jhi, 4)})
            print(f"  {PAIR_NAMES[pk]}: cos={c:.3f} [{clo:.3f},{chi:.3f}]  js={j:.3f} [{jlo:.3f},{jhi:.3f}]")

    pd.DataFrame(coh_rows).to_csv(OUT_COH, index=False)
    sim = pd.DataFrame(sim_rows)
    sim.to_csv(OUT_SIM, index=False)

    print("\n=== SUMMARY: most/least similar pair per k (cosine), SYMMETRIC FILTER ===")
    for k in K_SWEEP:
        s = sim[sim['k'] == k].sort_values('cosine', ascending=False)
        print(f"  k={k:>2}: most={s.iloc[0]['pair']} ({s.iloc[0]['cosine']:.3f})  "
              f"least={s.iloc[-1]['pair']} ({s.iloc[-1]['cosine']:.3f})")
    print(f"\nSaved: {OUT_SIM}")


if __name__ == "__main__":
    main()
