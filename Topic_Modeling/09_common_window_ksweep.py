"""
Item-level k-sweep restricted to the COMMON COLLECTION WINDOW (RP request).

Motivation: collection windows differ sharply (Guardian ~237 days, Reddit ~87,
Telegram ~41). Even without any diachronic claim, thematic distributions could
differ simply because the sources cover different historical events. The
collection window is therefore itself an unexamined analytical fork.

This script repeats the symmetric-filter item-level k-sweep on the intersection
of the three windows only, holding everything else identical to script 05:
  - same >=1-war-keyword relevance filter applied to all three sources;
  - same dictionary policy (filter_extremes no_below=5, no_above=0.5);
  - same k grid {5,7,10,15,21}, same seed, same 1000-replicate item bootstrap;
  - cosine and Jensen-Shannon similarity between per-platform distributions of
    dominant topics.

The only change is the date restriction, so any difference in the outcome is
attributable to the window.

Outputs (data/results/):
  commonwindow_corpus_profile.csv
  commonwindow_coherence_values.csv
  commonwindow_ksweep_similarity.csv
"""
import os
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
OUT_PROFILE = os.path.join(DATA_RESULTS, "commonwindow_corpus_profile.csv")
OUT_COH = os.path.join(DATA_RESULTS, "commonwindow_coherence_values.csv")
OUT_SIM = os.path.join(DATA_RESULTS, "commonwindow_ksweep_similarity.csv")

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
    df = df[(df['lingua_rilevata'] == 'en') & (df['fonte'].isin(PLATFORMS))]
    df = df.dropna(subset=['testo_lemmatizzato']).copy()
    df['parent'] = df['id_originale'].astype(str).str.replace(r'_s\d+$', '', regex=True)
    df['t'] = pd.to_datetime(df['data_originale_str'], format='mixed', errors='coerce', utc=True)

    items = df.groupby(['parent', 'fonte']).agg(
        lem=('testo_lemmatizzato', ' '.join),
        clean=('testo_pulito_base', lambda s: ' '.join(map(str, s))),
        t=('t', 'first'),
    ).reset_index()
    items = items[items['clean'].str.lower().apply(lambda x: any(k in x for k in WAR_KW))]
    items = items.dropna(subset=['t'])

    # common window = intersection of the three per-source windows
    start = max(items.loc[items['fonte'] == p, 't'].min() for p in PLATFORMS)
    end = min(items.loc[items['fonte'] == p, 't'].max() for p in PLATFORMS)
    print(f"\nCommon window: {start.date()} -> {end.date()} ({(end - start).days} days)")

    full = items.groupby('fonte').size()
    kept = items[(items['t'] >= start) & (items['t'] <= end)].copy()
    prof = pd.DataFrame({'items_full_window': full,
                         'items_common_window': kept.groupby('fonte').size()})
    prof['retained_pct'] = (prof['items_common_window'] / prof['items_full_window'] * 100).round(1)
    prof.index = [NAMES[i] for i in prof.index]
    prof.to_csv(OUT_PROFILE)
    print(prof.to_string())
    print(f"Total items in the common window: {len(kept)}")
    if prof['items_common_window'].min() < 200:
        print("NOTE: one source has fewer than 200 items; bootstrap CIs will be wide.")

    tokenized = [t.split() for t in kept['lem']]
    print("\nBuilding dictionary (same policy as script 05)...")
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
                             'cosine': round(c, 4), 'cos_ci_lo': round(clo, 4),
                             'cos_ci_hi': round(chi, 4),
                             'js': round(j, 4), 'js_ci_lo': round(jlo, 4),
                             'js_ci_hi': round(jhi, 4)})
            print(f"  {PAIR_NAMES[pk]}: cos={c:.3f} [{clo:.3f},{chi:.3f}]  "
                  f"js={j:.3f} [{jlo:.3f},{jhi:.3f}]")

    pd.DataFrame(coh_rows).to_csv(OUT_COH, index=False)
    sim = pd.DataFrame(sim_rows)
    sim.to_csv(OUT_SIM, index=False)

    print("\n=== SUMMARY: most/least similar pair per k, COMMON WINDOW ===")
    for k in K_SWEEP:
        s = sim[sim['k'] == k].sort_values('cosine', ascending=False)
        top, second = s.iloc[0], s.iloc[1]
        tie = top['cos_ci_lo'] <= second['cos_ci_hi']
        print(f"  k={k:>2}: most={top['pair']} ({top['cosine']:.3f})"
              f"{'  [tie with next]' if tie else '  [decisive]'}"
              f"   least={s.iloc[-1]['pair']} ({s.iloc[-1]['cosine']:.3f})")
    print("\nCompare with symfilter_ksweep_similarity.csv (full windows).")


if __name__ == "__main__":
    main()
