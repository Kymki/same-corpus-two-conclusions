"""
Item-level re-analysis + k-sweep (decisive experiment requested by RP).

Motivation: the corpus rows are SENTENCES (ids end in _s0, _s1, ...), so a long
Guardian article weighs ~46 rows while a Reddit comment weighs ~1.8 — sentence
percentages are length-weighted and sentences from the same parent are not
independent observations (pseudo-replication). This script:

  a) aggregates sentences to ITEMS (parent documents): lemmatized text is
     concatenated per parent; item sentiment = mean class probabilities
     (plus majority vote) over the parent's sentences;
  b) retrains LDA at item level (LDA on short sentences is unreliable anyway);
  c) sweeps k over {5, 7, 10, 15, 21} with fixed seed and the same
     filter_extremes dictionary policy as the main pipeline;
  d) computes cross-platform cosine and Jensen-Shannon similarity of the
     dominant-topic distributions for EVERY k;
  e) bootstraps items within platform (B=1000) for 95% CIs on similarities.

If the similarity ordering is stable across k, it is a result; if it flips,
the similarity analysis is a topic-model artifact and must be scaled down.

Outputs (data/results/):
  items_en_meta.csv            item id, fonte, n_sentences, item sentiment
  item_coherence_values.csv    C_v per k at item level
  item_ksweep_similarity.csv   k x pair: cosine, JS, bootstrap 95% CIs
  item_sentiment_by_source.csv item-level sentiment shares per source
  item_topics_k{K}.txt         top words per topic for each k
  document_topics_items_k{K}.csv  item -> dominant topic (per k)

BBC (9 items) is kept in LDA training but EXCLUDED from the similarity
analysis: with n=9 its topic distribution is statistically meaningless.
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
SENTIMENT_CSV = os.path.join(DATA_RESULTS, "sentiment_results_multilingua.csv")

K_SWEEP = [5, 7, 10, 15, 21]
LDA_PASSES = 20
LDA_WORKERS = 7
RANDOM_STATE = 100
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42
SIM_PLATFORMS = ["Guardian", "Reddit_Commento", "Telegram"]  # BBC excluded (n=9)


def parent_id(s):
    return re.sub(r'_s\d+$', '', str(s))


def build_items():
    print("Loading sentence-level corpus...")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    df = df[df['lingua_rilevata'] == 'en'].dropna(subset=['testo_lemmatizzato'])
    df['parent'] = df['id_originale'].map(parent_id)

    print("Aggregating sentences -> items (lemmatized text concatenated per parent)...")
    items = df.groupby('parent').agg(
        fonte=('fonte', 'first'),
        n_sentences=('id_originale', 'count'),
        testo_lemmatizzato=('testo_lemmatizzato', ' '.join),
    ).reset_index()
    print(f"  {len(df)} sentences -> {len(items)} items")
    print(items.groupby('fonte')['parent'].count().to_string())

    # Item-level sentiment: mean of the sentence class probabilities + majority vote
    print("Aggregating sentence sentiment -> item sentiment...")
    sent = pd.read_csv(SENTIMENT_CSV, low_memory=False,
                       usecols=['id_originale', 'neg_prob', 'neu_prob', 'pos_prob', 'sentiment_label'])
    sent['parent'] = sent['id_originale'].map(parent_id)
    probs = sent.groupby('parent')[['neg_prob', 'neu_prob', 'pos_prob']].mean()
    probs['sentiment_item_meanprob'] = probs.idxmax(axis=1).map(
        {'neg_prob': 'Negative', 'neu_prob': 'Neutral', 'pos_prob': 'Positive'})
    vote = (sent.groupby('parent')['sentiment_label']
            .agg(lambda s: s.value_counts().idxmax()).rename('sentiment_item_vote'))
    items = items.merge(probs, left_on='parent', right_index=True, how='left')
    items = items.merge(vote, left_on='parent', right_index=True, how='left')

    meta_cols = ['parent', 'fonte', 'n_sentences', 'neg_prob', 'neu_prob', 'pos_prob',
                 'sentiment_item_meanprob', 'sentiment_item_vote']
    items[meta_cols].to_csv(os.path.join(DATA_RESULTS, "items_en_meta.csv"), index=False)

    shares = (pd.crosstab(items['fonte'], items['sentiment_item_meanprob'], normalize='index') * 100).round(2)
    shares.to_csv(os.path.join(DATA_RESULTS, "item_sentiment_by_source.csv"))
    print("\nItem-level sentiment shares by source (mean-prob aggregation, %):")
    print(shares.to_string())
    return items


def pair_similarities(dist_matrix):
    """dist_matrix: platforms x topics (rows sum to 1). Returns dict pair -> (cos, js)."""
    out = {}
    n = dist_matrix.shape[0]
    cos = cosine_similarity(dist_matrix)
    for i in range(n):
        for j in range(i + 1, n):
            js = 1 - jensenshannon(dist_matrix[i], dist_matrix[j], base=2)
            out[(i, j)] = (cos[i, j], js)
    return out


def topic_distributions(items_df, topic_col, platforms, n_topics):
    mats = []
    for p in platforms:
        counts = items_df.loc[items_df['fonte'] == p, topic_col].value_counts()
        vec = np.array([counts.get(t, 0) for t in range(n_topics)], dtype=float)
        mats.append(vec / vec.sum() if vec.sum() > 0 else vec)
    return np.vstack(mats)


def main():
    items = build_items()
    tokenized = [t.split() for t in items['testo_lemmatizzato']]

    print("\nBuilding dictionary (filter_extremes no_below=5, no_above=0.5, same policy as main pipeline)...")
    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus_bow = [dictionary.doc2bow(t) for t in tokenized]
    print(f"  Dictionary: {len(dictionary)} tokens, corpus: {len(corpus_bow)} items")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    coh_rows, sim_rows = [], []
    pair_names = {(0, 1): "Guardian-Reddit", (0, 2): "Guardian-Telegram", (1, 2): "Reddit-Telegram"}

    for k in K_SWEEP:
        print(f"\n=== k={k}: training LDA (passes={LDA_PASSES}) ===", flush=True)
        model = LdaMulticore(corpus=corpus_bow, id2word=dictionary, num_topics=k,
                             random_state=RANDOM_STATE, passes=LDA_PASSES, workers=LDA_WORKERS)

        cm = CoherenceModel(model=model, texts=tokenized, dictionary=dictionary, coherence='c_v')
        cv = cm.get_coherence()
        coh_rows.append({'k': k, 'coherence_cv': cv})
        print(f"  C_v = {cv:.4f}")

        with open(os.path.join(DATA_RESULTS, f"item_topics_k{k}.txt"), 'w', encoding='utf-8') as f:
            for idx, topic in model.print_topics(-1):
                f.write(f"Topic #{idx}: {topic}\n")

        print("  Assigning dominant topics...", flush=True)
        dom = []
        for bow in corpus_bow:
            row = model.get_document_topics(bow)
            dom.append(max(row, key=lambda x: x[1])[0] if row else -1)
        col = f'topic_k{k}'
        items[col] = dom
        items[['parent', 'fonte', col]].to_csv(
            os.path.join(DATA_RESULTS, f"document_topics_items_k{k}.csv"), index=False)

        # Point estimates
        sub = items[items['fonte'].isin(SIM_PLATFORMS)]
        dist = topic_distributions(sub, col, SIM_PLATFORMS, k)
        point = pair_similarities(dist)

        # Bootstrap over items within platform
        print(f"  Bootstrapping similarities (B={N_BOOTSTRAP})...", flush=True)
        boot = {pk: {'cos': [], 'js': []} for pk in point}
        grouped = {p: sub.loc[sub['fonte'] == p, col].to_numpy() for p in SIM_PLATFORMS}
        for _ in range(N_BOOTSTRAP):
            mats = []
            for p in SIM_PLATFORMS:
                arr = grouped[p]
                res = arr[rng.integers(0, len(arr), len(arr))]
                counts = np.bincount(res[res >= 0], minlength=k).astype(float)
                mats.append(counts / counts.sum())
            bs = pair_similarities(np.vstack(mats))
            for pk, (c, j) in bs.items():
                boot[pk]['cos'].append(c)
                boot[pk]['js'].append(j)

        for pk, (c, j) in point.items():
            cos_lo, cos_hi = np.percentile(boot[pk]['cos'], [2.5, 97.5])
            js_lo, js_hi = np.percentile(boot[pk]['js'], [2.5, 97.5])
            sim_rows.append({'k': k, 'pair': pair_names[pk],
                             'cosine': round(c, 4), 'cos_ci_lo': round(cos_lo, 4), 'cos_ci_hi': round(cos_hi, 4),
                             'js': round(j, 4), 'js_ci_lo': round(js_lo, 4), 'js_ci_hi': round(js_hi, 4)})
            print(f"  {pair_names[pk]}: cos={c:.3f} [{cos_lo:.3f},{cos_hi:.3f}]  js={j:.3f} [{js_lo:.3f},{js_hi:.3f}]")

    pd.DataFrame(coh_rows).to_csv(os.path.join(DATA_RESULTS, "item_coherence_values.csv"), index=False)
    sim_df = pd.DataFrame(sim_rows)
    sim_df.to_csv(os.path.join(DATA_RESULTS, "item_ksweep_similarity.csv"), index=False)

    print("\n=== SUMMARY: most/least similar pair per k (cosine) ===")
    for k in K_SWEEP:
        sk = sim_df[sim_df['k'] == k].sort_values('cosine', ascending=False)
        print(f"  k={k:>2}: most={sk.iloc[0]['pair']} ({sk.iloc[0]['cosine']:.3f})  "
              f"least={sk.iloc[-1]['pair']} ({sk.iloc[-1]['cosine']:.3f})")
    print("\nDone. Outputs in data/results/ (item_ksweep_similarity.csv is the decisive table).")


if __name__ == "__main__":
    main()
