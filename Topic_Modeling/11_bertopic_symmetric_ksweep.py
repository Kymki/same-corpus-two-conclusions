"""
Sensitivity check: BERTopic instead of LDA on the symmetrically filtered subset.

Reviewers asked whether the similarity conclusions depend on LDA, a model known
to be unreliable on short texts. This script refits the item-level k-sweep of
script 05 with BERTopic (sentence embeddings -> UMAP -> clustering -> c-TF-IDF)
and reports the same cross-platform similarities, so the two models can be read
side by side. It is a sensitivity check, not a model comparison.

Held identical to script 05:
  - corpus: English items, symmetric relevance criterion (>=1 war keyword)
    applied to all three sources -> 6,292 items;
  - k grid {5,7,10,15,21};
  - platform profile = share of items assigned to each topic (hard assignment,
    as with the dominant LDA topic in 05);
  - cosine and Jensen-Shannon similarity, 1,000-replicate item bootstrap, seed 42.

Two choices are not inherited from 05 and are declared here.

1. Embedding unit (a length fork in its own right). The embedding model,
   all-MiniLM-L6-v2 (BERTopic's English default), truncates input at 256
   word-piece tokens. A Guardian article averages ~1,100 words, a Reddit or
   Telegram item a few dozen, so embedding whole items would let the model see
   only the opening of each article but almost all of every social item.
   Both treatments are run and reported:
     - "sentence"  (main):    every sentence embedded separately; item
                               embedding = mean of its sentence embeddings;
     - "truncated" (control): the whole item embedded once, truncated at 256
                               tokens by the model.
2. Clustering. BERTopic's default clusterer (HDBSCAN) chooses the number of
   topics itself and leaves part of the corpus in an outlier topic. To keep the
   k grid fixed and every item assigned, KMeans with k clusters is used instead.

UMAP is stochastic, so every configuration is run with three seeds. Topic
representations (c-TF-IDF) are computed on the lemmatised item text (stop words
already removed in preprocessing); they affect the topic labels only, not the
assignments.

Embeddings are cached in data/processed/ (derived from full text, so not
redistributed, like the corpus itself).

Outputs (data/results/):
  bertopic_symfilter_ksweep_similarity.csv  cosine/JS + bootstrap CIs per
                                            embedding, seed, k and pair
  bertopic_symfilter_topic_terms.csv        top terms and platform shares per topic
  bertopic_symfilter_truncation.csv         share of items truncated at 256 tokens
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon
from umap import UMAP
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_PROCESSED, DATA_RESULTS
from item_order import canonical_order

INPUT_CSV = os.path.join(DATA_PROCESSED, "dati_testuali_preproc_consolidati.csv")
EMB_CACHE = os.path.join(DATA_PROCESSED, "bertopic_embeddings_{mode}.npz")
OUT_SIM = os.path.join(DATA_RESULTS, "bertopic_symfilter_ksweep_similarity.csv")
OUT_TERMS = os.path.join(DATA_RESULTS, "bertopic_symfilter_topic_terms.csv")
OUT_TRUNC = os.path.join(DATA_RESULTS, "bertopic_symfilter_truncation.csv")

PLATFORMS = ["Guardian", "Reddit_Commento", "Telegram"]
NAMES = {"Guardian": "Guardian", "Reddit_Commento": "Reddit", "Telegram": "Telegram"}
PAIR_NAMES = {(0, 1): "Guardian-Reddit", (0, 2): "Guardian-Telegram", (1, 2): "Reddit-Telegram"}

WAR_KW = ['ukrain', 'russia', 'putin', 'zelensk', 'kyiv', 'kremlin', 'moscow',
          'donbas', 'crimea', 'kharkiv', 'kherson', 'mariupol', 'bakhmut', 'nato']

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODES = ["sentence", "truncated"]
K_SWEEP = [5, 7, 10, 15, 21]
UMAP_SEEDS = [100, 101, 102]
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42
N_TOP_TERMS = 10


def pair_similarities(mat):
    out = {}
    cos = cosine_similarity(mat)
    for i in range(mat.shape[0]):
        for j in range(i + 1, mat.shape[0]):
            out[(i, j)] = (cos[i, j], 1 - jensenshannon(mat[i], mat[j], base=2))
    return out


def load_corpus():
    """Sentence rows and items exactly as in script 05 (same filters, same order)."""
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    df = df[(df['lingua_rilevata'] == 'en') & (df['fonte'].isin(PLATFORMS))].copy()
    df = df.dropna(subset=['testo_lemmatizzato'])
    df['parent'] = df['id_originale'].astype(str).str.replace(r'_s\d+$', '', regex=True)

    items = df.groupby(['parent', 'fonte']).agg(
        testo_lemmatizzato=('testo_lemmatizzato', ' '.join),
        testo_pulito=('testo_pulito_base', lambda s: ' '.join(map(str, s))),
    ).reset_index()
    items = canonical_order(items)
    low = items['testo_pulito'].str.lower()
    items = items[low.apply(lambda t: any(k in t for k in WAR_KW))].reset_index(drop=True)

    keep = set(zip(items['parent'], items['fonte']))
    sents = df[[pf in keep for pf in zip(df['parent'], df['fonte'])]].copy()
    sents['testo_pulito_base'] = sents['testo_pulito_base'].astype(str)
    return items, sents


def embed(mode, items, sents, model):
    """Item embeddings for one mode, cached and checked against the item order."""
    cache = EMB_CACHE.format(mode=mode)
    keys = (items['parent'] + '|' + items['fonte']).to_numpy()
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if np.array_equal(z['keys'], keys):
            print(f"  [{mode}] embeddings loaded from cache")
            return z['emb']
        print(f"  [{mode}] cache does not match the item list; recomputing")

    if mode == "sentence":
        print(f"  [{mode}] encoding {len(sents):,} sentences...", flush=True)
        s_emb = model.encode(sents['testo_pulito_base'].tolist(), batch_size=256,
                             show_progress_bar=True, normalize_embeddings=True)
        idx = {k: i for i, k in enumerate(keys)}
        owner = np.array([idx[p + '|' + f] for p, f in zip(sents['parent'], sents['fonte'])])
        emb = np.zeros((len(items), s_emb.shape[1]))
        np.add.at(emb, owner, s_emb)
        emb /= np.bincount(owner, minlength=len(items)).reshape(-1, 1)
    else:
        print(f"  [{mode}] encoding {len(items):,} items "
              f"(max_seq_length={model.max_seq_length})...", flush=True)
        emb = model.encode(items['testo_pulito'].tolist(), batch_size=64,
                           show_progress_bar=True, normalize_embeddings=True)

    np.savez(cache, keys=keys, emb=emb)
    return emb


def truncation_profile(items, model):
    """How many items exceed the model's input window, per platform."""
    tok = model.tokenizer
    n_tok = [len(tok(t, add_special_tokens=True, truncation=False)['input_ids'])
             for t in items['testo_pulito']]
    d = items[['fonte']].assign(n_tokens=n_tok)
    d['truncated'] = d['n_tokens'] > model.max_seq_length
    prof = d.groupby('fonte').agg(items=('n_tokens', 'size'),
                                  median_tokens=('n_tokens', 'median'),
                                  truncated_items=('truncated', 'sum'))
    prof['truncated_pct'] = (prof['truncated_items'] / prof['items'] * 100).round(1)
    prof.index = [NAMES[i] for i in prof.index]
    return prof


def main():
    print("Loading corpus...")
    items, sents = load_corpus()
    print(items.groupby('fonte').size().rename(index=NAMES).to_string())
    print(f"Total items: {len(items)}, sentences: {len(sents):,}")

    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"Embedding model: {EMBEDDING_MODEL} on {model.device}")

    trunc = truncation_profile(items, model)
    trunc.to_csv(OUT_TRUNC)
    print("\n=== Items beyond the 256-token window ===\n" + trunc.to_string())

    docs = items['testo_lemmatizzato'].tolist()
    src = items['fonte'].to_numpy()
    sim_rows, term_rows = [], []

    for mode in EMBEDDING_MODES:
        print(f"\n########## EMBEDDING: {mode} ##########")
        emb = embed(mode, items, sents, model)

        for seed in UMAP_SEEDS:
            rng = np.random.default_rng(BOOTSTRAP_SEED)
            for k in K_SWEEP:
                topic_model = BERTopic(
                    umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                                    metric='cosine', random_state=seed),
                    hdbscan_model=KMeans(n_clusters=k, random_state=seed, n_init=10),
                    vectorizer_model=CountVectorizer(),
                    calculate_probabilities=False,
                )
                topics, _ = topic_model.fit_transform(docs, embeddings=emb)
                topics = np.asarray(topics)

                grouped = {p: topics[src == p] for p in PLATFORMS}
                mat = np.vstack([np.bincount(g, minlength=k) / len(g) for g in grouped.values()])
                point = pair_similarities(mat)

                boot = {pk: {'cos': [], 'js': []} for pk in point}
                for _ in range(N_BOOTSTRAP):
                    rr = []
                    for p in PLATFORMS:
                        arr = grouped[p]
                        res = arr[rng.integers(0, len(arr), len(arr))]
                        rr.append(np.bincount(res, minlength=k) / len(res))
                    for pk, (c, j) in pair_similarities(np.vstack(rr)).items():
                        boot[pk]['cos'].append(c)
                        boot[pk]['js'].append(j)

                for pk, (c, j) in point.items():
                    clo, chi = np.percentile(boot[pk]['cos'], [2.5, 97.5])
                    jlo, jhi = np.percentile(boot[pk]['js'], [2.5, 97.5])
                    sim_rows.append({'embedding': mode, 'umap_seed': seed, 'k': k,
                                     'pair': PAIR_NAMES[pk],
                                     'cosine': round(c, 4), 'cos_ci_lo': round(clo, 4),
                                     'cos_ci_hi': round(chi, 4),
                                     'js': round(j, 4), 'js_ci_lo': round(jlo, 4),
                                     'js_ci_hi': round(jhi, 4)})

                for t in range(k):
                    words = [w for w, _ in (topic_model.get_topic(t) or [])][:N_TOP_TERMS]
                    row = {'embedding': mode, 'umap_seed': seed, 'k': k, 'topic': t,
                           'n_items': int((topics == t).sum()), 'top_terms': ', '.join(words)}
                    for p in PLATFORMS:
                        row[f'share_{NAMES[p]}'] = round(float((grouped[p] == t).mean()), 4)
                    term_rows.append(row)

                best = max(point.items(), key=lambda kv: kv[1][0])
                print(f"  seed={seed} k={k:>2}: " + "  ".join(
                    f"{PAIR_NAMES[pk]}={c:.3f}" for pk, (c, _j) in point.items())
                    + f"  -> top: {PAIR_NAMES[best[0]]}", flush=True)

    sim = pd.DataFrame(sim_rows)
    sim.to_csv(OUT_SIM, index=False)
    pd.DataFrame(term_rows).to_csv(OUT_TERMS, index=False)

    print("\n================ SUMMARY: top pair per k ================")
    for mode in EMBEDDING_MODES:
        for seed in UMAP_SEEDS:
            sub = sim[(sim['embedding'] == mode) & (sim['umap_seed'] == seed)]
            tops = []
            for k in K_SWEEP:
                s = sub[sub['k'] == k].sort_values('cosine', ascending=False)
                top, second = s.iloc[0], s.iloc[1]
                tie = top['cos_ci_lo'] <= second['cos_ci_hi']
                short = top['pair'].replace('Guardian', 'G').replace('Reddit', 'R').replace('Telegram', 'T')
                tops.append(f"k={k}:{short}" + ("(tie)" if tie else ""))
            print(f"  {mode:<9} seed={seed}: " + "  ".join(tops))
    print(f"\nSaved: {OUT_SIM}\n       {OUT_TERMS}\n       {OUT_TRUNC}")


if __name__ == "__main__":
    main()
