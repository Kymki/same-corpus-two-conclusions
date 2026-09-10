"""
Length-matched cross-platform similarity (decisive test).

Two confounds have been identified in the similarity analysis:
  (1) asymmetric topical filtering — the relevance filter was applied to The
      Guardian only, so Reddit retained ~49% off-topic material (script 05);
  (2) document length — in the symmetrically filtered model a single topic of
      mean length ~30 words absorbs >50% of both short-form platforms, and The
      Guardian supplies ~88% of the token mass (script 06).

This script controls both. Every item is reduced to EXACTLY N lemmatised
tokens, sampled uniformly at random from the item (LDA is bag-of-words, so
random sampling preserves the item's topical mixture without privileging an
article's lead); items with fewer than N tokens are dropped. Under this design
no source can differ from another because of document length.

Conditions run:
  A. length-matched + symmetric relevance filter  (cleanest: both confounds controlled)
  B. length-matched only, no relevance filter      (isolates the length effect)
  C. condition A at a larger N                     (sensitivity to the choice of N)

If Guardian-Telegram leads under A, the paper's original claim is vindicated
on a clean comparison. If Reddit-Telegram leads, the reversal is substantive.
If the pairs are indistinguishable, the similarity analysis is underdetermined
by the data and must be reported as such.

Output: data/results/lengthmatched_ksweep_similarity.csv (all conditions)
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
OUT = os.path.join(DATA_RESULTS, "lengthmatched_ksweep_similarity.csv")

PLATFORMS = ["Guardian", "Reddit_Commento", "Telegram"]
NAMES = {"Guardian": "Guardian", "Reddit_Commento": "Reddit", "Telegram": "Telegram"}
PAIRS = {(0, 1): "Guardian-Reddit", (0, 2): "Guardian-Telegram", (1, 2): "Reddit-Telegram"}
WAR_KW = ['ukrain', 'russia', 'putin', 'zelensk', 'kyiv', 'kremlin', 'moscow',
          'donbas', 'crimea', 'kharkiv', 'kherson', 'mariupol', 'bakhmut', 'nato']

K_SWEEP = [5, 7, 10, 15, 21]
LDA_PASSES = 20
LDA_WORKERS = 7
RANDOM_STATE = 100
N_BOOT = 1000
SEED = 42

CONDITIONS = [
    ("A_len15_symfilter", 15, True),
    ("B_len15_nofilter", 15, False),
    ("C_len25_symfilter", 25, True),
]


def pair_sims(mat):
    out = {}
    cos = cosine_similarity(mat)
    for i in range(mat.shape[0]):
        for j in range(i + 1, mat.shape[0]):
            out[(i, j)] = (cos[i, j], 1 - jensenshannon(mat[i], mat[j], base=2))
    return out


def build_items():
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    df = df[(df['lingua_rilevata'] == 'en') & (df['fonte'].isin(PLATFORMS))]
    df = df.dropna(subset=['testo_lemmatizzato']).copy()
    df['parent'] = df['id_originale'].astype(str).str.replace(r'_s\d+$', '', regex=True)
    items = df.groupby(['parent', 'fonte']).agg(
        lem=('testo_lemmatizzato', ' '.join),
        clean=('testo_pulito_base', lambda s: ' '.join(map(str, s))),
    ).reset_index()
    items['relevant'] = items['clean'].str.lower().apply(lambda t: any(k in t for k in WAR_KW))
    items['tokens'] = items['lem'].str.split()
    return items


def run_condition(items, label, n_tokens, apply_filter, rng, rows):
    d = items[items['relevant']].copy() if apply_filter else items.copy()
    d = d[d['tokens'].map(len) >= n_tokens].copy()
    # exactly n_tokens per item, sampled at random (bag-of-words: order irrelevant)
    d['tok_matched'] = d['tokens'].map(
        lambda t: list(np.array(t)[rng.choice(len(t), n_tokens, replace=False)]))

    counts = d.groupby('fonte').size()
    print(f"\n########## CONDITION {label} "
          f"(N={n_tokens} tokens/item, relevance filter={'ON' if apply_filter else 'OFF'}) ##########")
    print("Items per source: " + ", ".join(f"{NAMES[k]}={v}" for k, v in counts.items())
          + f"  (total {len(d)})")
    if any(c < 100 for c in counts):
        print("  WARNING: a source has fewer than 100 items; treat with caution.")

    tok = d['tok_matched'].tolist()
    dic = corpora.Dictionary(tok)
    dic.filter_extremes(no_below=5, no_above=0.5)
    bow = [dic.doc2bow(t) for t in tok]
    print(f"  Dictionary: {len(dic)} tokens")

    for k in K_SWEEP:
        model = LdaMulticore(corpus=bow, id2word=dic, num_topics=k,
                             random_state=RANDOM_STATE, passes=LDA_PASSES, workers=LDA_WORKERS)
        dom = []
        for b in bow:
            r = model.get_document_topics(b)
            dom.append(max(r, key=lambda x: x[1])[0] if r else -1)
        d[f'topic_k{k}'] = dom
        grouped = {p: d.loc[d['fonte'] == p, f'topic_k{k}'].to_numpy() for p in PLATFORMS}
        mat = np.vstack([np.bincount(g[g >= 0], minlength=k) / max((g >= 0).sum(), 1)
                         for g in grouped.values()])
        point = pair_sims(mat)

        boot = {pk: [] for pk in point}
        for _ in range(N_BOOT):
            rr = []
            for p in PLATFORMS:
                a = grouped[p]
                s = a[rng.integers(0, len(a), len(a))]
                c = np.bincount(s[s >= 0], minlength=k).astype(float)
                rr.append(c / c.sum())
            for pk, (cv, _jv) in pair_sims(np.vstack(rr)).items():
                boot[pk].append(cv)

        line = []
        for pk, (c, j) in point.items():
            lo, hi = np.percentile(boot[pk], [2.5, 97.5])
            rows.append({'condition': label, 'n_tokens': n_tokens,
                         'relevance_filter': apply_filter, 'k': k, 'pair': PAIRS[pk],
                         'cosine': round(c, 4), 'cos_ci_lo': round(lo, 4),
                         'cos_ci_hi': round(hi, 4), 'js': round(j, 4)})
            line.append(f"{PAIRS[pk]}={c:.3f}[{lo:.2f},{hi:.2f}]")
        best = max(point.items(), key=lambda kv: kv[1][0])
        print(f"  k={k:>2}: " + "  ".join(line) + f"   -> top: {PAIRS[best[0]]}")


def main():
    items = build_items()
    rng = np.random.default_rng(SEED)
    rows = []
    for label, n_tokens, apply_filter in CONDITIONS:
        run_condition(items, label, n_tokens, apply_filter, rng, rows)

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)

    print("\n\n================ SUMMARY: top pair per k, per condition ================")
    for label, _, _ in CONDITIONS:
        sub = out[out['condition'] == label]
        tops = []
        for k in K_SWEEP:
            s = sub[sub['k'] == k].sort_values('cosine', ascending=False)
            top, second = s.iloc[0], s.iloc[1]
            tie = top['cos_ci_lo'] <= second['cos_ci_hi']
            tops.append(f"k={k}:{top['pair'].replace('Guardian','G').replace('Reddit','R').replace('Telegram','T')}"
                        + ("(tie)" if tie else ""))
        print(f"  {label:<20} " + "  ".join(tops))
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
