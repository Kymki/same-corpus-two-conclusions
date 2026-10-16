"""
Cluster bootstrap for the paper's confidence intervals (triage 2.5; reviewers 1, 2, 3).

Every interval in the paper resamples items as if they were independent. They are not:
Reddit comments share a thread, Telegram messages a channel and a news cycle, Guardian
articles a week of events. Here the same quantities are bootstrapped by resampling whole
clusters (data/results/item_clusters.csv), within each platform:

  Reddit    discussion thread
  Telegram  channel x ISO week   (sensitivity: channel x day)
  Guardian  ISO week             (sensitivity: day)

and the item bootstrap is recomputed alongside for comparison. Telegram has few weekly
clusters (19), so its weekly intervals are coarse; the daily version (87 clusters) is
reported beside it.

Quantities:
  1. platform means of negative probability, signed valence and affective charge, on the
     full English corpus (Section 4.4) and on the symmetric subset (Section 4.6), and the
     pairwise platform differences in negative probability;
  2. cross-platform cosine similarity of the canonical solution of script 05
     (symmetric subset, canonical order, seed 100, k in {5,7,10,15,21});
  3. the Section 4.8 contrasts on its canonical model (k = 10 and 21, main rule):
     written to file and NOT displayed, like the rest of Section 4.8 until its reading
     criterion is fixed.

Outputs (data/results/):
  cluster_bootstrap_sentiment.csv
  cluster_bootstrap_similarity.csv
  cluster_bootstrap_topic_charge.csv   (values not displayed)
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
import topic_category_rule as rule

CORPUS = os.path.join(DATA_PROCESSED, "dati_testuali_preproc_consolidati.csv")
ITEMS_META = os.path.join(DATA_RESULTS, "items_en_meta.csv")
CLUSTERS = os.path.join(DATA_RESULTS, "item_clusters.csv")
OUT_SENT = os.path.join(DATA_RESULTS, "cluster_bootstrap_sentiment.csv")
OUT_SIM = os.path.join(DATA_RESULTS, "cluster_bootstrap_similarity.csv")
OUT_CHARGE = os.path.join(DATA_RESULTS, "cluster_bootstrap_topic_charge.csv")

PLATFORMS = ["Guardian", "Reddit_Commento", "Telegram"]
NAMES = {"Guardian": "Guardian", "Reddit_Commento": "Reddit", "Telegram": "Telegram"}
WAR_KW = ['ukrain', 'russia', 'putin', 'zelensk', 'kyiv', 'kremlin', 'moscow',
          'donbas', 'crimea', 'kharkiv', 'kherson', 'mariupol', 'bakhmut', 'nato']
SCHEMES = {"item": None, "cluster_week": "cluster", "cluster_day": "cluster_day"}
K_SIM, K_CHARGE = [5, 7, 10, 15, 21], [10, 21]
SEED, LDA_PASSES, LDA_WORKERS = 100, 20, 7
B, BOOT_SEED = 1000, 42
DIP, KIN = "Diplomatic/institutional", "Kinetic"


def dominant_topics_as_published(model, bow):
    """Dominant topic per document, computed exactly as Sentiment_analysis/09 computes it.

    gensim's get_document_topics initialises its variational parameters with draws from the
    model's random_state, so every call advances that generator. Sentiment_analysis/09 calls
    it twice per document (once in the condition, once inside max); the same sequence of
    calls is needed to reproduce its topic assignments, and hence Section 4.8, exactly.
    """
    return np.array([max(model.get_document_topics(b), key=lambda x: x[1])[0]
                     if model.get_document_topics(b) else -1 for b in bow])


def resampler(groups, rng):
    """Index arrays for B resamples of one platform: items, or whole clusters."""
    if groups is None:
        return None
    codes, uniq = pd.factorize(groups)
    members = [np.where(codes == c)[0] for c in range(len(uniq))]
    def draw(n_items):
        pick = rng.integers(0, len(members), len(members))
        return np.concatenate([members[c] for c in pick])
    return draw


def boot_index(n, groups, rng):
    draw = resampler(groups, rng)
    for _ in range(B):
        yield rng.integers(0, n, n) if draw is None else draw(n)


def ci(values):
    lo, hi = np.percentile(values, [2.5, 97.5])
    return round(lo * 100, 2), round(hi * 100, 2)


def load_items():
    df = pd.read_csv(CORPUS, low_memory=False)
    df = df[(df['lingua_rilevata'] == 'en') & (df['fonte'].isin(PLATFORMS))]
    df = df.dropna(subset=['testo_lemmatizzato']).copy()
    df['parent'] = df['id_originale'].astype(str).str.replace(r'_s\d+$', '', regex=True)
    items = df.groupby(['parent', 'fonte']).agg(
        lem=('testo_lemmatizzato', ' '.join),
        clean=('testo_pulito_base', lambda s: ' '.join(map(str, s))),
    ).reset_index()
    items = canonical_order(items)
    items['relevant'] = items['clean'].str.lower().apply(lambda t: any(k in t for k in WAR_KW))
    meta = pd.read_csv(ITEMS_META, usecols=['parent', 'neg_prob', 'neu_prob', 'pos_prob'])
    items = items.merge(meta, on='parent', how='left').merge(
        pd.read_csv(CLUSTERS), on=['parent', 'fonte'], how='left')
    assert items['cluster'].notna().all() and items['neg_prob'].notna().all()
    items['valence'] = items['pos_prob'] - items['neg_prob']
    items['charge'] = 1 - items['neu_prob']
    return items


def sentiment(items, rng):
    rows = []
    for subset, d in [("full", items), ("symmetric", items[items['relevant']])]:
        for scheme, col in SCHEMES.items():
            means = {}
            for p in PLATFORMS:
                dp = d[d['fonte'] == p]
                groups = None if col is None else dp[col].to_numpy()
                idx = list(boot_index(len(dp), groups, rng))
                for m in ('neg_prob', 'valence', 'charge'):
                    x = dp[m].to_numpy()
                    boot = np.array([x[i].mean() for i in idx])
                    means[(p, m)] = boot
                    lo, hi = ci(boot)
                    rows.append(dict(subset=subset, scheme=scheme, quantity=f"mean_{m}", platform=NAMES[p],
                                     estimate=round(x.mean() * 100, 2), ci_lo=lo, ci_hi=hi,
                                     n_items=len(dp), n_clusters=None if col is None else dp[col].nunique()))
            for a, b in [("Guardian", "Reddit_Commento"), ("Guardian", "Telegram"), ("Reddit_Commento", "Telegram")]:
                diff = means[(a, 'neg_prob')] - means[(b, 'neg_prob')]
                est = d.loc[d['fonte'] == a, 'neg_prob'].mean() - d.loc[d['fonte'] == b, 'neg_prob'].mean()
                lo, hi = ci(diff)
                rows.append(dict(subset=subset, scheme=scheme, quantity="diff_neg_prob",
                                 platform=f"{NAMES[a]} - {NAMES[b]}", estimate=round(est * 100, 2),
                                 ci_lo=lo, ci_hi=hi, n_items=None, n_clusters=None))
        print(f"  sentiment, {subset} corpus: done", flush=True)
    return pd.DataFrame(rows)


def fit(d, K, as_published_48=False):
    """One model per k; dominant topics as script 05 (similarity) or as
    Sentiment_analysis/09 (Section 4.8), whose assignments differ (see
    dominant_topics_as_published)."""
    tok = [t.split() for t in d['lem']]
    dic = corpora.Dictionary(tok)
    dic.filter_extremes(no_below=5, no_above=0.5)
    bow = [dic.doc2bow(t) for t in tok]
    out = {}
    for k in K:
        m = LdaMulticore(corpus=bow, id2word=dic, num_topics=k, random_state=SEED,
                         passes=LDA_PASSES, workers=LDA_WORKERS)
        dom = (dominant_topics_as_published(m, bow) if as_published_48 else
               np.array([max(r, key=lambda x: x[1])[0] if r else -1
                         for r in (m.get_document_topics(b) for b in bow)]))
        terms = {i: [w.split('*')[1].strip().strip('"') for w in t.split(' + ')]
                 for i, t in m.print_topics(-1, num_words=10)}
        out[k] = (dom, terms)
    return out


def similarity(sym, models, rng):
    rows = []
    src = sym['fonte'].to_numpy()
    for k in K_SIM:
        dom = models[k][0]
        def profile(ix_by_p):
            mat = []
            for p in PLATFORMS:
                g = dom[ix_by_p[p]]
                g = g[g >= 0]
                mat.append(np.bincount(g, minlength=k) / max(len(g), 1))
            return cosine_similarity(np.vstack(mat))
        pos = {p: np.where(src == p)[0] for p in PLATFORMS}
        point = profile(pos)
        for scheme, col in SCHEMES.items():
            gens = {p: boot_index(len(pos[p]), None if col is None else sym[col].to_numpy()[pos[p]], rng)
                    for p in PLATFORMS}
            boots = [profile({p: pos[p][next(gens[p])] for p in PLATFORMS}) for _ in range(B)]
            for (i, j), name in {(0, 1): "Guardian-Reddit", (0, 2): "Guardian-Telegram",
                                 (1, 2): "Reddit-Telegram"}.items():
                vals = np.array([b[i, j] for b in boots])
                lo, hi = np.percentile(vals, [2.5, 97.5])
                rows.append(dict(k=k, scheme=scheme, pair=name, cosine=round(point[i, j], 4),
                                 ci_lo=round(lo, 4), ci_hi=round(hi, 4)))
        print(f"  similarity k={k}: done", flush=True)
    return pd.DataFrame(rows)


def topic_charge(sym, models, rng):
    rows = []
    src, charge = sym['fonte'].to_numpy(), sym['charge'].to_numpy()
    for k in K_CHARGE:
        dom, terms = models[k]
        cat = np.array([rule.categorise(terms[t]) if t >= 0 else "Other" for t in dom], dtype=object)
        def contrasts(ix_by_p):
            def diff(p):
                ix = ix_by_p[p]
                c, kk = charge[ix], cat[ix]
                d, kin = c[kk == DIP], c[kk == KIN]
                return d.mean() - kin.mean() if len(d) and len(kin) else np.nan
            t = diff('Telegram')
            return {'within_telegram': t, 'did_vs_guardian': t - diff('Guardian'),
                    'did_vs_reddit': t - diff('Reddit_Commento')}
        pos = {p: np.where(src == p)[0] for p in PLATFORMS}
        point = contrasts(pos)
        for scheme, col in SCHEMES.items():
            gens = {p: boot_index(len(pos[p]), None if col is None else sym[col].to_numpy()[pos[p]], rng)
                    for p in PLATFORMS}
            boots = [contrasts({p: pos[p][next(gens[p])] for p in PLATFORMS}) for _ in range(B)]
            for e in point:
                lo, hi = np.nanpercentile([b[e] for b in boots], [2.5, 97.5])
                rows.append(dict(k=k, scheme=scheme, variant='main', effect=e,
                                 estimate=round(point[e] * 100, 2), ci_lo=round(lo * 100, 2),
                                 ci_hi=round(hi * 100, 2)))
        print(f"  topic charge k={k}: done (values not displayed)", flush=True)
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(BOOT_SEED)
    items = load_items()
    print(f"Items: {len(items)}; clusters (weekly): "
          + ", ".join(f"{NAMES[p]} {items.loc[items.fonte == p, 'cluster'].nunique()}" for p in PLATFORMS), flush=True)
    sent = sentiment(items, rng)
    sent.to_csv(OUT_SENT, index=False)
    sym = items[items['relevant']].reset_index(drop=True)
    models = fit(sym, K_SIM)
    sim = similarity(sym, models, rng)
    sim.to_csv(OUT_SIM, index=False)
    topic_charge(sym, fit(sym, K_CHARGE, as_published_48=True), rng).to_csv(OUT_CHARGE, index=False)

    show = sent[(sent['quantity'].isin(['mean_neg_prob', 'diff_neg_prob']))]
    print("\nNegative probability: estimate and 95% interval by bootstrap scheme")
    print(show.pivot_table(index=['subset', 'quantity', 'platform'], columns='scheme',
                           values=['ci_lo', 'ci_hi'], aggfunc='first').round(2).to_string())
    print(f"\nSaved: {OUT_SENT}\n       {OUT_SIM}\n       {OUT_CHARGE} (values not displayed)")


if __name__ == "__main__":
    main()
