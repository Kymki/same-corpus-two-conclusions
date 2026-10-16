"""
Section 4.8 under solution variability: the diplomatic/kinetic contrast in affective
charge, re-derived for every model of the order x seed repetitions (triage 2.4 + 2.6).

Because LDA's topics change with the document order and the seed, the diplomatic/kinetic
classification cannot be carried over from one model to another: it is re-derived for
every model by the declared rule in topic_category_rule.py, applied unchanged (the rule
was fixed and validated by two blind coders before any charge was computed).

Per model (symmetric subset as in Sentiment_analysis/09; k in {10, 21}; canonical order
plus 10 permutations, seeds 100 and 200 = 22 models per k):
  - dominant topic per item and top-10 terms, exactly as Sentiment_analysis/09;
  - topic category under the main rule and the two declared variants;
  - affective charge (1 - mean neutral probability) per topic x platform cell
    (cells below MIN_CELL items flagged, as in the paper).
The contrast is tested directly rather than read off topic rankings (reviewer 2):
  within_telegram   mean charge of Telegram items in diplomatic topics minus kinetic topics
  did_vs_guardian   within_telegram minus the same difference for The Guardian
  did_vs_reddit     within_telegram minus the same difference for Reddit
each with a 95% bootstrap interval over items (B_EFFECT replicates).

Per channel (triage 2.14): the Telegram subcorpus is dominated by one channel
(WarTranslated, ~55% of the symmetric subset), so the within-Telegram contrast is also
computed separately for each of the three channels that contribute materially
(WarTranslated, Slavyangrad, Clash Report), with a bootstrap over that channel's items:
  within_channel:<name>   mean charge of the channel's items in diplomatic topics minus
                          kinetic topics
The channel of each Telegram item is read from data/results/item_clusters.csv (script
build_item_clusters.py), whose Telegram labels encode the channel.

Rule variants: the three declared in topic_category_rule.py (main, ALT_PROCESS, ALT_TOP5),
plus one POST-HOC variant defined here and labelled as such. In the blind validation two
topics were judged balanced by both coders but diplomatic by the rule, because its kinetic
lexicon lacks "weapon", "military" and "nuclear"; POSTHOC_KINETIC adds those three terms.
It was defined after the validation and before any charge was inspected, and is reported
only as a sensitivity check beside the main rule, which stays unchanged.

The script prints progress only; charge values are written to file and not displayed.
Repetition (order 0, seed 100) reproduces Sentiment_analysis/09; this is checked and the
check is reported as a yes/no.

Outputs (data/results/):
  topic_charge_repetitions_topics.csv   one row per k, order, seed, topic (terms, categories, cells)
  topic_charge_repetitions_effects.csv  one row per k, order, seed, rule variant, effect
"""
import os
import sys
import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models import LdaMulticore

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_PROCESSED, DATA_RESULTS
from item_order import canonical_order
import topic_category_rule as rule

CORPUS = os.path.join(DATA_PROCESSED, "dati_testuali_preproc_consolidati.csv")
ITEMS_META = os.path.join(DATA_RESULTS, "items_en_meta.csv")
CLUSTERS = os.path.join(DATA_RESULTS, "item_clusters.csv")
OUT_TOPICS = os.path.join(DATA_RESULTS, "topic_charge_repetitions_topics.csv")
OUT_EFFECTS = os.path.join(DATA_RESULTS, "topic_charge_repetitions_effects.csv")

PLATFORMS = ["Guardian", "Reddit_Commento", "Telegram"]
NAMES = {"Guardian": "Guardian", "Reddit_Commento": "Reddit", "Telegram": "Telegram"}
WAR_KW = ['ukrain', 'russia', 'putin', 'zelensk', 'kyiv', 'kremlin', 'moscow',
          'donbas', 'crimea', 'kharkiv', 'kherson', 'mariupol', 'bakhmut', 'nato']
K_LIST = [10, 21]
N_PERMUTATIONS, SEEDS = 10, [100, 200]
LDA_PASSES, LDA_WORKERS = 20, 7
MIN_CELL = 30
B_EFFECT = 500
BOOT_SEED = 42
DIP, KIN = "Diplomatic/institutional", "Kinetic"
POSTHOC_KINETIC_TERMS = {"weapon", "military", "nuclear"}
VARIANTS = dict(rule.VARIANTS,
                POSTHOC_KINETIC=dict(kinetic=rule.KINETIC | POSTHOC_KINETIC_TERMS))
# channel labels as written by build_item_clusters.py (channels sorted by title)
CHANNEL_NAMES = {"channel_1": "Clash Report", "channel_2": "Militare Semplice",
                 "channel_3": "Rybar in English", "channel_4": "Slavyangrad",
                 "channel_5": "WarTranslated"}
CHANNELS_TESTED = ["channel_5", "channel_4", "channel_1"]


def dominant_topics_as_published(model, bow):
    """Dominant topic per document, computed exactly as Sentiment_analysis/09 computes it.

    gensim's get_document_topics initialises its variational parameters with draws from the
    model's random_state, so every call advances that generator. Sentiment_analysis/09 calls
    it twice per document (once in the condition, once inside max); the same sequence of
    calls is needed to reproduce its topic assignments, and hence Section 4.8, exactly.
    """
    return np.array([max(model.get_document_topics(b), key=lambda x: x[1])[0]
                     if model.get_document_topics(b) else -1 for b in bow])


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
    items = items[items['clean'].str.lower().apply(lambda t: any(k in t for k in WAR_KW))].copy()
    meta = pd.read_csv(ITEMS_META, usecols=['parent', 'neu_prob'])
    items = items.merge(meta, on='parent', how='inner').reset_index(drop=True)
    items['charge'] = 1 - items['neu_prob']
    cl = pd.read_csv(CLUSTERS, usecols=['parent', 'fonte', 'cluster'])
    items = items.merge(cl, on=['parent', 'fonte'], how='left')
    tel = items['fonte'] == 'Telegram'
    items['channel'] = items['cluster'].str.extract(r'^telegram_(channel_\d+)_')[0].where(tel)
    assert items.loc[tel, 'channel'].notna().all(), "a Telegram item has no channel"
    return items


def channel_effect(charge, chan, cat, channel, rng):
    """Diplomatic minus kinetic mean charge within one channel, bootstrap over its items."""
    sel = np.where(chan == channel)[0]
    c, k = charge[sel], cat[sel]
    def diff(cc, kk):
        d, kin = cc[kk == DIP], cc[kk == KIN]
        return (d.mean() - kin.mean()) if len(d) and len(kin) else np.nan
    boot = []
    for _ in range(B_EFFECT):
        ix = rng.integers(0, len(sel), len(sel))
        boot.append(diff(c[ix], k[ix]))
    lo, hi = np.nanpercentile(boot, [2.5, 97.5]) if np.isfinite(boot).any() else (np.nan, np.nan)
    return diff(c, k), lo, hi, int((k == DIP).sum()), int((k == KIN).sum())


def effects(charge, src, cat, rng):
    """Within-Telegram contrast and its difference from the other two platforms."""
    def diff(p, idx=None):
        c, s, k = (charge, src, cat) if idx is None else (charge[idx], src[idx], cat[idx])
        d, kk = c[(s == p) & (k == DIP)], c[(s == p) & (k == KIN)]
        return (d.mean() - kk.mean()) if len(d) and len(kk) else np.nan
    point = {'within_telegram': diff('Telegram'),
             'did_vs_guardian': diff('Telegram') - diff('Guardian'),
             'did_vs_reddit': diff('Telegram') - diff('Reddit_Commento')}
    boot = {e: [] for e in point}
    n = len(charge)
    for _ in range(B_EFFECT):
        idx = rng.integers(0, n, n)
        t = diff('Telegram', idx)
        boot['within_telegram'].append(t)
        boot['did_vs_guardian'].append(t - diff('Guardian', idx))
        boot['did_vs_reddit'].append(t - diff('Reddit_Commento', idx))
    return {e: (point[e], *np.nanpercentile(boot[e], [2.5, 97.5])) for e in point}


def main():
    items = load_items()
    print(f"Symmetric subset with item measures: {len(items)} items", flush=True)
    topic_rows, effect_rows = [], []
    for o in range(N_PERMUTATIONS + 1):
        perm = np.arange(len(items)) if o == 0 else np.random.default_rng(o).permutation(len(items))
        d = items.iloc[perm].reset_index(drop=True)
        tok = [t.split() for t in d['lem']]
        dic = corpora.Dictionary(tok)
        dic.filter_extremes(no_below=5, no_above=0.5)
        bow = [dic.doc2bow(t) for t in tok]
        src, charge = d['fonte'].to_numpy(), d['charge'].to_numpy()
        chan = d['channel'].to_numpy(dtype=object)
        for seed in SEEDS:
            for K in K_LIST:
                m = LdaMulticore(corpus=bow, id2word=dic, num_topics=K, random_state=seed,
                                 passes=LDA_PASSES, workers=LDA_WORKERS)
                dom = dominant_topics_as_published(m, bow)
                terms = {i: [w.split('*')[1].strip().strip('"') for w in t.split(' + ')]
                         for i, t in m.print_topics(-1, num_words=10)}
                cats = {v: {t: rule.categorise(terms[t], **kw) for t in terms}
                        for v, kw in VARIANTS.items()}
                for t in range(K):
                    row = dict(k=K, order=o, seed=seed, topic=t, top_terms=', '.join(terms[t]),
                               **{f'category_{v}': cats[v][t] for v in cats})
                    for p in PLATFORMS:
                        sel = (dom == t) & (src == p)
                        row[f'n_{NAMES[p]}'] = int(sel.sum())
                        row[f'charge_{NAMES[p]}'] = round(float(charge[sel].mean()) * 100, 2) if sel.any() else np.nan
                    topic_rows.append(row)
                for v in cats:
                    cat = np.array([cats[v].get(t, "Other") for t in dom], dtype=object)
                    n_tel = {c: int(((src == 'Telegram') & (cat == c)).sum()) for c in (DIP, KIN)}
                    rng = np.random.default_rng(BOOT_SEED)
                    for e, (pt, lo, hi) in effects(charge, src, cat, rng).items():
                        effect_rows.append(dict(k=K, order=o, seed=seed, variant=v, effect=e,
                                                estimate=round(pt * 100, 2), ci_lo=round(lo * 100, 2),
                                                ci_hi=round(hi * 100, 2), n_telegram_dip=n_tel[DIP],
                                                n_telegram_kin=n_tel[KIN]))
                    for ch in CHANNELS_TESTED:
                        pt, lo, hi, nd, nk = channel_effect(charge, chan, cat, ch, rng)
                        effect_rows.append(dict(k=K, order=o, seed=seed, variant=v,
                                                effect=f"within_channel:{CHANNEL_NAMES[ch]}",
                                                estimate=round(pt * 100, 2), ci_lo=round(lo * 100, 2),
                                                ci_hi=round(hi * 100, 2), n_telegram_dip=nd,
                                                n_telegram_kin=nk))
                print(f"  k={K:>2} order={o:>2} seed={seed} done", flush=True)
        pd.DataFrame(topic_rows).to_csv(OUT_TOPICS, index=False)      # checkpoint
        pd.DataFrame(effect_rows).to_csv(OUT_EFFECTS, index=False)

    # check, reported as yes/no: the canonical run reproduces Sentiment_analysis/09's cells
    tp = pd.DataFrame(topic_rows)
    ok = True
    for K in K_LIST:
        pub = pd.read_csv(os.path.join(DATA_RESULTS, f"symfilter_continuous_by_topic_platform_k{K}.csv"))
        can = tp[(tp.k == K) & (tp.order == 0) & (tp.seed == 100)]
        for r in pub.itertuples():
            row = can[can.topic == r.topic].iloc[0]
            ok &= (row[f'n_{r.platform}'] == r.n_items) and abs(row[f'charge_{r.platform}'] - r.charge_mean) < 0.006
    print(f"\nCanonical run reproduces Sentiment_analysis/09 cells: {'yes' if ok else 'NO'}")
    print(f"Saved (values not displayed): {OUT_TOPICS}\n                              {OUT_EFFECTS}")


if __name__ == "__main__":
    main()
