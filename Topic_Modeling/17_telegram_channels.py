"""
Telegram per channel: item counts through the pipeline, and the diplomatic/kinetic
gradient within each channel (triage 2.14; reviewer 2).

The Telegram subcorpus pools five channels. Two questions follow: how unevenly the
channels contribute once language and relevance filters are applied, and whether the
within-Telegram contrast of Section 4.8 is a property of the channels in general or is
carried by one of them.

The channel of each message is recovered from the raw collection
(data/raw/telegram_messaggi_raccolti.csv, available on request): message ids are unique
across the five channels, so the mapping is unambiguous.

Counts: messages collected; items in the English corpus; items in the symmetric subset;
Italian items (excluded from the analysis).

Gradient: on the canonical model of Section 4.8 (symmetric subset, canonical order,
seed 100, k = 10 and 21), topics are classified by the validated rule in
topic_category_rule.py, and for each channel the mean affective charge of its items in
diplomatic topics minus kinetic topics is computed, with a 95% bootstrap interval over
the channel's items. Gradient values are written to file and not displayed.

Outputs (data/results/):
  telegram_channel_counts.csv
  telegram_channel_gradient.csv
"""
import os
import sys
import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models import LdaMulticore

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_PROCESSED, DATA_RAW, DATA_RESULTS
from item_order import canonical_order
import topic_category_rule as rule

CORPUS = os.path.join(DATA_PROCESSED, "dati_testuali_preproc_consolidati.csv")
RAW_TELEGRAM = os.path.join(DATA_RAW, "telegram_messaggi_raccolti.csv")
ITEMS_META = os.path.join(DATA_RESULTS, "items_en_meta.csv")
OUT_COUNTS = os.path.join(DATA_RESULTS, "telegram_channel_counts.csv")
OUT_GRADIENT = os.path.join(DATA_RESULTS, "telegram_channel_gradient.csv")

PLATFORMS = ["Guardian", "Reddit_Commento", "Telegram"]
WAR_KW = ['ukrain', 'russia', 'putin', 'zelensk', 'kyiv', 'kremlin', 'moscow',
          'donbas', 'crimea', 'kharkiv', 'kherson', 'mariupol', 'bakhmut', 'nato']
K_LIST, SEED, LDA_PASSES, LDA_WORKERS = [10, 21], 100, 20, 7
B, BOOT_SEED, MIN_CELL = 1000, 42, 30
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


def main():
    raw = pd.read_csv(RAW_TELEGRAM, usecols=['message_id', 'chat_title'])
    assert raw['message_id'].is_unique, "message ids are not unique across channels"
    channel = dict(zip(raw['message_id'].astype(str), raw['chat_title']))

    df = pd.read_csv(CORPUS, low_memory=False)
    df['parent'] = df['id_originale'].astype(str).str.replace(r'_s\d+$', '', regex=True)
    tel = df[df['fonte'] == 'Telegram'].copy()
    tel['channel'] = tel['parent'].map(channel)
    assert tel['channel'].notna().all(), "a Telegram item has no channel in the raw collection"

    items_all = tel.groupby(['parent', 'lingua_rilevata', 'channel']).agg(
        clean=('testo_pulito_base', lambda s: ' '.join(map(str, s)))).reset_index()
    en = items_all[items_all['lingua_rilevata'] == 'en']
    sym = en[en['clean'].str.lower().apply(lambda t: any(k in t for k in WAR_KW))]
    counts = pd.DataFrame({
        'messages_collected': raw.groupby('chat_title').size(),
        'items_english': en.groupby('channel').size(),
        'items_italian': items_all[items_all['lingua_rilevata'] == 'it'].groupby('channel').size(),
        'items_symmetric_subset': sym.groupby('channel').size(),
    }).fillna(0).astype(int)
    counts['share_of_telegram_symmetric_pct'] = (100 * counts['items_symmetric_subset']
                                                 / counts['items_symmetric_subset'].sum()).round(1)
    counts.index.name = 'channel'
    counts.to_csv(OUT_COUNTS)
    print("Telegram items per channel:\n" + counts.to_string())

    # canonical model of Section 4.8 (as Sentiment_analysis/09)
    d = df[(df['lingua_rilevata'] == 'en') & (df['fonte'].isin(PLATFORMS))].dropna(subset=['testo_lemmatizzato'])
    items = d.groupby(['parent', 'fonte']).agg(
        lem=('testo_lemmatizzato', ' '.join),
        clean=('testo_pulito_base', lambda s: ' '.join(map(str, s)))).reset_index()
    items = canonical_order(items)
    items = items[items['clean'].str.lower().apply(lambda t: any(k in t for k in WAR_KW))].copy()
    items = items.merge(pd.read_csv(ITEMS_META, usecols=['parent', 'neu_prob']), on='parent', how='inner')
    items = items.reset_index(drop=True)
    items['charge'] = 1 - items['neu_prob']
    items['channel'] = items['parent'].map(channel).where(items['fonte'] == 'Telegram')
    tok = [t.split() for t in items['lem']]
    dic = corpora.Dictionary(tok)
    dic.filter_extremes(no_below=5, no_above=0.5)
    bow = [dic.doc2bow(t) for t in tok]

    rng = np.random.default_rng(BOOT_SEED)
    rows = []
    for K in K_LIST:
        m = LdaMulticore(corpus=bow, id2word=dic, num_topics=K, random_state=SEED,
                         passes=LDA_PASSES, workers=LDA_WORKERS)
        dom = dominant_topics_as_published(m, bow)
        terms = {i: [w.split('*')[1].strip().strip('"') for w in t.split(' + ')]
                 for i, t in m.print_topics(-1, num_words=10)}
        cat = np.array([rule.categorise(terms[t]) if t >= 0 else "Other" for t in dom], dtype=object)
        for ch in sorted(counts.index):
            sel = (items['channel'] == ch).to_numpy()
            c, k = items['charge'].to_numpy()[sel], cat[sel]
            dip, kin = c[k == DIP], c[k == KIN]
            row = dict(k=K, channel=ch, n_diplomatic=len(dip), n_kinetic=len(kin))
            if len(dip) >= MIN_CELL and len(kin) >= MIN_CELL:
                boot = [rng.choice(dip, len(dip)).mean() - rng.choice(kin, len(kin)).mean() for _ in range(B)]
                lo, hi = np.percentile(boot, [2.5, 97.5])
                row.update(gradient=round((dip.mean() - kin.mean()) * 100, 2),
                           ci_lo=round(lo * 100, 2), ci_hi=round(hi * 100, 2))
            rows.append(row)
        print(f"  k={K}: gradients computed (values not displayed)", flush=True)
    pd.DataFrame(rows).to_csv(OUT_GRADIENT, index=False)
    print(f"\nSaved: {OUT_COUNTS}\n       {OUT_GRADIENT} (values not displayed)")


if __name__ == "__main__":
    main()
