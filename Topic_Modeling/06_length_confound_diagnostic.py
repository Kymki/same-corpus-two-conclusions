"""
Diagnostic: document length confounds the topic model too (not only sentiment).

Context. The symmetric-filter robustness check (05) reverses the main similarity
result: with the same relevance criterion applied to all sources, Reddit-Telegram
becomes the most similar pair (cosine 0.963 at k=10) instead of Guardian-Telegram.
Before accepting either result, this script tests whether the reversal is itself
a length artefact.

Finding. In the symmetrically filtered k=10 model, one topic absorbs the short
documents of both short-form platforms:
  - Reddit puts 54.9% and Telegram 52.6% of their items in the SAME topic, while
    The Guardian puts only 3.0% there;
  - that topic's mean item length is ~30 lemmatised words against 141-401 for
    every other topic;
  - The Guardian supplies ~88% of the corpus token mass, so LDA topics are
    estimated essentially on Guardian prose, and short items from the other
    sources are absorbed by a single generic short-text topic.

Consequence. The high Reddit-Telegram similarity is driven by co-assignment to a
length-defined topic rather than by a shared substantive agenda. Symmetrically,
the original Guardian-Telegram result was confounded by asymmetric filtering.
Neither configuration yields a clean estimate of thematic similarity, and the
cross-platform similarity analysis should be reported as not robust.

This is the same class of artefact the paper documents for sentiment
aggregation, now affecting topic modelling: it extends the methodological
contribution rather than merely undermining a result.

Output: data/results/length_confound_diagnostic.csv
"""
import os
import sys
import pandas as pd
from gensim import corpora
from gensim.models import LdaMulticore

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_PROCESSED, DATA_RESULTS

INPUT_CSV = os.path.join(DATA_PROCESSED, "dati_testuali_preproc_consolidati.csv")
OUT = os.path.join(DATA_RESULTS, "length_confound_diagnostic.csv")

PLATFORMS = ["Guardian", "Reddit_Commento", "Telegram"]
WAR_KW = ['ukrain', 'russia', 'putin', 'zelensk', 'kyiv', 'kremlin', 'moscow',
          'donbas', 'crimea', 'kharkiv', 'kherson', 'mariupol', 'bakhmut', 'nato']
K = 10


def main():
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    df = df[(df['lingua_rilevata'] == 'en') & (df['fonte'].isin(PLATFORMS))]
    df = df.dropna(subset=['testo_lemmatizzato']).copy()
    df['parent'] = df['id_originale'].astype(str).str.replace(r'_s\d+$', '', regex=True)

    items = df.groupby(['parent', 'fonte']).agg(
        lem=('testo_lemmatizzato', ' '.join),
        clean=('testo_pulito_base', lambda s: ' '.join(map(str, s))),
    ).reset_index()

    # token mass per source (why LDA topics are Guardian's topics)
    items['n_words'] = items['lem'].str.split().str.len()
    mass = items.groupby('fonte')['n_words'].sum()
    print("Token mass share per source (%):")
    print((mass / mass.sum() * 100).round(1).to_string())

    # symmetric relevance filter, then k=10 LDA
    items = items[items['clean'].str.lower().apply(lambda t: any(k in t for k in WAR_KW))].copy()
    tok = [t.split() for t in items['lem']]
    d = corpora.Dictionary(tok)
    d.filter_extremes(no_below=5, no_above=0.5)
    bow = [d.doc2bow(t) for t in tok]
    model = LdaMulticore(corpus=bow, id2word=d, num_topics=K,
                         random_state=100, passes=20, workers=7)
    items['topic'] = [max(model.get_document_topics(b), key=lambda x: x[1])[0]
                      if model.get_document_topics(b) else -1 for b in bow]

    share = pd.crosstab(items['fonte'], items['topic'], normalize='index') * 100
    length = items.groupby('topic')['n_words'].agg(['mean', 'count'])
    out = length.join(share.T).rename(columns={'mean': 'mean_item_words', 'count': 'n_items'})
    out.index.name = 'topic'
    out = out.round(1)
    out.to_csv(OUT)

    print("\nPer topic: mean item length and platform shares (%)")
    print(out.to_string())
    short = out['mean_item_words'].idxmin()
    print(f"\nShortest-document topic: {short} "
          f"(mean {out.loc[short, 'mean_item_words']:.0f} words) — platform shares: "
          + ", ".join(f"{p} {out.loc[short, p]:.1f}%" for p in PLATFORMS if p in out.columns))
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
