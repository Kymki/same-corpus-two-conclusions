"""
C_v coherence beyond the k grid of script 05 (triage 2.7; reviewer 2).

On the symmetric subset the coherence optimum of script 05 falls at k = 21, the edge of
the grid {5, 7, 10, 15, 21}. Script 13 extends the similarity analysis to k = 25 and 30;
this script computes the coherence of those models, exactly as script 05 does (canonical
item order, same dictionary policy, seed 100, 20 passes, C_v), so the reader can see
whether the optimum is still at the edge. k = 21 is recomputed as a check against
data/results/symfilter_coherence_values.csv.

Output: data/results/symfilter_coherence_extended.csv
"""
import os
import sys
import pandas as pd
from gensim import corpora
from gensim.models import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_PROCESSED, DATA_RESULTS
from item_order import canonical_order

INPUT_CSV = os.path.join(DATA_PROCESSED, "dati_testuali_preproc_consolidati.csv")
PUBLISHED = os.path.join(DATA_RESULTS, "symfilter_coherence_values.csv")
OUT = os.path.join(DATA_RESULTS, "symfilter_coherence_extended.csv")

PLATFORMS = ["Guardian", "Reddit_Commento", "Telegram"]
WAR_KW = ['ukrain', 'russia', 'putin', 'zelensk', 'kyiv', 'kremlin', 'moscow',
          'donbas', 'crimea', 'kharkiv', 'kherson', 'mariupol', 'bakhmut', 'nato']
K_LIST = [21, 25, 30]
LDA_PASSES, LDA_WORKERS, RANDOM_STATE = 20, 7, 100


def main():
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    df = df[(df['lingua_rilevata'] == 'en') & (df['fonte'].isin(PLATFORMS))].copy()
    df = df.dropna(subset=['testo_lemmatizzato'])
    df['parent'] = df['id_originale'].astype(str).str.replace(r'_s\d+$', '', regex=True)
    items = df.groupby(['parent', 'fonte']).agg(
        testo_lemmatizzato=('testo_lemmatizzato', ' '.join),
        testo_pulito=('testo_pulito_base', lambda s: ' '.join(map(str, s))),
    ).reset_index()
    items = canonical_order(items)
    items = items[items['testo_pulito'].str.lower().apply(lambda t: any(k in t for k in WAR_KW))]
    tokenized = [t.split() for t in items['testo_lemmatizzato']]
    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus_bow = [dictionary.doc2bow(t) for t in tokenized]

    rows = []
    for k in K_LIST:
        model = LdaMulticore(corpus=corpus_bow, id2word=dictionary, num_topics=k,
                             random_state=RANDOM_STATE, passes=LDA_PASSES, workers=LDA_WORKERS)
        cv = CoherenceModel(model=model, texts=tokenized, dictionary=dictionary,
                            coherence='c_v').get_coherence()
        rows.append({'k': k, 'coherence_cv': cv})
        print(f"  k={k}: C_v = {cv:.4f}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    pub = pd.read_csv(PUBLISHED).set_index('k')['coherence_cv']
    print(f"\nk=21 check: recomputed {out.set_index('k').loc[21, 'coherence_cv']:.4f} vs published {pub.loc[21]:.4f}")
    print("Published grid:", {k: round(v, 4) for k, v in pub.items()})
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
