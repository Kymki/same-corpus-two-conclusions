"""
Continuous measures on the SYMMETRICALLY FILTERED corpus (RP request).

Rationale: the previously reported topic x platform measures were computed on the
unfiltered corpus with the unfiltered k=10 topic model. Since the paper now
argues that asymmetric filtering invalidates conclusions, a conclusion drawn
from the asymmetric corpus cannot be retained. Everything is therefore
recomputed on the symmetric subset (>=1 war keyword in every source), with the
topic model retrained on that subset.

Reported:
  - C_v coherence on the symmetric subset (from script 05) to choose the
    operational k; the optimum is NOT 10, so both the optimum and k=10 are run;
  - the three continuous measures (p_neg, signed valence p_pos-p_neg, affective
    charge 1-p_neu) per source, on the symmetric subset;
  - the same three measures per topic x platform cell, with bootstrap 95% CIs
    and cell n, discarding cells below MIN_CELL items;
  - top terms per topic for the retrained model, so Figure 5 can be relabelled.

Key question: does the "flat register" result survive? On the unfiltered corpus
Telegram's affective charge on peace talks was 59.8% [57.4; 62.2] against
Guardian 67.4 and Reddit 71.9, with separated CIs.
"""
import os
import sys
import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models import LdaMulticore

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_PROCESSED, DATA_RESULTS

CORPUS = os.path.join(DATA_PROCESSED, "dati_testuali_preproc_consolidati.csv")
ITEMS_META = os.path.join(DATA_RESULTS, "items_en_meta.csv")
COH = os.path.join(DATA_RESULTS, "symfilter_coherence_values.csv")
OUT_SRC = os.path.join(DATA_RESULTS, "symfilter_continuous_by_source.csv")

PLATFORMS = ["Guardian", "Reddit_Commento", "Telegram"]
NAMES = {"Guardian": "Guardian", "Reddit_Commento": "Reddit", "Telegram": "Telegram"}
WAR_KW = ['ukrain', 'russia', 'putin', 'zelensk', 'kyiv', 'kremlin', 'moscow',
          'donbas', 'crimea', 'kharkiv', 'kherson', 'mariupol', 'bakhmut', 'nato']
MEASURES = {"neg_prob": "p_neg", "valence": "signed valence", "charge": "affective charge"}
MIN_CELL = 30
B = 1000
SEED = 42
LDA_PASSES, LDA_WORKERS, RS = 20, 7, 100
# terms used only to FLAG which retrained topic looks like diplomacy / peace talks
DIPLO = {'peace', 'talk', 'negotiation', 'summit', 'ceasefire', 'deal',
         'zelenskyy', 'zelensky', 'putin', 'trump', 'diplomatic', 'agreement'}


def boot_ci(x, rng, b=B):
    x = np.asarray(x, float)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(b)]
    return np.percentile(m, [2.5, 97.5])


def main():
    coh = pd.read_csv(COH).sort_values('k')
    k_opt = int(coh.loc[coh['coherence_cv'].idxmax(), 'k'])
    print("=== C_v on the symmetric subset ===")
    print(coh.to_string(index=False))
    print(f"  -> coherence optimum: k={k_opt} "
          f"(k=10 gives {float(coh.loc[coh.k == 10, 'coherence_cv'].iloc[0]):.4f})")

    df = pd.read_csv(CORPUS, low_memory=False)
    df = df[(df['lingua_rilevata'] == 'en') & (df['fonte'].isin(PLATFORMS))]
    df = df.dropna(subset=['testo_lemmatizzato']).copy()
    df['parent'] = df['id_originale'].astype(str).str.replace(r'_s\d+$', '', regex=True)
    items = df.groupby(['parent', 'fonte']).agg(
        lem=('testo_lemmatizzato', ' '.join),
        clean=('testo_pulito_base', lambda s: ' '.join(map(str, s))),
    ).reset_index()
    items = items[items['clean'].str.lower().apply(lambda t: any(k in t for k in WAR_KW))].copy()
    print(f"\nItems in the symmetric subset: {len(items)}")
    print(items.groupby('fonte').size().rename(index=NAMES).to_string())

    meta = pd.read_csv(ITEMS_META, usecols=['parent', 'neg_prob', 'neu_prob', 'pos_prob'])
    items = items.merge(meta, on='parent', how='inner')
    items['valence'] = items['pos_prob'] - items['neg_prob']
    items['charge'] = 1 - items['neu_prob']
    rng = np.random.default_rng(SEED)

    # --- three measures per source, symmetric subset ---
    rows = []
    for p in PLATFORMS:
        d = items[items['fonte'] == p]
        r = {'source': NAMES[p], 'n_items': len(d)}
        for m in MEASURES:
            lo, hi = boot_ci(d[m].values, rng)
            r[f'{m}_mean'] = round(d[m].mean() * 100, 2)
            r[f'{m}_lo'] = round(lo * 100, 2)
            r[f'{m}_hi'] = round(hi * 100, 2)
        rows.append(r)
    by_src = pd.DataFrame(rows)
    by_src.to_csv(OUT_SRC, index=False)
    print("\n=== THREE MEASURES BY SOURCE (symmetric subset, %) ===")
    for m, lab in MEASURES.items():
        print(f"  {lab}:")
        for _, r in by_src.iterrows():
            print(f"    {r['source']:<10} {r[f'{m}_mean']:>7.2f}  "
                  f"[{r[f'{m}_lo']:.2f}, {r[f'{m}_hi']:.2f}]")

    # --- retrain LDA on the symmetric subset ---
    tok = [t.split() for t in items['lem']]
    dic = corpora.Dictionary(tok)
    dic.filter_extremes(no_below=5, no_above=0.5)
    bow = [dic.doc2bow(t) for t in tok]

    for K in sorted({k_opt, 10}):
        note = "coherence optimum" if K == k_opt else "for continuity with the previous analysis"
        print(f"\n\n########## RETRAINED MODEL, k={K}  ({note}) ##########")
        model = LdaMulticore(corpus=bow, id2word=dic, num_topics=K,
                             random_state=RS, passes=LDA_PASSES, workers=LDA_WORKERS)
        items[f'topic_{K}'] = [max(model.get_document_topics(b), key=lambda x: x[1])[0]
                               if model.get_document_topics(b) else -1 for b in bow]

        lab_path = os.path.join(DATA_RESULTS, f"symfilter_topic_terms_k{K}.txt")
        terms = {}
        with open(lab_path, 'w', encoding='utf-8') as f:
            for idx, t in model.print_topics(-1, num_words=10):
                words = [w.split('*')[1].strip().strip('"') for w in t.split(' + ')]
                terms[idx] = words
                f.write(f"Topic {idx}: {', '.join(words)}\n")
        print(f"  top terms written to {os.path.basename(lab_path)}")

        crows = []
        for t in sorted(items[f'topic_{K}'].unique()):
            for p in PLATFORMS:
                d = items[(items[f'topic_{K}'] == t) & (items['fonte'] == p)]
                if len(d) < MIN_CELL:
                    continue
                r = {'topic': int(t), 'top_terms': ', '.join(terms[int(t)][:5]),
                     'platform': NAMES[p], 'n_items': len(d)}
                for m in MEASURES:
                    lo, hi = boot_ci(d[m].values, rng)
                    r[f'{m}_mean'] = round(d[m].mean() * 100, 2)
                    r[f'{m}_lo'] = round(lo * 100, 2)
                    r[f'{m}_hi'] = round(hi * 100, 2)
                crows.append(r)
        cells = pd.DataFrame(crows)
        out_cells = os.path.join(DATA_RESULTS, f"symfilter_continuous_by_topic_platform_k{K}.csv")
        cells.to_csv(out_cells, index=False)
        n_full = items[f'topic_{K}'].nunique() * len(PLATFORMS)
        print(f"  cells with n>={MIN_CELL}: {len(cells)} of {n_full} possible "
              f"-> {os.path.basename(out_cells)}")

        diplo_topics = [t for t, w in terms.items() if len(DIPLO & set(w)) >= 2]
        print(f"  topics with diplomacy vocabulary: {diplo_topics if diplo_topics else 'none'}")
        for t in diplo_topics:
            sub = cells[cells['topic'] == t]
            if sub.empty:
                print(f"    T{t}: no cell above threshold")
                continue
            print(f"    T{t} ({', '.join(terms[t][:6])})")
            for _, r in sub.iterrows():
                print(f"      {r['platform']:<10} charge={r['charge_mean']:>6.2f} "
                      f"[{r['charge_lo']:.2f}, {r['charge_hi']:.2f}]  "
                      f"p_neg={r['neg_prob_mean']:>6.2f}  n={r['n_items']}")
            tel = sub[sub['platform'] == 'Telegram']
            oth = sub[sub['platform'] != 'Telegram']
            if not tel.empty and not oth.empty:
                sep = all(tel.iloc[0]['charge_hi'] < o['charge_lo'] for _, o in oth.iterrows())
                print(f"      -> Telegram below all others with separated CIs: "
                      f"{'YES' if sep else 'NO'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
