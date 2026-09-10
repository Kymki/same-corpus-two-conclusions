"""
Generates a BLIND annotation sample for manual sentiment validation,
stratified by SOURCE x PREDICTED CLASS (RP review): N_PER_CELL per cell so
per-class metrics are estimable in every source.

The unit is the SENTENCE — the unit the classifier operates on and therefore
the unit we validate (the sample is drawn from the sentence-level predictions;
the paper must call these "sentences", not "documents").

ENGLISH ONLY: the analysed corpus excludes Italian, so the sample and the
post-stratification weights are computed on the English subset only. This keeps
the corpus-weighted accuracy describing the corpus we actually analyse
(previously Telegram's weight was inflated ~18% by Italian sentences).

BBC is excluded (243 sentences: the positive cell cannot reach N_PER_CELL).

The predicted class is used for sampling only and is NOT written to the blind
file; it is saved separately in the cell-design file for later reweighting.

Workflow:
  1. Run this script -> validation_annotation_TODO.csv + validation_sampling_cells.csv
  2. Fill 'manual_label' (Negative/Neutral/Positive); 'manual_label_2' optional
     second annotator.
  3. Run 02_sentiment_validation.py (accuracy reweighted, macro-F1, per-class
     report, and calibration: reliability + ECE overall and per source).
"""
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_RESULTS

INPUT_CSV = os.path.join(DATA_RESULTS, "sentiment_results_multilingua.csv")
OUTPUT_CSV = os.path.join(DATA_RESULTS, "validation_annotation_TODO.csv")
OUTPUT_CELLS = os.path.join(DATA_RESULTS, "validation_sampling_cells.csv")

SOURCES = ["Guardian", "Reddit_Commento", "Telegram"]
CLASSES = ["Negative", "Neutral", "Positive"]
N_PER_CELL = 40
RANDOM_SEED = 42


def generate():
    print(f"Loading data from {INPUT_CSV}")
    try:
        df = pd.read_csv(INPUT_CSV, low_memory=False)
    except FileNotFoundError:
        print("ERROR: run sentiment analysis (01_sent.py) first.")
        sys.exit(1)

    df = df.dropna(subset=['testo_pulito_base'])
    df = df[df['fonte'].isin(SOURCES)]
    # English only — the analysed corpus (Italian is a declared limitation).
    df = df[df['lingua_rilevata'] == 'en']

    parts, cell_rows = [], []
    total_in_scope = len(df)
    for fonte in SOURCES:
        for cls in CLASSES:
            cell = df[(df['fonte'] == fonte) & (df['sentiment_label'] == cls)]
            n_avail = len(cell)
            n_take = min(N_PER_CELL, n_avail)
            if n_take < N_PER_CELL:
                print(f"WARNING: cell {fonte} x {cls} has only {n_avail} sentences "
                      f"(< {N_PER_CELL}); taking all of them.")
            parts.append(cell.sample(n=n_take, random_state=RANDOM_SEED))
            cell_rows.append({
                'fonte': fonte, 'predicted_class': cls,
                'n_sampled': n_take, 'n_corpus_en': n_avail,
                'corpus_share_en': round(n_avail / total_in_scope, 6),
            })

    df_sample = pd.concat(parts)

    blind_cols = ['id_originale', 'fonte', 'lingua_rilevata', 'testo_pulito_base']
    df_blind = df_sample[blind_cols].sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    df_blind['manual_label'] = ""
    df_blind['manual_label_2'] = ""
    df_blind.to_csv(OUTPUT_CSV, index=False)

    pd.DataFrame(cell_rows).to_csv(OUTPUT_CELLS, index=False)

    print(f"\nBlind annotation sample ({len(df_blind)} English sentences) -> {OUTPUT_CSV}")
    print(f"Cell design + English corpus shares (for reweighting) -> {OUTPUT_CELLS}")
    print("\nCell design:")
    print(pd.DataFrame(cell_rows).to_string(index=False))
    print(f"\nLanguage check (should be all 'en'): {df_blind['lingua_rilevata'].unique().tolist()}")
    print("\nNext: fill 'manual_label' (Negative/Neutral/Positive), "
          "then run 02_sentiment_validation.py.")


if __name__ == "__main__":
    generate()
