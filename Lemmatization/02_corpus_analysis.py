import pandas as pd
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_PROCESSED, DATA_RESULTS

# --- CONFIGURAZIONE ---
INPUT_FILE = os.path.join(DATA_PROCESSED, "dati_testuali_preproc_consolidati.csv")
OUTPUT_LATEX = os.path.join(DATA_RESULTS, "corpus_analysis.tex")

def analizza_corpus():
    print(f"Caricamento dati per l'analisi del corpus da: {INPUT_FILE}")
    try:
        df = pd.read_csv(INPUT_FILE, low_memory=False)
    except FileNotFoundError:
        print(f"ERRORE: File '{INPUT_FILE}' non trovato.")
        return

    print("Computing the volumetric and distributional profile...")
    
    # Rimuove le righe senza testo pulito
    df.dropna(subset=['testo_pulito_base'], inplace=True)
    
    # Word count (whitespace-separated words) — NOT transformer subword tokens.
    # The XLM-RoBERTa subword-token statistics are computed separately in
    # Sentiment_analysis/03_truncation_assessment.py: keep the two table captions distinct.
    df['num_parole'] = df['testo_pulito_base'].apply(lambda x: len(str(x).split()))

    # Statistiche raggruppate per Fonte e Lingua
    stats = df.groupby(['fonte', 'lingua_rilevata']).agg(
        Totale_Documenti=('id_originale', 'count'),
        Media_Parole=('num_parole', 'mean'),
        Std_Parole=('num_parole', 'std'),
        Max_Parole=('num_parole', 'max'),
        Min_Parole=('num_parole', 'min')
    ).reset_index()

    print("\n--- STATISTICHE DEL CORPUS ---")
    print(stats.round(2).to_string(index=False))

    # Aggregazione totale
    totale = pd.DataFrame({
        'fonte': ['TOTALE'],
        'lingua_rilevata': ['-'],
        'Totale_Documenti': [df['id_originale'].count()],
        'Media_Parole': [df['num_parole'].mean()],
        'Std_Parole': [df['num_parole'].std()],
        'Max_Parole': [df['num_parole'].max()],
        'Min_Parole': [df['num_parole'].min()]
    })
    
    stats_finale = pd.concat([stats, totale], ignore_index=True)

    # Esportazione in LaTeX
    latex_out_path = OUTPUT_LATEX
    try:
        with open(latex_out_path, 'w', encoding='utf-8') as f:
            try:
                f.write(stats_finale.style.format(precision=2).hide(axis="index").to_latex(
                    caption="Volumetric profile of the corpus after preprocessing. Lengths are whitespace-delimited word counts, not transformer subword tokens.",
                    label="tab:corpus_analysis"
                ))
            except AttributeError:
                f.write(stats_finale.round(2).to_latex(
                    index=False,
                    caption="Volumetric profile of the corpus after preprocessing. Lengths are whitespace-delimited word counts, not transformer subword tokens.",
                    label="tab:corpus_analysis",
                    float_format="%.2f"
                ))
        print(f"\nTabella LaTeX esportata con successo in: '{latex_out_path}'")
    except Exception as e:
        print(f"Errore durante l'esportazione in LaTeX: {e}")

if __name__ == "__main__":
    analizza_corpus()
