import pandas as pd
from transformers import AutoTokenizer
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_PROCESSED, DATA_RESULTS

# --- CONFIGURAZIONE ---
INPUT_FILE = os.path.join(DATA_PROCESSED, "dati_testuali_preproc_consolidati.csv")
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
MAX_LENGTH = 512

def valuta_troncamento():
    print(f"Caricamento Tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print(f"Lettura Dati: {INPUT_FILE} ...")
    try:
        df = pd.read_csv(INPUT_FILE, low_memory=False)
        df.dropna(subset=['testo_pulito_base'], inplace=True)
        # Filtriamo le lingue che verranno processate nel sentiment
        df = df[df['lingua_rilevata'].isin(['en', 'it'])]
    except Exception as e:
        print(f"Errore lettura file: {e}")
        return

    print("Analisi della lunghezza tokens...")
    total_docs = len(df)
    
    # Esegue il conto token usando il tokenizer. Senza applicare padding o truncation per avere la len reale.
    # Usando apply può essere lento su enormi dataset, usiamo batch via tokenizer
    
    # Per evitare MemoryError sul batch di encoding, eseguiamo in list comprehension
    # Attenzione: encode è veloce
    testi = df['testo_pulito_base'].tolist()
    
    print(f"Processamento {total_docs} documenti contro la Window Size: {MAX_LENGTH}")
    
    # Conta tokens (add_special_tokens=True simula lo state effettivo)
    lunghezze_tokens = [len(tokenizer.encode(str(testo), add_special_tokens=True)) for testo in testi]
    df['num_tokens_xlm_roberta'] = lunghezze_tokens
    
    df['is_truncated'] = df['num_tokens_xlm_roberta'] > MAX_LENGTH
    
    # Statistiche raggruppate per Fonte (conteggi in SUBWORD-token del tokenizer XLM-RoBERTa,
    # da non confondere con il word count whitespace di 02_corpus_analysis.py)
    riepilogo = df.groupby('fonte').agg(
        Totale_Documenti=('id_originale', 'count'),
        Doc_Troncati=('is_truncated', 'sum'),
        Max_Subword_Tokens=('num_tokens_xlm_roberta', 'max'),
        Media_Subword_Tokens=('num_tokens_xlm_roberta', 'mean')
    ).reset_index()
    
    riepilogo['Rate_Of_Truncation_Perc'] = (riepilogo['Doc_Troncati'] / riepilogo['Totale_Documenti']) * 100
    
    print("\n--- RISULTATO LOSS OF CONTEXT ---")
    print(riepilogo.round(2))
    
    # LaTeX Export
    latex_out = os.path.join(DATA_RESULTS, "loss_of_context_assessment.tex")
    with open(latex_out, 'w', encoding='utf-8') as f:
        try:
            f.write(riepilogo.style.format(precision=2).hide(axis="index").to_latex(
                caption="Assessment of the potential loss of context caused by the 512 subword-token limit of XLM-RoBERTa. Counts are transformer subword tokens, not words.",
                label="tab:truncation_loss"
            ))
        except AttributeError:
            f.write(riepilogo.round(2).to_latex(
                index=False,
                caption="Assessment of the potential loss of context caused by the 512 subword-token limit of XLM-RoBERTa. Counts are transformer subword tokens, not words.",
                label="tab:truncation_loss",
                float_format="%.2f"
            ))
    
    print(f"\nTabella di Assessment Troncamento LaTeX generata in: '{latex_out}'")

if __name__ == "__main__":
    valuta_troncamento()
