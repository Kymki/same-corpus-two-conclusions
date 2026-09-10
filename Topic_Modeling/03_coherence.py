import pandas as pd
from gensim import corpora
from gensim.models import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import os
import json
import sys

# Add root directory to sys.path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_PROCESSED, DATA_RESULTS

# --- CONFIGURATION ---
INPUT_PROCESSED_CSV = os.path.join(DATA_PROCESSED, "dati_testuali_preproc_consolidati.csv")
COLONNA_TESTO_PROCESSATO = "testo_lemmatizzato"
COLONNA_LINGUA = "lingua_rilevata"

# Exploration range for the number of topics
START_TOPICS = 5
LIMIT_TOPICS = 25
STEP_TOPICS = 2

# CPU Optimization for Ryzen 7 
LDA_WORKERS = 7

# --- FUNCTIONS ---
def prepara_corpus_per_lda(documenti_testuali):
    """
    Prepares the corpus for LDA analysis.
    Supports both raw strings and pre-tokenized lists.
    """
    if not documenti_testuali:
        return None, None, None

    # If documents are already lists of tokens, use them directly
    if len(documenti_testuali) > 0 and isinstance(documenti_testuali[0], list):
        tokenized_docs = documenti_testuali
    else:
        # Otherwise, split by whitespace
        tokenized_docs = [str(doc).split() for doc in documenti_testuali if isinstance(doc, (str, bytes)) and str(doc).strip()]
    
    # Remove any residual empty lists
    tokenized_docs = [doc for doc in tokenized_docs if len(doc) > 0]
    
    if not tokenized_docs:
        return None, None, None

    dictionary = corpora.Dictionary(tokenized_docs)
    # Optional filter for statistical noise: removes rare or too frequent terms
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    
    corpus_bow = [dictionary.doc2bow(doc) for doc in tokenized_docs]
    return tokenized_docs, dictionary, corpus_bow

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Computes C_v coherence for various numbers of topics.
    """
    coherence_values = []
    model_list = []
    
    for num_topics in range(start, limit, step):
        print(f"  Training LDA model with k={num_topics} topics (workers={LDA_WORKERS})...")
        model = LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=100,
            passes=5,
            workers=LDA_WORKERS
        )
        model_list.append(model)
        
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        c_v_score = coherencemodel.get_coherence()
        print(f"    -> Coherence Score (C_v) for k={num_topics}: {c_v_score:.4f}")
        coherence_values.append(c_v_score)

    return model_list, coherence_values

def visualizza_e_salva_coerenza(coherence_values, start, limit, step, lingua, file_out):
    """
    Plots and saves the coherence scores.
    """
    x = range(start, limit, step)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, coherence_values, marker='o', linestyle='-', color='b')
    plt.xlabel("Number of Topics (k)", fontsize=12)
    plt.ylabel("Coherence Score ($C_v$)", fontsize=12)
    plt.title(f"Optimal Coherence Score (Language: {lingua.upper()})", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    file_out_path = os.path.join(DATA_RESULTS, file_out)
    plt.savefig(file_out_path, dpi=300)
    print(f"Coherence plot saved at 300 DPI in '{file_out_path}'")
    plt.close()

def salva_valori_coerenza(coherence_values, start, step, lingua):
    """
    Saves the raw (k, C_v) pairs to CSV so fig4 and the paper tables
    are generated from the same run as the final model.
    """
    ks = [start + i * step for i in range(len(coherence_values))]
    df_out = pd.DataFrame({'k': ks, 'coherence_cv': coherence_values})
    out_path = os.path.join(DATA_RESULTS, f"coherence_values_{lingua.lower()}.csv")
    df_out.to_csv(out_path, index=False)
    print(f"Coherence values saved to '{out_path}'")

def salva_ottimo_k(coherence_values, start, step, lingua):
    """
    Identifies the best k and saves it to a JSON file.
    """
    best_index = coherence_values.index(max(coherence_values))
    best_k = start + best_index * step
    best_cv = coherence_values[best_index]
    
    output_file = os.path.join(DATA_RESULTS, f"optimal_k_{lingua.lower()}.json")
    with open(output_file, 'w') as f:
        json.dump({"optimal_k": best_k, "max_cv": best_cv}, f)
    print(f"Mathematical optimum for {lingua.upper()} found: k={best_k} (Cv={best_cv:.4f}). Saved in '{output_file}'")
    return best_k

# --- MAIN FLOW ---
if __name__ == "__main__":
    print(f"Loading data for coherence analysis from: {INPUT_PROCESSED_CSV}")
    try:
        df_processed = pd.read_csv(INPUT_PROCESSED_CSV, low_memory=False)
    except FileNotFoundError:
        print(f"ERROR: File '{INPUT_PROCESSED_CSV}' not found.")
        sys.exit(1)

    # Pre-clean the dataframe
    df_processed.dropna(subset=[COLONNA_TESTO_PROCESSATO], inplace=True)
    df_processed[COLONNA_TESTO_PROCESSATO] = df_processed[COLONNA_TESTO_PROCESSATO].astype(str)
    df_processed = df_processed[df_processed[COLONNA_TESTO_PROCESSATO].str.strip() != '']

    # --- ENGLISH ---
    print("\n\n--- C_v Extraction for ENGLISH ---")
    df_en = df_processed[df_processed[COLONNA_LINGUA] == 'en'].copy()
    if not df_en.empty:
        # Preventive filtering for alignment
        df_en['temp_tokens'] = df_en[COLONNA_TESTO_PROCESSATO].fillna('').astype(str).str.split()
        df_en = df_en[df_en['temp_tokens'].map(len) > 0]
        
        texts_en, dict_en, corpus_en = prepara_corpus_per_lda(df_en['temp_tokens'].tolist())
        if texts_en:
            _, coherence_vals_en = compute_coherence_values(dict_en, corpus_en, texts_en, start=START_TOPICS, limit=LIMIT_TOPICS, step=STEP_TOPICS)
            visualizza_e_salva_coerenza(coherence_vals_en, START_TOPICS, LIMIT_TOPICS, STEP_TOPICS, "EN", "coherence_plot_en.png")
            salva_valori_coerenza(coherence_vals_en, START_TOPICS, STEP_TOPICS, "EN")
            salva_ottimo_k(coherence_vals_en, START_TOPICS, STEP_TOPICS, "EN")

    # --- ITALIAN ---
    print("\n\n--- C_v Extraction for ITALIAN ---")
    df_it = df_processed[df_processed[COLONNA_LINGUA] == 'it'].copy()
    if not df_it.empty:
        # Preventive filtering for alignment
        df_it['temp_tokens'] = df_it[COLONNA_TESTO_PROCESSATO].fillna('').astype(str).str.split()
        df_it = df_it[df_it['temp_tokens'].map(len) > 0]
        
        texts_it, dict_it, corpus_it = prepara_corpus_per_lda(df_it['temp_tokens'].tolist())
        if texts_it:
            _, coherence_vals_it = compute_coherence_values(dict_it, corpus_it, texts_it, start=START_TOPICS, limit=LIMIT_TOPICS, step=STEP_TOPICS)
            visualizza_e_salva_coerenza(coherence_vals_it, START_TOPICS, LIMIT_TOPICS, STEP_TOPICS, "IT", "coherence_plot_it.png")
            salva_valori_coerenza(coherence_vals_it, START_TOPICS, STEP_TOPICS, "IT")
            salva_ottimo_k(coherence_vals_it, START_TOPICS, STEP_TOPICS, "IT")

    print("\nCoherence Score (C_v) analysis finished. Analyze the plots to find the optimal k for LDA labeling.")
