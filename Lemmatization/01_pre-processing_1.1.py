import pandas as pd
import spacy
import re
import os
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import sys

# Add root directory to sys.path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_RAW, DATA_PROCESSED

# --- CONFIGURATION ---
MODEL_EN = "en_core_web_sm"
MODEL_IT = "it_core_news_sm"

# Global spaCy models (will be initialized per process)
nlp_en = None
nlp_it = None

def init_spacy():
    global nlp_en, nlp_it
    try:
        # Load English model, disable heavy components, add fast sentencizer
        nlp_en = spacy.load(MODEL_EN, disable=['ner', 'parser'])
        nlp_en.add_pipe('sentencizer')
        nlp_en.max_length = 2000000
        
        # Load Italian model, disable heavy components, add fast sentencizer
        nlp_it = spacy.load(MODEL_IT, disable=['ner', 'parser'])
        nlp_it.add_pipe('sentencizer')
        nlp_it.max_length = 2000000
    except Exception as e:
        pass

def is_spam(text):
    if not isinstance(text, str): return True
    text_lower = text.lower()
    
    # Check for common bot, spam, or promotional keywords
    spam_keywords = [
        "t.me/", "subscribe", "join my channel", "follow us",
        "i am a bot", "action was performed automatically", 
        "contact the moderators", "message the moderators"
    ]
    if any(k in text_lower for k in spam_keywords):
        return True
        
    urls = re.findall(r'http\S+|www\S+|https\S+', text_lower)
    if len(urls) >= 2:
        return True
    return False

def clean_text_basic(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#\w+', '', text)
    text = re.sub(r'[^a-zA-ZàèìòùÀÈÌÒÙáéíóúÁÉÍÓÚ\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def process_document(row):
    """
    Process a single row: split into sentences, clean, lemmatize, filter by semantic density.
    Returns a list of valid sentence dictionaries.
    """
    try:
        global nlp_en, nlp_it
        text = str(row['text'])
        lang = row['lingua_rilevata']
        
        if lang not in ['en', 'it']: return []
        if is_spam(text): return []
        
        # Safety truncation for massively long malformed text blocks
        if len(text) > 1500000:
            text = text[:1500000]
            
        nlp_model = nlp_it if lang == 'it' else nlp_en
        if not nlp_model: return []
        
        doc = nlp_model(text)
        # Split into sentences
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
        
        valid_rows = []
        # If it's a social media post that is only 1 sentence, we don't append '_s0'
        is_multi = len(sentences) > 1
        
        for i, sent in enumerate(sentences):
            clean = clean_text_basic(sent)
            
            # We need a doc object for the cleaned sentence to extract lemmas
            sent_doc = nlp_model(clean)
            lemmas = [token.lemma_ for token in sent_doc if not token.is_stop and not token.is_punct and len(token.lemma_) > 2]
            
            # Semantic Density check: At least 4 valid words
            if len(lemmas) >= 4:
                valid_rows.append({
                    'id_originale': f"{row['id_originale']}_s{i}" if is_multi else row['id_originale'],
                    'fonte': row['fonte'],
                    'data_originale_str': row['data_originale_str'],
                    'lingua_rilevata': lang,
                    'testo_pulito_base': clean,
                    'testo_lemmatizzato': " ".join(lemmas)
                })
                
        return valid_rows
    except Exception as e:
        # If anything fails (like memory or length errors), silently drop this specific document to prevent deadlock
        return []

def process_chunk(chunk_df):
    """Multiprocessing helper for DataFrame chunk."""
    if nlp_en is None or nlp_it is None:
        init_spacy()
    
    results = []
    for _, row in chunk_df.iterrows():
        results.extend(process_document(row))
    return results

if __name__ == "__main__":
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    
    raw_files = {
        "BBC_News": "BBC_News_contenuti_articoli_estratti.csv",
        "Kyiv_Independent": "Kyiv_Independent_contenuti_articoli_estratti.csv",
        "Guardian": "Guardian_contenuti_articoli_estratti.csv",
        "Reddit_Commento": "reddit_comments_raccolti.csv",
        "Telegram": "telegram_messaggi_raccolti.csv"
    }

    all_dfs = []

    for source_name, filename in raw_files.items():
        path = os.path.join(DATA_RAW, filename)
        if not os.path.exists(path):
            print(f"Skipping: {filename} (File not found)")
            continue
        
        print(f"--- Loading: {filename} ---")
        df = pd.read_csv(path, low_memory=False)
        
        # Normalize text column
        text_cols = ['testo_articolo', 'text_content', 'testo_commento', 'text', 'testo_pulito_base']
        for col in text_cols:
            if col in df.columns:
                df['text'] = df[col]
                break
                
        if 'text' not in df.columns or df['text'].dropna().empty:
            print(f"WARNING: No valid text column found for {filename}")
            continue
            
        # Normalize ID column
        id_cols = ['id_originale', 'commento_id', 'message_id', 'url_articolo', 'url']
        for col in id_cols:
            if col in df.columns:
                df['id_originale'] = df[col]
                break
                
        if 'id_originale' not in df.columns:
            df['id_originale'] = df.index.astype(str)
            
        # Normalize Date
        date_cols = ['data_pubblicazione_iso', 'timestamp_utc_commento', 'timestamp_utc', 'data_originale_str']
        for col in date_cols:
            if col in df.columns:
                df['data_originale_str'] = df[col]
                break
        
        # Temporal Filter (Strict: >= 2025-05-25)
        df['parsed_date'] = pd.to_datetime(df['data_originale_str'], format='mixed', errors='coerce', utc=True)
        df = df[df['parsed_date'] >= '2025-05-25'].copy()
        
        if len(df) == 0:
            print(f"  -> 0 valid rows found after 1-year time filter.")
            continue
            
        df['fonte'] = source_name

        # Topical relevance filter for The Guardian.
        # Unlike the other sources (BBC: war archive; Telegram/Reddit: war-specific
        # channels/subreddits; Kyiv Independent scraper: keyword check), the Guardian
        # collection used a broad full-text API query ("ukraine OR russia") that also
        # matches passing mentions. Keep an article iff:
        #   - a war keyword appears in the TITLE (covers dedicated war live-blogs), or
        #   - it is a regular article (not a multi-topic /live/ blog) with >= 3
        #     war-keyword occurrences in the body.
        if source_name == 'Guardian':
            WAR_KW = ['ukrain', 'russia', 'putin', 'zelensk', 'kyiv', 'kremlin', 'moscow',
                      'donbas', 'crimea', 'kharkiv', 'kherson', 'mariupol', 'bakhmut', 'nato']
            def kw_count(t):
                t = str(t).lower()
                return sum(t.count(k) for k in WAR_KW)
            n_before = len(df)
            title_match = df['titolo'].apply(lambda t: kw_count(t) > 0)
            body_count = df['text'].apply(kw_count)
            is_live = df['id_originale'].astype(str).str.contains('/live/')
            df = df[title_match | (~is_live & (body_count >= 3))].copy()
            print(f"  Guardian topical relevance filter: kept {len(df)}/{n_before} articles")

        # Per-document language detection (langdetect), applied uniformly to every source.
        # Documents detected as neither EN nor IT are dropped downstream in process_document.
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0  # deterministic detection for reproducibility

        def detect_lang(text):
            try:
                return detect(str(text))
            except Exception:
                return 'unknown'

        print(f"  Running langdetect on {len(df)} documents...")
        df['lingua_rilevata'] = df['text'].apply(detect_lang)
        lang_counts = df['lingua_rilevata'].value_counts()
        print(f"  Language distribution for {source_name}: {lang_counts.to_dict()}")

        df['lingua_rilevata'] = df['lingua_rilevata'].astype(str).str.lower().str.strip()
        
        all_dfs.append(df[['id_originale', 'fonte', 'data_originale_str', 'lingua_rilevata', 'text']])

    if not all_dfs:
        print("CRITICAL ERROR: No data survived the time filter! Check your dates.")
        sys.exit(1)

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all['text'] = df_all['text'].astype(str)
    print(f"\nConsolidated {len(df_all)} raw documents from the last year.")

    # --- Multiprocessing ---
    num_processes = 7
    print(f"\n--- Starting Multiprocess Lemmatization & Splitting (n_process={num_processes}) ---")
    
    # Split dataframe into chunks
    chunk_size = len(df_all) // num_processes + 1
    chunks = [df_all.iloc[i:i + chunk_size] for i in range(0, len(df_all), chunk_size)]

    with ProcessPoolExecutor(max_workers=num_processes, initializer=init_spacy) as executor:
        results = list(tqdm(executor.map(process_chunk, chunks), total=len(chunks), desc="Processing"))

    # Flatten
    final_rows = [item for sublist in results for item in sublist]
    df_final = pd.DataFrame(final_rows)
    
    print(f"\nTotal valid sentences extracted: {len(df_final)}")
    
    # STRICT DEDUPLICATION
    df_final = df_final.drop_duplicates(subset=['testo_pulito_base'], keep='first')
    
    print(f"Total sentences after deduplication (spam/echo removal): {len(df_final)}")
    print(df_final.groupby('fonte').size())

    output_path = os.path.join(DATA_PROCESSED, "dati_testuali_preproc_consolidati.csv")
    df_final.to_csv(output_path, index=False)
    print(f"\n--- Robust Preprocessing Completed ---")
    print(f"Output saved to: {output_path}")