import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from tqdm import tqdm
import os
import sys

# Add root directory to sys.path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_PROCESSED, DATA_RESULTS

# --- CONFIGURATION ---
# Multilingual Transformer model (XLM-RoBERTa)
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

INPUT_CSV = os.path.join(DATA_PROCESSED, "dati_testuali_preproc_consolidati.csv")
OUTPUT_CSV = os.path.join(DATA_RESULTS, "sentiment_results_multilingua.csv")

# Processing parameters
BATCH_SIZE = 32  # Optimized for 8GB+ VRAM. Reduce to 16 or 8 if memory errors occur.
MAX_LENGTH = 512 # Standard XLM-RoBERTa limit

# --- MODEL INITIALIZATION ---
print(f"Checking hardware availability...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print(f"Loading model and tokenizer: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
model.eval() # Set to evaluation mode

# Mapping of model output labels to human-readable strings
# 0 -> Negative, 1 -> Neutral, 2 -> Positive
ID_TO_LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}

def analyze_sentiment_batch(texts):
    """
    Analyzes sentiment for a batch of texts using the GPU.
    """
    # Tokenization with padding and truncation
    encoded_input = tokenizer(
        texts, 
        return_tensors='pt', 
        padding=True, 
        truncation=True, 
        max_length=MAX_LENGTH
    ).to(device)

    with torch.no_grad():
        output = model(**encoded_input)
    
    # Apply softmax to get probabilities
    scores = output.logits.detach().cpu().numpy()
    scores = softmax(scores, axis=1)
    
    results = []
    for score in scores:
        # Index of the highest probability
        label_id = score.argmax()
        label_name = ID_TO_LABEL[label_id]
        confidence = score[label_id]
        results.append({
            "sentiment_label": label_name,
            "sentiment_score": float(confidence),
            "neg_prob": float(score[0]),
            "neu_prob": float(score[1]),
            "pos_prob": float(score[2])
        })
    return results

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print(f"Loading data from: {INPUT_CSV}")
    try:
        df = pd.read_csv(INPUT_CSV)
        print(f"Total rows to analyze: {len(df)}")
    except FileNotFoundError:
        print(f"ERROR: Input file not found at {INPUT_CSV}")
        sys.exit(1)

    # Filter out empty or invalid texts
    df = df.dropna(subset=['testo_pulito_base'])
    texts = df['testo_pulito_base'].astype(str).tolist()

    all_results = []
    print(f"Starting sentiment analysis (Batch size: {BATCH_SIZE})...")
    
    # Process in batches to optimize GPU usage
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Sentiment Inference"):
        batch_texts = texts[i : i + BATCH_SIZE]
        batch_results = analyze_sentiment_batch(batch_texts)
        all_results.extend(batch_results)

    # Integrate results back into the dataframe
    df_results = pd.concat([df.reset_index(drop=True), pd.DataFrame(all_results)], axis=1)

    # Save to CSV
    print(f"Saving results to: {OUTPUT_CSV}")
    df_results.to_csv(OUTPUT_CSV, index=False)
    print("Sentiment analysis completed successfully.")