"""
Figure 8 (step 1/2) – Extract REAL candidate narrative fragments from the corpus.

For each platform, selects candidate documents from the sentiment results
(so every candidate has a traceable id_originale and a real model-assigned
sentiment label), then traces the cleaned sentence back to the original raw
CSV to recover the original wording/casing.

Output: data/results/fig8_candidates.csv
Then: pick one fragment per platform and record its id_originale in
data/results/fig8_selected_ids.json — fig8_narrative_fragments.py builds the
figure ONLY from those IDs, failing loudly if any ID is not in the corpus.

Social-media fragments are quoted without any author/username metadata
(anonymized); the id_originale is kept for internal traceability only.
"""
import pandas as pd
import re
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_RAW, DATA_RESULTS

INPUT_SENTIMENT = os.path.join(DATA_RESULTS, "sentiment_results_multilingua.csv")
OUTPUT_CANDIDATES = os.path.join(DATA_RESULTS, "fig8_candidates.csv")

N_PER_PLATFORM = 8

# fonte -> (raw file, id column in raw file, text column in raw file)
RAW_MAP = {
    "Reddit_Commento": ("reddit_comments_raccolti.csv", "id_originale", "text"),
    "Telegram": ("telegram_messaggi_raccolti.csv", "message_id", "text"),
    "BBC_News": ("BBC_News_contenuti_articoli_estratti.csv", "url", "testo_articolo"),
    # NB: preprocessing uses url_articolo (not id_articolo) as the Guardian document id
    "Guardian": ("Guardian_contenuti_articoli_estratti.csv", "url_articolo", "text_content"),
}

_raw_cache = {}

def load_raw(fonte):
    if fonte not in _raw_cache:
        fname, id_col, text_col = RAW_MAP[fonte]
        path = os.path.join(DATA_RAW, fname)
        df = pd.read_csv(path, usecols=[id_col, text_col], dtype=str, low_memory=False).dropna()
        # BBC/Guardian ids in the processed corpus come from the url/id column
        df = df.set_index(id_col)
        _raw_cache[fonte] = (df, text_col)
    return _raw_cache[fonte]


def simplify(s):
    return re.sub(r'[^a-z]', '', str(s).lower())


def trace_original_sentence(fonte, id_originale, clean_text):
    """Find the original sentence in the raw document that produced clean_text."""
    base_id = re.sub(r'_s\d+$', '', str(id_originale))
    try:
        raw_df, text_col = load_raw(fonte)
        raw_text = raw_df.loc[base_id, text_col]
        if isinstance(raw_text, pd.Series):
            raw_text = raw_text.iloc[0]
    except KeyError:
        return None
    target = simplify(clean_text)
    if not target:
        return None
    for sent in re.split(r'(?<=[.!?])\s+|\n', str(raw_text)):
        s = simplify(sent)
        if target in s or (len(s) > 20 and s in target):
            return sent.strip()
    return None


def pick(df, fonte, criterion, n=N_PER_PLATFORM):
    """criterion: 'negative' (most negative), 'neutral' (closest to neutral)."""
    sub = df[(df['fonte'] == fonte) & (df['testo_pulito_base'].str.len().between(80, 260))].copy()
    if sub.empty:
        return sub.assign(original_text=None)
    if criterion == 'negative':
        sub = sub.nlargest(n * 5, 'neg_prob')
    else:
        sub['dist'] = (sub['neu_prob'] - sub[['neg_prob', 'pos_prob']].max(axis=1))
        sub = sub.nlargest(n * 5, 'dist')

    rows = []
    for _, row in sub.iterrows():
        orig = trace_original_sentence(fonte, row['id_originale'], row['testo_pulito_base'])
        if orig and len(orig) > 60:
            rows.append({
                'fonte': fonte,
                'id_originale': row['id_originale'],
                'original_text': orig,
                'testo_pulito_base': row['testo_pulito_base'],
                'sentiment_label': row['sentiment_label'],
                'sentiment_score': row['sentiment_score'],
            })
        if len(rows) >= n:
            break
    return pd.DataFrame(rows)


def main():
    print(f"Loading sentiment results: {INPUT_SENTIMENT}")
    df = pd.read_csv(INPUT_SENTIMENT, low_memory=False)
    df = df[df['lingua_rilevata'] == 'en'].dropna(subset=['testo_pulito_base'])

    parts = [
        pick(df, 'Reddit_Commento', 'negative'),   # polarized community discourse
        pick(df, 'Telegram', 'neutral'),           # tactical OSINT register
        pick(df, 'BBC_News', 'neutral'),           # institutional register
        pick(df, 'Guardian', 'neutral'),           # analytical register
    ]
    out = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    out.to_csv(OUTPUT_CANDIDATES, index=False)
    print(f"\n{len(out)} candidates written to {OUTPUT_CANDIDATES}")
    print(out.groupby('fonte').size().to_string())
    print("\nNext: choose one id per platform into fig8_selected_ids.json "
          "and run fig8_narrative_fragments.py")


if __name__ == "__main__":
    main()
