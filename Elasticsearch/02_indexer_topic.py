import pandas as pd
from elasticsearch import Elasticsearch, helpers
import os
import sys

# Add root directory to sys.path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_RESULTS

# --- CONFIGURATION ---
ES_URL = "http://localhost:9200"
ES_INDEX_TOPIC = "war_narratives_topics"

# CSV files containing dominant topics for both languages
INPUT_EN = os.path.join(DATA_RESULTS, "document_topics_en.csv")
INPUT_IT = os.path.join(DATA_RESULTS, "document_topics_it.csv")

def index_topics():
    """
    Consolidates English and Italian topic results and indexes them into Elasticsearch.
    """
    es = Elasticsearch(ES_URL)
    if not es.ping():
        print("ERROR: Connection to Elasticsearch failed.")
        return

    all_data = []
    
    # Load and flag English data
    if os.path.exists(INPUT_EN):
        print(f"Loading English topics from: {INPUT_EN}")
        df_en = pd.read_csv(INPUT_EN)
        df_en['language_group'] = 'en'
        all_data.append(df_en)
    
    # Load and flag Italian data
    if os.path.exists(INPUT_IT):
        print(f"Loading Italian topics from: {INPUT_IT}")
        df_it = pd.read_csv(INPUT_IT)
        df_it['language_group'] = 'it'
        all_data.append(df_it)

    if not all_data:
        print("No topic data found to index.")
        return

    df_consolidated = pd.concat(all_data, ignore_index=True)
    df_consolidated = df_consolidated.where(pd.notnull(df_consolidated), None)

    print(f"Indexing {len(df_consolidated)} documents with topics...")

    def doc_generator(dataframe):
        for idx, row in dataframe.iterrows():
            yield {
                "_index": ES_INDEX_TOPIC,
                "_id": f"topic_{row['language_group']}_{idx}",
                "_source": row.to_dict()
            }

    try:
        success, failed = helpers.bulk(es, doc_generator(df_consolidated))
        print(f"Indexing completed: {success} successful, {failed} failed.")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    index_topics()