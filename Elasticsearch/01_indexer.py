import pandas as pd
from elasticsearch import Elasticsearch, helpers
import os
import sys

# Add root directory to sys.path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_RESULTS

# --- CONFIGURATION ---
# Elasticsearch local or cloud URL
ES_URL = "http://localhost:9200"
ES_INDEX = "war_narratives_sentiment"

INPUT_CSV = os.path.join(DATA_RESULTS, "sentiment_results_multilingua.csv")

def index_data():
    """
    Reads the sentiment-processed CSV and indexes it into Elasticsearch.
    """
    print(f"Connecting to Elasticsearch at {ES_URL}...")
    es = Elasticsearch(ES_URL)

    if not es.ping():
        print("ERROR: Could not connect to Elasticsearch. Ensure the service is running.")
        return

    print(f"Loading data from: {INPUT_CSV}")
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"ERROR: File not found at {INPUT_CSV}")
        return

    # Clean data: Replace NaN with None for Elasticsearch compatibility
    df = df.where(pd.notnull(df), None)

    print(f"Preparing {len(df)} documents for indexing...")

    def doc_generator(dataframe):
        for idx, row in dataframe.iterrows():
            yield {
                "_index": ES_INDEX,
                "_id": f"{row['fonte']}_{idx}",
                "_source": row.to_dict()
            }

    print(f"Starting bulk indexing into index: {ES_INDEX}...")
    try:
        success, failed = helpers.bulk(es, doc_generator(df))
        print(f"Indexing completed: {success} documents indexed, {failed} failed.")
    except Exception as e:
        print(f"ERROR during bulk indexing: {e}")

if __name__ == "__main__":
    index_data()