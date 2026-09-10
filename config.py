import os

# Determine the root directory of the project (where config.py is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directory definitions
DATA_DIR = os.path.join(BASE_DIR, 'data')
DATA_RAW = os.path.join(DATA_DIR, 'raw')
DATA_PROCESSED = os.path.join(DATA_DIR, 'processed')
DATA_RESULTS = os.path.join(DATA_DIR, 'results')

# Ensure directories exist
os.makedirs(DATA_RAW, exist_ok=True)
os.makedirs(DATA_PROCESSED, exist_ok=True)
os.makedirs(DATA_RESULTS, exist_ok=True)

# Global variables (e.g., Elasticsearch settings)
# ELASTIC_URL = "http://localhost:9200"
# ELASTIC_INDEX = "war_narratives_corpus"
