import requests
import pandas as pd
import time
import os
import sys

# Add root directory to sys.path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import DATA_RAW

def scrape_guardian():
    print("Scraping The Guardian for 'Ukraine' articles (2022-present)...")
    url = "https://content.guardianapis.com/search"
    params = {
        "q": "ukraine OR russia",
        "from-date": "2025-05-25",
        "show-fields": "bodyText,headline",
        "page-size": 50,
        "api-key": "test",
        "order-by": "newest"
    }

    articles = []
    max_pages = 60  # 60 * 50 = 3000 articles
    
    for page in range(1, max_pages + 1):
        params['page'] = page
        try:
            response = requests.get(url, params=params)
            if response.status_code != 200:
                print(f"Error {response.status_code} on page {page}")
                break
            
            data = response.json().get('response', {})
            results = data.get('results', [])
            
            if not results:
                break
                
            for res in results:
                fields = res.get('fields', {})
                body_text = fields.get('bodyText', '').strip()
                if len(body_text) < 100:
                    continue  # Skip short articles or empty bodies
                    
                articles.append({
                    "id_articolo": res.get("id"),
                    "titolo": fields.get("headline", res.get("webTitle", "")),
                    "data_pubblicazione_iso": res.get("webPublicationDate"),
                    "url_articolo": res.get("webUrl"),
                    "text_content": body_text
                })
                
            print(f"  Fetched page {page}/{max_pages} - Total articles: {len(articles)}")
            time.sleep(0.5)  # Be nice to the API
        except Exception as e:
            print(f"Exception on page {page}: {e}")
            break

    df = pd.DataFrame(articles)
    out_path = os.path.join(DATA_RAW, "Guardian_contenuti_articoli_estratti.csv")
    df.to_csv(out_path, index=False)
    print(f"Successfully saved {len(df)} Guardian articles to {out_path}")

if __name__ == "__main__":
    scrape_guardian()
