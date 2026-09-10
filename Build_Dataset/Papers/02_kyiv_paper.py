import csv
from newspaper import Article, Config
import datetime
import time
import os
from dateutil import parser

# --- GESTIONE PERCORSI DINAMICA ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
os.makedirs(DATA_RAW, exist_ok=True)

# --- CONFIGURAZIONE ---
NOME_SITO = "Kyiv_Independent" 
INPUT_URL_CSV_FILE = os.path.join(DATA_RAW, f"{NOME_SITO}_archive_article_urls.csv")
OUTPUT_ARTICLES_CSV_FILE = os.path.join(DATA_RAW, f"{NOME_SITO}_contenuti_articoli_estratti.csv")
URL_COLUMN_NAME_IN_CSV = "url" 
START_DATE_LIMIT = datetime.datetime(2025, 5, 25, tzinfo=datetime.timezone.utc)

config = Config()
config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
config.request_timeout = 20 
config.memoize_articles = True 
DELAY_PER_ARTICLE = 2 

def carica_url_da_csv(filename, url_column):
    urls_data = []
    try:
        with open(filename, 'r', newline='', encoding='utf-8') as file_csv:
            reader = csv.DictReader(file_csv)
            for row in reader:
                url = row.get(url_column)
                if url: urls_data.append({'url': url, 'date_from_archive': None})
        return urls_data
    except Exception as e:
        print(f"ERRORE lettura CSV: {e}")
        return None

def estrai_contenuto_articolo_newspaper(url, newspaper_config):
    try:
        print(f"  Download: {url}")
        article = Article(url, config=newspaper_config)
        article.download()
        article.parse()
        
        publish_date_obj = None
        if article.publish_date:
            try:
                if isinstance(article.publish_date, datetime.datetime):
                    publish_date_obj = article.publish_date
                elif isinstance(article.publish_date, str):
                    publish_date_obj = parser.parse(article.publish_date)
                
                if publish_date_obj and publish_date_obj.tzinfo is None:
                    publish_date_obj = publish_date_obj.replace(tzinfo=datetime.timezone.utc)
                elif publish_date_obj:
                    publish_date_obj = publish_date_obj.astimezone(datetime.timezone.utc)
            except Exception:
                pass
        
        return {
            "url": url, "titolo": article.title, "autori": ", ".join(article.authors if article.authors else []),
            "data_pubblicazione_newspaper": publish_date_obj, "testo_articolo": article.text, "immagine_principale": article.top_image,
        }
    except Exception as e:
        return None

def salva_articoli_estratti_csv(lista_dati_articoli, nome_file):
    if not lista_dati_articoli: return
    fieldnames = ["url", "titolo", "autori", "data_pubblicazione_iso", "testo_articolo", "immagine_principale"]
    try:
        with open(nome_file, 'w', newline='', encoding='utf-8') as file_csv:
            writer = csv.DictWriter(file_csv, fieldnames=fieldnames)
            writer.writeheader()
            for art in lista_dati_articoli:
                d_pub = art.get("data_pubblicazione_newspaper")
                writer.writerow({
                    "url": art.get("url"), "titolo": art.get("titolo"), "autori": art.get("autori"),
                    "data_pubblicazione_iso": d_pub.isoformat() if isinstance(d_pub, datetime.datetime) else "DATA NON TROVATA",
                    "testo_articolo": art.get("testo_articolo"), "immagine_principale": art.get("immagine_principale"),
                })
    except IOError: pass

if __name__ == "__main__":
    lista_url_data = carica_url_da_csv(INPUT_URL_CSV_FILE, URL_COLUMN_NAME_IN_CSV)
    contenuti_articoli_finali = []

    if lista_url_data:
        for i, url_info in enumerate(lista_url_data):
            url_articolo = url_info['url']
            dati_estratto = estrai_contenuto_articolo_newspaper(url_articolo, config)
            
            if dati_estratto:
                testo_estratto = str(dati_estratto.get('testo_articolo', '')).lower()
                titolo_estratto = str(dati_estratto.get('titolo', '')).lower()
                
                KEYWORDS = ['ukrain', 'russia', 'putin', 'zelensk']
                if not any(kw in testo_estratto for kw in KEYWORDS) and not any(kw in titolo_estratto for kw in KEYWORDS):
                    time.sleep(DELAY_PER_ARTICLE)
                    continue

                d_finale = dati_estratto.get('data_pubblicazione_newspaper')
                if isinstance(d_finale, datetime.datetime):
                    if d_finale >= START_DATE_LIMIT:
                        contenuti_articoli_finali.append(dati_estratto)
                else:
                    contenuti_articoli_finali.append(dati_estratto)
            time.sleep(DELAY_PER_ARTICLE)

    salva_articoli_estratti_csv(contenuti_articoli_finali, OUTPUT_ARTICLES_CSV_FILE)
