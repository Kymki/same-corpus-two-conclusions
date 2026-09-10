import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import csv
import time
import datetime
import os
from urllib.parse import urljoin, urlparse

# --- GESTIONE PERCORSI DINAMICA ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
os.makedirs(DATA_RAW, exist_ok=True)

# --- CONFIGURAZIONE ---
NOME_SITO = "BBC_News"
STARTING_ARCHIVE_URL = "https://www.bbc.com/news/war-in-ukraine" 
BASE_URL = "https://www.bbc.com" 

OUTPUT_CSV_URLS = os.path.join(DATA_RAW, f"{NOME_SITO}_archive_article_urls.csv")
START_DATE_LIMIT = datetime.datetime(2025, 5, 25, tzinfo=datetime.timezone.utc)
MAX_PAGES_TO_SCRAPE = 200 

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept-Language': 'en-GB,en;q=0.5' 
}
DELAY_TRA_RICHIESTE = 2

# --- FUNZIONI ---
def get_requests_session():
    session = requests.Session()
    retry = Retry(connect=5, read=5, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

sessione_resiliente = get_requests_session()

def scarica_pagina(url):
    print(f"  Download pagina: {url}")
    try:
        response = sessione_resiliente.get(url, headers=HEADERS, timeout=20)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"    ERRORE CRITICO: {e}")
        return None

def estrai_link_e_date_da_pagina_bbc(html_content, base_url):
    if not html_content: return []
    soup = BeautifulSoup(html_content, 'html.parser')
    articoli_nella_pagina = []
    
    KEYWORDS_VALIDI = ['ukrain', 'russia', 'putin', 'zelensk', 'kyiv', 'moscow', 'donbas']
    tutti_i_link = soup.find_all('a', href=True)
    
    for link_tag in tutti_i_link:
        href = link_tag['href'].lower()
        if '/videos/' in href or '/live/' in href: continue
        if '/news/articles/' in href or '/news/world-europe' in href:
            testo_link = link_tag.get_text().lower()
            if any(kw in href for kw in KEYWORDS_VALIDI) or any(kw in testo_link for kw in KEYWORDS_VALIDI):
                url_articolo = urljoin(base_url, link_tag['href'])
                articoli_nella_pagina.append({'url': url_articolo, 'date_on_archive': None})

    articoli_unici = {v['url']:v for v in articoli_nella_pagina}.values()
    print(f"    Estratti {len(articoli_unici)} link PERTINENTI da questa pagina.")
    return list(articoli_unici)

def trova_url_pagina_successiva_bbc(html_content, current_page_base_url):
    if not html_content: return None
    soup = BeautifulSoup(html_content, 'html.parser')
    link_rel_next = soup.find('link', rel='next')
    if link_rel_next and link_rel_next.get('href'):
        return urljoin(current_page_base_url, link_rel_next['href'])
    next_button_tag = soup.find('a', {'data-testid': 'pagination-next-button', 'href': True})
    if next_button_tag:
        return urljoin(current_page_base_url, next_button_tag['href'])
    return None

def salva_url_csv_incremental(lista_url_dati, nome_file):
    if not lista_url_dati: return
    file_exists = os.path.exists(nome_file)
    try:
        with open(nome_file, 'a', newline='', encoding='utf-8') as file_csv:
            writer = csv.writer(file_csv)
            if not file_exists:
                writer.writerow(['url', 'date_on_archive_utc_iso'])
            for item in lista_url_dati:
                writer.writerow([item['url'], 'N/A_DATE'])
    except IOError as e: print(f"Errore I/O: {e}")

if __name__ == "__main__":
    tutti_gli_articoli_info = []
    url_visitati = set() 
    current_page_url = STARTING_ARCHIVE_URL
    pagine_scansionate = 0
    
    print(f"Inizio scraping URL. Salvataggio in: {OUTPUT_CSV_URLS}")

    while current_page_url and current_page_url not in url_visitati and pagine_scansionate < MAX_PAGES_TO_SCRAPE:
        url_visitati.add(current_page_url)
        pagine_scansionate += 1
        parsed_current_url = urlparse(current_page_url)
        current_base = f"{parsed_current_url.scheme}://{parsed_current_url.netloc}"

        contenuto_pagina_archivio = scarica_pagina(current_page_url)
        if not contenuto_pagina_archivio: break

        articoli_nella_pagina_corrente = estrai_link_e_date_da_pagina_bbc(contenuto_pagina_archivio, current_base)
        
        for art_info in articoli_nella_pagina_corrente:
            if art_info.get('url') not in (item['url'] for item in tutti_gli_articoli_info):
                tutti_gli_articoli_info.append(art_info)

        current_page_url = trova_url_pagina_successiva_bbc(contenuto_pagina_archivio, current_base)
        
        if tutti_gli_articoli_info:
            salva_url_csv_incremental(tutti_gli_articoli_info, OUTPUT_CSV_URLS)
            tutti_gli_articoli_info = []

        if current_page_url: time.sleep(DELAY_TRA_RICHIESTE)
