import csv
import time
import datetime
import os
import sys
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import DATA_RAW

# --- CONFIGURAZIONE ---
NOME_SITO = "Kyiv_Independent"
STARTING_ARCHIVE_URL = "https://kyivindependent.com/news-archive/"
BASE_URL = "https://kyivindependent.com"
OUTPUT_CSV_URLS_FROM_ARCHIVE = os.path.join(DATA_RAW, f"{NOME_SITO}_archive_article_urls.csv")
DELAY_TRA_RICHIESTE = 3
TARGET_COUNT = 20

def salva_url_raccolti_csv_incremental(lista_url_dati, nome_file):
    if not lista_url_dati:
        return
    file_exists = os.path.exists(nome_file)
    try:
        with open(nome_file, 'a', newline='', encoding='utf-8') as file_csv:
            writer = csv.writer(file_csv)
            if not file_exists or os.path.getsize(nome_file) == 0:
                writer.writerow(['url', 'date_on_archive_utc_iso'])
            for item in lista_url_dati:
                writer.writerow([item['url'], 'N/A_DATE'])
    except IOError as e:
        print(f"Errore I/O nel salvataggio del file CSV: {e}")

def flush_print(msg):
    print(msg)
    sys.stdout.flush()

if __name__ == "__main__":
    url_visitati_globali = set()
    
    if os.path.exists(OUTPUT_CSV_URLS_FROM_ARCHIVE) and os.path.getsize(OUTPUT_CSV_URLS_FROM_ARCHIVE) > 0:
        with open(OUTPUT_CSV_URLS_FROM_ARCHIVE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            url_visitati_globali.update(row['url'] for row in reader if row.get('url'))
            flush_print(f"Caricati {len(url_visitati_globali)} URL esistenti.")

    flush_print(f"Avvio Playwright. Obiettivo: {TARGET_COUNT} articoli.")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36")
        page = context.new_page()
        
        flush_print(f"Navigazione: {STARTING_ARCHIVE_URL}")
        try:
            page.goto(STARTING_ARCHIVE_URL, wait_until="networkidle", timeout=60000)
        except Exception as e:
            flush_print(f"Errore caricamento: {e}")
            browser.close()
            exit(1)

        esclusioni = [
            'membership', 'tag/', 'category/', 'author/', 
            'commercial-services', 'guidelines', 'assets/', 
            'about', 'contact', 'privacy', 'podcasts', 'newsletters',
            'jobs', 'team', 'cookie-policy', 'writing-op-ed', 'comments-policy',
            'news-archive'
        ]
        
        nuovi_articoli_totali = 0
        
        while nuovi_articoli_totali < TARGET_COUNT:
            # Aspetta un po' per il rendering dei componenti React
            time.sleep(2)
            
            html_content = page.content()
            soup = BeautifulSoup(html_content, 'html.parser')
            
            articoli_scoperti_in_questo_passo = []
            
            # Troviamo tutti i link e filtriamo quelli che sembrano articoli
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                # Pulisci il link
                if href.startswith(BASE_URL):
                    href = href.replace(BASE_URL, "")
                
                # Deve essere un link interno, lungo (slug), e non in blacklist
                # Gli articoli di solito hanno molti trattini e nessun punto finale (tranne slash)
                if href.startswith('/') and len(href) > 35 and '-' in href and not any(excl in href for excl in esclusioni):
                    url_completo = urljoin(BASE_URL, href)
                    if url_completo not in url_visitati_globali:
                        url_visitati_globali.add(url_completo)
                        articoli_scoperti_in_questo_passo.append({'url': url_completo})
            
            if articoli_scoperti_in_questo_passo:
                nuovi_articoli_totali += len(articoli_scoperti_in_questo_passo)
                flush_print(f"  Trovati {len(articoli_scoperti_in_questo_passo)} nuovi link. Totale: {nuovi_articoli_totali}")
                salva_url_raccolti_csv_incremental(articoli_scoperti_in_questo_passo, OUTPUT_CSV_URLS_FROM_ARCHIVE)
            
            if nuovi_articoli_totali >= TARGET_COUNT:
                flush_print(f"Raggiunto l'obiettivo.")
                break
                
            # Clicca 'Load More'
            btn_selector = 'button[class*="NewsArchiveList_btn"]'
            if page.is_visible(btn_selector):
                flush_print("Clicco 'Load More'...")
                page.click(btn_selector)
                time.sleep(DELAY_TRA_RICHIESTE)
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            else:
                # Se non è visibile, forse dobbiamo scrollare per vederlo
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                time.sleep(2)
                if not page.is_visible(btn_selector):
                    flush_print("Pulsante 'Load More' non trovato.")
                    break
                else:
                    page.click(btn_selector)
                
        browser.close()
        flush_print(f"Fine. Salvati {nuovi_articoli_totali} articoli.")
