Flusso di Lavoro per l'Analisi della Tesi con Elastic Stack
Questo documento delinea il flusso di lavoro completo, dalla raccolta dei dati grezzi fino alla visualizzazione interattiva dei risultati tramite Elasticsearch e Kibana.

Fase 1: Raccolta Dati

Obiettivo: Acquisire i dati testuali grezzi da tutte le fonti definite nella metodologia (Reddit, Telegram, Kyiv Independent, BBC).

Strumenti:
Script Python per Reddit (basato su praw).
Script Python per Telegram (basato su telethon).
Script Python per il Web Scraping (basati su BeautifulSoup, newspaper3k).
Processo: Eseguire gli script di raccolta per ciascuna fonte.
Output: Una serie di file .csv contenenti i dati grezzi di ciascuna fonte (es. reddit_comments_raccolti.csv, BBC_News_contenuti_articoli_estratti.csv, ecc.).
Fase 2: Preprocessing del Testo

Obiettivo: Pulire, normalizzare e preparare i dati testuali per le analisi NLP.

Strumenti:
Script Python di preprocessing (basato su pandas, spaCy, langdetect).
Processo: Eseguire lo script di preprocessing sui file CSV grezzi generati nella Fase 1. Lo script eseguirà operazioni come:
Pulizia (rimozione URL, mentions, ecc.).
Rilevamento della lingua.
Lemmatizzazione e rimozione delle stopwords.
Output: Un unico file CSV consolidato, ad esempio: dati_testuali_preproc_consolidati.csv.
Fase 3: Analisi NLP (Topic Modeling & Sentiment)

Obiettivo: Estrarre i temi principali (topic) e analizzare il sentiment dei testi.

Strumenti:
Script Python per il Topic Modeling (basato su gensim).
Script Python per l'Analisi del Sentiment (basato su transformers).
Processo:
Eseguire lo script di Topic Modeling sull'output della Fase 2 per generare i temi.
Eseguire lo script di Analisi del Sentiment sull'output della Fase 2 per calcolare la polarità dei testi.
Output: Un file CSV finale arricchito con tutte le informazioni: dati_con_sentiment_finale.csv. Questo file conterrà il testo, il topic dominante, il sentiment, la fonte, la data, ecc.
Fase 4: Indicizzazione in Elasticsearch

Obiettivo: Caricare i dati finali e arricchiti in un motore di ricerca per permettere analisi veloci e complesse.

Strumenti:
Script Python di indicizzazione (basato sulla libreria elasticsearch).
Servizio Elasticsearch in esecuzione (tramite Docker o Cloud).
Processo: Eseguire lo script indicizza_dati.py che legge il file dati_con_sentiment_finale.csv e invia ogni riga come un documento a un indice in Elasticsearch.
Output: Un indice Elasticsearch popolato con tutti i dati.
Fase 5: Visualizzazione e Analisi Interattiva

Obiettivo: Esplorare i risultati in modo interattivo e creare le visualizzazioni finali per la tesi.

Strumenti:
Kibana (interfaccia web che si collega a Elasticsearch).
Processo:
Accedere a Kibana tramite browser (http://localhost:5601).
Creare una "Data View" per connettersi all'indice creato nella Fase 4.
Utilizzare gli strumenti "Discover", "Lens" e "Dashboard" per creare:
Grafici a barre sulla distribuzione dei topic e del sentiment per fonte.
Word Clouds (nuvole di parole) per i temi più importanti.
Grafici temporali per tracciare l'evoluzione del sentiment.
Una dashboard di "intelligence semantica" interattiva.
Output: Visualizzazioni e insight per il Capitolo 4 e 5 della tesi.