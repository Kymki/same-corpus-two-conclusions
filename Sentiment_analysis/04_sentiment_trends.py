import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_RESULTS

# --- CONFIGURAZIONE ---
INPUT_FILE = os.path.join(DATA_RESULTS, "dati_con_sentiment_finale.csv")
OUTPUT_PLOT = os.path.join(DATA_RESULTS, "sentiment_trends_300dpi.png")

def genera_trend_sentiment():
    print(f"Caricamento dati sentiment da: {INPUT_FILE}")
    try:
        df = pd.read_csv(INPUT_FILE, low_memory=False)
    except FileNotFoundError:
        print(f"ERRORE: File '{INPUT_FILE}' non trovato.")
        return

    # Filtra documenti senza data o senza sentiment calcolato
    df.dropna(subset=['data_originale_str', 'sentiment_label'], inplace=True)
    
    # Conversione data
    df['data'] = pd.to_datetime(df['data_originale_str'], errors='coerce', utc=True)
    df.dropna(subset=['data'], inplace=True)
    
    # Raggruppamento settimanale ('W')
    # Resample richiede un index datetime
    df.set_index('data', inplace=True)
    
    # Calcolo media settimanale dei punteggi continui
    trend_data = df.groupby('fonte').resample('W').agg(
        Media_Positivo=('sentiment_score_positive', 'mean'),
        Media_Negativo=('sentiment_score_negative', 'mean'),
        Volume=('id_originale', 'count')
    ).reset_index()

    # Visualizzazione
    print("Generazione grafico 300 DPI per trend sentiment...")
    sns.set_theme(style="whitegrid")
    
    fonti = trend_data['fonte'].unique()
    if len(fonti) == 0:
        print("Nessuna fonte trovata con dati temporali validi per il plot.")
        return

    fig, axes = plt.subplots(len(fonti), 1, figsize=(14, 6 * len(fonti)), sharex=True)
    
    if len(fonti) == 1:
        axes = [axes]

    for ax, fonte in zip(axes, fonti):
        data_fonte = trend_data[trend_data['fonte'] == fonte]
        
        # Plot dei trend delle probabilità medie
        sns.lineplot(data=data_fonte, x='data', y='Media_Positivo', ax=ax, label='Positivo (Score)', color='green', linewidth=2.5)
        sns.lineplot(data=data_fonte, x='data', y='Media_Negativo', ax=ax, label='Negativo (Score)', color='red', linewidth=2.5)
        
        # Overlay del volume (opzionale, ma utile per contesto scientifico)
        ax2 = ax.twinx()
        sns.barplot(data=data_fonte, x='data', y='Volume', ax=ax2, alpha=0.2, color='gray')
        ax2.set_ylabel('Volume (Numero di Documenti)', color='gray')
        
        ax.set_title(f'Trend Temporale del Sentiment - Fonte: {fonte}', fontsize=16, fontweight='bold', pad=15)
        ax.set_ylabel('Score Medio Sentiment (Softmax)', fontsize=12)
        ax.set_xlabel('Data (Aggr. Settimanale)', fontsize=12)
        
        # Legend configuration
        lines, labels = ax.get_legend_handles_labels()
        ax.legend(lines, labels, loc='upper left')

    plt.tight_layout()
    output_path = OUTPUT_PLOT
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Grafico ad alta risoluzione (300 DPI) salvato in '{output_path}'")
    except Exception as e:
        print(f"Errore durante il salvataggio del plot: {e}")
    finally:
        plt.close()

if __name__ == "__main__":
    genera_trend_sentiment()
