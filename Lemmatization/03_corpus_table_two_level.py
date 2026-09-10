"""
Two-level corpus table (Table 1 of the paper), as requested by RP:
sentence-level and item-level statistics side by side, so the
pseudo-replication (a Guardian article = ~46 sentence-rows, a Reddit
comment = ~1.8) is visible instead of hidden.

Unit definitions:
  - sentence: one row of the consolidated corpus (id ends in _sN for
    multi-sentence parents);
  - item: the parent document (article / comment / message), obtained by
    stripping the trailing _sN from id_originale.

Scope: English corpus (the corpus actually analysed in the paper).
Output: data/results/corpus_analysis_two_level.tex (+ console preview).
"""
import os
import re
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_PROCESSED, DATA_RESULTS

INPUT_FILE = os.path.join(DATA_PROCESSED, "dati_testuali_preproc_consolidati.csv")
OUTPUT_TEX = os.path.join(DATA_RESULTS, "corpus_analysis_two_level.tex")

DISPLAY = {"Guardian": "The Guardian", "Reddit_Commento": "Reddit",
           "Telegram": "Telegram", "BBC_News": "BBC News"}
ORDER = ["Reddit_Commento", "Telegram", "Guardian", "BBC_News"]


def main():
    print(f"Loading corpus: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    df = df[df['lingua_rilevata'] == 'en'].dropna(subset=['testo_pulito_base'])
    df['parent'] = df['id_originale'].astype(str).map(lambda s: re.sub(r'_s\d+$', '', s))
    df['n_words'] = df['testo_pulito_base'].astype(str).str.split().str.len()

    rows = []
    for fonte in ORDER:
        sub = df[df['fonte'] == fonte]
        if sub.empty:
            continue
        items = sub.groupby('parent')['n_words'].agg(['count', 'sum'])
        rows.append({
            'Source': DISPLAY.get(fonte, fonte),
            'Items': len(items),
            'Sentences': len(sub),
            'Sent_per_item': len(sub) / len(items),
            'Words_per_item': items['sum'].mean(),
            'Share_items_pct': 0.0,   # filled below
            'Share_sentences_pct': 0.0,
        })
    tab = pd.DataFrame(rows)
    tab['Share_items_pct'] = tab['Items'] / tab['Items'].sum() * 100
    tab['Share_sentences_pct'] = tab['Sentences'] / tab['Sentences'].sum() * 100

    total = {
        'Source': 'Total', 'Items': tab['Items'].sum(), 'Sentences': tab['Sentences'].sum(),
        'Sent_per_item': tab['Sentences'].sum() / tab['Items'].sum(),
        'Words_per_item': df.groupby('parent')['n_words'].sum().mean(),
        'Share_items_pct': 100.0, 'Share_sentences_pct': 100.0,
    }
    tab = pd.concat([tab, pd.DataFrame([total])], ignore_index=True)

    print("\nTwo-level corpus profile (EN corpus):")
    print(tab.round(1).to_string(index=False))

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Volumetric profile of the English corpus at both units of analysis. "
        r"\emph{Items} are parent documents (articles, comments, messages); \emph{sentences} "
        r"are the rows produced by sentence-level preprocessing. The two share columns make "
        r"the length imbalance explicit: long-form articles dominate the sentence count while "
        r"representing a minority of items; all comparative analyses are therefore conducted "
        r"at item level. Word counts are whitespace-delimited words, not transformer subword tokens.}",
        r"\label{tab:corpus_two_level}",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r" & & & \textbf{Sent./} & \textbf{Words/} & \multicolumn{2}{c}{\textbf{Corpus share (\%)}} \\",
        r"\cmidrule(lr){6-7}",
        r"\textbf{Source} & \textbf{Items} & \textbf{Sentences} & \textbf{item} & \textbf{item} & \textbf{items} & \textbf{sentences} \\",
        r"\midrule",
    ]
    for _, r_ in tab.iterrows():
        prefix = r"\midrule" + "\n" if r_['Source'] == 'Total' else ""
        lines.append(f"{prefix}{r_['Source']} & {int(r_['Items']):,} & {int(r_['Sentences']):,} & "
                     f"{r_['Sent_per_item']:.1f} & {r_['Words_per_item']:.0f} & "
                     f"{r_['Share_items_pct']:.1f} & {r_['Share_sentences_pct']:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    with open(OUTPUT_TEX, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\nLaTeX table written to: {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
