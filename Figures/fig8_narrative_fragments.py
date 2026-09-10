"""
Table 2: verbatim sentences from each source.

Renders the table as published: three sources, three columns (Source, Excerpt,
Sentiment) and the caption used in the paper. The excerpts are resolved from
fig8_candidates.csv by the identifiers recorded in fig8_selected_ids.json, so no
text is retyped here and every row is traceable to a corpus document.

On identifiers. The Reddit entry is addressed by its keyed pseudonym, not by the
comment id: the raw id is not in this repository (see pseudonymise_reddit_ids.py).
The Guardian entry is addressed by URL. BBC News is collected but excluded from
the analysis (Section 4.1), so it is not a row of this table.

Inputs (data/results/):
  fig8_selected_ids.json    which document is shown for each source
  fig8_candidates.csv       the candidate excerpts, with text and model label

Outputs (data/results/figures/):
  fig8_narrative_fragments.tex          the LaTeX table used in the paper
  fig8_narrative_fragments.png / .pdf   a rendered version for quick reading
"""
import json
import os
import sys
import textwrap

import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fig_config import FIGURES_DIR  # noqa: E402

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_RESULTS  # noqa: E402

SELECTED_JSON = os.path.join(DATA_RESULTS, "fig8_selected_ids.json")
CANDIDATES_CSV = os.path.join(DATA_RESULTS, "fig8_candidates.csv")
OUT_TEX = os.path.join(FIGURES_DIR, "fig8_narrative_fragments.tex")
OUT_PNG = os.path.join(FIGURES_DIR, "fig8_narrative_fragments.png")
OUT_PDF = os.path.join(FIGURES_DIR, "fig8_narrative_fragments.pdf")

DISPLAY = {"Reddit_Commento": "Reddit", "Telegram": "Telegram",
           "Guardian": "The Guardian"}
ORDER = ["Reddit_Commento", "Telegram", "Guardian"]

CAPTION = (
    "Verbatim sentences from each source, shown to illustrate differences in "
    "register. The excerpts are single documents selected for illustration and "
    "are not evidence of platform-level properties; the quantitative basis for "
    "the comparisons in this paper is given in Sections 4.6 and 4.8. The "
    "sentiment column reports the model-assigned label for the source document."
)


def latex_escape(s):
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s


def load_fragments():
    for p in (SELECTED_JSON, CANDIDATES_CSV):
        if not os.path.exists(p):
            sys.exit(f"ERROR: {p} not found. Run fig8_extract_candidates.py first.")
    with open(SELECTED_JSON, encoding="utf-8") as f:
        selected = json.load(f)
    cands = pd.read_csv(CANDIDATES_CSV, dtype={"id_originale": str})
    cands = cands.set_index("id_originale")

    by_source = {}
    for e in selected:
        doc_id = e.get("id_originale")
        if not doc_id:
            continue
        doc_id = str(doc_id)
        if doc_id not in cands.index:
            sys.exit(f"ERROR: selected id {doc_id} is not in "
                     f"{os.path.basename(CANDIDATES_CSV)}: every excerpt must be a "
                     "real, traceable corpus document.")
        row = cands.loc[doc_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        by_source[row["fonte"]] = {
            "source": DISPLAY.get(row["fonte"], row["fonte"]),
            "id": doc_id,
            "text": str(row["original_text"]).strip(),
            "sentiment": row["sentiment_label"],
        }

    frags = []
    for src in ORDER:
        if src not in by_source:
            sys.exit(f"ERROR: no excerpt selected for {src}. The published table "
                     f"has one row per source in {ORDER}.")
        frags.append(by_source[src])
    return frags


def write_tex(frags):
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{" + CAPTION + "}",
             r"\label{tab:narrative_fragments}", r"\small",
             r"\begin{tabularx}{\textwidth}{l X l}", r"\toprule",
             r"\textbf{Source} & \textbf{Excerpt} & \textbf{Sentiment} \\",
             r"\midrule"]
    for i, f in enumerate(frags):
        if i:
            lines.append(r"\addlinespace")
        lines.append(f'{latex_escape(f["source"])} & {latex_escape(f["text"])} '
                     f'& {f["sentiment"]} \\\\')
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    with open(OUT_TEX, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def write_image(frags):
    fig, ax = plt.subplots(figsize=(11, 1.1 + 1.05 * len(frags)))
    ax.axis("off")
    tbl = ax.table(
        cellText=[[f["source"], textwrap.fill(f["text"], 74), f["sentiment"]]
                  for f in frags],
        colLabels=["Source", "Excerpt", "Sentiment"],
        colWidths=[0.15, 0.68, 0.13], cellLoc="left", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, _), cell in tbl.get_celld().items():
        cell.set_linewidth(0.5)
        cell.set_edgecolor("0.8")
        cell.set_height(0.30 if r == 0 else 0.16)
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("0.94")
    ax.set_title("Verbatim sentences from each source", fontsize=12, pad=14)
    fig.tight_layout()
    for p in (OUT_PNG, OUT_PDF):
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    frags = load_fragments()
    write_tex(frags)
    write_image(frags)
    print(f"{len(frags)} righe, nell'ordine pubblicato:")
    for f in frags:
        print(f"  {f['source']:<13} {f['sentiment']:<9} id={f['id'][:52]}")
    print(f"\nscritto: {OUT_TEX}\n         {OUT_PNG}\n         {OUT_PDF}")


if __name__ == "__main__":
    main()
