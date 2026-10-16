"""
Section 4.6: the similarity designs side by side, and what each design costs.

Reads the k-sweep results already in data/results/ and recomputes nothing about
similarity. For every design and every k it reports the most similar platform
pair (cosine), its lead over the runner-up, and whether the leader's 95%
bootstrap interval clears the runner-up's. Designs:

  full length, relevance filter on The Guardian only   item_ksweep_similarity.csv (04)
  full length, symmetric relevance filter              symfilter_ksweep_similarity.csv (05)
  A / B / C, length matching                           lengthmatched_ksweep_similarity.csv (07)
  chunked, C=10 and C=20                               chunked_balanced_ksweep_similarity.csv (08)

It also counts, from the consolidated corpus and with the same filter and length
rules as scripts 07 and 08, how many items each design retains per source, and
how many LDA training chunks each source contributes to the chunked designs.
The length-controlled designs drop short items, and nearly all of them come from
Reddit and Telegram; that cost is part of the result.

Outputs (data/results/):
  table_2_1.tex           both tables, LaTeX (requires booktabs)
  table_2_1_summary.csv   leading pair, lead, separation and JS agreement per design and k
  design_costs.csv        items retained per design and source
"""
import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_PROCESSED, DATA_RESULTS

INPUT_CSV = os.path.join(DATA_PROCESSED, "dati_testuali_preproc_consolidati.csv")
OUT_TEX = os.path.join(DATA_RESULTS, "table_2_1.tex")
OUT_SUMMARY = os.path.join(DATA_RESULTS, "table_2_1_summary.csv")
OUT_COSTS = os.path.join(DATA_RESULTS, "design_costs.csv")

PLATFORMS = ["Guardian", "Reddit_Commento", "Telegram"]
WAR_KW = ['ukrain', 'russia', 'putin', 'zelensk', 'kyiv', 'kremlin', 'moscow',
          'donbas', 'crimea', 'kharkiv', 'kherson', 'mariupol', 'bakhmut', 'nato']
K_SWEEP = [5, 7, 10, 15, 21]
TRAIN_CHUNKS_CAP = 4000          # per-platform training cap used by script 08
SHORT = {"Guardian-Reddit": "G--R", "Guardian-Telegram": "G--T", "Reddit-Telegram": "R--T"}


def load_designs():
    res = lambda f: pd.read_csv(os.path.join(DATA_RESULTS, f))
    lm = res("lengthmatched_ksweep_similarity.csv")
    ch = res("chunked_balanced_ksweep_similarity.csv")
    # (summary name, LaTeX design label, LaTeX filter label, results)
    return [
        ("Asymmetric filter (original)", "Full length", "Guardian only", res("item_ksweep_similarity.csv")),
        ("Symmetric filter", "Full length", "symmetric", res("symfilter_ksweep_similarity.csv")),
        ("A_len15_symfilter", "A: $N=15$", "symmetric", lm[lm.condition == "A_len15_symfilter"]),
        ("B_len15_nofilter", "B: $N=15$", "none", lm[lm.condition == "B_len15_nofilter"]),
        ("C_len25_symfilter", "C: $N=25$", "symmetric", lm[lm.condition == "C_len25_symfilter"]),
        ("Chunked-balanced C=10", "Chunked, $C=10$", "symmetric", ch[ch.chunk_size == 10]),
        ("Chunked-balanced C=20", "Chunked, $C=20$", "symmetric", ch[ch.chunk_size == 20]),
    ]


def leaders(designs):
    rows = []
    for name, _, _, d in designs:
        for k in K_SWEEP:
            s = d[d.k == k].sort_values("cosine", ascending=False)
            a, b = s.iloc[0], s.iloc[1]
            js = d[d.k == k].sort_values("js", ascending=False).iloc[0].pair
            rows.append(dict(design=name, k=k, leader=a.pair, cos=a.cosine, runner_up=b.pair,
                             lead=round(a.cosine - b.cosine, 3), lead_exact=a.cosine - b.cosine,
                             separated=bool(a.cos_ci_lo > b.cos_ci_hi),
                             js_leader=js, js_agrees=js == a.pair))
    return pd.DataFrame(rows)


def load_items():
    """Items exactly as in scripts 05/07/08 (English, BBC excluded)."""
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    df = df[(df['lingua_rilevata'] == 'en') & (df['fonte'].isin(PLATFORMS))]
    df = df.dropna(subset=['testo_lemmatizzato']).copy()
    df['parent'] = df['id_originale'].astype(str).str.replace(r'_s\d+$', '', regex=True)
    it = df.groupby(['parent', 'fonte']).agg(
        lem=('testo_lemmatizzato', ' '.join),
        clean=('testo_pulito_base', lambda s: ' '.join(map(str, s))),
    ).reset_index()
    it['rel'] = it['clean'].str.lower().apply(lambda t: any(k in t for k in WAR_KW))
    it['n'] = it['lem'].str.split().map(len)
    return it


def costs(it):
    sym = it[it.rel]
    plan = [("all_en", it, it), ("symfilter", sym, it),
            ("A_len15_sym", sym[sym.n >= 15], sym), ("B_len15_nofilter", it[it.n >= 15], it),
            ("C_len25_sym", sym[sym.n >= 25], sym),
            ("chunk10_sym", sym[sym.n >= 10], sym), ("chunk20_sym", sym[sym.n >= 20], sym)]
    rows = []
    for name, d, base in plan:
        for p in PLATFORMS:
            b, k = int((base['fonte'] == p).sum()), int((d['fonte'] == p).sum())
            rows.append(dict(design=name, source=p, items=k, base=b, pct=round(100 * k / b, 1)))
    # training chunks per platform in script 08: min(cap, floor(len/C) summed over items)
    chunks = {C: [min(TRAIN_CHUNKS_CAP, int((sym[(sym.fonte == p) & (sym.n >= C)].n // C).sum()))
                  for p in PLATFORMS] for C in (10, 20)}
    medians = sym.groupby('fonte').n.median()
    return pd.DataFrame(rows), chunks, medians, len(sym)


def num(x):
    return f"{int(x):,}" if float(x).is_integer() else f"{x:,}"


def latex(summary, designs, cost, chunks, medians, n_sym):
    L = []
    L += [r"% Triage 2.1 -- length-control designs side by side (for Section 4.6).",
          r"% Generated from data/results/{item_ksweep,symfilter_ksweep,lengthmatched_ksweep,",
          r"% chunked_balanced_ksweep}_similarity.csv; item counts recomputed from the",
          r"% consolidated corpus with the same filter and length rules as scripts 07/08.",
          r"% Requires \usepackage{booktabs}.", ""]
    L += [r"\begin{table}[H]",
          r"\caption{Most similar platform pair by cosine similarity under seven corpus designs.",
          r"Each cell gives the leading pair and its lead over the runner-up; bold marks a lead",
          r"whose 95\% bootstrap interval does not overlap the runner-up's. G = \textit{The Guardian},",
          r"R = Reddit, T = Telegram. Designs A--C truncate every item to exactly $N$ lemmatised",
          r"tokens (randomly sampled) and drop shorter items; chunked-balanced designs split items",
          r"into consecutive $C$-token chunks and average soft topic distributions per item.",
          r"Item retention for each design is given in Table~\ref{tab:design_costs}.\label{tab:length_designs}}",
          r"\centering", r"\small", r"\begin{tabular}{l l ccccc}", r"\toprule",
          r"Design & Filter & $k=5$ & $k=7$ & $k=10$ & $k=15$ & $k=21$ \\", r"\midrule"]
    for i, (name, lab, filt, _) in enumerate(designs):
        if i in (2, 5):
            L.append(r"\midrule")
        cells = []
        for _, r in summary[summary.design == name].iterrows():
            c = f"{SHORT[r.leader]} (+{r.lead_exact:.2f})"
            cells.append(r"\textbf{" + c + "}" if r.separated else c)
        L.append(f"{lab:<20} & {filt:<13} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    def cell(design, p):
        # shares are relative to the subcorpus each design starts from (see caption):
        # the symmetric subset itself is that base for the symmetric designs
        r = cost[(cost.design == design) & (cost.source == p)].iloc[0]
        share = 100.0 if design == "symfilter" else r.pct
        pct = "100" if share == 100 else f"{share:.1f}"
        return f"{r['items']:,} ({pct}\\%)"
    G, R, T = PLATFORMS
    md = {p: num(medians[p]) for p in PLATFORMS}
    base_all = cost[cost.design == "all_en"].set_index("source")["items"]
    L += [r"\begin{table}[H]",
          r"\caption{Items retained by each design, as a share of the subcorpus it starts from",
          f"(symmetric designs: the {n_sym:,}-item symmetrically filtered subset; design B: the full",
          r"English corpus). \textit{The Guardian} loses no items under any design; the",
          r"length-controlled designs retain only the longer Reddit and Telegram items",
          f"(median item length in the symmetric subset: Reddit {md[R]}, Telegram {md[T]}, "
          + r"\textit{Guardian} " + f"{md[G]} lemmatised tokens).",
          f"For the chunked designs, LDA training chunks per source are capped at {TRAIN_CHUNKS_CAP:,}; Telegram",
          r"falls short of the cap at both chunk sizes and Reddit at $C=20$, so training is not",
          r"balanced across sources.\label{tab:design_costs}}",
          r"\centering", r"\small", r"\begin{tabular}{l rrr r}", r"\toprule",
          r"Design & \textit{Guardian} & Reddit & Telegram & Training chunks (G / R / T) \\", r"\midrule"]
    rows = [("Symmetric filter, full length", "symfilter", "--"),
            ("A: $N=15$, symmetric", "A_len15_sym", "--"),
            ("B: $N=15$, no filter", "B_len15_nofilter", "--"),
            ("C: $N=25$, symmetric", "C_len25_sym", "--"),
            ("Chunked, $C=10$, symmetric", "chunk10_sym", " / ".join(f"{c:,}" for c in chunks[10])),
            ("Chunked, $C=20$, symmetric", "chunk20_sym", " / ".join(f"{c:,}" for c in chunks[20]))]
    for lab, d, tr in rows:
        g, r_, t = cell(d, G), cell(d, R), cell(d, T)
        if d == "B_len15_nofilter":
            r_, t = r_ + "$^{a}$", t + "$^{a}$"
        L.append(f"{lab:<29} & {g} & {r_} & {t} & {tr} \\\\")
    L += [r"\bottomrule",
          r"\multicolumn{5}{l}{\footnotesize $^{a}$ Share of the full English corpus "
          f"(Reddit {base_all[R]:,}; Telegram {base_all[T]:,}).}}",
          r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(L)


def main():
    designs = load_designs()
    summary = leaders(designs)
    summary.drop(columns="lead_exact").to_csv(OUT_SUMMARY, index=False)
    print(summary.drop(columns="lead_exact").to_string())

    cost, chunks, medians, n_sym = costs(load_items())
    cost.to_csv(OUT_COSTS, index=False)
    print("\n" + cost.to_string())
    print(f"\nTraining chunks per platform (G/R/T): C=10 {chunks[10]}, C=20 {chunks[20]}")

    with open(OUT_TEX, "w", encoding="utf-8", newline="\n") as f:
        f.write(latex(summary, designs, cost, chunks, medians, n_sym))
    print(f"\nSaved: {OUT_TEX}\n       {OUT_SUMMARY}\n       {OUT_COSTS}")


if __name__ == "__main__":
    main()
