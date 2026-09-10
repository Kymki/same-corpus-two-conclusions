"""
Figure 1: document length inflates item-level sentiment shares.

Reproduces the simulation described in Section 4.3. The sentence-level
probability distribution is held fixed at the values observed for The Guardian
and only the number of sentences per item varies: n sentence-level probability
vectors are drawn with replacement from the observed Guardian pool, averaged, and
the argmax taken. As n grows the averaged vector converges on the pool mean, so
the argmax-negative share rises towards 100% while the mean negative probability
stays flat. Observed platform values are overlaid.

Inputs (data/results/):
  fig_length_artifact_sentence_pool.csv   every Guardian sentence-level
                                          probability vector. Built automatically
                                          from sentiment_results_multilingua.csv
                                          the first time this script runs, then
                                          committed so the figure regenerates
                                          without the full sentiment output.

Note on the two means. The simulated flat line converges on the *sentence-level*
mean negative probability (52.2%), because that is what resampling sentences
estimates. The 53.6% quoted in Section 4.3 is the *item-level* mean, which
weights every article equally regardless of length; it is plotted separately as
the observed Guardian value. Both appear in the figure, and both are written to
the values CSV.
  item_sentiment_length_artifact.csv      observed per-source values
  items_en_meta.csv                       sentences per item, per source

Outputs (data/results/figures/):
  fig_length_artifact.png / .pdf
  fig_length_artifact_values.csv          the simulated curve and the overlaid
                                          observations, as plotted
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fig_config import FIGURES_DIR, PLATFORM_COLORS, PLATFORM_LABELS  # noqa: E402

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_RESULTS  # noqa: E402

POOL = os.path.join(DATA_RESULTS, "fig_length_artifact_sentence_pool.csv")
SENT = os.path.join(DATA_RESULTS, "sentiment_results_multilingua.csv")
OBS = os.path.join(DATA_RESULTS, "item_sentiment_length_artifact.csv")
META = os.path.join(DATA_RESULTS, "items_en_meta.csv")
OUT_PNG = os.path.join(FIGURES_DIR, "fig_length_artifact.png")
OUT_PDF = os.path.join(FIGURES_DIR, "fig_length_artifact.pdf")
OUT_VALS = os.path.join(FIGURES_DIR, "fig_length_artifact_values.csv")

# The full Guardian sentence population is used, not a subsample: it reproduces
# the sentence-level argmax-negative share of 59.3% reported in Section 4.3
# exactly, and costs about 2 MB.
SIM_SEED = 42
SIM_REPS = 4000         # simulated items per n
N_GRID = [1, 2, 3, 5, 10, 20, 46, 100]

# Markers duplicate the colour so the three platforms remain distinguishable in
# greyscale print and for colour-blind readers.
MARKERS = {"Guardian": "o", "Reddit_Commento": "s", "Telegram": "^"}


def build_pool():
    """Extract the Guardian sentence-level probability pool from the full
    sentiment output, once, and cache it as a small committed CSV."""
    if not os.path.exists(SENT):
        sys.exit(
            f"ERROR: neither {os.path.basename(POOL)} nor "
            f"{os.path.basename(SENT)} is present.\n"
            "The pool file is part of the replication package; if it is missing, "
            "the full sentiment output is needed to rebuild it."
        )
    print(f"building {os.path.basename(POOL)} from {os.path.basename(SENT)} ...")
    df = pd.read_csv(SENT, low_memory=False,
                     usecols=["fonte", "lingua_rilevata", "neg_prob", "neu_prob", "pos_prob"])
    g = df[(df["fonte"] == "Guardian") & (df["lingua_rilevata"] == "en")]
    g = g[["neg_prob", "neu_prob", "pos_prob"]].dropna()
    g.round(6).to_csv(POOL, index=False)
    print(f"  {len(g)} sentence vectors written")
    return g


def main():
    pool = pd.read_csv(POOL) if os.path.exists(POOL) else build_pool()
    P = pool[["neg_prob", "neu_prob", "pos_prob"]].to_numpy(float)
    mean_neg = P[:, 0].mean() * 100
    sent_level_argmax_neg = (P.argmax(axis=1) == 0).mean() * 100
    print(f"pool: {len(P)} frasi Guardian")
    print(f"  media p_neg a livello di frase     : {mean_neg:.2f}%  "
          f"(la retta piatta simulata converge qui)")
    print(f"  argmax-negative a livello di frase : {sent_level_argmax_neg:.2f}%  "
          f"(Sezione 4.3: 59.3%)")
    print("  NB: il 53.6% citato nella 4.3 e' la media p_neg a livello di ITEM,")
    print("      riportata separatamente fra i valori osservati.")

    rng = np.random.default_rng(SIM_SEED)
    rows = []
    for n in N_GRID:
        idx = rng.integers(0, len(P), size=(SIM_REPS, n))
        avg = P[idx].mean(axis=1)                     # SIM_REPS x 3
        share = (avg.argmax(axis=1) == 0).mean() * 100
        # The mean negative probability is analytically invariant in n: the
        # expectation of the mean of n draws is the pool mean for every n. It is
        # therefore reported as that exact constant rather than as a Monte Carlo
        # estimate, whose residual wobble would be noise and not signal. The
        # simulated argmax share, by contrast, has no closed form here.
        rows.append({"n_sentences": n,
                     "simulated_argmax_negative_pct": round(share, 2),
                     "simulated_mean_negprob_pct": round(mean_neg, 2)})
    sim = pd.DataFrame(rows)
    print("\nsimulazione:")
    print(sim.to_string(index=False))

    obs = pd.read_csv(OBS)
    meta = pd.read_csv(META)
    spi = meta.groupby("fonte")["n_sentences"].mean()

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.plot(sim["n_sentences"], sim["simulated_argmax_negative_pct"],
            color="#c0392b", linewidth=2, zorder=2,
            label="Simulated argmax-negative share (identical sentence distribution)")
    ax.plot(sim["n_sentences"], sim["simulated_mean_negprob_pct"],
            color="#27ae60", linewidth=2, zorder=2,
            label="Simulated mean negative probability (length-invariant)")

    obs_rows = []
    for _, r in obs.iterrows():
        src = r["fonte"]
        if src not in spi.index:
            continue
        x = spi[src]
        col, mk = PLATFORM_COLORS[src], MARKERS[src]
        ax.scatter(x, r["item_neg_meanprob_pct"], s=78, color=col, marker=mk,
                   edgecolor="white", linewidth=1.1, zorder=4)
        ax.scatter(x, r["item_mean_negprob_pct"], s=78, facecolor="white",
                   edgecolor=col, marker=mk, linewidth=1.8, zorder=4)
        ax.annotate(PLATFORM_LABELS[src], (x, r["item_neg_meanprob_pct"]),
                    textcoords="offset points", xytext=(0, 11), ha="center",
                    fontsize=9.5, color=col, fontweight="bold")
        obs_rows.append({"source": PLATFORM_LABELS[src], "sentences_per_item": round(x, 2),
                         "observed_argmax_negative_pct": r["item_neg_meanprob_pct"],
                         "observed_mean_negprob_pct": r["item_mean_negprob_pct"]})

    ax.scatter([], [], s=78, color="0.35", marker="o", edgecolor="white",
               label="Observed argmax-negative share")
    ax.scatter([], [], s=78, facecolor="white", edgecolor="0.35", marker="o",
               linewidth=1.8, label="Observed mean negative probability")

    ax.set_xscale("log")
    ax.set_xticks(N_GRID)
    ax.set_xticklabels(N_GRID)
    ax.set_xlabel("Sentences per item (log scale)")
    ax.set_ylabel("Negativity (%)")
    ax.set_ylim(40, 102)
    ax.set_title("Document length inflates item-level sentiment shares\n"
                 "(observed platforms overlaid on simulation)", fontsize=12.5)
    ax.legend(loc="center right", fontsize=8.5)
    fig.tight_layout()
    for p in (OUT_PNG, OUT_PDF):
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    pd.concat([sim.assign(kind="simulated"),
               pd.DataFrame(obs_rows).assign(kind="observed")],
              ignore_index=True).to_csv(OUT_VALS, index=False)
    print(f"\nscritto: {OUT_PNG}\n         {OUT_PDF}\n         {OUT_VALS}")


if __name__ == "__main__":
    main()
