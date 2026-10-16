"""
Cross-platform similarity under document order x seed (Section 4.6, triage 2.6).

The published similarity results are one LDA solution each: one order of the items
and one seed. This figure shows the 22 solutions of the symmetric design (canonical
order plus 10 permutations, seeds 100 and 200) for every k:

  (a) cosine similarity of each platform pair in every repetition; the bar marks the
      median, the outlined marker the canonical run (canonical order, seed 100), which
      is the published result for k <= 21 (k = 25 and 30 extend the grid, triage 2.7);
  (b) how many of the 22 repetitions each pair is the most similar pair.

Regenerated from data/results/repetitions_symmetric.csv (script 13); profiles are
dominant-topic shares, as in the paper. Alongside the image it writes
fig_order_repetitions_values.csv with every plotted number and the SHA-256 of its
source file.

Outputs (data/results/figures/):
  fig_order_repetitions.png / .pdf
  fig_order_repetitions_values.csv
"""
import os
import sys
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fig_config import FIGURES_DIR  # noqa: E402  (also applies the house style)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_RESULTS  # noqa: E402

SRC = os.path.join(DATA_RESULTS, "repetitions_symmetric.csv")
OUT_PNG = os.path.join(FIGURES_DIR, "fig_order_repetitions.png")
OUT_PDF = os.path.join(FIGURES_DIR, "fig_order_repetitions.pdf")
OUT_VALS = os.path.join(FIGURES_DIR, "fig_order_repetitions_values.csv")

# Same pair colours and markers as Figure 3 (fig_preprocessing_regimes.py), so each
# pair keeps one identity across the paper's figures.
PAIR_STYLE = {
    "Guardian-Reddit":   {"color": "#D95F02", "marker": "o", "label": "The Guardian - Reddit"},
    "Guardian-Telegram": {"color": "#1B9E77", "marker": "s", "label": "The Guardian - Telegram"},
    "Reddit-Telegram":   {"color": "#7570B3", "marker": "^", "label": "Reddit - Telegram"},
}
PAIR_ORDER = list(PAIR_STYLE)
PROFILE = "argmax"
CANONICAL = dict(order=0, seed=100)
DODGE = {"Guardian-Reddit": -0.24, "Guardian-Telegram": 0.0, "Reddit-Telegram": 0.24}
JITTER = 0.07


def sha(path, n=12):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:n]


def main():
    if not os.path.exists(SRC):
        sys.exit(f"ERROR: missing input {SRC} (run Topic_Modeling/13_order_seed_repetitions.py)")
    df = pd.read_csv(SRC)
    df = df[df["profile"] == PROFILE]
    ks = sorted(df["k"].unique())
    n_runs = df.groupby(["order", "seed"]).ngroups
    rng = np.random.default_rng(0)                      # jitter only; does not touch the data
    rows = []

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.6),
                                 gridspec_kw={"width_ratios": [1.6, 1]})

    # (a) every repetition, median bar, published run
    for xi, k in enumerate(ks):
        dk = df[df["k"] == k]
        for pair in PAIR_ORDER:
            st, d = PAIR_STYLE[pair], dk[dk["pair"] == pair]
            x0 = xi + DODGE[pair]
            xs = x0 + rng.uniform(-JITTER, JITTER, len(d))
            ax.scatter(xs, d["cosine"], s=16, color=st["color"], marker=st["marker"],
                       alpha=0.55, linewidths=0, zorder=2)
            med = d["cosine"].median()
            ax.plot([x0 - 0.1, x0 + 0.1], [med, med], color=st["color"], linewidth=2.2, zorder=3)
            can = d[(d["order"] == CANONICAL["order"]) & (d["seed"] == CANONICAL["seed"])]["cosine"].iloc[0]
            ax.scatter([x0], [can], s=70, facecolor="white", edgecolor=st["color"],
                       marker=st["marker"], linewidths=1.8, zorder=4)
            for r in d.itertuples():
                rows.append({"panel": "a", "k": k, "pair": pair, "order": r.order, "seed": r.seed,
                             "cosine": r.cosine, "is_canonical_run": (r.order, r.seed) == (0, 100), "is_published": (r.order, r.seed) == (0, 100) and k <= 21,
                             "source_file": os.path.basename(SRC), "source_sha256_12": sha(SRC)})
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels(ks)
    ax.set_xlabel("Number of topics ($k$)")
    ax.set_ylabel("Cosine similarity")
    ax.set_title(f"(a) {n_runs} solutions per $k$", fontsize=12, loc="left", pad=14)
    ax.text(0, 1.015, "dots: order x seed repetitions; bar: median; open marker: canonical run",
            transform=ax.transAxes, fontsize=9.5, color="0.35", style="italic")
    handles = [plt.Line2D([], [], color=PAIR_STYLE[p]["color"], marker=PAIR_STYLE[p]["marker"],
                          linestyle="", markersize=7, label=PAIR_STYLE[p]["label"]) for p in PAIR_ORDER]
    handles.append(plt.Line2D([], [], color="0.3", marker="o", markerfacecolor="white",
                              linestyle="", markersize=8,
                              label="canonical run (published result for $k \leq 21$)"))

    # (b) how often each pair leads
    lead = (df.sort_values("cosine", ascending=False)
              .groupby(["k", "order", "seed"]).head(1)
              .groupby(["k", "pair"]).size().unstack(fill_value=0)
              .reindex(index=ks, columns=PAIR_ORDER, fill_value=0))
    bottom = np.zeros(len(ks))
    for pair in PAIR_ORDER:
        vals = lead[pair].to_numpy()
        bx.bar(range(len(ks)), vals, bottom=bottom, width=0.62, color=PAIR_STYLE[pair]["color"],
               edgecolor="white", linewidth=2, label=PAIR_STYLE[pair]["label"], zorder=2)
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            rows.append({"panel": "b", "k": ks[xi], "pair": pair, "runs_leading": int(v),
                         "source_file": os.path.basename(SRC), "source_sha256_12": sha(SRC)})
        bottom += vals
    for xi, v in enumerate(lead["Reddit-Telegram"].to_numpy()):
        bx.text(xi, n_runs + 0.4, f"{v}", ha="center", va="bottom", fontsize=9, color="0.25")
    bx.set_xticks(range(len(ks)))
    bx.set_xticklabels(ks)
    bx.set_xlabel("Number of topics ($k$)")
    bx.set_ylabel(f"Repetitions (of {n_runs})")
    bx.set_ylim(0, n_runs + 2.5)
    bx.set_title("(b) Most similar pair", fontsize=12, loc="left", pad=14)
    bx.text(0, 1.015, "number above each bar: repetitions led by Reddit - Telegram",
            transform=bx.transAxes, fontsize=9.5, color="0.35", style="italic")

    fig.suptitle("Cross-platform similarity across document orders and seeds (symmetric filter)",
                 fontsize=13.5, y=1.03)
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for p in (OUT_PNG, OUT_PDF):
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT_VALS, index=False)
    print(f"input: {os.path.basename(SRC)}  sha256[:12]={sha(SRC)}")
    print(f"written: {OUT_PNG}\n         {OUT_PDF}\n         {OUT_VALS}")
    print("\nRepetitions led by each pair:\n" + lead.to_string())


if __name__ == "__main__":
    main()
