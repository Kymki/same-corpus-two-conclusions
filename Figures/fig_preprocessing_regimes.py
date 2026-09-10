"""
The same corpus under two preprocessing regimes (paper's central figure).

Regenerates the asymmetric-vs-symmetric similarity comparison directly from the
two canonical k-sweep CSVs, so the figure's provenance is a repository file
rather than a transcription:

  (a) data/results/item_ksweep_similarity.csv       relevance filter on The Guardian only
  (b) data/results/symfilter_ksweep_similarity.csv  same criterion on all three sources

Corpus, pipeline, similarity measures and bootstrap procedure are identical
across panels; only the symmetry of the relevance criterion differs.

Alongside the image it writes fig_preprocessing_regimes_values.csv -- the exact
numbers plotted, with their source file and row -- so every point in the figure
can be traced back without rerunning anything.

Outputs (data/results/figures/):
  fig_preprocessing_regimes.png / .pdf
  fig_preprocessing_regimes_values.csv
"""
import os
import sys
import hashlib
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fig_config import FIGURES_DIR  # noqa: E402  (also applies the house style)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_RESULTS  # noqa: E402

ASYM = os.path.join(DATA_RESULTS, "item_ksweep_similarity.csv")
SYM = os.path.join(DATA_RESULTS, "symfilter_ksweep_similarity.csv")
OUT_PNG = os.path.join(FIGURES_DIR, "fig_preprocessing_regimes.png")
OUT_PDF = os.path.join(FIGURES_DIR, "fig_preprocessing_regimes.pdf")
OUT_VALS = os.path.join(FIGURES_DIR, "fig_preprocessing_regimes_values.csv")

# Categorical palette, assigned in fixed order and never cycled. Validated for
# CVD separation (min OKLab dE 11.6 over all pairs under protan/deutan
# simulation), normal-vision separation (19.9) and contrast against a light
# surface (>= 3.3:1). Markers duplicate the identity so the series stay
# distinguishable in greyscale print and for colour-blind readers.
PAIR_STYLE = {
    "Guardian-Reddit":   {"color": "#D95F02", "marker": "o", "label": "The Guardian - Reddit"},
    "Guardian-Telegram": {"color": "#1B9E77", "marker": "s", "label": "The Guardian - Telegram"},
    "Reddit-Telegram":   {"color": "#7570B3", "marker": "^", "label": "Reddit - Telegram"},
}
PAIR_ORDER = list(PAIR_STYLE)
PANELS = [(ASYM, "(a) Asymmetric filtering", "filter applied to The Guardian only"),
          (SYM, "(b) Symmetric filtering", "same criterion applied to all three sources")]


def sha(path, n=12):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:n]


def main():
    frames = {}
    for path, _, _ in PANELS:
        if not os.path.exists(path):
            sys.exit(f"ERROR: missing canonical input {path}")
        frames[path] = pd.read_csv(path)

    ks = sorted(frames[ASYM]["k"].unique())
    if sorted(frames[SYM]["k"].unique()) != ks:
        sys.exit("ERROR: the two CSVs do not share the same k grid")
    x = range(len(ks))  # evenly spaced categorical positions

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    rows = []

    for ax, (path, title, subtitle) in zip(axes, PANELS):
        df = frames[path]
        for pair in PAIR_ORDER:
            st = PAIR_STYLE[pair]
            d = df[df["pair"] == pair].set_index("k").reindex(ks)
            ax.fill_between(x, d["cos_ci_lo"], d["cos_ci_hi"],
                            color=st["color"], alpha=0.16, linewidth=0)
            ax.plot(x, d["cosine"], color=st["color"], marker=st["marker"],
                    markersize=6.5, linewidth=2, markeredgecolor="white",
                    markeredgewidth=0.9, label=st["label"], zorder=3,
                    clip_on=False)
            for k in ks:
                r = df[(df["pair"] == pair) & (df["k"] == k)].iloc[0]
                rows.append({"panel": title[:3].strip("() "), "regime": title,
                             "k": k, "pair": pair, "cosine": r["cosine"],
                             "cos_ci_lo": r["cos_ci_lo"], "cos_ci_hi": r["cos_ci_hi"],
                             "source_file": os.path.basename(path),
                             "source_sha256_12": sha(path)})

        ax.set_title(title, fontsize=12, pad=14, loc="left")
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=9.5,
                color="0.35", style="italic")
        ax.set_xticks(list(x))
        ax.set_xticklabels(ks)
        ax.set_xlabel("Number of topics ($k$)")
        ax.set_xlim(-0.25, len(ks) - 0.75)

    axes[0].set_ylabel("Cosine similarity")
    axes[0].set_ylim(0.15, 1.02)
    axes[0].legend(loc="lower left", frameon=True, fontsize=9)

    fig.suptitle("The same corpus, two preprocessing regimes", fontsize=13.5,
                 y=1.015)
    fig.tight_layout()
    for p in (OUT_PNG, OUT_PDF):
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    vals = pd.DataFrame(rows)
    vals.to_csv(OUT_VALS, index=False)

    print(f"asymmetric input : {os.path.basename(ASYM)}  sha256[:12]={sha(ASYM)}")
    print(f"symmetric  input : {os.path.basename(SYM)}  sha256[:12]={sha(SYM)}")
    print(f"\nwritten: {OUT_PNG}\n         {OUT_PDF}\n         {OUT_VALS}")

    print("\nMost similar pair per k, as plotted:")
    for path, title, _ in PANELS:
        df = frames[path]
        print(f"  {title}")
        for k in ks:
            s = df[df["k"] == k].sort_values("cosine", ascending=False)
            top, sec = s.iloc[0], s.iloc[1]
            sep = "separated" if top["cos_ci_lo"] > sec["cos_ci_hi"] else "overlapping"
            print(f"    k={k:>2}: {top['pair']:<18} {top['cosine']:.3f}  "
                  f"(vs {sec['pair']} {sec['cosine']:.3f}, CIs {sep})")


if __name__ == "__main__":
    main()
