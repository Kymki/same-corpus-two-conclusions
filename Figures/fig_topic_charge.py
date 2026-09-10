"""
Figure 4: topic-conditional affective charge on the symmetrically filtered subset.

(a) cross-platform comparison at k = 10, restricted to topics with at least two
    platforms above the 30-item threshold;
(b) Telegram's internal profile at k = 21, with topics grouped as
    diplomatic/institutional, kinetic, or other.

Panel (b) compares a single source against itself and is therefore unaffected by
the between-source calibration differential of Section 4.2.

On the grouping in panel (b). The paper assigns the categories by inspection of
the leading terms. Assigning them by hand in a figure script would make the
grouping unauditable, so the rule is declared below instead and the resulting
assignment is written to the values CSV. A topic counts as
diplomatic/institutional only if its leading terms name an office, an
institution or a head of state (minister, president, zelenskyy, putin, china,
beijing, poland, nato, summit, talk, negotiation, diplomatic); a place name
alone is not enough. That distinction is what separates "say, european, russia,
minister" (institutional) from "trump, say, europe, war" (other), and the
grouping is sensitive to it: were the latter counted as institutional, the
categories would interleave rather than separate. It is an interpretive choice,
and it is reported as one.

Inputs (data/results/):
  symfilter_continuous_by_topic_platform_k10.csv
  symfilter_continuous_by_topic_platform_k21.csv

Outputs (data/results/figures/):
  fig_topic_charge.png / .pdf
  fig_topic_charge_values.csv
"""
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fig_config import FIGURES_DIR, PLATFORM_COLORS, PLATFORM_LABELS  # noqa: E402

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_RESULTS  # noqa: E402

K10 = os.path.join(DATA_RESULTS, "symfilter_continuous_by_topic_platform_k10.csv")
K21 = os.path.join(DATA_RESULTS, "symfilter_continuous_by_topic_platform_k21.csv")
OUT_PNG = os.path.join(FIGURES_DIR, "fig_topic_charge.png")
OUT_PDF = os.path.join(FIGURES_DIR, "fig_topic_charge.pdf")
OUT_VALS = os.path.join(FIGURES_DIR, "fig_topic_charge_values.csv")

# CSV platform labels -> fig_config keys
KEY = {"Guardian": "Guardian", "Reddit": "Reddit_Commento", "Telegram": "Telegram"}
ORDER = ["Guardian", "Reddit", "Telegram"]
MARKERS = {"Guardian": "o", "Reddit": "s", "Telegram": "^"}

INSTITUTIONAL = {"minister", "president", "zelenskyy", "zelensky", "putin", "china",
                 "beijing", "poland", "nato", "summit", "talk", "negotiation",
                 "diplomatic", "ceasefire"}
KINETIC = {"missile", "strike", "drone", "attack", "damage", "village", "soldier",
           "force", "injure"}
CAT_STYLE = {"Diplomatic/institutional": "#1B9E77",
             "Kinetic": "#D95F02",
             "Other": "#8d8d8d"}
CAT_MARKER = {"Diplomatic/institutional": "o", "Kinetic": "s", "Other": "D"}


def categorise(top_terms):
    w = {t.strip() for t in str(top_terms).split(",")}
    if w & KINETIC:
        return "Kinetic"
    if w & INSTITUTIONAL:
        return "Diplomatic/institutional"
    return "Other"


def main():
    a = pd.read_csv(K10)
    keep = a.groupby("topic")["platform"].nunique()
    a = a[a["topic"].isin(keep[keep >= 2].index)].copy()
    a["label"] = a["top_terms"].str.split(", ").str[:4].str.join(", ")
    order_a = (a.groupby("label")["charge_mean"].mean().sort_values().index.tolist())

    b = pd.read_csv(K21)
    b = b[b["platform"] == "Telegram"].copy()
    b["label"] = b["top_terms"].str.split(", ").str[:3].str.join(", ")
    b["category"] = b["top_terms"].apply(categorise)
    b = b.sort_values("charge_mean")

    print(f"pannello (a): k=10, {a['topic'].nunique()} topic con >=2 piattaforme")
    print(f"pannello (b): k=21, {len(b)} celle Telegram\n")
    print("classificazione dichiarata, pannello (b):")
    for _, r in b.iterrows():
        print(f"  {r.charge_mean:>6.2f}  {r.category:<26} {r.top_terms}")
    for cat in ("Diplomatic/institutional", "Kinetic"):
        v = b.loc[b["category"] == cat, "charge_mean"]
        if len(v):
            print(f"  {cat}: n={len(v)}, min {v.min():.2f}, max {v.max():.2f}")
    di = b.loc[b["category"] == "Diplomatic/institutional", "charge_mean"]
    ki = b.loc[b["category"] == "Kinetic", "charge_mean"]
    if len(di) and len(ki):
        print(f"  ogni topic cinetico sopra ogni topic diplomatico: "
              f"{'SI' if di.max() < ki.min() else 'NO'}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4),
                                   gridspec_kw={"width_ratios": [1, 1]})

    for i, lab in enumerate(order_a):
        for plat in ORDER:
            r = a[(a["label"] == lab) & (a["platform"] == plat)]
            if r.empty:
                continue
            r = r.iloc[0]
            ax1.errorbar(r["charge_mean"], i,
                         xerr=[[r["charge_mean"] - r["charge_lo"]],
                               [r["charge_hi"] - r["charge_mean"]]],
                         fmt=MARKERS[plat], color=PLATFORM_COLORS[KEY[plat]],
                         markersize=6.5, markeredgecolor="white", markeredgewidth=0.8,
                         capsize=2.5, elinewidth=1.5,
                         label=PLATFORM_LABELS[KEY[plat]] if i == 0 else None)
    ax1.set_yticks(range(len(order_a)))
    ax1.set_yticklabels(order_a, fontsize=8.5)
    ax1.set_xlabel(r"Affective charge, $1-\bar{p}_{\mathrm{neu}}$ (%)")
    ax1.set_title("(a) Topic-conditional charge by platform ($k$=10)",
                  fontsize=11.5, loc="left")
    ax1.legend(fontsize=8.5, loc="lower right")

    for i, (_, r) in enumerate(b.iterrows()):
        ax2.errorbar(r["charge_mean"], i,
                     xerr=[[r["charge_mean"] - r["charge_lo"]],
                           [r["charge_hi"] - r["charge_mean"]]],
                     fmt=CAT_MARKER[r["category"]], color=CAT_STYLE[r["category"]],
                     markersize=6.5, markeredgecolor="white", markeredgewidth=0.8,
                     capsize=2.5, elinewidth=1.5)
    for cat, col in CAT_STYLE.items():
        if (b["category"] == cat).any():
            ax2.errorbar([], [], fmt=CAT_MARKER[cat], color=col, markersize=6.5,
                         markeredgecolor="white", markeredgewidth=0.8, label=cat)
    ax2.set_yticks(range(len(b)))
    ax2.set_yticklabels(b["label"], fontsize=8.5)
    ax2.set_xlabel(r"Affective charge, $1-\bar{p}_{\mathrm{neu}}$ (%)")
    ax2.set_title("(b) Telegram's internal profile ($k$=21)", fontsize=11.5, loc="left")
    ax2.legend(fontsize=8.5, loc="lower right")

    fig.suptitle("Topic-conditional affective charge (symmetric subset)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    for p in (OUT_PNG, OUT_PDF):
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    cols = ["topic", "top_terms", "platform", "n_items",
            "charge_mean", "charge_lo", "charge_hi"]
    pd.concat([a[cols].assign(panel="a", k=10, category=""),
               b[cols].assign(panel="b", k=21, category=b["category"])],
              ignore_index=True).to_csv(OUT_VALS, index=False)
    print(f"\nscritto: {OUT_PNG}\n         {OUT_PDF}\n         {OUT_VALS}")


if __name__ == "__main__":
    main()
