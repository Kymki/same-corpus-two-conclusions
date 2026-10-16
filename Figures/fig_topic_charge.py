"""
Topic-conditional affective charge on the symmetrically filtered subset (Section 4.8).

(a) cross-platform comparison at k = 10, restricted to topics with at least two
    platforms above the 30-item threshold;
(b) the within-Telegram contrast (mean charge of items in diplomatic topics minus
    kinetic topics) for Telegram pooled and for each channel separately, in the 22
    solutions per k of the order x seed repetitions (11 document orders x seeds 100
    and 200), at k = 10 and k = 21.

Panel (b) replaces the submitted version, which showed Telegram's per-topic profile
at k = 21 as a single source and grouped topics with a word list declared in this
script. Section 4.8 now reports that the pooled contrast does not describe the
channels: two go in the stated direction, Clash Report in the opposite one. Split by
channel and topic, most cells fall below 30 items, so the panel shows the contrast
per channel instead. Topics are classified by the validated rule
(topic_category_rule.py, main variant), as in scripts 14-15; nothing is recomputed
here. Repetitions in which a cell has fewer than 30 items are not plotted (as in the
counts of script 15); they are flagged in the values CSV. Rybar in English (1.9% of
the Telegram subset) has fewer than 30 items in both categories and is not shown.

Inputs (data/results/):
  symfilter_continuous_by_topic_platform_k10.csv          panel (a)
  topic_charge_repetitions_effects.csv (script 14)        panel (b)
  telegram_channel_counts.csv (script 17)                 channel shares

Outputs (data/results/figures/):
  fig_topic_charge.png / .pdf
  fig_topic_charge_values.csv  (every plotted value, with the SHA-256 of its source)
"""
import os
import sys
import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fig_config import FIGURES_DIR, PLATFORM_COLORS, PLATFORM_LABELS  # noqa: E402

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_RESULTS  # noqa: E402

K10 = os.path.join(DATA_RESULTS, "symfilter_continuous_by_topic_platform_k10.csv")
EFFECTS = os.path.join(DATA_RESULTS, "topic_charge_repetitions_effects.csv")
COUNTS = os.path.join(DATA_RESULTS, "telegram_channel_counts.csv")
OUT_PNG = os.path.join(FIGURES_DIR, "fig_topic_charge.png")
OUT_PDF = os.path.join(FIGURES_DIR, "fig_topic_charge.pdf")
OUT_VALS = os.path.join(FIGURES_DIR, "fig_topic_charge_values.csv")

# CSV platform labels -> fig_config keys
KEY = {"Guardian": "Guardian", "Reddit": "Reddit_Commento", "Telegram": "Telegram"}
ORDER = ["Guardian", "Reddit", "Telegram"]
MARKERS = {"Guardian": "o", "Reddit": "s", "Telegram": "^"}

# panel (b): rows top to bottom; effect names as in script 14
ROWS = [("within_telegram", "Telegram, pooled", None),
        ("within_channel:WarTranslated", "WarTranslated", "WarTranslated"),
        ("within_channel:Slavyangrad", "Slavyangrad", "Slavyangrad"),
        ("within_channel:Clash Report", "Clash Report", "Clash Report")]
# One colour, Telegram's (panel (b) is Telegram only); k is told apart by position
# (upper / lower dot row), marker shape and the legend.
K_STYLE = {10: {"color": PLATFORM_COLORS["Telegram"], "marker": "o", "dy": -0.17},
           21: {"color": PLATFORM_COLORS["Telegram"], "marker": "s", "dy": 0.17}}
CANONICAL = (0, 100)
MIN_CELL = 30
JITTER = 0.06


def sha(path, n=12):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:n]


def main():
    a = pd.read_csv(K10)
    keep = a.groupby("topic")["platform"].nunique()
    a = a[a["topic"].isin(keep[keep >= 2].index)].copy()
    a["label"] = a["top_terms"].str.split(", ").str[:4].str.join(", ")
    order_a = (a.groupby("label")["charge_mean"].mean().sort_values().index.tolist())

    e = pd.read_csv(EFFECTS)
    e = e[(e["variant"] == "main") & e["effect"].isin([r[0] for r in ROWS])].copy()
    e["estimable"] = (e["n_telegram_dip"] >= MIN_CELL) & (e["n_telegram_kin"] >= MIN_CELL)
    counts = pd.read_csv(COUNTS)
    share = {c: counts.loc[counts["channel"].str.startswith(c),
                           "share_of_telegram_symmetric_pct"].iloc[0]
             for _, _, c in ROWS if c}

    print(f"pannello (a): k=10, {a['topic'].nunique()} topic con >=2 piattaforme")
    print("pannello (b): contrasto diplomatico - cinetico, regola principale")
    for eff, lab, _ in ROWS:
        for k in K_STYLE:
            d = e[(e["effect"] == eff) & (e["k"] == k)]
            ok = d[d["estimable"]]
            print(f"  {lab:<17} k={k}: stimabili {len(ok)}/{len(d)}, "
                  f"negativi {(ok.estimate < 0).sum()}, positivi {(ok.estimate > 0).sum()}, "
                  f"mediana {ok.estimate.median():.2f}")

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

    rng = np.random.default_rng(0)                      # jitter only; does not touch the data
    for i, (eff, lab, _) in enumerate(ROWS):
        for k, st in K_STYLE.items():
            d = e[(e["effect"] == eff) & (e["k"] == k) & e["estimable"]]
            y0 = i + st["dy"]
            ax2.scatter(d["estimate"], y0 + rng.uniform(-JITTER, JITTER, len(d)), s=16,
                        color=st["color"], marker=st["marker"], alpha=0.55, linewidths=0,
                        zorder=2)
            med = d["estimate"].median()
            ax2.plot([med, med], [y0 - 0.1, y0 + 0.1], color=st["color"], linewidth=2.2,
                     zorder=3)
            can = d[(d["order"] == CANONICAL[0]) & (d["seed"] == CANONICAL[1])]["estimate"]
            if len(can):
                ax2.scatter(can, [y0], s=70, facecolor="white", edgecolor=st["color"],
                            marker=st["marker"], linewidths=1.8, zorder=4)
            neg, pos = int((d["estimate"] < 0).sum()), int((d["estimate"] > 0).sum())
            note = f"{neg} of {len(d)} < 0" if neg >= pos else f"{pos} of {len(d)} > 0"
            ax2.text(1.01, y0, note, transform=ax2.get_yaxis_transform(), va="center",
                     fontsize=8.5, color="0.3")
    ax2.axvline(0, color="0.35", linewidth=1, zorder=1)
    ax2.set_yticks(range(len(ROWS)))
    ax2.set_yticklabels([lab if c is None else f"{lab}\n({share[c]:.1f}% of Telegram)"
                         for _, lab, c in ROWS], fontsize=9)
    ax2.set_ylim(len(ROWS) - 0.5, -0.5)
    ax2.set_xlabel("Diplomatic minus kinetic charge (percentage points)")
    ax2.set_title("(b) Within-Telegram contrast, pooled and by channel", fontsize=11.5,
                  loc="left", pad=14)
    ax2.text(0, 1.015, "dots: 22 order x seed solutions per $k$; bar: median; "
             "open marker: canonical run", transform=ax2.transAxes, fontsize=8.5,
             color="0.35", style="italic")
    handles = [plt.Line2D([], [], color=st["color"], marker=st["marker"], linestyle="",
                          markersize=7, label=f"$k$={k}") for k, st in K_STYLE.items()]
    handles.append(plt.Line2D([], [], color="0.3", marker="o", markerfacecolor="white",
                              linestyle="", markersize=8, label="canonical run"))
    ax2.legend(handles=handles, fontsize=8.5, loc="lower left")

    fig.suptitle("Topic-conditional affective charge (symmetric subset)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    for p in (OUT_PNG, OUT_PDF):
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    cols = ["topic", "top_terms", "platform", "n_items",
            "charge_mean", "charge_lo", "charge_hi"]
    va = a[cols].assign(panel="a", k=10, source_file=os.path.basename(K10),
                        source_sha256_12=sha(K10))
    vb = e.assign(panel="b", row=e["effect"].map({r[0]: r[1] for r in ROWS}),
                  is_canonical_run=(e["order"] == CANONICAL[0]) & (e["seed"] == CANONICAL[1]),
                  plotted=e["estimable"],
                  source_file=os.path.basename(EFFECTS), source_sha256_12=sha(EFFECTS))
    vb = vb[["panel", "k", "row", "order", "seed", "estimate", "ci_lo", "ci_hi",
             "n_telegram_dip", "n_telegram_kin", "is_canonical_run", "plotted",
             "source_file", "source_sha256_12"]]
    pd.concat([va, vb], ignore_index=True).to_csv(OUT_VALS, index=False)
    print(f"\nscritto: {OUT_PNG}\n         {OUT_PDF}\n         {OUT_VALS}")


if __name__ == "__main__":
    main()
