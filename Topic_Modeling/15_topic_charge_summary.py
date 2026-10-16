"""
Section 4.8: reading the order x seed repetitions against criteria fixed in advance.

The repetitions (script 14) were computed and stored without their charge values being
inspected. The criteria below were fixed by the authors before the results were opened
(corresponding author's e-mail of 9-10 October 2026), and this script applies them
mechanically. The actual counts are reported whatever the verdict: a threshold prevents
moving the goalposts after seeing the numbers, it does not replace the distribution.

Rule: the main classification rule (topic_category_rule.py, validated by two blind coders).
ALT_PROCESS and the post-hoc variant are summarised beside it as declared checks.

POOLED. The within-Telegram contrast (mean affective charge of items in diplomatic topics
minus kinetic topics) is negative with a 95% interval excluding zero in at least
MIN_REPS of the 22 repetitions, at k=10 AND at k=21.

PER CHANNEL (WarTranslated, Slavyangrad, Clash Report): direction only, since requiring
each channel's interval to exclude zero would measure its size, not the effect. For each
channel and each k, out of the 22 repetitions:
  holds              negative estimate in >= MIN_REPS
  direction failure  positive estimate in a majority (>= MAJORITY)
  power              negative estimate in a majority, but fewer than MIN_REPS
  undetermined       none of the above (neither sign has a majority)
A channel holds if it holds at both k; it fails by direction if it fails by direction at
either k. Repetitions in which the channel has fewer than MIN_CELL items in either topic
category count as neither negative nor positive.

CLASS (the 4.8 claim about a class of channels), given POOLED:
  - all three channels hold                          -> the class claim stands;
  - two hold, the third only lacks power             -> it stands, reported as holding
                                                        clearly in two channels and
                                                        underdetermined in the third;
  - any channel fails by direction                   -> the class claim falls; the three
                                                        channels are reported separately;
  - WarTranslated fails by direction                 -> Section 4.8 is withdrawn, whatever
                                                        the other two show (55.5% of the
                                                        Telegram subset goes the other way).
Any other combination is reported as not covered by the rules fixed in advance.

Input:  data/results/topic_charge_repetitions_effects.csv (script 14)
Output: data/results/topic_charge_summary.csv
"""
import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_RESULTS

EFFECTS = os.path.join(DATA_RESULTS, "topic_charge_repetitions_effects.csv")
OUT = os.path.join(DATA_RESULTS, "topic_charge_summary.csv")

# --- fixed in advance with the corresponding author -----------------------------------
RULE_VARIANT = "main"
N_REPS = 22
MIN_REPS = 18          # of 22
MAJORITY = 12          # of 22
# ---------------------------------------------------------------------------------------
MIN_CELL = 30
K_BOTH = (10, 21)
CHANNELS = ["WarTranslated", "Slavyangrad", "Clash Report"]


def counts(g):
    est = (g.n_telegram_dip >= MIN_CELL) & (g.n_telegram_kin >= MIN_CELL)
    return dict(repetitions=len(g), estimable=int(est.sum()),
                negative=int((est & (g.estimate < 0)).sum()),
                negative_ci_excl_zero=int((est & (g.estimate < 0) & (g.ci_hi < 0)).sum()),
                positive=int((est & (g.estimate > 0)).sum()),
                positive_ci_excl_zero=int((est & (g.estimate > 0) & (g.ci_lo > 0)).sum()),
                estimate_median=round(g.estimate.median(), 2),
                estimate_min=round(g.estimate.min(), 2), estimate_max=round(g.estimate.max(), 2))


def channel_status(c):
    if c['negative'] >= MIN_REPS:
        return "holds"
    if c['positive'] >= MAJORITY:
        return "direction failure"
    if c['negative'] >= MAJORITY:
        return "power"
    return "undetermined"


def main():
    e = pd.read_csv(EFFECTS)
    assert e.groupby(['k', 'variant', 'effect']).size().eq(N_REPS).all()
    s = pd.DataFrame([dict(k=k, variant=v, effect=eff, **counts(g))
                      for (k, v, eff), g in e.groupby(['k', 'variant', 'effect'])])
    s.to_csv(OUT, index=False)
    with pd.option_context('display.width', 250):
        print(s.to_string(index=False))

    sel = s[s.variant == RULE_VARIANT].set_index(['effect', 'k'])
    print(f"\n=== Verdicts (rule: {RULE_VARIANT}; {MIN_REPS} of {N_REPS} repetitions at k=10 and k=21) ===")
    pooled = {k: sel.loc[('within_telegram', k), 'negative_ci_excl_zero'] for k in K_BOTH}
    pooled_holds = all(v >= MIN_REPS for v in pooled.values())
    print("POOLED within Telegram, negative with interval excluding zero: "
          + ", ".join(f"k={k}: {v} of {N_REPS}" for k, v in pooled.items())
          + f"  -> {'HOLDS' if pooled_holds else f'BELOW the threshold of {MIN_REPS} fixed in advance'}")

    status = {}
    for c in CHANNELS:
        per_k = {k: channel_status(sel.loc[(f'within_channel:{c}', k)]) for k in K_BOTH}
        if all(v == "holds" for v in per_k.values()):
            status[c] = "holds"
        elif any(v == "direction failure" for v in per_k.values()):
            status[c] = "direction failure"
        elif all(v in ("holds", "power") for v in per_k.values()):
            status[c] = "power"
        else:
            status[c] = "undetermined"
        detail = ", ".join(f"k={k}: negative {sel.loc[(f'within_channel:{c}', k), 'negative']}, "
                           f"positive {sel.loc[(f'within_channel:{c}', k), 'positive']} ({per_k[k]})"
                           for k in K_BOTH)
        print(f"  {c:<14} {status[c].upper():<18} [{detail}]")

    if not pooled_holds:
        verdict = "pooled contrast below the threshold fixed in advance; class question not reached"
    elif status["WarTranslated"] == "direction failure":
        verdict = "WarTranslated fails by direction -> Section 4.8 is withdrawn"
    elif any(v == "direction failure" for v in status.values()):
        verdict = "a channel fails by direction -> class claim falls; report channels separately"
    elif all(v == "holds" for v in status.values()):
        verdict = "all three channels hold -> class claim stands"
    elif sorted(status.values()) == ["holds", "holds", "power"]:
        third = [c for c, v in status.items() if v == "power"][0]
        verdict = f"two channels hold, {third} lacks power -> class claim stands, underdetermined in {third}"
    else:
        verdict = "combination not covered by the rules fixed in advance -> refer to the authors"
    print(f"CLASS: {verdict}")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
