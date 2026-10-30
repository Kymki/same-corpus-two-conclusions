"""
Document order versus seed: how much each contributes to the spread of the similarity results
(Section 4.6; reviewer 2, second round).

Script 13 fits each k on 11 document orders x 2 seeds (100, 200), one run per cell. This
script separates the two factors for every k, pair and profile of the symmetric design:

  - the spread across orders with the seed held fixed (SD over the 11 orders, per seed);
  - the spread across seeds with the order held fixed (pooled SD over the 2 seeds of each
    order) and the correlation between the two seeds across orders;
  - a two-way ANOVA without replication (order: 10 df; seed: 1 df; residual: 10 df). With
    one run per cell the order x seed interaction cannot be separated from run-to-run
    noise (LdaMulticore is not bit-for-bit deterministic), so it is the residual.

It also summarises the 22 values themselves, so that their spread can be compared with a
bootstrap interval: mean, median, SD, min, max, range, and the 2.5 and 97.5 percentiles
(numpy's default linear interpolation; with 22 values these lie between the two smallest
and the two largest). A range grows with the number of runs; a percentile interval does not.

Finally, it sets the canonical run (the published result) against the 22 solutions for
the difference the paper reports, Reddit-Telegram minus Guardian-Telegram cosine: its
value in the canonical run, its median and range over the 22, the rank of the canonical
run among them, and in how many runs Reddit-Telegram is the more similar of the two.

Input:  data/results/repetitions_symmetric.csv (script 13)
Output: data/results/order_seed_decomposition.csv
        data/results/order_seed_pair_gaps.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_RESULTS  # noqa: E402

SRC = os.path.join(DATA_RESULTS, "repetitions_symmetric.csv")
OUT = os.path.join(DATA_RESULTS, "order_seed_decomposition.csv")
OUT_GAPS = os.path.join(DATA_RESULTS, "order_seed_pair_gaps.csv")
CANONICAL = (0, 100)


def pair_gaps(r):
    """Reddit-Telegram minus Guardian-Telegram cosine: canonical run against the 22."""
    rows = []
    for (k, prof), g in r.groupby(['k', 'profile']):
        w = g.pivot_table(index=['order', 'seed'], columns='pair', values='cosine')
        gap = w['Reddit-Telegram'] - w['Guardian-Telegram']
        can = w.loc[CANONICAL]
        rows.append({
            'k': k, 'profile': prof, 'n_runs': len(w),
            'canonical_RT': can['Reddit-Telegram'], 'canonical_GT': can['Guardian-Telegram'],
            'canonical_gap': gap.loc[CANONICAL],
            'median_RT': w['Reddit-Telegram'].median(), 'median_GT': w['Guardian-Telegram'].median(),
            'median_gap': gap.median(), 'min_gap': gap.min(), 'max_gap': gap.max(),
            'canonical_gap_rank_from_largest': int((gap > gap.loc[CANONICAL]).sum()) + 1,
            'runs_RT_above_GT': int((gap > 0).sum()),
        })
    return pd.DataFrame(rows)


def decompose(g):
    w = g.pivot(index='order', columns='seed', values='cosine').sort_index()
    y = w.to_numpy()
    a, b = y.shape
    gm = y.mean()
    ss_order = b * ((y.mean(axis=1) - gm) ** 2).sum()
    ss_seed = a * ((y.mean(axis=0) - gm) ** 2).sum()
    ss_total = ((y - gm) ** 2).sum()
    ss_res = ss_total - ss_order - ss_seed
    df_o, df_s, df_r = a - 1, b - 1, (a - 1) * (b - 1)
    ms_o, ms_s, ms_r = ss_order / df_o, ss_seed / df_s, ss_res / df_r
    v = y.ravel()
    lo, hi = np.percentile(v, [2.5, 97.5])
    out = {
        'n_runs': v.size,
        'canonical_run': w.loc[CANONICAL[0], CANONICAL[1]],
        'mean': v.mean(), 'median': np.median(v), 'sd': v.std(ddof=1),
        'min': v.min(), 'max': v.max(), 'range': v.max() - v.min(),
        'p2.5': lo, 'p97.5': hi, 'p2.5_p97.5_width': hi - lo,
        'canonical_rank_from_lowest': int((v < w.loc[CANONICAL[0], CANONICAL[1]]).sum()) + 1,
    }
    for j, seed in enumerate(w.columns):
        out[f'sd_across_orders_seed{seed}'] = y[:, j].std(ddof=1)
    d = y[:, 0] - y[:, 1]
    out['sd_across_seeds_within_order'] = np.sqrt((d ** 2 / 2).mean())
    out['corr_seeds_across_orders'] = np.corrcoef(y[:, 0], y[:, 1])[0, 1]
    out.update({
        'ss_order': ss_order, 'ss_seed': ss_seed, 'ss_residual': ss_res,
        'share_order': ss_order / ss_total, 'share_seed': ss_seed / ss_total,
        'share_residual': ss_res / ss_total,
        'F_order': ms_o / ms_r, 'p_order': stats.f.sf(ms_o / ms_r, df_o, df_r),
        'F_seed': ms_s / ms_r, 'p_seed': stats.f.sf(ms_s / ms_r, df_s, df_r),
    })
    return out


def main():
    r = pd.read_csv(SRC)
    r = r[r['design'] == 'symmetric']
    rows = [{'k': k, 'profile': prof, 'pair': pair, **decompose(g)}
            for (k, prof, pair), g in r.groupby(['k', 'profile', 'pair'])]
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, float_format='%.4f')
    sel = out[(out.k == 10) & (out.profile == 'argmax') & (out.pair == 'Guardian-Telegram')].iloc[0]
    print("Guardian-Telegram, k=10, dominant-topic profile (as in the paper):")
    for c in out.columns[3:]:
        print(f"  {c:<32} {sel[c]:.4f}" if isinstance(sel[c], float) else f"  {c:<32} {sel[c]}")

    gaps = pair_gaps(r)
    gaps.to_csv(OUT_GAPS, index=False, float_format='%.4f')
    print("\nReddit-Telegram minus Guardian-Telegram, dominant-topic profile:")
    print(gaps[gaps.profile == 'argmax'].drop(columns='profile').to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nSaved: {OUT}\n       {OUT_GAPS}")


if __name__ == "__main__":
    main()
