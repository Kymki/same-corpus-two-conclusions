"""
Document order versus seed with ten seeds (Section 4.6; reviewer 2, third round).

Script 20 separated document order and seed on the 11 x 2 grid of script 13. With two
seeds the seed effect has one degree of freedom, and the correlation between the two
seeds across the 11 orders has a 95% interval of roughly +/- 0.6. This script refits the
symmetric design of script 13 at k = 10 and k = 21 on the same 11 orders (the canonical
order and permutations 1-10) with ten seeds, 100, 200, ..., 1000: 110 runs per k.

Seeds 100 and 200 are refitted rather than copied from script 13, so that every run here
comes from the same code. Their archived values in repetitions_symmetric.csv are the same
order and seed fitted a second time, and the differences between the two measure the
run-to-run noise of LdaMulticore, which is not bit-for-bit deterministic.

  python 21_order_seed_ten_seeds.py run       fit; rows are appended as each run ends,
                                              seed by seed, so an interrupted run still
                                              leaves complete 11-order designs
  python 21_order_seed_ten_seeds.py summary   decomposition, for every k, pair and profile:
      - SD across the 11 orders with the seed fixed, for each seed;
      - SD across the seeds with the order fixed, for each order;
      - two-way ANOVA, order x seed, one run per cell: order (10 df), seed (s - 1 df),
        residual ((10)(s - 1) df) = interaction plus run-to-run noise; variance
        components from the mean squares;
      - the correlation between every pair of seeds across the 11 orders, their mean,
        and a 95% bootstrap interval for the mean (orders resampled, 2,000 replicates);
      - run-to-run noise from the refitted seeds 100 and 200 (variance and SD, and how many
        refitted runs differ from the archived ones), and the interaction variance left
        once it is subtracted from the residual.

Outputs (data/results/):
  ten_seed_repetitions.csv     one row per k, order, seed, profile, pair
  ten_seed_decomposition.csv   the summary above, one row per k, profile, pair
  ten_seed_correlations.csv    seed-by-seed correlation matrices across the 11 orders
"""
import os
import sys
import time
import importlib.util
import itertools

import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models import LdaMulticore
from scipy import stats

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_RESULTS  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "s13", os.path.join(os.path.dirname(os.path.abspath(__file__)), "13_order_seed_repetitions.py"))
s13 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s13)

K_LIST = [10, 21]
SEEDS = list(range(100, 1001, 100))
OUT_RUNS = os.path.join(DATA_RESULTS, "ten_seed_repetitions.csv")
OUT_SUMMARY = os.path.join(DATA_RESULTS, "ten_seed_decomposition.csv")
OUT_CORR = os.path.join(DATA_RESULTS, "ten_seed_correlations.csv")
ARCHIVED = os.path.join(DATA_RESULTS, "repetitions_symmetric.csv")
N_BOOT = 2000
BOOT_SEED = 42


def run():
    items = s13.load_items()
    done = set()
    if os.path.exists(OUT_RUNS):
        prev = pd.read_csv(OUT_RUNS)
        done = set(zip(prev['seed'], prev['order'], prev['k']))
    print(f"items: {len(items)}; runs already done: {len(done)}", flush=True)
    bows = {}
    for o, perm in s13.orders(len(items)):
        d = items.iloc[perm].reset_index(drop=True)
        dic = corpora.Dictionary(d['tokens'])
        dic.filter_extremes(no_below=5, no_above=0.5)
        bows[o] = (dic, [dic.doc2bow(t) for t in d['tokens']], d['fonte'].to_numpy())
    for seed in SEEDS:
        for o, (dic, bow, src) in bows.items():
            for k in K_LIST:
                if (seed, o, k) in done:
                    continue
                t0 = time.time()
                m = LdaMulticore(corpus=bow, id2word=dic, num_topics=k, random_state=seed,
                                 passes=s13.LDA_PASSES, workers=s13.LDA_WORKERS)
                theta = np.zeros((len(bow), k))
                for i, b in enumerate(bow):
                    for t, p in m.get_document_topics(b, minimum_probability=0.0):
                        theta[i, t] = p
                dom = theta.argmax(axis=1)
                profiles = {
                    'argmax': np.vstack([np.bincount(dom[src == p], minlength=k) / (src == p).sum()
                                         for p in s13.PLATFORMS]),
                    'mean': np.vstack([theta[src == p].mean(axis=0) for p in s13.PLATFORMS]),
                }
                rows = [dict(k=k, order=o, seed=seed, profile=prof, pair=pair,
                             cosine=round(c, 4), js=round(j, 4))
                        for prof, mat in profiles.items()
                        for pair, (c, j) in s13.pair_sims(mat).items()]
                pd.DataFrame(rows).to_csv(OUT_RUNS, mode='a', index=False,
                                          header=not os.path.exists(OUT_RUNS))
                print(f"  seed={seed:>4} order={o:>2} k={k:>2}  {time.time() - t0:5.0f}s", flush=True)


def decompose(w):
    """w: orders x seeds matrix of cosines, one run per cell."""
    y = w.to_numpy()
    a, b = y.shape
    gm = y.mean()
    ss_o = b * ((y.mean(axis=1) - gm) ** 2).sum()
    ss_s = a * ((y.mean(axis=0) - gm) ** 2).sum()
    ss_t = ((y - gm) ** 2).sum()
    ss_r = ss_t - ss_o - ss_s
    df_o, df_s, df_r = a - 1, b - 1, (a - 1) * (b - 1)
    ms_o, ms_s, ms_r = ss_o / df_o, ss_s / df_s, ss_r / df_r
    var_o, var_s = max((ms_o - ms_r) / b, 0.0), max((ms_s - ms_r) / a, 0.0)
    out = {'n_orders': a, 'n_seeds': b,
           'sd_across_orders_mean_over_seeds': y.std(axis=0, ddof=1).mean(),
           'sd_across_seeds_mean_over_orders': y.std(axis=1, ddof=1).mean(),
           'ss_order': ss_o, 'ss_seed': ss_s, 'ss_residual': ss_r,
           'df_order': df_o, 'df_seed': df_s, 'df_residual': df_r,
           'F_order': ms_o / ms_r, 'p_order': stats.f.sf(ms_o / ms_r, df_o, df_r),
           'F_seed': ms_s / ms_r, 'p_seed': stats.f.sf(ms_s / ms_r, df_s, df_r),
           'var_order': var_o, 'var_seed': var_s, 'var_residual': ms_r,
           'share_var_order': var_o / (var_o + var_s + ms_r),
           'share_var_seed': var_s / (var_o + var_s + ms_r),
           'share_var_residual': ms_r / (var_o + var_s + ms_r)}
    for j, seed in enumerate(w.columns):
        out[f'sd_across_orders_seed{seed}'] = y[:, j].std(ddof=1)
    for i, order in enumerate(w.index):
        out[f'sd_across_seeds_order{order}'] = y[i, :].std(ddof=1)
    corr = np.corrcoef(y.T)
    iu = np.triu_indices(b, 1)
    out['mean_pairwise_seed_corr'] = corr[iu].mean()
    rng = np.random.default_rng(BOOT_SEED)
    boot = []
    for _ in range(N_BOOT):
        yb = y[rng.integers(0, a, a)]
        if np.any(yb.std(axis=0) == 0):
            continue
        boot.append(np.corrcoef(yb.T)[iu].mean())
    out['mean_pairwise_seed_corr_lo'], out['mean_pairwise_seed_corr_hi'] = np.percentile(boot, [2.5, 97.5])
    return out, pd.DataFrame(corr, index=w.columns, columns=w.columns)


def summary():
    r = pd.read_csv(OUT_RUNS)
    full = r.groupby(['seed', 'k']).order.nunique()
    seeds = sorted({s for (s, k), n in full.items() if n == 11 and all(full.get((s, kk), 0) == 11 for kk in K_LIST)})
    r = r[r.seed.isin(seeds)]
    arch = pd.read_csv(ARCHIVED)
    rows, mats = [], []
    for (k, prof, pair), g in r.groupby(['k', 'profile', 'pair']):
        w = g.pivot(index='order', columns='seed', values='cosine').sort_index()
        out, corr = decompose(w)
        a = arch[(arch.k == k) & (arch.profile == prof) & (arch.pair == pair)]
        dup = g.merge(a[['order', 'seed', 'cosine']], on=['order', 'seed'], suffixes=('', '_archived'))
        diff = dup['cosine'] - dup['cosine_archived']
        noise = (diff ** 2).mean() / 2 if len(dup) else np.nan
        out.update({'n_duplicate_runs': len(dup), 'n_duplicate_runs_differing': int((diff != 0).sum()),
                    'noise_var_run_to_run': noise, 'sd_run_to_run': np.sqrt(noise),
                    'sd_residual': np.sqrt(out['var_residual']),
                    'max_abs_diff_vs_archived': diff.abs().max() if len(dup) else np.nan,
                    'var_interaction': max(out['var_residual'] - noise, 0.0) if len(dup) else np.nan})
        rows.append({'k': k, 'profile': prof, 'pair': pair, 'seeds': ' '.join(map(str, seeds)), **out})
        mats.append(corr.assign(k=k, profile=prof, pair=pair).reset_index().rename(columns={'seed': 'seed_row'}))
    pd.DataFrame(rows).to_csv(OUT_SUMMARY, index=False, float_format='%.4f')
    pd.concat(mats).to_csv(OUT_CORR, index=False, float_format='%.3f')
    sel = pd.DataFrame(rows)
    sel = sel[(sel.profile == 'argmax') & (sel.pair == 'Guardian-Telegram')]
    cols = ['k', 'n_seeds', 'sd_across_orders_mean_over_seeds', 'sd_across_seeds_mean_over_orders',
            'F_order', 'p_order', 'F_seed', 'p_seed', 'share_var_order', 'share_var_seed',
            'share_var_residual', 'mean_pairwise_seed_corr', 'mean_pairwise_seed_corr_lo',
            'mean_pairwise_seed_corr_hi', 'sd_residual', 'sd_run_to_run', 'n_duplicate_runs_differing',
            'max_abs_diff_vs_archived']
    print(f"seeds with complete designs: {seeds}")
    print(sel[cols].T.to_string(header=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Saved: {OUT_SUMMARY}\n       {OUT_CORR}")


if __name__ == "__main__":
    {"run": run, "summary": summary}[sys.argv[1] if len(sys.argv) > 1 else "summary"]()
