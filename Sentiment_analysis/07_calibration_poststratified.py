"""
Post-stratified calibration of the negative probability (RP request).

Problem: the validation sample is stratified by source x predicted class
(40 sentences per cell => 1/3-1/3-1/3), while the corpus is ~60% predicted
negative. Raw calibration gaps and ECEs therefore describe the sample, not the
corpus: the low-probability region is heavily over-represented.

Fix: post-stratify. Each annotated sentence in cell (source s, predicted class c)
receives weight
        w(s,c) = corpus_share(s,c) / n_sampled(s,c)
with corpus_share(s,c) computed WITHIN source s on the English corpus, so the
weights reconstruct each source's true predicted-class composition.

Outputs (data/results/):
  calibration_poststratified.csv       gap and ECE per source, raw vs weighted
  calibration_bins_by_source.csv       weighted gap per probability bin x source

Sensitivity note also computed: the headline quantity in the paper is an
ITEM-level mean, but these gaps are estimated over SENTENCES sampled uniformly
within cells; Guardian sentences come disproportionately from long articles.
The column `gap_ps_itemweighted` additionally weights each sentence by
1/n_sentences(parent item), approximating an item-level calibration estimate.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_RESULTS

ANNOTATIONS = os.path.join(DATA_RESULTS, "validation_annotation_labeled.csv")
PREDICTIONS = os.path.join(DATA_RESULTS, "sentiment_results_multilingua.csv")
CELLS = os.path.join(DATA_RESULTS, "validation_sampling_cells.csv")
ITEMS = os.path.join(DATA_RESULTS, "items_en_meta.csv")
OUT_MAIN = os.path.join(DATA_RESULTS, "calibration_poststratified.csv")
OUT_BINS = os.path.join(DATA_RESULTS, "calibration_bins_by_source.csv")

SOURCES = ["Guardian", "Reddit_Commento", "Telegram"]
NAMES = {"Guardian": "Guardian", "Reddit_Commento": "Reddit", "Telegram": "Telegram"}
LABEL_MAP = {"negativo": "Negative", "neutro": "Neutral", "positivo": "Positive",
             "negative": "Negative", "neutral": "Neutral", "positive": "Positive"}
# Wide bins: cell counts are small once split by source, so 4 bins not 10.
BIN_EDGES = [0.0, 0.25, 0.50, 0.75, 1.0]

# Item-level platform means reported in the paper (Section 4.4), for the
# indicative calibration-corrected figures.
ITEM_MEANS = {"Guardian": 53.6, "Reddit": 56.8, "Telegram": 53.4}


def weighted_ece(pred, obs, w, edges=BIN_EDGES):
    """Weighted ECE: sum_b (W_b/W) * |weighted conf_b - weighted obs_b|."""
    idx = np.clip(np.digitize(pred, edges) - 1, 0, len(edges) - 2)
    total_w = w.sum()
    ece = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        wb = w[m].sum()
        conf = np.average(pred[m], weights=w[m])
        ob = np.average(obs[m], weights=w[m])
        ece += (wb / total_w) * abs(conf - ob)
    return ece


def main():
    ann = pd.read_csv(ANNOTATIONS)
    ann['manual_label'] = ann['manual_label'].map(lambda s: LABEL_MAP.get(str(s).strip().lower()))
    pred = pd.read_csv(PREDICTIONS, low_memory=False,
                       usecols=['id_originale', 'sentiment_label', 'neg_prob'])
    df = ann.merge(pred, on='id_originale', how='inner')
    df['sentiment_label'] = df['sentiment_label'].str.capitalize()
    df = df[df['fonte'].isin(SOURCES)].copy()
    df['obs_neg'] = (df['manual_label'] == 'Negative').astype(float)

    # --- post-stratification weights (within source) ---
    cells = pd.read_csv(CELLS)
    share_col = 'n_corpus_en' if 'n_corpus_en' in cells.columns else 'n_corpus'
    cells['within_source_share'] = cells.groupby('fonte')[share_col].transform(lambda x: x / x.sum())
    cells['w_per_sentence'] = cells['within_source_share'] / cells['n_sampled']
    df = df.merge(cells[['fonte', 'predicted_class', 'within_source_share', 'w_per_sentence']],
                  left_on=['fonte', 'sentiment_label'], right_on=['fonte', 'predicted_class'],
                  how='left')

    # --- item-level sensitivity weight: 1/n_sentences of the parent item ---
    items = pd.read_csv(ITEMS, usecols=['parent', 'n_sentences'])
    df['parent'] = df['id_originale'].astype(str).str.replace(r'_s\d+$', '', regex=True)
    df = df.merge(items, on='parent', how='left')
    df['n_sentences'] = df['n_sentences'].fillna(1.0)
    df['w_item'] = df['w_per_sentence'] / df['n_sentences']

    rng = np.random.default_rng(42)
    B = 2000

    def boot_gap(p, o, w, b=B):
        """Bootstrap CI for a weighted gap (resampling sentences within source)."""
        n = len(p)
        vals = []
        for _ in range(b):
            idx = rng.integers(0, n, n)
            ww = w[idx]
            if ww.sum() <= 0:
                continue
            vals.append(np.average(p[idx], weights=ww) - np.average(o[idx], weights=ww))
        return np.percentile(vals, [2.5, 97.5])

    rows = []
    boot_draws = {}
    for src in SOURCES:
        d = df[df['fonte'] == src]
        p, o = d['neg_prob'].to_numpy(float), d['obs_neg'].to_numpy(float)
        w = d['w_per_sentence'].to_numpy(float)
        wi = d['w_item'].to_numpy(float)

        gap_raw = p.mean() - o.mean()
        gap_ps = np.average(p, weights=w) - np.average(o, weights=w)
        gap_ps_item = np.average(p, weights=wi) - np.average(o, weights=wi)
        ci_lo, ci_hi = boot_gap(p, o, w)
        # keep draws to test the Guardian-Reddit differential directly
        boot_draws[src] = np.array([
            np.average(p[i], weights=w[i]) - np.average(o[i], weights=w[i])
            for i in (rng.integers(0, len(p), len(p)) for _ in range(B))
        ])
        ece_raw = weighted_ece(p, o, np.ones_like(p))
        ece_ps = weighted_ece(p, o, w)

        # Kish effective sample size: weighting costs precision, and the
        # item-weighted estimate costs a lot of it for long-form sources.
        ess_ps = w.sum() ** 2 / (w ** 2).sum()
        ess_item = wi.sum() ** 2 / (wi ** 2).sum()

        rows.append({
            'source': NAMES[src], 'n': len(d), 'n_parent_items': d['parent'].nunique(),
            'gap_raw': round(gap_raw, 4), 'gap_poststrat': round(gap_ps, 4),
            'gap_ps_itemweighted': round(gap_ps_item, 4),
            'gap_ps_ci_lo': round(ci_lo, 4), 'gap_ps_ci_hi': round(ci_hi, 4),
            'ECE_raw': round(ece_raw, 4), 'ECE_poststrat': round(ece_ps, 4),
            'ESS_poststrat': round(ess_ps, 1), 'ESS_itemweighted': round(ess_item, 1),
        })
    out = pd.DataFrame(rows)

    # indicative corrected platform means (same naive arithmetic RP used,
    # but with post-stratified gaps)
    out['item_mean_reported'] = out['source'].map(ITEM_MEANS)
    out['item_mean_corrected_naive'] = (out['item_mean_reported'] - out['gap_raw'] * 100).round(1)
    out['item_mean_corrected_ps'] = (out['item_mean_reported'] - out['gap_poststrat'] * 100).round(1)
    out.to_csv(OUT_MAIN, index=False)

    print("=== POST-STRATIFIED CALIBRATION (per source) ===")
    print(out[['source', 'n', 'n_parent_items', 'gap_raw', 'gap_poststrat',
               'gap_ps_ci_lo', 'gap_ps_ci_hi', 'gap_ps_itemweighted',
               'ECE_raw', 'ECE_poststrat',
               'ESS_poststrat', 'ESS_itemweighted']].to_string(index=False))

    # Is the Guardian-Reddit differential distinguishable from zero?
    diff = boot_draws['Guardian'] - boot_draws['Reddit_Commento']
    d_lo, d_hi = np.percentile(diff, [2.5, 97.5])
    print(f"\nGuardian-Reddit calibration differential (post-stratified): "
          f"{(out.set_index('source').loc['Guardian','gap_poststrat'] - out.set_index('source').loc['Reddit','gap_poststrat'])*100:.1f} pts, "
          f"bootstrap 95% CI [{d_lo*100:.1f}, {d_hi*100:.1f}] pts")
    print("  -> CI excludes 0" if (d_lo > 0 or d_hi < 0) else
          "  -> CI INCLUDES 0: the differential is not distinguishable from zero at n=360")
    print()
    spread_raw = out['item_mean_corrected_naive'].max() - out['item_mean_corrected_naive'].min()
    spread_ps = out['item_mean_corrected_ps'].max() - out['item_mean_corrected_ps'].min()
    print("Indicative correction of the Section 4.4 platform means "
          "(reported spread = 3.4 pts):")
    print(out[['source', 'item_mean_reported', 'item_mean_corrected_naive',
               'item_mean_corrected_ps']].to_string(index=False))
    print(f"  spread with RAW gaps        : {spread_raw:.1f} pts")
    print(f"  spread with POST-STRAT gaps : {spread_ps:.1f} pts")
    gd = out.set_index('source')
    print(f"  Guardian-Reddit differential: raw {abs(gd.loc['Guardian','gap_raw']-gd.loc['Reddit','gap_raw'])*100:.1f} pts"
          f"  ->  post-strat {abs(gd.loc['Guardian','gap_poststrat']-gd.loc['Reddit','gap_poststrat'])*100:.1f} pts")
    print(f"Saved: {OUT_MAIN}")

    # --- per-bin gap by source ---
    bin_rows = []
    df['bin'] = np.clip(np.digitize(df['neg_prob'], BIN_EDGES) - 1, 0, len(BIN_EDGES) - 2)
    for src in SOURCES:
        for b in range(len(BIN_EDGES) - 1):
            d = df[(df['fonte'] == src) & (df['bin'] == b)]
            if d.empty:
                continue
            w = d['w_per_sentence'].to_numpy(float)
            conf = np.average(d['neg_prob'], weights=w)
            ob = np.average(d['obs_neg'], weights=w)
            bin_rows.append({
                'source': NAMES[src],
                'bin': f"[{BIN_EDGES[b]:.2f},{BIN_EDGES[b+1]:.2f})",
                'n': len(d),
                'weight_share_of_source': round(w.sum() / df[df['fonte'] == src]['w_per_sentence'].sum(), 3),
                'mean_pred_neg': round(conf, 4), 'obs_neg_freq': round(ob, 4),
                'gap': round(conf - ob, 4),
            })
    bins = pd.DataFrame(bin_rows)
    bins.to_csv(OUT_BINS, index=False)
    print("\n=== GAP PER PROBABILITY BIN x SOURCE (post-stratified weights) ===")
    print(bins.to_string(index=False))
    print(f"Saved: {OUT_BINS}")


if __name__ == "__main__":
    main()
