"""
Third reading of the sentences on which the two annotators disagree (Section 4.2; reviewer 2).

The 360 validation sentences were labelled by two annotators. The paper uses the first
as the reference (manual_label) and reports their agreement (Cohen's kappa 0.824). They
disagree on 42 sentences. Those 42 were read by a third reader, who saw only each
sentence and its source, under neutral codes and in shuffled order, and labelled them
negative, neutral or positive. The third reader is not one of the two annotators and
had not seen their labels.

Rule, fixed before the third reading: the label is the majority of the three; if all
three differ, the sentence is unresolved and is reported as such. The script reports

  - how many of the 42 the third reader assigns to annotator 1's label and how many to
    annotator 2's. Had the third reader sided systematically with annotator 2, the
    choice of annotator 1 as the reference would have mattered;
  - accuracy, macro-F1, corpus-weighted accuracy and calibration recomputed with the
    adjudicated labels (unresolved sentences excluded), beside the same quantities
    with annotator 1 as the reference (the submitted values). Argmax metrics and
    calibration are computed as in 02_sentiment_validation.py, with its functions;
    the post-stratified calibration by running 07_calibration_poststratified.py
    unchanged on each set of labels.

Inputs (data/results/):
  validation_annotation_labeled.csv     annotators 1 and 2, as in the paper
  validation_third_reader.csv           third reader's labels for the 42 sentences
  sentiment_results_multilingua.csv     sentence-level predictions (not in the
                                        repository, as for scripts 02 and 07)
  validation_sampling_cells.csv, items_en_meta.csv

Outputs (data/results/):
  validation_adjudicated_labels.csv            all 360 sentences: the three labels, the
                                               adjudicated label and its status
  validation_third_reader_summary.csv          the 42 by outcome, overall and by source
  validation_metrics_adjudicated.csv           annotator 1 vs adjudicated, metric by metric
  calibration_poststratified_adjudicated.csv   script 07's main table, adjudicated labels
"""
import os
import sys
import tempfile
import importlib.util

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_RESULTS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LABELED = os.path.join(DATA_RESULTS, "validation_annotation_labeled.csv")
THIRD = os.path.join(DATA_RESULTS, "validation_third_reader.csv")
PREDICTIONS = os.path.join(DATA_RESULTS, "sentiment_results_multilingua.csv")
CELLS = os.path.join(DATA_RESULTS, "validation_sampling_cells.csv")
OUT_LABELS = os.path.join(DATA_RESULTS, "validation_adjudicated_labels.csv")
OUT_SUMMARY = os.path.join(DATA_RESULTS, "validation_third_reader_summary.csv")
OUT_METRICS = os.path.join(DATA_RESULTS, "validation_metrics_adjudicated.csv")
OUT_PS = os.path.join(DATA_RESULTS, "calibration_poststratified_adjudicated.csv")

STATUS = ["third reader with annotator 1", "third reader with annotator 2", "unresolved"]


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v02 = load("02_sentiment_validation")
c07 = load("07_calibration_poststratified")
VALID = v02.VALID_LABELS


def adjudicate():
    lab = pd.read_csv(LABELED)
    third = pd.read_csv(THIRD)[['id_originale', 'manual_label_3']]
    d = lab.merge(third, on='id_originale', how='left')
    a1, a2, a3 = (d[c].map(v02.normalize_label) for c in ('manual_label', 'manual_label_2', 'manual_label_3'))
    disputed = a1 != a2
    if set(d.loc[disputed, 'id_originale']) != set(third['id_originale']):
        sys.exit("ERROR: the third reader's sentences are not the 42 disputed ones")
    d['status'] = 'annotators agree'
    d.loc[disputed & (a3 == a1), 'status'] = STATUS[0]
    d.loc[disputed & (a3 == a2), 'status'] = STATUS[1]
    d.loc[disputed & (a3 != a1) & (a3 != a2), 'status'] = STATUS[2]
    d['adjudicated_label'] = a1.where(d['status'].isin(['annotators agree', STATUS[0]]),
                                      a2.where(d['status'] == STATUS[1]))
    return d


def argmax_and_calibration(ann, pred, cells):
    """The quantities of 02_sentiment_validation.py, computed with its own functions."""
    d = ann.merge(pred, on='id_originale', how='inner')
    y_true = d['manual_label']
    y_pred = d['sentiment_label'].astype(str).str.capitalize()
    out = {('ALL', 'n'): len(d),
           ('ALL', 'accuracy'): accuracy_score(y_true, y_pred),
           ('ALL', 'macro_F1'): f1_score(y_true, y_pred, average='macro', labels=VALID)}
    share_col = 'corpus_share_en' if 'corpus_share_en' in cells.columns else 'corpus_share'
    d['correct'] = (y_true == y_pred)
    cell_acc = (d.groupby(['fonte', 'sentiment_label'])['correct']
                .agg(['mean', 'count']).reset_index()
                .rename(columns={'mean': 'cell_accuracy', 'count': 'n_annotated'}))
    merged = cells.merge(cell_acc, left_on=['fonte', 'predicted_class'],
                         right_on=['fonte', 'sentiment_label'], how='left')
    covered = merged.dropna(subset=['cell_accuracy'])
    w = covered[share_col] / covered[share_col].sum()
    out[('ALL', 'corpus_weighted_accuracy')] = float((w * covered['cell_accuracy']).sum())
    for lab, f1 in zip(VALID, f1_score(y_true, y_pred, average=None, labels=VALID)):
        out[('ALL', f'F1_{lab}')] = f1
    d['pred_neg'] = d['neg_prob'].astype(float)
    d['obs_neg'] = (y_true == 'Negative').astype(int)
    for scope, g in [('ALL', d)] + [(s, d[d['fonte'] == s]) for s in sorted(d['fonte'].unique())]:
        if len(g) < 10:
            continue
        ece, _ = v02.expected_calibration_error(g['pred_neg'].values, g['obs_neg'].values)
        out[(scope, 'n_calibration')] = len(g)
        out[(scope, 'mean_pred_neg')] = g['pred_neg'].mean()
        out[(scope, 'obs_neg_freq')] = g['obs_neg'].mean()
        out[(scope, 'gap_pred_minus_obs')] = g['pred_neg'].mean() - g['obs_neg'].mean()
        out[(scope, 'ECE')] = ece
    return out


def poststratified(ann, out_main):
    """Script 07, unchanged, on the given labels."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "annotations.csv")
        ann.to_csv(path, index=False)
        c07.ANNOTATIONS, c07.OUT_MAIN = path, out_main
        c07.OUT_BINS = os.path.join(tmp, "bins.csv")
        c07.main()
    return pd.read_csv(out_main)


def main():
    d = adjudicate()
    d[['id_originale', 'fonte', 'manual_label', 'manual_label_2', 'manual_label_3',
       'status', 'adjudicated_label']].to_csv(OUT_LABELS, index=False)

    dis = d[d['status'] != 'annotators agree']
    summary = (pd.crosstab(dis['fonte'], dis['status'], margins=True, margins_name='all')
               .reindex(columns=STATUS + ['all'], fill_value=0))
    summary.index.name = 'source'
    summary.to_csv(OUT_SUMMARY)
    print(f"Disputed sentences: {len(dis)} of {len(d)}")
    print(summary.to_string(), "\n")

    pred = pd.read_csv(PREDICTIONS, low_memory=False,
                       usecols=['id_originale', 'sentiment_label', 'neg_prob'])
    cells = pd.read_csv(CELLS)
    refs = {
        'annotator 1 (submitted)': d.assign(manual_label=d['manual_label'].map(v02.normalize_label)),
        'adjudicated': d[d['adjudicated_label'].notna()].assign(manual_label=d['adjudicated_label']),
    }
    rows = {}
    with tempfile.TemporaryDirectory() as tmp:
        for ref, ann in refs.items():
            ann = ann[['id_originale', 'fonte', 'manual_label']]
            for key, v in argmax_and_calibration(ann, pred, cells).items():
                rows.setdefault(key, {})[ref] = v
            ps_path = OUT_PS if ref == 'adjudicated' else os.path.join(tmp, "ps_submitted.csv")
            ps = poststratified(ann, ps_path)
            for r in ps.itertuples():
                for col in ('gap_poststrat', 'ECE_poststrat', 'item_mean_corrected_ps'):
                    rows.setdefault((r.source, col), {})[ref] = getattr(r, col)
    m = pd.DataFrame([{'scope': s, 'metric': k, **v} for (s, k), v in rows.items()])
    m['difference'] = m['adjudicated'] - m['annotator 1 (submitted)']
    m.to_csv(OUT_METRICS, index=False, float_format='%.4f')
    with pd.option_context('display.width', 200, 'display.max_rows', 200):
        print("\n" + m.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved: {OUT_LABELS}\n       {OUT_SUMMARY}\n       {OUT_METRICS}\n       {OUT_PS}")


if __name__ == "__main__":
    main()
