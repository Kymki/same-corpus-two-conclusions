"""
Validates the XLM-RoBERTa sentiment predictions against REAL manual annotations.

Unit = SENTENCE (what the classifier scores). Reads the blind annotation file
from generate_annotation_sample.py (after a human fills 'manual_label'), joins
to the model predictions by id_originale, and reports:

  ARGMAX validation (the discrete label):
  - raw sample accuracy and macro-F1 (macro-F1 essential given class imbalance)
  - corpus-weighted accuracy (post-stratification on the ENGLISH corpus shares)
  - per-class precision/recall/F1 (LaTeX export) + confusion matrix
  - Cohen's kappa between two annotators, if 'manual_label_2' is filled

  CALIBRATION (the continuous negative probability — the paper's headline
  measure, so it must be validated, not just the argmax label):
  - (a) mean predicted negative probability vs observed frequency of "Negative"
        in the annotations, overall and per source
  - (b) reliability diagram (binned) + Expected Calibration Error (ECE)
  - (c) ECE disaggregated per source (decisive: if the calibration gap is
        constant across sources it cancels in cross-platform comparisons; if it
        is source-dependent it threatens the 4.4 / 4.7 comparisons)
"""
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DATA_RESULTS

# Prefer the human-labeled file (never overwritten by the sample generator);
# fall back to the blank TODO template.
_LABELED = os.path.join(DATA_RESULTS, "validation_annotation_labeled.csv")
_TODO = os.path.join(DATA_RESULTS, "validation_annotation_TODO.csv")
INPUT_ANNOTATIONS_CSV = _LABELED if os.path.exists(_LABELED) else _TODO
INPUT_PREDICTIONS_CSV = os.path.join(DATA_RESULTS, "sentiment_results_multilingua.csv")
OUTPUT_LATEX_REPORT = os.path.join(DATA_RESULTS, "sentiment_metrics.tex")
OUTPUT_CALIB_CSV = os.path.join(DATA_RESULTS, "sentiment_calibration.csv")
OUTPUT_RELIABILITY_PNG = os.path.join(DATA_RESULTS, "sentiment_reliability_diagram.png")

VALID_LABELS = ["Negative", "Neutral", "Positive"]
N_BINS = 10

# Annotators may label in Italian or English; normalise both to the canonical set.
LABEL_MAP = {
    "negativo": "Negative", "neutro": "Neutral", "positivo": "Positive",
    "negative": "Negative", "neutral": "Neutral", "positive": "Positive",
}


def normalize_label(s):
    return LABEL_MAP.get(str(s).strip().lower(), str(s).strip().capitalize())


def expected_calibration_error(pred_prob, is_positive, n_bins=N_BINS):
    """ECE for a binary target (here: is the sentence truly 'Negative')."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(pred_prob, bins) - 1, 0, n_bins - 1)
    rows, ece, n = [], 0.0, len(pred_prob)
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        conf = pred_prob[m].mean()          # mean predicted neg probability
        obs = is_positive[m].mean()         # observed fraction truly negative
        w = m.sum() / n
        ece += w * abs(conf - obs)
        rows.append({'bin_lo': round(bins[b], 2), 'bin_hi': round(bins[b + 1], 2),
                     'n': int(m.sum()), 'mean_pred_neg': round(conf, 4),
                     'obs_neg_freq': round(obs, 4)})
    return ece, pd.DataFrame(rows)


def validate_sentiment():
    print(f"Loading manual annotations from: {INPUT_ANNOTATIONS_CSV}")
    if not os.path.exists(INPUT_ANNOTATIONS_CSV):
        print("ERROR: annotation file not found. Run generate_annotation_sample.py first.")
        return

    print(f"(reading annotations from {os.path.basename(INPUT_ANNOTATIONS_CSV)})")
    df_ann = pd.read_csv(INPUT_ANNOTATIONS_CSV)
    df_ann['manual_label'] = df_ann['manual_label'].map(normalize_label)
    df_ann = df_ann[df_ann['manual_label'].isin(VALID_LABELS)].copy()
    if df_ann.empty:
        print("WARNING: no valid manual labels found yet (fill 'manual_label' with "
              "Negative/Neutral/Positive). Nothing to validate.")
        return
    print(f"Manually annotated sentences found: {len(df_ann)}")

    df_pred = pd.read_csv(INPUT_PREDICTIONS_CSV, low_memory=False,
                          usecols=['id_originale', 'sentiment_label', 'neg_prob'])
    df_valid = df_ann.merge(df_pred, on='id_originale', how='inner')
    if df_valid.empty:
        print("ERROR: no overlap between annotations and predictions.")
        return

    y_true = df_valid['manual_label']
    y_pred = df_valid['sentiment_label'].astype(str).str.capitalize()

    # ===== ARGMAX VALIDATION =====
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', labels=VALID_LABELS)
    print(f"\n[ARGMAX] sample size: {len(df_valid)}")
    print(f"[ARGMAX] raw sample accuracy (not corpus accuracy): {accuracy:.2%}")
    print(f"[ARGMAX] macro-F1: {macro_f1:.4f}")

    # Post-stratification on ENGLISH corpus shares
    cells_path = os.path.join(DATA_RESULTS, "validation_sampling_cells.csv")
    weighted_note = ""
    if os.path.exists(cells_path):
        cells = pd.read_csv(cells_path)
        share_col = 'corpus_share_en' if 'corpus_share_en' in cells.columns else 'corpus_share'
        df_valid['correct'] = (y_true == y_pred)
        cell_acc = (df_valid.groupby(['fonte', 'sentiment_label'])['correct']
                    .agg(['mean', 'count']).reset_index()
                    .rename(columns={'mean': 'cell_accuracy', 'count': 'n_annotated'}))
        merged = cells.merge(cell_acc, left_on=['fonte', 'predicted_class'],
                             right_on=['fonte', 'sentiment_label'], how='left')
        covered = merged.dropna(subset=['cell_accuracy'])
        w = covered[share_col] / covered[share_col].sum()
        weighted_acc = float((w * covered['cell_accuracy']).sum())
        print(f"[ARGMAX] corpus-weighted accuracy (English corpus): {weighted_acc:.2%}")
        weighted_note = f" Corpus-weighted accuracy (English) = {weighted_acc:.3f}."

    print("\n[ARGMAX] classification report:")
    print(classification_report(y_true, y_pred, target_names=VALID_LABELS, labels=VALID_LABELS))

    kappa_note = ""
    if 'manual_label_2' in df_valid.columns:
        second = df_valid['manual_label_2'].map(normalize_label)
        mask = second.isin(VALID_LABELS)
        if mask.sum() > 0:
            kappa = cohen_kappa_score(y_true[mask], second[mask])
            print(f"[ARGMAX] inter-annotator Cohen's kappa (n={mask.sum()}): {kappa:.4f}")
            kappa_note = f" Cohen's kappa = {kappa:.3f} (n={mask.sum()})."
        else:
            print("[ARGMAX] single-annotator validation (no second annotator labels).")

    try:
        report_df = pd.DataFrame(classification_report(
            y_true, y_pred, output_dict=True, target_names=VALID_LABELS, labels=VALID_LABELS)).transpose()
        caption = (f"Sentiment validation on {len(df_valid)} manually annotated English "
                   f"sentences, sampled with source x predicted-class stratification "
                   f"(raw accuracy = {accuracy:.3f}, macro-F1 = {macro_f1:.3f})."
                   + weighted_note + kappa_note)
        report_df.to_latex(OUTPUT_LATEX_REPORT, float_format="%.3f",
                           caption=caption, label="tab:sentiment_validation")
        print(f"LaTeX report saved to: {OUTPUT_LATEX_REPORT}")
    except Exception as e:
        print(f"Error during LaTeX export: {e}")

    cm = confusion_matrix(y_true, y_pred, labels=VALID_LABELS)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=VALID_LABELS, yticklabels=VALID_LABELS)
    plt.xlabel('Predicted Label'); plt.ylabel('True Label (Manual)'); plt.title('Sentiment Confusion Matrix')
    plt.savefig(os.path.join(DATA_RESULTS, "sentiment_confusion_matrix.png"), dpi=300)
    plt.close()

    # ===== CALIBRATION OF THE NEGATIVE PROBABILITY =====
    df_valid['pred_neg'] = df_valid['neg_prob'].astype(float)
    df_valid['obs_neg'] = (y_true == 'Negative').astype(int)

    print("\n[CALIBRATION] (a) mean predicted neg-prob vs observed 'Negative' frequency")
    calib_rows = []
    scopes = [('ALL', df_valid)] + [(s, df_valid[df_valid['fonte'] == s]) for s in sorted(df_valid['fonte'].unique())]
    for scope, d in scopes:
        if len(d) < 10:
            continue
        mean_pred = d['pred_neg'].mean()
        obs = d['obs_neg'].mean()
        ece, _ = expected_calibration_error(d['pred_neg'].values, d['obs_neg'].values)
        calib_rows.append({'scope': scope, 'n': len(d),
                           'mean_pred_neg': round(mean_pred, 4),
                           'obs_neg_freq': round(obs, 4),
                           'gap_pred_minus_obs': round(mean_pred - obs, 4),
                           'ECE': round(ece, 4)})
        print(f"  {scope:<16} n={len(d):<4} mean_pred={mean_pred:.3f}  observed={obs:.3f}  "
              f"gap={mean_pred-obs:+.3f}  ECE={ece:.3f}")
    calib_df = pd.DataFrame(calib_rows)
    calib_df.to_csv(OUTPUT_CALIB_CSV, index=False)
    print(f"[CALIBRATION] table saved to: {OUTPUT_CALIB_CSV}")
    print("[CALIBRATION] (c) decisive check: compare the per-source ECE / gap rows above. "
          "Constant gap => cross-platform comparisons survive; source-dependent gap => threat to 4.4/4.7.")

    # (b) reliability diagram (overall)
    _, rel = expected_calibration_error(df_valid['pred_neg'].values, df_valid['obs_neg'].values)
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], '--', color='#95a5a6', label='Perfect calibration')
    plt.plot(rel['mean_pred_neg'], rel['obs_neg_freq'], 'o-', color='#c0392b',
             label='Model (binned)')
    for _, r in rel.iterrows():
        plt.annotate(f"n={int(r['n'])}", (r['mean_pred_neg'], r['obs_neg_freq']),
                     fontsize=7, xytext=(3, -8), textcoords='offset points')
    plt.xlabel('Mean predicted negative probability')
    plt.ylabel("Observed 'Negative' frequency")
    plt.title('Reliability Diagram — negative class')
    plt.xlim(0, 1); plt.ylim(0, 1); plt.legend(loc='upper left'); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_RELIABILITY_PNG, dpi=300, facecolor='white')
    plt.close()
    print(f"[CALIBRATION] reliability diagram saved to: {OUTPUT_RELIABILITY_PNG}")


if __name__ == "__main__":
    validate_sentiment()
