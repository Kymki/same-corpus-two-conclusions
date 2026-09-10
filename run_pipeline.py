"""
Full pipeline, in execution order, for

    The Same Corpus, Two Conclusions:
    How Analyst Choices Determine Cross-Platform Findings

Each stage names the section of the paper it produces. Two things are worth
knowing before running it.

First, one stage is not automatic. `generate_annotation_sample.py` writes a blank
template that has to be annotated by hand; `02_sentiment_validation.py` and
`07_calibration_poststratified.py` then score it. The annotations used in the
paper are committed as data/results/validation_annotation_labeled.csv, so those
stages reproduce without re-annotating. If that file is absent, they are skipped
with a notice rather than failing.

Second, the raw corpora are not redistributed (see README), so stages 1-3 cannot
be run from a fresh clone. Everything downstream of
data/results/items_en_meta.csv can.

The figure scripts named fig2_ to fig7_ in Figures/ belong to an earlier,
sentence-level version of the analysis and do not correspond to any figure in the
published paper. They are deliberately not part of this pipeline.

Usage:
    python run_pipeline.py                 # run every stage
    python run_pipeline.py --list          # print the plan and exit
    python run_pipeline.py --from 8        # start at stage 8
"""
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "data", "results")
LABELLED = os.path.join(RESULTS, "validation_annotation_labeled.csv")

# (stage number, title, [(script relative to BASE, required input or None)])
PLAN = [
    (1, "Preprocessing: langdetect per document, sentence split, lemmatisation", [
        ("Lemmatization/01_pre-processing_1.1.py", None),
    ]),
    (2, "Corpus profile - Table 1, and the subword truncation check", [
        ("Lemmatization/02_corpus_analysis.py", None),
        ("Lemmatization/03_corpus_table_two_level.py", None),
        ("Sentiment_analysis/03_truncation_assessment.py", None),
    ]),
    (3, "Sentence-level sentiment classification (XLM-RoBERTa)", [
        ("Sentiment_analysis/01_sent.py", None),
    ]),
    (4, "Topic model: coherence sweep, then the final model and its labels", [
        ("Topic_Modeling/03_coherence.py", None),
        ("Topic_Modeling/01_topic.py", None),
        ("Topic_Modeling/02_labeling.py", None),
    ]),
    (5, "RQ1 - item-level re-analysis: aggregation rules and the length artefact "
        "(Sections 4.3, 4.4)", [
        ("Topic_Modeling/04_item_level_ksweep.py", None),
        ("Sentiment_analysis/05_item_sentiment_aggregation.py", None),
        ("Sentiment_analysis/06_item_topic_sentiment.py", None),
    ]),
    (6, "Validation and calibration - Section 4.2 (manual annotation required)", [
        ("Sentiment_analysis/generate_annotation_sample.py", None),
        ("Sentiment_analysis/02_sentiment_validation.py", LABELLED),
        ("Sentiment_analysis/07_calibration_poststratified.py", LABELLED),
    ]),
    (7, "RQ2 - measure dependence across continuous summaries (Section 4.5)", [
        ("Sentiment_analysis/08_continuous_measures.py", None),
    ]),
    (8, "RQ3 - the symmetric relevance criterion: the paper's central result "
        "(Section 4.6, Figure 3)", [
        ("Topic_Modeling/05_symmetric_filter_ksweep.py", None),
        ("Sentiment_analysis/09_symfilter_continuous_measures.py", None),
    ]),
    (9, "Robustness: length confound, length matching, chunked+balanced design, "
        "common collection window (Sections 4.6, 4.8)", [
        ("Topic_Modeling/06_length_confound_diagnostic.py", None),
        ("Topic_Modeling/07_length_matched_ksweep.py", None),
        ("Topic_Modeling/08_chunked_balanced_ksweep.py", None),
        ("Topic_Modeling/09_common_window_ksweep.py", None),
    ]),
    (10, "Table 2 - verbatim fragments with traceable identifiers (Section 4.9)", [
        ("Figures/fig8_extract_candidates.py", None),
        ("Figures/fig8_narrative_fragments.py", None),
    ]),
    (11, "Figures 1-4, each regenerated from the committed derived tables", [
        ("Figures/fig_length_artifact.py", None),
        ("Figures/fig_item_negativity.py", None),
        ("Figures/fig_preprocessing_regimes.py", None),
        ("Figures/fig_topic_charge.py", None),
    ]),
    (12, "Replace Reddit identifiers with keyed pseudonyms before publication", [
        ("pseudonymise_reddit_ids.py", None),
    ]),
]


def run(rel):
    path = os.path.join(BASE, rel)
    if not os.path.exists(path):
        sys.exit(f"ERROR: script not found: {rel}")
    print(f"\n--- {rel}")
    r = subprocess.run([sys.executable, path], cwd=os.path.dirname(path) or BASE, text=True)
    if r.returncode != 0:
        sys.exit(f"\nPipeline stopped: {rel} exited with code {r.returncode}")


def main():
    if "--list" in sys.argv:
        for n, title, scripts in PLAN:
            print(f"{n:>3}. {title}")
            for rel, _ in scripts:
                print(f"       {rel}")
        return

    start = 1
    if "--from" in sys.argv:
        try:
            start = int(sys.argv[sys.argv.index("--from") + 1])
        except (IndexError, ValueError):
            sys.exit("ERROR: --from needs a stage number, e.g. --from 8")

    skipped = []
    for n, title, scripts in PLAN:
        if n < start:
            continue
        print(f"\n{'=' * 74}\n STAGE {n}: {title}\n{'=' * 74}")
        for rel, needs in scripts:
            if needs and not os.path.exists(needs):
                print(f"\n--- {rel}\n    SKIPPED: requires {os.path.relpath(needs, BASE)},"
                      f" which does not exist yet.\n"
                      f"    Annotate the template written by generate_annotation_sample.py"
                      f" first.")
                skipped.append(rel)
                continue
            run(rel)

    print(f"\n{'=' * 74}")
    print(" PIPELINE COMPLETE")
    if skipped:
        print(" Skipped (awaiting manual annotation):")
        for s in skipped:
            print(f"   {s}")
    print(f" Figures and tables are in {os.path.relpath(RESULTS, BASE)}")
    print(f"{'=' * 74}")


if __name__ == "__main__":
    main()
