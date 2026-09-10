# Data

This directory holds the artefacts that every number, table and figure in the
paper derives from. The top-level `README.md` explains the package as a whole;
this file describes the data files themselves.

## Included

| Path | Content |
|---|---|
| `raw/Guardian_articles_metadata.csv` | Dehydrated Guardian dataset: id, URL, title, publication date and body word count for all 3,000 collected articles (official Guardian Content API, query `ukraine OR russia`, from 2025-05-25). Full body text is **not** redistributed (publisher terms); re-fetch with `Build_Dataset/Papers/05_guardian.py` or request it from the authors. |
| `results/item_ksweep_similarity.csv` | Cross-platform similarity, k-sweep, **asymmetric** regime — the relevance filter applied to The Guardian only (Section 4.6, Figure 3a). |
| `results/symfilter_ksweep_similarity.csv` | The same sweep under the **symmetric** criterion, applied to all three sources (Section 4.6, Figure 3b). Reading these two against each other reproduces the retraction. |
| `results/commonwindow_*.csv` | The symmetric sweep restricted to the 40-day window common to all three sources, plus its coherence values and corpus profile (Section 4.6). |
| `results/items_en_meta.csv` | One row per item: source, sentence count, mean negative/neutral/positive probability, and the two argmax-based aggregations. The join key for most of the analysis. |
| `results/item_coherence_values.csv` | C_v coherence per k for the **item-level** model over k ∈ {5, 7, 10, 15, 21}. This is the model selection reported in Section 3.5: the optimum is k = 10 at C_v = 0.488. |
| `results/item_topics_k*.txt`, `results/document_topics_items_k*.csv` | Top terms and per-item dominant topic for each k of the item-level sweep. |
| `results/item_sentiment_aggregation_compare.csv`, `results/item_sentiment_length_artifact.csv`, `results/item_sentiment_length_correlation.csv` | The aggregation-rule comparison and the length-artefact diagnostics of Section 4.3. |
| `results/fig_length_artifact_sentence_pool.csv` | Every Guardian sentence-level probability vector (81,582 rows). Input to the Figure 1 simulation, so that it regenerates without the full sentiment output. |
| `results/validation_annotation_labeled.csv` | The **360**-sentence validation sample as annotated: source × predicted-class stratified, 40 per cell, with both annotators' labels (Sections 3.4 and 4.2). |
| `results/validation_annotation_TODO.csv` | The same sample as issued to the annotators, with the labels blank and the model predictions withheld. |
| `results/validation_sampling_cells.csv` | Cell sizes and corpus shares used to post-stratify accuracy. |
| `results/sentiment_calibration.csv`, `results/calibration_poststratified.csv`, `results/calibration_bins_by_source.csv` | Calibration of the predicted negative probability, in aggregate and per source (Section 4.2). |
| `results/continuous_measures_by_*.csv` | The three continuous summaries per source and per topic × platform on the full corpus (Section 4.5). |
| `results/symfilter_continuous_by_*.csv`, `results/symfilter_topic_terms_k*.txt` | The same measures recomputed on the symmetric subset, at k = 10 and k = 21, with the retrained model's top terms (Section 4.8, Figure 4). |
| `results/length_confound_diagnostic.csv`, `results/lengthmatched_ksweep_similarity.csv`, `results/chunked_balanced_ksweep_similarity.csv` | The length-confound diagnostics and the two length-controlled designs. |
| `results/*.tex` | LaTeX tables: corpus volumetrics, the 512-subword-token truncation assessment, and the sentiment validation metrics. |
| `results/fig8_candidates.csv`, `results/fig8_selected_ids.json` | Candidate excerpts for Table 2 and the document shown for each source. |
| `results/figures/` | The published figures (PNG + PDF) and, beside each, a `*_values.csv` with the numbers it plots. |

## Superseded files, kept for transparency

The analysis moved from sentences to items as the unit of comparison (Section
3.3). These files are the **sentence-level** stage, which Section 3.5 records as
having produced a lower-coherence model. They are not the model the paper
reports:

- `results/coherence_values_en.csv` — C_v per k over a different grid (k = 5..23)
- `results/optimal_k_en.json` — the sentence-level optimum, **k = 7**, not the
  k = 10 of Section 3.5
- `results/lda_topics_en.txt` — top terms of that k = 7 model

The Italian counterparts (`coherence_values_it.csv`, `optimal_k_it.json`,
`lda_topics_it.txt`) are a by-product of the multilingual pipeline. Per-document
language detection assigns under 1% of the corpus to Italian, almost entirely
from one platform, and that material is **excluded** from the analysis and
reported as a limitation (Sections 4.1 and 5.5). No Italian result is reported in
the paper.

`Figures/fig1_pipeline_architecture.tex` and the scripts `Figures/fig2_` to
`fig7_` likewise belong to the earlier sentence-level analysis and do not
correspond to any figure in the published paper. `fig8_extract_candidates.py` and
`fig8_narrative_fragments.py` are current: they produce Table 2.

## Not included

- **Raw full-text collections** (`raw/*_contenuti_articoli_estratti.csv`, the
  Reddit and Telegram dumps): redistribution of full article and message text is
  restricted by the platforms' terms of service. Re-collect with the scrapers in
  `Build_Dataset/`, or contact the authors.
- **The consolidated preprocessed corpus** (`processed/dati_testuali_preproc_consolidati.csv`)
  and the **sentence-level sentiment output** (`results/sentiment_results_multilingua.csv`):
  derived full-text data, regenerable from the raw collections.
- **Reddit comment identifiers.** Replaced throughout by keyed pseudonyms of the
  form `r_<12 hex>`, including in `fig8_selected_ids.json`, where the Reddit row
  of Table 2 is addressed by its pseudonym rather than by the comment id. A
  comment id resolves to the comment and therefore to its author, which is
  inconsistent with Section 3.2. The salt is not in this repository; the original
  identifiers are available from the corresponding author on request.

Telegram message ids are retained: Section 3.2 names the channels and treats them
as publications, since channel posts carry no individual author attribution.

## Reproducing

```
python run_pipeline.py --list      # the stage order, without running anything
python run_pipeline.py             # run it
```

Because the raw collections are not distributed, the stages before sentiment
classification cannot be run from a fresh clone. Everything downstream of
`results/items_en_meta.csv` can. One stage is not automatic: the validation
sample has to be annotated by hand between `generate_annotation_sample.py` and
`02_sentiment_validation.py`, and the annotations used in the paper are committed
so that it reproduces without re-annotating.
