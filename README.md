# The Same Corpus, Two Conclusions

Replication package for:

> Laquintana, M.; Pareschi, R. **The Same Corpus, Two Conclusions: How Analyst
> Choices Determine Cross-Platform Findings.** *Information*, 2026.

The paper shows that three routine analyst choices — the unit of analysis, the
summary measure, and the criterion delimiting each subcorpus — can each determine
the substantive conclusion drawn about platforms, and reports and retracts a
finding of its own to demonstrate it.

---

## Start here: the comparison the paper is about

Two files carry the central result (Section 4.6, Figure 3). They have identical
schemas, so they can be read side by side:

| File | Regime |
|---|---|
| `data/results/item_ksweep_similarity.csv` | relevance filter applied to **The Guardian only** |
| `data/results/symfilter_ksweep_similarity.csv` | the **same criterion applied to all three sources** |

Columns: `k`, `pair`, `cosine`, `cos_ci_lo`, `cos_ci_hi`, `js`, `js_ci_lo`,
`js_ci_hi`. Confidence intervals are 95% bootstrap over items, 1,000 replicates.

Reading them against each other reproduces the retraction directly. Under the
asymmetric filter, *Guardian–Telegram* is the most similar pair at k = 5, 7, 10
and 15. Under the symmetric criterion, *Reddit–Telegram* becomes the most similar
pair at k = 7, 10, 15 and 21, with cosine 0.97, 0.96, 0.95 and 0.75 against 0.43,
0.40, 0.39 and 0.42 for Guardian–Telegram. Nothing changes between the two files
except the symmetry of the relevance criterion.

These are the figures of the canonical run (document order of the manifest, seed 100),
the one the submitted analysis used. LDA is sensitive to document order, so
`Topic_Modeling/13_order_seed_repetitions.py` refits each k on 22 order × seed
combinations. Across them, Reddit–Telegram is more similar than Guardian–Telegram in
20, 19, 21 and 21 of 22 at k = 7, 10, 15 and 21 (between 17 and 22 of 22 over the whole
grid, k = 5 to 30), but the median difference between the
two (0.29, 0.23, 0.11 and 0.30) is smaller than in the canonical run (0.54, 0.56, 0.56
and 0.33), which has the largest difference of the 22 at k = 7, 10 and 15. Per-k values
are in `data/results/order_seed_pair_gaps.csv` (`Topic_Modeling/20_order_seed_decomposition.py`).
With ten seeds on the same 11 orders (`Topic_Modeling/21_order_seed_ten_seeds.py`, 110
runs per k), Reddit–Telegram is more similar in 99 and 96 of 110 at k = 10 and 21, with
median differences of 0.21 and 0.24; the canonical difference is the largest of the 22
and the third largest of the 110 at k = 10. Of the variance of the Guardian–Telegram
similarity, the order accounts for 1% at k = 10, the seed for 0% and the residual for
99% (0%, 0% and 100% at k = 21; variance components from the mean squares). Refitted
runs reproduce the archived ones to within 0.0011, so the residual is the interaction
of the two: a solution is fixed by order and seed together, not by either alone. Per-k
values are in `data/results/ten_seed_decomposition.csv`.

A third file, `data/results/commonwindow_ksweep_similarity.csv`, repeats the
symmetric sweep on the 40-day window common to all three sources, to rule out the
unequal collection windows as an explanation.

To regenerate Figure 3 from these two files:

```bash
python Figures/fig_preprocessing_regimes.py
```

It also writes `fig_preprocessing_regimes_values.csv`, listing every plotted
value with the SHA-256 of the file it came from, so each point in the figure is
traceable to a file in this repository.

---

## What is in this repository

**Code.** The full pipeline: collection (`Build_Dataset/`), preprocessing and
lemmatisation (`Lemmatization/`), sentiment classification and validation
(`Sentiment_analysis/`), topic modelling and cross-platform similarity
(`Topic_Modeling/`), figures (`Figures/`), and Elasticsearch indexing
(`Elasticsearch/`).

**Derived tables** (`data/results/`), including:

| File | Section |
|---|---|
| `item_ksweep_similarity.csv`, `symfilter_ksweep_similarity.csv` | 4.6 |
| `commonwindow_*.csv` | 4.6, common-window check |
| `items_en_meta.csv` | item-level sentiment, all sections |
| `item_sentiment_aggregation_compare.csv`, `item_sentiment_length_artifact.csv` | 4.3 |
| `validation_annotation_labeled.csv`, `sentiment_calibration.csv`, `calibration_poststratified.csv` | 4.2 |
| `continuous_measures_by_*.csv`, `symfilter_continuous_by_*.csv` | 4.5, 4.8 |
| `length_confound_diagnostic.csv`, `lengthmatched_*.csv`, `chunked_balanced_*.csv` | robustness checks |
| `document_topics_items_k*.csv`, `item_topics_k*.txt`, `symfilter_topic_terms_k*.txt` | topic assignments and top terms |

**Figure data.** `data/results/figures/` holds the rendered figures and, beside
each one, a `*_values.csv` with the numbers it plots.

**Guardian metadata** (`data/raw/Guardian_articles_metadata.csv`): identifiers,
URLs, titles and dates only, without full text. Article text can be re-fetched
with `Build_Dataset/Papers/05_guardian.py`.

---

## What is *not* in this repository

**The raw corpora.** Reddit and Telegram terms of service make redistribution
problematic, and Guardian full text is subject to the publisher's terms. The
derived tables above are sufficient to reproduce every reported number; they are
not sufficient to re-run preprocessing from scratch.

**Reddit comment identifiers.** Section 3.2 reports Reddit material "without
thread context sufficient to locate a poster". A comment id resolves to the
comment and therefore to its author, so the ids are replaced throughout by keyed
pseudonyms of the form `r_<12 hex>`, produced by `pseudonymise_reddit_ids.py`.
Joins across files are unaffected — the pseudonym is stable — but the original
ids cannot be recovered from this repository. They are available from the
corresponding author on request, as stated in the Data Availability statement.

**Identifiers for the quoted social-media excerpts** in Table 2, for the same
reason. `data/results/fig8_selected_ids.json` retains the Guardian and BBC URLs
and records the withheld entries explicitly.

---

## Figure generators, and the older scripts

The scripts in `Figures/` named `fig2_`–`fig8_` belong to an earlier,
sentence-level version of the analysis and do **not** correspond to the figures
in the published paper. `fig8_extract_candidates.py` and
`fig8_narrative_fragments.py` are the exception: they still produce Table 2. The
generators for the published figures are:

| Figure | Script |
|---|---|
| 1 — length artefact | `Figures/fig_length_artifact.py` |
| 2 — item-level negativity | `Figures/fig_item_negativity.py` |
| 3 — the two preprocessing regimes | `Figures/fig_preprocessing_regimes.py` |
| 4 — topic-conditional charge | `Figures/fig_topic_charge.py` |

Each writes a `*_values.csv` beside the image, listing the numbers it plotted, so
every point in every figure is traceable to a file in this repository without
re-running anything.

---

## Reproducing the analysis

Requires Python 3.11, the spaCy models `en_core_web_sm` and `it_core_news_sm`,
and a CUDA GPU for the sentiment stage (it runs on CPU, more slowly).

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download it_core_news_sm
python run_pipeline.py
```

`run_pipeline.py` prints the stage order and stops with a clear message at the
one point that is not automatic: the validation sample
(`generate_annotation_sample.py`) writes a blank template that has to be
annotated by hand before `02_sentiment_validation.py` can score it. The
annotations used in the paper are committed as
`data/results/validation_annotation_labeled.csv`, so that stage reproduces
without re-annotating.

Because the raw corpora are not distributed, the stages before sentiment
classification cannot be run from this repository alone. Everything downstream of
`data/results/items_en_meta.csv` can.

**Note.** If preprocessing is ever re-run from raw collections, real Reddit
identifiers re-enter the derived tables; `pseudonymise_reddit_ids.py` must be run
again afterwards. It is idempotent, and `--check` reports without modifying
anything.

---

## Added in the revision

**Document order.** LDA in gensim is sensitive to the order of the documents, so the
order is part of the analysis. `item_order.py` fixes it from
`data/results/item_order_manifest.csv`, the order used for the submitted results.
Every topic-model script on the analysed corpus applies it, including those added in
the revision. Where the items go beyond that corpus (script 16, all 3,000 Guardian
articles), they are ordered by the SHA-256 of source and lemmatised text instead, which
needs no identifiers. `LdaMulticore` is not bit-for-bit deterministic (parallel
reduction): in our checks, re-runs differed by up to about 0.015 in cosine.
`Topic_Modeling/verify_lda_order.py` runs the three checks behind these statements
(`verify_lda_order.csv`).

**Section 4.8 was re-tested blind.** The protocol, its timeline, the hashes and what was
seen before the opening are in [`BLIND_PROTOCOL.md`](BLIND_PROTOCOL.md).

| Script | Output (`data/results/`) | What it does |
|---|---|---|
| `Topic_Modeling/10_chunked_rebalanced_ksweep.py` | `chunked_rebalanced_ksweep_similarity.csv` | chunked design, training capped at the smallest platform |
| `Topic_Modeling/11_bertopic_symmetric_ksweep.py` | `bertopic_symfilter_*.csv` | BERTopic on the symmetric subset |
| `Topic_Modeling/12_length_designs_table.py` | `table_2_1.tex`, `table_2_1_summary.csv`, `design_costs.csv` | the similarity designs side by side, and what each costs |
| `Topic_Modeling/13_order_seed_repetitions.py` | `repetitions_*.csv` | 22 solutions per k (11 document orders × 2 seeds), k up to 30 |
| `Topic_Modeling/20_order_seed_decomposition.py` | `order_seed_decomposition.csv`, `order_seed_pair_gaps.csv` | spread due to document order versus seed; the 22 solutions summarised beside the bootstrap intervals; the canonical run set against the 22 |
| `Topic_Modeling/21_order_seed_ten_seeds.py` | `ten_seed_*.csv` | order and seed separated on 11 orders × 10 seeds at k = 10 and 21 |
| `Topic_Modeling/14_topic_charge_repetitions.py`, `15_topic_charge_summary.py`, `topic_category_rule.py` | `topic_charge_*.csv`, `topic_category_agreement.csv` | Section 4.8 on the same 22 solutions; see `BLIND_PROTOCOL.md` |
| `Topic_Modeling/16_symfilter_all_guardian.py` (with `Lemmatization/01_pre-processing_1.1.py --all-guardian`) | `symfilter_all_guardian_*.csv` | the symmetric relevance filter applied to all 3,000 Guardian articles |
| `Topic_Modeling/17_telegram_channels.py` | `telegram_channel_counts.csv`, `telegram_channel_gradient.csv` | Telegram by channel |
| `Topic_Modeling/18_cluster_bootstrap.py`, `build_item_clusters.py` | `item_clusters.csv`, `cluster_bootstrap_*.csv` | bootstrap by Reddit thread, Telegram channel × week, Guardian week |
| `Topic_Modeling/19_coherence_extended_grid.py` | `symfilter_coherence_extended.csv` | C_v coherence at k = 25 and 30 (single run) |
| `Sentiment_analysis/10_common_window_sentiment.py` | `common_window_sentiment.csv` | sentiment within the window common to the three sources |
| `Sentiment_analysis/11_guardian_short_sample.py` | `guardian_short_sample.csv` | Guardian articles cut to a random sample of their own sentences, at the item lengths of Reddit and Telegram |
| `Sentiment_analysis/12_third_reader_adjudication.py` | `validation_third_reader*.csv`, `validation_adjudicated_labels.csv`, `validation_metrics_adjudicated.csv`, `calibration_poststratified_adjudicated.csv` | third reading of the 42 sentences on which the two annotators disagree; validation metrics with the adjudicated labels |
| `Sentiment_analysis/13_continuous_measures_k21.py` | `continuous_measures_by_topic_platform_k21.csv`, `continuous_measures_inversions_k21.csv` | the three continuous measures by topic and platform on the k = 21 model, with the topics whose platform ordering changes across measures |
| `Figures/fig_order_repetitions.py` | `figures/fig_order_repetitions.*` | Section 4.6 across orders and seeds |

`Figures/fig_topic_charge.py` was revised. Its panel (b) now shows the within-Telegram
contrast pooled and by channel across the 22 solutions, instead of Telegram as a single
source. `Figures/fig_length_artifact.py` was revised to plot the share of items classified
negative and the mean negative probability in two panels instead of on one axis; the
values it plots are unchanged.

`data/results/figures/submitted_version/` keeps these two figures as they appeared in the
submitted manuscript. The numbers they plot are in `fig_topic_charge_values.csv` beside
them and, for the length figure, in `figures/fig_length_artifact_values.csv`, which the
revision did not change. No other result file of the submitted analysis was changed.

`build_item_clusters.py` needs the raw collections and the private pseudonym map, so it
cannot be run from this repository. It is included to document how the clusters were
built; its output contains only pseudonyms and anonymous labels.

---

## Licence

Code is released under the MIT Licence (`LICENSE`). Documentation and derived
data tables are released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), consistent with the
journal's open-access terms.

## Citing

Please cite the article. If you use the code or the derived tables directly,
please also cite the archived release on Zenodo: the article cites release v1.0 by its
version DOI.
