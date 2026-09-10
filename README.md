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

## Licence

Code is released under the MIT Licence (`LICENSE`). Documentation and derived
data tables are released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), consistent with the
journal's open-access terms.

## Citing

Please cite the article. If you use the code or the derived tables directly,
please also cite the archived release (DOI to be added on publication).
