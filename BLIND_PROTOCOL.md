# Section 4.8: the blind protocol

The order sensitivity of LDA (Section 3.5) meant that the submitted Section 4.8 rested
on one solution out of many. In the revision the claim was therefore re-tested on 22
solutions per k (11 document orders × seeds 100 and 200). The results were computed and
stored without being inspected, the criteria for reading them were fixed in advance,
and the results were then opened once. This file documents each step and states what
was and was not seen before the opening.

All times are local time in Italy (CEST, UTC+2), taken from the timestamps of the
files listed.

## Timeline

| When (2026) | Step | File |
|---|---|---|
| 4 Oct, 21:39:56 | Topic-classification rule written and frozen; unchanged since | `topic_category_rule.py` |
| 5 Oct, 09:18 | Coding kit prepared: the 31 topics of the two canonical models (k = 10 and k = 21), each described only by its 10 leading terms, under neutral codes. No charges, platforms or category labels were included. | — |
| 5 Oct, 10:21 and 10:26 | Sheets returned by two independent coders who worked separately. They are not among the original sentiment annotators. | `data/results/topic_category_agreement.csv` |
| 5 Oct | Agreement computed (below). Repetitions computed for the first time, sealed. | — |
| 9 Oct, 11:20:59 | Repetitions re-computed, still sealed. Per-channel contrasts were added, and topic assignment was aligned with `Sentiment_analysis/09` (see note 2). | `Topic_Modeling/14_topic_charge_repetitions.py` → `topic_charge_repetitions_effects.csv`, `topic_charge_repetitions_topics.csv` |
| 9 Oct, 11:22–11:28 | Per-channel contrasts and cluster-bootstrap intervals on the canonical model computed, sealed | `17_telegram_channels.py` → `telegram_channel_gradient.csv`; `18_cluster_bootstrap.py` → `cluster_bootstrap_topic_charge.csv` |
| 9–10 Oct | Reading criteria proposed and then fixed by the authors | below |
| 10 Oct, 13:00:51 | Criteria written into the summary script | `Topic_Modeling/15_topic_charge_summary.py` |
| 10 Oct, 13:01:01 | Summary script executed, once: the opening | output `topic_charge_summary.csv`, 13:01:03 |

The opening was recorded as follows:

```
criteria script sha256: 0f937d1e975c7528 | rule sha256: d40b68bc | opened at 2026-10-10 13:01:01
```

## Integrity

```
d40b68bc74b49aaa29d32378ab0bcfec1c3d17ee01cc7e817d11529a424848ac  topic_category_rule.py
0f937d1e975c752839b48c7bc48f8cd0fd1da29627585dff048a9dd858ae00d9  Topic_Modeling/15_topic_charge_summary.py
```

Both files are stored with LF line endings and are excluded from line-ending conversion
by `.gitattributes`, so `sha256sum` gives these values on any platform.

Running `python Topic_Modeling/15_topic_charge_summary.py` on the archived
`topic_charge_repetitions_effects.csv` reproduces `topic_charge_summary.csv` byte for
byte. It needs no corpus and no model.

## The rule and its validation

`topic_category_rule.py` classifies a topic as diplomatic/institutional, kinetic or
other from its 10 leading terms. It declares two alternatives in advance: `ALT_PROCESS`,
which adds diplomatic-process terms, and `ALT_TOP5`, which uses 5 leading terms
instead of 10.

The two coders were given the definition in words, not the rule. Their codings, topic
by topic, are in `data/results/topic_category_agreement.csv`, beside the
classifications of the rule and its two declared alternatives. On the 31 topics:

| | Cohen's κ | Agreement |
|---|---|---|
| coder 1 vs coder 2 | 0.79 | 27 / 31 |
| rule (main) vs coder 1 | 0.69 | 25 / 31 |
| rule (main) vs coder 2 | 0.89 | 29 / 31 |

On these 31 topics `ALT_PROCESS` gives the same classification as the main rule.

A third variant, `POSTHOC_KINETIC`, adds "weapon", "military" and "nuclear" to the
kinetic lexicon. It was defined after the coding was seen: both coders disagreed with
the rule on topics T05 and T09. It was not defined after any charge was seen. It is
reported as a post-hoc check and cannot change the verdict.

## Criteria fixed before the opening

These were fixed by the authors before the opening and written into the docstring and
constants of `15_topic_charge_summary.py`:

- **Rule.** The main rule decides; the variants are reported beside it.
- **Pooled.** The within-Telegram contrast (diplomatic minus kinetic) must be negative,
  with a 95% interval excluding zero, in at least 18 of the 22 repetitions, at k = 10
  and at k = 21. The actual counts are reported whatever the verdict.
- **Per channel.** WarTranslated, Slavyangrad and Clash Report are judged by direction
  only, because requiring each channel's interval to exclude zero would measure its
  size, not the effect. Out of the 22 repetitions, a channel:
  - **holds** if its contrast is negative in at least 18;
  - **fails by direction** if its contrast is positive in a majority (at least 12);
  - **lacks power** if its contrast is negative in a majority but in fewer than 18.
- **Class.**
  - If all three channels hold, the class claim stands.
  - If two hold and the third only lacks power, the claim stands and is reported as
    underdetermined in the third.
  - If any channel fails by direction, the class claim falls and the channels are
    reported separately.
  - If WarTranslated (55.5% of the Telegram subset) fails by direction, Section 4.8 is
    withdrawn.

Two details were not covered by these criteria. Both were made explicit in the script
before it was run:

1. A channel holds if it holds at both k, and fails by direction if it fails at either.
2. A repetition in which the channel has fewer than 30 items in either category counts
   as neither negative nor positive.

Neither changes the outcome: each channel has the same status at both k.

## What was seen before the opening

- **The canonical repetition** (canonical order, seed 100). It was compared with the
  cells of the submitted Section 4.8, which were already public, to establish that the
  scripts reproduce the submitted analysis. They do: the cells are reproduced exactly.
- **Cell sizes**, i.e. the number of items per channel and category and the number of
  repetitions with at least 30 items in a cell. These contain no charge values.

## What was not seen

The charge values of the other 21 repetitions, the per-channel contrasts and the
cluster-bootstrap intervals were not inspected before 13:01:01 on 10 October.

The header of `topic_category_rule.py` still reads "DRAFT for approval". The file was
frozen as written on 4 October and approved without changes, so the word was left in
place rather than altering a file whose hash had already been recorded.

The same header says the rule was "frozen before any affective
charge was computed or inspected for the models it will be applied to". That holds for
the 21 non-canonical repetitions. It does not hold for the two canonical models: their
charges were published in the submitted paper and were known to the author of the
rule, who also knew their leading terms. This is why the rule was validated against
coders who had seen neither.

## Result

The criteria were applied as fixed. The full counts are in `topic_charge_summary.csv`.

- **Pooled.** The contrast is negative in 22 of 22 repetitions at both k. Its interval
  excludes zero in 19 of 22 at k = 10 and in 17 of 22 at k = 21, below the threshold of
  18 fixed in advance.
- **WarTranslated holds:** negative in 22 of 22 at both k.
- **Slavyangrad holds:** negative in 22 of 22 at k = 10 and in 18 of 22 at k = 21.
- **Clash Report fails by direction:** positive in 21 of 21 estimable repetitions at
  k = 10 and in 22 of 22 at k = 21.

The class claim therefore falls by two independent routes. The script stops at the
pooled threshold. Applied regardless, the per-channel rules give the same outcome.

## Notes

1. **Median and range.** In `topic_charge_summary.csv`, the median and min/max include
   the non-estimable repetitions, which the counts exclude. The only one under the main
   rule is Clash Report, k = 10, order 1, seed 200 (18 kinetic items, estimate +11.68).
   Over the 21 estimable repetitions its median is 5.71 and its maximum 8.24.
2. **Topic assignment.** `get_document_topics` in gensim draws its initialisation from
   the model's random state, so the number of calls per document changes the
   assignments. `Sentiment_analysis/09`, which produced the submitted Section 4.8, calls
   it twice per document. Scripts 14, 17 and 18 do the same
   (`dominant_topics_as_published`); with a single call the canonical repetition would
   not reproduce the submitted cells.
3. **Reruns.** `LdaMulticore` is not bit-for-bit deterministic (parallel reduction,
   Section 3.5), so a re-run of script 14 may differ slightly from the archived output.
   The verdict was applied to the archived output.
4. **Coding.** The coders are anonymous; `topic_category_agreement.csv` identifies
   them only as coder 1 and coder 2. Its codes T01–T31 can be matched to the topics of
   the canonical models through their terms (`topic_charge_repetitions_topics.csv`,
   order 0, seed 100), and the κ values above can be recomputed from it with
   `sklearn.metrics.cohen_kappa_score`.
