# News Corner - Statistical Media Consistency

_Experiment report - WP12 Stockholm sprint, 16-18 June 2026._

> **Note on status.** Unlike the other two sprint outputs, News Corner is reported here as an
> **experiment rather than a runnable prototype**. The sprint work consisted of a designed
> architecture and a series of model tests establishing whether the core task is feasible at all.
> No packaged, runnable codebase is delivered with it. The value of the contribution lies in the
> test evidence and in the architecture that the evidence justifies.

## The question

Can a large language model determine whether a news article accurately reflects the official
statistical release it reports on?

This matters because monitoring how official statistics are reported in the media is currently
manual work, and because the judgement required is genuinely difficult: an article and a release may
share a topic and still differ on figures, reference period, attribution or framing. A system that
gets this right must compare two texts - often in two different languages - along several dimensions
at once and be specific about which dimension failed.

## What was tested

Newspaper articles were compared against official NSI releases along five binary dimensions:
**topic match**, **figures accuracy**, **reference period accuracy**, **source attribution** and
**framing consistency**. A test passed when the model's overall consistency judgement matched the
expected one.

Seven tests were run: six with `gemma4-26b-moe` hosted on SSPCloud Onyxia, and one supplementary
comparison with a frontier commercial model. Source pairs were CBS / NL Times and SURS / RTVSLO;
languages were English, Dutch and Slovenian.

| # | Model | Source pair | Language | Expected | Result | Outcome |
|---|---|---|---|---|---|---|
| 1 | GEMMA4-26B-MOE | CBS vs NL Times | English | Consistent | Consistent | Passed |
| 1B | GEMMA4-26B-MOE | CBS vs NL Times | English | Consistent | Consistent | Passed |
| 2 | GEMMA4-26B-MOE | CBS vs NL Times | English | Inconsistent | Inconsistent | Passed |
| 3 | GEMMA4-26B-MOE | CBS vs NL Times | English | Inconsistent | Inconsistent | Passed |
| 4 | GEMMA4-26B-MOE | CBS vs NL Times | Dutch / English | Consistent | Consistent | Passed |
| 5 | GEMMA4-26B-MOE | SURS vs RTVSLO | Slovenian | Inconsistent | Consistent | Failed |
| 5B | ChatGPT 5.5 | SURS vs RTVSLO | Slovenian | Inconsistent | Inconsistent | Passed |

Full test detail, dimension-level results and the evaluation prompt are in [Report.md](Report.md).

## What the experiment shows is possible

The model performed very well given the difficulty of the task. It correctly identified consistent
reporting, topic mismatches, incorrect figures and incorrect reference months, and it handled a
**Dutch release against an English article** correctly - a directly relevant result, since NSIs
publish in national languages while media coverage is often in English.

The one failure is informative rather than disqualifying. In the Slovenian case the article and the
release shared a topic and a quarter label but referred to different years (Q3 2022 vs Q3 2025), and
the model accepted them as consistent. The supplementary test showed the case was solvable, which
locates the problem precisely: reference-period comparison must be forced to include the year, not
just the period label. That is a prompt-design finding, and an actionable one - future prompts should
require explicit extraction and comparison of the full reference period before any consistency
judgement is made.

Taken together: an open-weights model hosted on infrastructure an NSI could realistically operate is
a credible basis for statistical media monitoring, with a known weak spot that structured prompting
can address. Note that all test material was public, as the shared environment permits only
non-sensitive data; these results say nothing about processing sensitive material.

## Architecture designed

The tests were run manually, but the sprint also produced a full architecture specification for the
operational system the tests justify - eleven modules covering configuration, scheduling, feed
ingestion, storage (SQLite plus raw files), deduplication, candidate matching, LLM analysis,
alignment scoring, benchmarking, reporting and a graphical interface. See
[Draft-architecture.md](Draft-architecture.md).

Two design principles in that specification are worth carrying into other WP12 work:

- **The LLM is used only where semantic interpretation is required** - identifying the statistical
  topic, extracting claims, comparing article to release, producing the alignment assessment.
  Ingestion, parsing, deduplication, scheduling, storage and score aggregation are deterministic Python.
- **API round-trips are minimised by design** - content is persisted locally and candidate pairs are
  generated deterministically before any model call, so cost scales with genuinely new comparisons
  rather than with feed volume.

## Evaluation summary

| Criterion | Assessment |
|---|---|
| Efficiency gain | High, compared to manual work |
| Reusability | High - designed so that any NSI can download, configure and run it, though the runnable implementation does not yet exist |
| Data accessibility | Medium/low, due to paywalls on news articles |
| On-prem compatibility | High - the task requires only a chat completion endpoint, with no tool-calling requirement, so it places the lowest infrastructure demand of the three sprint use cases |
| Low-hanging fruit for NSIs | Medium/low |
| Evaluation robustness | High by design - the benchmark method (synthetic articles at known alignment levels, scored by human evaluators as a reference) is specified and is the most rigorous evaluation design produced during the sprint; it has not yet been executed at scale |
| Feasibility | High - demonstrated by the tests |
| Lifespan | Medium, because of fast changes in media |
| Cost effectiveness | High, compared to manual work |

## Limitations

The test set is small: six model tests and one comparison test. That is enough to establish
feasibility and to locate one specific weakness, but not enough to support claims about reliability
across languages, topics or statistical domains. No runnable, reproducible implementation was
produced during the sprint, so the results cannot currently be re-executed by a third party - this
is the main thing standing between the experiment and a prototype.

## Next steps

1. Execute the specified benchmark: generate synthetic articles at known alignment levels, score
   them with human evaluators, and measure system agreement with human judgement.
2. Revise the prompt to force explicit full-reference-period extraction before judgement, and re-run
   the Slovenian case as a regression test.
3. Implement the ingestion and storage modules so the experiment becomes reproducible.
