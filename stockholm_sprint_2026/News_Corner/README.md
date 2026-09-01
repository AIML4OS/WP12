# News Corner Prototype

> **Status:** reported as an **experiment**, not a runnable prototype. The sprint produced a
> designed architecture and a series of model tests establishing feasibility; no packaged codebase
> is delivered with it. See [Experiment-Report.md](Experiment-Report.md) for the short write-up and
> [Report.md](Report.md) for the full test detail.

## Description

This project is a prototype for comparing news articles collected from RSS feeds with official publications released by National Statistical Institutes (NSIs). It uses an LLM to interpret and compare web content, allowing the system to identify when media articles refer to specific statistical releases and how accurately those releases are reported.

The goal is to create a module that can monitor official statistics in the media, extract quoted figures, compare narratives, and detect potential misinterpretations or distortions.

The system is designed to be reusable by any NSI.

## Input and Output

The system takes as input two sources of information: RSS feeds from news websites and official releases from national statistical institutes, also provided as RSS feeds. News articles represent how statistical information is reported in the media, while official releases provide the reference source, or ground truth.

The output is an alignment indication measuring how closely each news article matches the corresponding official release. This indicates whether the article refers to the correct statistical topic, reports the correct figures, preserves the right context, and reflects the main message of the source.

## Evaluation Criteria

The system is evaluated using synthetic news articles generated from official releases by national statistical institutes. For selected releases, articles are created with different alignment levels, such as accurate, partially accurate, missing context, or factually wrong.

Human evaluators score each synthetic article against the official release, creating a reference benchmark. The same articles are then injected into the system, which produces its own alignment score.

The evaluation is based on how well the system scores match human judgement. The system should assign high scores to accurate articles and lower scores to articles with wrong figures, missing context, incorrect reference periods, or misleading framing.

## Architecture

The prototype is implemented in Python, using standard libraries wherever possible to maximise reusability, portability, and long-term compatibility. The architecture is kept lightweight and modular, so that individual components can be replaced or extended without changing the overall workflow.

The system retrieves RSS feeds and official statistical releases, persists the collected content in a local database, and applies a limited number of LLM API calls for analysis. This design avoids redundant processing, supports reproducible results, and keeps API roundtrips to a minimum.

The LLM is used only where semantic interpretation is required, such as identifying the relevant statistical topic, extracting claims, comparing media articles with official releases, and producing the final alignment assessment. Deterministic Python logic is used for ingestion, parsing, deduplication, scheduling, database operations, and score aggregation.

This design keeps the prototype simple, reusable, and cost-aware while allowing the LLM to focus on the tasks where domain understanding and contextual reasoning provide the most value.

## Evaluation Summary

| Criterion | Assessment |
|---|---|
| Efficiency gain | High, compared to manual work |
| Reusability | High by design - intended so that any NSI can download, configure and run it, though the runnable implementation does not yet exist |
| Data accessibility | Medium/Low, due to paywalls on news articles |
| On-prem compatibility | High - the task requires only a chat completion endpoint, with no tool-calling requirement, so it places the lowest infrastructure demand of the three sprint use cases |
| Low-hanging fruit for NSIs | Medium/Low |
| Evaluation robustness | High by design - the benchmark method (synthetic articles at known alignment levels, scored by human evaluators) is specified but has not yet been executed at scale |
| Feasibility | High - demonstrated by the sprint tests |
| Lifespan | Medium, because of fast changes in media |
| Cost effectiveness | High, compared to manual work |

## Results

Seven tests were run (six with `gemma4-26b-moe` on SSPCloud Onyxia, one supplementary comparison
with a frontier commercial model), across English, Dutch and Slovenian. Six passed. The model handled
consistent reporting, topic mismatches, incorrect figures, incorrect reference months and a
Dutch-release-versus-English-article comparison correctly. It did not detect a reference-*year*
mismatch in the Slovenian case where the quarter label matched - a prompt-design finding, since the
supplementary test showed the case was solvable.

See [Experiment-Report.md](Experiment-Report.md) for the summary and [Report.md](Report.md) for
dimension-level detail and the evaluation prompt.

## Next steps

1. Execute the specified benchmark: synthetic articles at known alignment levels, scored by human
   evaluators, measuring system agreement with human judgement.
2. Revise the prompt to force explicit full-reference-period extraction before judgement, and re-run
   the Slovenian case as a regression test.
3. Implement the ingestion and storage modules so the experiment becomes reproducible.
