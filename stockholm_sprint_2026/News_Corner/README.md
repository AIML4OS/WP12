# News Corner Prototype

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
| Reusability | High, just download, configure, and run |
| Data accessibility | Medium/Low, due to paywalls on news articles |
| On-prem compatibility | High |
| Low-hanging fruit for NSIs | Medium/Low |
| Evaluation robustness | High, using a benchmark dataset |
| Feasibility | High |
| Lifespan | Medium, because of fast changes in media |
| Cost effectiveness | High, compared to manual work |

## TODO

