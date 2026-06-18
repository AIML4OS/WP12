# NSI News Alignment Architecture

## Overview

This document describes the architecture of a local prototype system for comparing newspaper articles with official National Statistical Institute (NSI) releases.

The system retrieves RSS feeds from newspapers and NSIs, stores metadata and raw content locally, removes duplicates, generates plausible article-release candidate pairs, uses an LLM for semantic comparison, computes alignment scores, and produces operational reports for inspection. A graphical user interface can be used to inspect feeds, browse results, compare matched items, and search across collected content. A separate benchmarking workflow evaluates system performance against human-scored test cases.

The architecture is designed to be:

- modular;
- reproducible;
- cost-aware;
- country-aware;
- suitable for local prototyping;
- extensible into a larger monitoring tool;
- usable through both command-line workflows and a graphical interface.

---

## 1. Configuration Module

The configuration module loads and validates the project configuration.

It defines:

- RSS feeds from newspapers;
- RSS feeds from NSIs;
- source type: `newspaper` or `nsi`;
- country or nationality of each feed;
- language of each feed;
- fetching frequency;
- local database path;
- raw storage paths;
- matching parameters;
- LLM provider and model settings.

The nationality of each RSS feed is explicitly stored so that news articles can be compared primarily with official releases from the same national context.

---

## 2. Scheduling Module

The scheduling module determines when each RSS feed should be fetched.

Different sources can have different fetching schedules. For example, newspaper feeds may be checked frequently, while NSI feeds may be checked less often.

The scheduler reads the fetching rules from the configuration file and identifies which feeds are due for retrieval.

---

## 3. Feed Ingestion Module

The feed ingestion module retrieves RSS feeds from newspapers and NSIs.

It extracts basic metadata from each feed item, including:

- title;
- URL;
- publication date;
- summary;
- source name;
- source type;
- country;
- language.

The module should rely on standard Python libraries where possible

---

## 4. Storage Module

The storage module persists metadata in a local database and stores article or release content as raw files on disk.

The local database stores structured information such as:

- configured sources;
- feed entries;
- article and release metadata;
- file paths to raw content;
- file paths to processed content;
- content hashes;
- candidate matches;
- LLM responses;
- alignment scores;
- benchmarking records;
- search indexes or searchable text references for GUI queries.

Raw article and release content is stored separately in files, for example as HTML, XML, JSON, or plain text. This keeps the database lightweight while preserving the original collected content for reproducibility, auditing, and future reprocessing.

A possible storage structure is:

```text
data/
├── alignment.db
├── raw/
│   ├── news/
│   └── nsi/
├── processed/
│   ├── news/
│   └── nsi/
└── llm_cache/
```

The database references these files through stable paths. This design avoids storing large text or HTML blobs directly in the database while still allowing the system to retrieve, inspect, and reprocess the original content when needed.

SQLite is sufficient for the prototype, as it supports local execution, simple deployment, and reproducible experiments.

---

## 5. Deduplication Module

The deduplication module prevents repeated processing of the same articles or releases.

Duplicates can be detected using:

- canonical URL;
- content hash;
- title;
- publication date;
- source identifier.

This module helps avoid duplicate records, repeated LLM calls, and distorted reporting metrics.

---

## 6. Candidate Matching Module

The candidate matching module selects possible article-release pairs before LLM analysis.

Matching is based on deterministic signals such as:

- same country;
- compatible language;
- close publication dates;
- shared keywords;
- statistical topic;
- source type;
- title similarity;
- explicit mention of the NSI or indicator.

This module reduces the number of comparisons and ensures that only plausible matches are sent to the LLM.

---

## 7. LLM Analysis Module

The LLM analysis module performs semantic interpretation.

It is responsible for:

- identifying the statistical topic;
- extracting statistical claims;
- comparing a news article with an official release;
- detecting missing context;
- identifying incorrect figures;
- assessing narrative or framing differences;
- producing a structured alignment assessment.

The LLM should return structured JSON so that downstream modules can process the output deterministically.

---

## 8. Alignment Scoring Module

The alignment scoring module converts deterministic checks and LLM output into an alignment indication.

The score may consider:

- topic relevance;
- numerical accuracy;
- reference period accuracy;
- geographical accuracy;
- context preservation;
- source attribution;
- framing consistency.

The final output can be expressed as both a numerical score and a qualitative label, such as:

- high alignment;
- partial alignment;
- low alignment;
- misleading or factually wrong;
- unrelated.

---

## 9. Benchmarking Module

The benchmarking module supports system validation and benchmark testing.

This module is run separately from the normal operational pipeline. It does not run automatically after the Alignment Scoring Module. Instead, it provides an independent workflow for testing whether the system's scores agree with human judgement.

Synthetic news articles are generated from selected official NSI releases with different levels of alignment, such as:

- accurate;
- partially accurate;
- missing context;
- factually wrong.

Human evaluators score these synthetic articles against the official releases. The benchmarking workflow then runs the system on the same benchmark cases and compares the system's alignment scores with the human benchmark.

The benchmarking module may reuse the same LLM analysis and scoring components used by the operational pipeline, but it is invoked as a separate mode with separate inputs and outputs.

---

## 10. Reporting Module

The reporting module produces outputs for inspection and analysis.

It can generate:

- matched article-release pairs;
- alignment scores;
- explanations of detected issues;
- logs of processed feeds;
- benchmark summaries, when the separate benchmarking workflow is run;
- dashboard-ready tables or files.

The reporting layer should remain separate from the core pipeline so that the prototype can later support different interfaces, such as command-line output, CSV exports, dashboards, or APIs.

---

## 11. Graphical User Interface Module

The graphical user interface module provides a local interface for inspecting the system state and exploring results.

The GUI should be a presentation and interaction layer on top of the database, raw storage, processed content, reporting outputs, and search indexes. It should not contain the core ingestion, matching, LLM analysis, scoring, or benchmarking logic. Instead, it should call existing modules or read their outputs through a stable application interface.

The GUI can support:

- viewing configured feeds;
- showing feed type, country, language, URL, and fetching frequency;
- displaying last fetch time, next scheduled fetch time, ingestion status, and errors;
- browsing ingested newspaper articles and NSI releases;
- browsing candidate article-release matches;
- displaying alignment scores and qualitative labels;
- opening a detailed comparison view for a matched article and release;
- showing extracted claims, detected issues, missing context, incorrect figures, and LLM explanations;
- searching by keyword across titles, summaries, processed text, source names, countries, languages, topics, dates, and score labels;
- filtering results by source type, country, date range, topic, alignment score, and processing status;
- exporting visible results to CSV or JSON;
- linking back to raw and processed files for audit and reproducibility.

For the prototype, the GUI can be implemented as a local web interface. It can use SQLite queries directly for structured filters and SQLite full-text search, where available, for keyword search over titles, summaries, and processed content.

The GUI should be optional. The command-line pipeline should remain fully usable without it.

---

## Configuration File

The system should use an external configuration file, for example `config.json`.

Example:

```json
{
  "database": {
    "path": "data/alignment.db"
  },
  "storage": {
    "raw_path": "data/raw",
    "processed_path": "data/processed",
    "llm_cache_path": "data/llm_cache"
  },
  "countries": [
    {
      "country_code": "LU",
      "country_name": "Luxembourg",
      "timezone": "Europe/Luxembourg",
      "languages": ["fr", "de", "en", "lb"]
    }
  ],
  "feeds": [
    {
      "name": "Example NSI Feed",
      "type": "nsi",
      "country_code": "LU",
      "language": "fr",
      "url": "https://example-nsi.lu/rss",
      "fetching": {
        "mode": "daily",
        "times": ["08:00", "14:00"]
      }
    },
    {
      "name": "Example Newspaper Feed",
      "type": "newspaper",
      "country_code": "LU",
      "language": "en",
      "url": "https://example-news.lu/rss",
      "fetching": {
        "mode": "interval",
        "minutes": 60
      }
    }
  ],
  "matching": {
    "candidate_window_days": 7,
    "same_country_priority": true,
    "cross_country_matching": false,
    "max_candidates_per_release": 10
  },
  "llm": {
    "provider": "openai",
    "model": "gpt-4.1-mini",
    "temperature": 0,
    "cache_results": true
  }
}
```

---

## Suggested Project Structure

```text
nsi_news_alignment/
│
├── README.md
├── ARCHITECTURE.md
├── config.json
├── main.py
│
├── data/
│   ├── alignment.db
│   ├── raw/
│   │   ├── news/
│   │   └── nsi/
│   ├── processed/
│   │   ├── news/
│   │   └── nsi/
│   └── llm_cache/
│
├── src/
│   ├── config_loader.py
│   ├── scheduler.py
│   ├── feeds.py
│   ├── database.py
│   ├── raw_storage.py
│   ├── deduplication.py
│   ├── matching.py
│   ├── llm_analysis.py
│   ├── alignment.py
│   ├── benchmarking.py
│   ├── reporting.py
│   ├── search_index.py
│   └── gui.py
│
└── prompts/
    └── alignment_prompt.txt
```

---

## Execution Flow

The prototype can expose a simple command-line workflow for the operational pipeline:

```bash
python main.py ingest
python main.py match
python main.py score
python main.py report
```

Or a full operational run:

```bash
python main.py run
```

The full operational run should:

1. Load and validate the configuration.
2. Determine which feeds are due for fetching.
3. Retrieve RSS feeds.
4. Store metadata in the local database.
5. Store raw article and release content as files.
6. Remove or ignore duplicates.
7. Generate candidate article-release pairs.
8. Run LLM analysis only for unresolved candidates.
9. Store alignment results.
10. Produce operational reports.

The benchmarking workflow is separate and should be invoked explicitly, for example:

```bash
python main.py benchmark
```

The benchmarking run should:

1. Load benchmark cases or generate synthetic cases from selected NSI releases.
2. Load human benchmark scores.
3. Run the system on the benchmark cases.
4. Compare system scores with human scores.
5. Produce benchmark metrics and benchmark reports.

Benchmarking is therefore not part of `python main.py run` unless explicitly enabled by a separate command-line option.

The graphical interface can be started separately, for example:

```bash
python main.py gui
```

The GUI should read from the existing database, processed content, search index, and reporting outputs. It should allow users to inspect feeds, browse matches, view alignment results, and search collected content without requiring the operational pipeline or benchmarking workflow to be running at the same time.

---

## LLM Roundtrip Minimisation

To reduce unnecessary API calls, the system should:

- cache LLM responses;
- hash article-release pairs before analysis;
- reuse previous results when inputs have not changed;
- perform deterministic candidate matching before LLM analysis;
- send compact text representations rather than full raw HTML;
- store prompt version and model name with each LLM output.

A possible cache table is:

```text
llm_cache
- id
- input_hash
- prompt_version
- model_name
- response_json
- created_at
```

---

## Summary

This architecture keeps the prototype simple, reusable, and cost-aware while allowing the LLM to focus on the tasks where domain understanding and contextual reasoning provide the most value.

The operational pipeline, benchmarking workflow, and GUI are intentionally separated. The operational pipeline monitors and scores real article-release pairs, the benchmarking workflow validates the system against human-scored test cases, and the GUI provides an interactive way to inspect feeds, search collected material, and review results.

The modular structure allows the system to evolve from a local prototype into a more complete monitoring tool without changing the core design.
