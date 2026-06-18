# Deliverable D12.2 - Large Language Models for Official Statistics

## Overview

This document provides the draft structure and initial content for **Deliverable D12.2** of **Work Package 12 (WP12)** within the AIML4OS project. D12.2 builds on the work done for D12.1 (Lisbon hackathon, June 2025) and incorporates the outputs, reflections, and evaluations from the Stockholm sprint (June 2026).

The deliverable documents practical LLM-based prototypes for official statistics, including use-case descriptions, architectural considerations, evaluation results, and guidance for implementation and reuse by National Statistical Institutes (NSIs).

## Structure

The deliverable follows the structure established by D12.1, extended with additional sections on scope, limitations, quality assessment of GenAI approaches, and future considerations.

---

## 1. Introduction

### 1.1 Background

WP12 explores how large language models (LLMs) and generative AI can be used in the context of official statistics. The work includes both practical prototype development and reflection on how such systems can be implemented in a way that supports reuse, data protection, transparency, evaluation and operational relevance.

### 1.2 Objectives of D12.2

- Document the prototypes developed and extended during the Stockholm sprint (June 2026)
- Provide evaluation results using a shared evaluation framework
- Describe architectural choices and their implications
- Capture lessons learned and guidance for NSIs considering LLM-based approaches
- Identify scope, limitations, and future directions

### 1.3 Relationship to D12.1

D12.1 was based on the Lisbon hackathon (June 2025) and included three prototypes:
1. Dissemination summary (PDF summarisation and keyword extraction)
2. From PDF to figures (structured data extraction from PDFs)
3. Web Corner (web scraping and LLM content classification)

D12.2 continues and extends this work, with a focus on:
- **News Corner** (statistical media consistency checking) — new use case
- **Web Corner** (agentic web scraping) — extended from D12.1
- **Metadata management using AI** (graph-based metadata interaction) — new use case

---

## 2. Evaluation Framework

To support sustainability and maintainability of AI systems developed within the project, each use case and prototype is evaluated along the following dimensions:

| Criterion | Description |
|---|---|
| Efficiency gain | How much effort does the prototype save compared to manual or existing approaches? |
| Reusability | Can other NSIs download, configure, and run the prototype with minimal changes? |
| Data accessibility | Are the required input data and resources openly available or easily obtainable? |
| On-prem compatibility | Can the prototype run in an on-premises environment without external API dependencies? |
| Low-hanging fruit for NSIs | How easy is it for an NSI to adopt the prototype with existing infrastructure? |
| Evaluation robustness | How well can the quality of outputs be assessed and benchmarked? |
| Feasibility | Is the prototype technically achievable within realistic resource constraints? |
| Lifespan | How long is the prototype expected to remain useful given evolving technologies and data? |
| Cost effectiveness | What are the operational costs (API calls, compute, maintenance) relative to the value? |

---

## 3. Architecture Perspective

### 3.1 General Approach

The prototypes share a common architectural approach:
- Use of LLMs only where semantic interpretation is required
- Deterministic logic for ingestion, parsing, deduplication, scheduling, and scoring
- Modular design allowing individual components to be replaced
- Cost-aware design minimising unnecessary API calls

### 3.2 Shared Infrastructure

- **SSPCloud (Onyxia)**: shared development and hosting environment provided by WP3
- **LLM endpoints**: both SSPCloud-hosted open-source models (e.g., GEMMA4-26B-MOE) and commercial APIs
- **Tool-calling**: used by agentic prototypes for structured interaction with external systems

### 3.3 Architecture Diagrams

<!-- TODO: Add architecture diagrams for each prototype -->
_Placeholder: architecture diagrams to be added._

---

## 4. Use Cases and Prototypes

### 4.1 News Corner - Statistical Media Consistency

#### 4.1.1 Description

A prototype system for comparing newspaper articles with official National Statistical Institute (NSI) releases. It uses an LLM to interpret and compare web content, identifying when media articles refer to specific statistical releases and how accurately those releases are reported.

The system retrieves RSS feeds from newspapers and NSIs, stores metadata and content locally, generates candidate article-release pairs, uses an LLM for semantic comparison, and produces alignment scores.

#### 4.1.2 Input and Output

- **Input**: RSS feeds from news websites and official releases from NSIs
- **Output**: Alignment indication measuring how closely each news article matches the corresponding official release, across five dimensions (topic match, figures accuracy, reference period, source attribution, framing consistency)

#### 4.1.3 Architecture

The prototype is implemented in Python, using standard libraries wherever possible. The architecture is modular with the following components:

1. Configuration Module
2. Scheduling Module
3. Feed Ingestion Module
4. Storage Module (SQLite + raw files)
5. Deduplication Module
6. Candidate Matching Module
7. LLM Analysis Module
8. Alignment Scoring Module
9. Benchmarking Module
10. Reporting Module
11. Graphical User Interface Module

See [Draft-architecture.md](../News_Corner/Draft-architecture.md) for the full architecture specification.

#### 4.1.4 Evaluation Results

The system was tested using GEMMA4-26B-MOE hosted on SSPCloud Onyxia and supplementary tests with ChatGPT 5.5.

| Test | Languages | Expected | Result | Outcome |
|---|---|---|---|---|
| CBS vs NL Times (consistent) | English | Consistent | Consistent | Passed |
| CBS vs NL Times (different topic) | English | Inconsistent | Inconsistent | Passed |
| CBS vs NL Times (wrong figures/period) | English | Inconsistent | Inconsistent | Passed |
| CBS (Dutch) vs NL Times (English) | Dutch/English | Consistent | Consistent | Passed |
| SURS vs RTVSLO (year mismatch) | Slovenian | Inconsistent | Consistent | Failed |
| Same case with ChatGPT 5.5 | Slovenian | Inconsistent | Inconsistent | Passed |

See [Report.md](../News_Corner/Report.md) for detailed test results.

#### 4.1.5 Evaluation Summary

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

---

### 4.2 Web Corner - Agentic Web Scraping

#### 4.2.1 Description

A proof of concept for using LLMs as agentic web scrapers. The system takes a natural language prompt including a starting URL and a specific goal, and returns structured information along with a step-by-step reasoning trace.

#### 4.2.2 Input and Output

- **Input**: A natural language prompt including a starting URL and a specific goal (e.g., "Find all job listings related to information security on the SCB website. Start at www.scb.se")
- **Output**: A structured response containing the requested information, the relevant URLs found, and a step-by-step reasoning trace

#### 4.2.3 Architecture

The system uses an LLM with tool-calling capabilities. The tools are kept "atomic" (performing only one small task) for higher reliability:
- `fetch_page_urls`: Retrieve hyperlinks from a URL
- `fetch_page_content`: Fetch page content from a URL
- `interact_with_web`: Playwright-based interaction for dynamic content

Integration with Playwright has been initiated to handle JavaScript-heavy websites.

#### 4.2.4 Evaluation Results

The proof of concept was evaluated across several diverse use cases:
- **Comparative Analysis**: Finding specific articles on a topic across multiple sources
- **Job Discovery**: Matching job offerings to specific professional descriptions
- **E-commerce Extraction**: Fetching product catalogs and real-time pricing
- **NSI Data Comparison**: Fetching and comparing inflation numbers from SCB and CBS

#### 4.2.5 Evaluation Summary

| Criterion | Assessment |
|---|---|
| Efficiency gain | High in generic applicability; Low in raw runtime speed |
| Reusability | High |
| On-prem compatibility | Medium/High (requires local LLM with tool-calling) |
| Feasibility | Medium |
| Lifespan | Medium/High |
| Performance vs. chatbots | Comparable; often superior due to specialized system prompting |

#### 4.2.6 Key Takeaways

- **System Prompting** is the most critical component for tuning behaviour from a generic LLM into a specialised scraping agent
- **Reasoning Trade-offs**: Enabling chain-of-thought reasoning significantly improves output quality but increases inference latency
- **Tool Design**: Keeping tools atomic ensures higher reliability during the LLM's tool-calling phase

---

### 4.3 Metadata Management using AI

#### 4.3.1 Description

A prototype exploring the use of an AI-powered knowledge graph as a foundation for metadata management in statistical offices. Users interact with statistical metadata (variables, concepts, data structures, code lists, etc.) through natural language, while the underlying graph captures and enforces the relationships defined by standards like GSIM and SDMX.

#### 4.3.2 Input and Output

- **Input**: Natural language queries and/or uploaded documents describing statistical metadata
- **Output**: A navigable, visual knowledge graph and conversational answers grounded in the graph's content

#### 4.3.3 Architecture

The software is a profile-based knowledge graph platform with:
- **Frontend**: React (React Flow visualisation)
- **Backend**: FastAPI + NetworkX
- **AI layer**: Claude or any OpenAI-compatible LLM with tool calling
- **Domain model**: 8 node types (Actor, StatisticalProgramme, DataSet, DataStructure, InstanceVariable, Concept, UnitType, CodeList) and 9 relationship types modelled after GSIM

Configuration is fully driven by a profile schema — no code changes are needed to adapt the domain model.

#### 4.3.4 Evaluation Criteria

- **Discoverability**: Can users find related metadata faster than in existing registries?
- **Interoperability**: Does the graph model align with GSIM/SDMX well enough to support real metadata exchange workflows?
- **Collaboration**: Can multiple statistical offices contribute to and benefit from a shared metadata landscape?
- **Flexibility**: How easily can the schema be adapted to other statistical domains?

#### 4.3.5 Evaluation Summary

| Criterion | Assessment |
|---|---|
| Efficiency gain | High in discoverability and cross-office collaboration |
| Reusability | High — profile system supports any metadata domain without code changes |
| Data accessibility | <!-- TODO: to be assessed --> |
| On-prem compatibility | Medium/High — requires an LLM endpoint with tool-calling support |
| Low-hanging fruit for NSIs | <!-- TODO: to be assessed --> |
| Evaluation robustness | <!-- TODO: to be assessed --> |
| Feasibility | Medium — seed data demonstrates the concept, real-world coverage requires curation |
| Lifespan | Medium/High — standards-aligned schema (GSIM/SDMX) and LLM-agnostic design |
| Cost effectiveness | <!-- TODO: to be assessed --> |

#### 4.3.6 Current Status

<!-- TODO: Update with final sprint outputs -->
- Graph seeded with ESS actors and their relationships
- Statistical programmes linked to datasets and data structures
- Variables, concepts, unit types, and code lists modelled and connected
- Natural language queries resolve against the metadata graph
- AI extracts entities and relationships from uploaded documents
- Duplicate detection flags overlapping definitions across offices

---

## 5. Scope and Limitations

### 5.1 General Limitations

- All prototypes are **proof-of-concept implementations** developed during time-limited sprints and are not production-ready.
- Results are **model-dependent**: switching LLM provider, version, or configuration may produce different outcomes.
- **Evaluation datasets are small**: broader, systematic evaluation is needed before generalising findings.
- **Reproducibility** is limited by the non-deterministic nature of LLM outputs; the same prompt may yield different results across runs.

### 5.2 Data and Access Limitations

- Some use cases depend on **paywalled or restricted content** (e.g., newspaper articles behind paywalls).
- **Language coverage** is uneven: English performs best, while other European languages show variable quality.
- Access to **internal NSI metadata** and production systems was not available during the sprint.

### 5.3 Technical Limitations

- **On-premises deployment** of open-source models with tool-calling support remains challenging for some NSIs.
- **Latency and cost** of LLM API calls may limit operational-scale deployment.
- **Dynamic web content** (JavaScript-heavy sites) is not yet reliably handled by the web scraping prototype.
- The metadata prototype requires **curation effort** to populate the knowledge graph with real-world data.

### 5.4 Methodological Limitations

- The **evaluation framework** relies partly on subjective assessments by sprint participants.
- **Human benchmark data** is limited to a small number of test cases for the News Corner prototype.
- No **longitudinal evaluation** has been performed to assess how outputs change as models are updated.

---

## 6. Quality of GenAI-based Approaches

This section captures observations and reflections on the quality of results produced by generative AI approaches during the sprint.

### 6.1 Strengths Observed

- LLMs performed well on **structured English-language tasks** such as comparing statistical releases with news articles and extracting information from web pages.
- **Multilingual comparison** (e.g., Dutch source vs. English article) worked correctly in tested cases.
- **Agentic tool-calling** enabled flexible, multi-step workflows that adapted to different website structures without manual code changes.
- **Prompt engineering** had a significant and positive impact on result quality when done carefully.

### 6.2 Weaknesses and Failure Modes

- Models can **miss subtle mismatches**, such as reference-period year differences in non-English text (Slovenian test case).
- **Chain-of-thought reasoning** improves quality but increases latency and cost, creating a practical trade-off.
- **Open-source models** (e.g., GEMMA4-26B-MOE on SSPCloud) showed promising but uneven quality compared to commercial alternatives (ChatGPT 5.5).
- Model outputs are **non-deterministic**: the same input may produce different quality outputs across runs.

### 6.3 Practical Recommendations

- Use **structured prompts** with explicit step-by-step instructions (e.g., extract reference period before making a judgement).
- Implement **benchmark datasets** to track quality over time and across model changes.
- Consider **model comparison** as part of the evaluation workflow (e.g., running difficult cases through multiple models).
- Keep **tool interfaces atomic** and well-defined to reduce failure points in agentic setups.

---

## 7. Future Considerations

### 7.1 Prototype Development

- Extend News Corner with automated RSS feed monitoring and dashboard reporting.
- Complete Playwright integration for Web Corner to handle dynamic web content reliably.
- Expand the metadata knowledge graph with real-world ESS data and test federation across organisations.

### 7.2 Evaluation and Benchmarking

- Create shared benchmark datasets for each use case to enable systematic model comparison.
- Develop human-in-the-loop evaluation workflows for ongoing quality assessment.
- Compare LLM-based approaches with rule-based baselines and existing tools.

### 7.3 Deployment and Reuse

- Package prototypes for easy deployment by other NSIs (containerisation, configuration templates).
- Test on-premises deployment with open-source models to support NSIs with data-protection constraints.
- Document deployment guides and minimum infrastructure requirements.

### 7.4 Integration

- Explore integration with existing statistical production systems and metadata registries.
- Investigate how prototype outputs can feed into quality assurance and dissemination workflows.
- Consider alignment with GSBPM and other statistical process models.

### 7.5 Research Directions

- Investigate fine-tuning or domain adaptation of open-source models for statistical tasks.
- Explore multi-agent architectures for complex statistical workflows.
- Assess the impact of retrieval-augmented generation (RAG) on output quality for metadata and dissemination tasks.

---

## 8. Conclusions

<!-- TODO: To be written after sprint outputs are finalised -->

_Placeholder: conclusions to be drafted based on final sprint results and evaluations._

---

## Appendices

### Appendix A: Sprint Agenda

See [agenda.md](../agenda.md)

### Appendix B: Background Material

See [background.md](../background.md)

### Appendix C: Use-case Planning

See [use-case-planning.md](../use-case-planning.md)

### Appendix D: News Corner Detailed Test Report

See [News_Corner/Report.md](../News_Corner/Report.md)

### Appendix E: News Corner Architecture

See [News_Corner/Draft-architecture.md](../News_Corner/Draft-architecture.md)
