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

The work package proceeds from the observation that LLMs already offer substantial opportunities in the statistical domain — text automation, code translation, quality control, chatbot interfaces — while the pace of development makes it difficult to forecast which specific applications will be relevant across 2024–2027. WP12 therefore identifies five presumptive high-value areas and adopts an iterative approach, selecting specific applications as late as possible before each piece of work starts:

| # | High-value area |
|---|---|
| A | Data and metadata handling through usage of LLM |
| B | Generating draft text for the process step *Analyse* |
| C | Improving and translating production code |
| D | Dissemination process by using chatbots |
| E | Analysis of large documents and web page data |

Across these areas WP12 pursues three objectives: to explore the ability of LLMs available in 2024–2027 to integrate into the production and support of official statistics; to explore what benefits fine-tuning existing LLMs brings to the use cases; and — pervasive to both — to define the architectural enablers and constraints that apply when LLMs are used.

The work package is organised in three tasks. **T12.1** analyses how LLMs can be integrated into production in ways that align data-protection strategy with data sensitivity, and is to produce architectural guidance supporting the prototypes. **T12.2** demonstrates the use of pre-existing LLMs through at least two prototypes covering at least two of the high-value areas. **T12.3** examines fine-tuning as a route to a specialised statistical LLM.

**Status of T12.1 at the time of writing.** Collection of input for the architecture work has begun, but no guidance documents or other T12.1 material have yet been produced. Nothing in this deliverable should be read as applying or summarising established T12.1 guidance. The architectural observations and the evaluation criteria recorded here are the other way round: they are an indication, drawn from building and running the prototypes, of what may be worth having in mind when an architecture or design for AI solutions in these application areas is developed — and they are offered as input to T12.1 rather than as output from it.

### 1.2 Objectives of D12.2

D12.2 is *WP12 LLM prototype B*, delivered under **T12.2 — Use of pre-existing LLM**. Its purpose is therefore to demonstrate what existing, unmodified LLMs can do for official statistics, and to record what was learned in enough detail that other NSIs can judge whether to try the same thing.

T12.2 requires at least two prototypes covering at least two of the high-value areas. This deliverable meets that requirement as follows:

| Contribution | High-value area | Form |
|---|---|---|
| Metadata Graph | **A** — Data and metadata handling through usage of LLM | Runnable prototype |
| Web Corner | **E** — Analysis of large documents and web page data | Runnable prototype |
| News Corner | **E** — Analysis of large documents and web page data, applied to quality control of statistical communication | Experiment report |

Two prototypes, two distinct high-value areas. News Corner extends the coverage of area E into a quality-control setting without being a third prototype; §1.4 explains how the three fit together.

Against that framing, the objectives of this deliverable are to:

- Document the prototypes developed and extended during the Stockholm sprint (June 2026), with enough code, documentation and getting-started guidance that they can be run by people who were not part of the sprint groups
- Provide evaluation results using a shared evaluation framework
- Describe architectural choices and their implications, and record them as input to the architecture work under T12.1 — in particular the consequences of choosing open-source over proprietary models for reusability, openness and data sensitivity
- Capture lessons learned and guidance for NSIs considering LLM-based approaches
- Identify scope, limitations, and future directions

**Out of scope.** D12.2 concerns pre-existing models used as they are. Fine-tuning and domain adaptation belong to T12.3 and are not attempted here; where the sprint results bear on that question, they are noted as input to it rather than as findings about fine-tuning. The deliverable also does not attempt to cover the remaining high-value areas — draft text for the *Analyse* step (B), production code translation (C), or dissemination chatbots (D) — beyond noting in §7 where the sprint architectures could be extended towards them.

### 1.3 Relationship to D12.1

D12.1 was based on the Lisbon hackathon (June 2025) and included three prototypes:
1. Dissemination summary (PDF summarisation and keyword extraction)
2. From PDF to figures (structured data extraction from PDFs)
3. Web Corner (web scraping and LLM content classification)

D12.2 continues and extends this work, with a focus on:
- **News Corner** (statistical media consistency checking) — new use case
- **Web Corner** (agentic web scraping) — extended from D12.1
- **Metadata management using AI** (graph-based metadata interaction) — new use case

The extension of Web Corner is the clearest line of continuity. In D12.1 the Web Corner prototype used an LLM to classify content that had already been retrieved by conventional means; the retrieval itself was still script-driven. In D12.2 the LLM drives the retrieval as well, deciding which pages to visit through tool-calling. The step from *LLM as classifier* to *LLM as navigator* is the substantive advance, and it is what makes the maintenance argument — no per-site code — possible.

### 1.4 Composition of the deliverable

D12.2 consists of a report and a prototype. In practice the prototype requirement is met by **two technical prototypes**, each demonstrating different properties and architectural approaches to generative AI, complemented by **one experiment report** on a third use case:

| Contribution | Area | Form | What it demonstrates |
|---|---|---|---|
| **Web Corner** — agentic web scraper | E | Runnable prototype, code in this repository | Agentic tool-calling: a small, self-contained agent loop where capability is carried by the system prompt and a few atomic tools rather than by application code |
| **Metadata Graph** | A | Runnable prototype, [own repository](https://github.com/AIML4OS/WP12_MetadataGraph) | A full application architecture: a domain-configurable knowledge graph with a human interface and a parallel machine interface (MCP), skills-based domain knowledge injection, and profile-driven adaptation |
| **News Corner** — statistical media consistency | E | Experiment report, no runnable code | Feasibility evidence: whether an open-weights model hosted on infrastructure an NSI could realistically operate can perform a genuinely difficult multilingual comparison task, and where its limits lie |

The two prototypes were chosen to sit at opposite ends of a spectrum rather than to duplicate each other. Web Corner is minimal — one control loop, three tools, one configuration file — and shows how much capability can be obtained with very little code when the model is doing the reasoning. The Metadata Graph is a structured application and shows what is required when the AI capability must be embedded in an organisation's information model, governed by a schema, and made available to other systems. Between them they cover the two realistic shapes an NSI's first LLM system is likely to take.

News Corner is reported as an experiment rather than a prototype: the sprint produced a designed architecture and a series of model tests, but no packaged, runnable codebase. It is included because the test evidence is directly useful to NSIs considering the same task, and because the architecture it justifies is specified in enough detail to be implemented by others.

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

**Two levels of evaluation, only one of which this deliverable reaches.** It is useful to separate two questions that the word "evaluation" covers.

The first is whether a prototype is *reasonable* — whether it can be built, run, reused and operated under realistic conditions, and whether it respects sensible architectural principles. That is what the nine criteria above address, and it is what this deliverable evaluates. These criteria were formulated during the work rather than derived from finished guidance; they should be read as an indication of the dimensions that appear to matter when designing AI solutions in this domain, not as a settled framework.

The second is whether a prototype is *effective* — how accurate, reliable and useful its output actually is, measured systematically against ground truth. The work so far has not had the preconditions for this. The sprint was three days long, the shared environment permits only non-sensitive material (see §3.2), and no benchmark datasets existed to measure against. Where this deliverable reports on output quality it therefore reports observations and individual test cases, not systematic measurement.

That distinction is not a caveat to apologise for; it defines what the prototypes are for. Their value is precisely that they make the second kind of evaluation possible: they are working systems that could serve as the basis for systematic effectiveness evaluation, carried out in environments where sensitive material may also be used and where an NSI's own data provides the ground truth. §8 returns to this.

A note on how the *on-prem compatibility* rating should be read. All three use cases are architecturally on-prem compatible: none of them depends on a specific vendor, and each talks to an LLM through a replaceable, standard interface. The rating therefore reflects **what an NSI must provide locally for the system to work well**, not whether the software can be pointed at a local endpoint. Two of the three require a model with tool-calling support, which in practice means hosting a medium-sized model on local GPU infrastructure — a real prerequisite, and the main reason those two are rated medium/high rather than high.

---

## 3. Architecture Perspective

### 3.1 General Approach

The prototypes share a common architectural approach:
- Use of LLMs only where semantic interpretation is required
- Deterministic logic for ingestion, parsing, deduplication, scheduling, and scoring
- Modular design allowing individual components to be replaced
- Cost-aware design minimising unnecessary API calls

### 3.2 Shared Infrastructure

- **SSPCloud (Onyxia)**: shared development and hosting environment made available to the project through WP3
- **LLM endpoint**: all three use cases were tested against `gemma4-26b-moe`, served from the SSPCloud-hosted vLLM endpoint (`https://llm.lab.sspcloud.fr/api`), with supplementary comparisons against commercial APIs
- **Tool-calling**: used by both prototypes for structured interaction with external systems

**What SSPCloud is, and what it is not.** SSPCloud is in practice a cloud service available to NSIs, operated by INSEE, the French statistical office. It carries the conditions of such a service — among them a requirement that only non-sensitive material may be handled in the environment. All sprint work therefore used public or otherwise non-sensitive data, and nothing here demonstrates that these prototypes may be used on sensitive statistical material. Doing that would require an environment cleared for it, which is a separate matter from whether the software runs.

What the setup does provide is a realistic picture of the conditions many NSIs could establish — either locally, or within whatever arrangements are available in their own country for using comparable services. The same applies to serving open LLMs: an office able to host a medium-sized open-weights model reproduces the essential property of this environment without depending on this particular service.

That all three use cases ran against the same open-weights model on the same infrastructure is itself a result. It means the differences observed between use cases reflect the tasks and the architectures rather than differences in model access, and it shows that a shared inference service of this kind is sufficient for prototype work.

**Openness, reusability and data sensitivity.** These three questions turn out to dominate the architectural picture, and the prototypes speak to each of them. On **openness and practicality**, the open-weights model was sufficient for every task attempted, and was outperformed by a commercial model only on the single hardest comparison case, where the gap was closed by prompt design rather than by changing model — so choosing an open model did not cost capability at this level of ambition. On **reusability**, both prototypes reach their model through a standard OpenAI-compatible interface, so the choice between the SSPCloud endpoint, a locally hosted vLLM or Ollama server, and a commercial API is a configuration change rather than a code change; this is the property that lets the same prototype be adopted by offices working under different constraints. On **data sensitivity**, the relevant observation is about the shape of the solution rather than about what was processed: because the model endpoint is replaceable and the metadata prototype runs fully without an LLM key at all, an office can place the boundary between its data and any external service wherever its own rules require, and can deploy the non-AI parts before taking a model decision at all. Whether a given deployment is then permitted to handle sensitive material is a question for that office's own assessment, not something these prototypes settle.

Taken together, the prototypes indicate that for several application areas within official statistics it is possible to establish IT solutions built on the strengths of generative AI under conditions that an NSI can realistically meet. The cost of that portability is the infrastructure prerequisite recorded in §2: a tool-calling model of useful size still has to be served from somewhere. These are observations offered as input to the T12.1 architecture work, not conclusions drawn from it.

### 3.3 Tool-calling and the Model Context Protocol

Both prototypes are built on tool-calling, but they use it in structurally different ways, and the contrast is one of the more transferable architectural findings of the sprint.

**Tool-calling as an internal control loop (Web Corner).** The agent owns its tools. Three atomic functions — fetch hyperlinks, fetch content, interact with the page — are passed to the model with each request, and a thin loop executes whatever the model calls, appends the result to the message history, and asks again. The tools are private to the application; nothing outside it can invoke them. This is the lightest possible agentic architecture and it is appropriate when the agent's job is bounded and self-contained.

**Tool-calling as a published interface (Metadata Graph).** The same underlying capability is exposed through the **Model Context Protocol (MCP)**, an open protocol that standardises how LLMs discover and invoke external tools and data sources. It is to agent–tool integration what the Language Server Protocol is to editor–language integration: a common contract that removes the need for bespoke, one-off wiring between each agent and each system.

In the Metadata Graph the graph operations are exposed on two parallel surfaces backed by the same service layer: a **REST API**, designed for a human operating a browser and consumed by the React frontend, and an **MCP tool layer**, designed for an LLM agent operating autonomously. Both are mounted on the same uvicorn server on port 8000 on different URL paths, with no separate infrastructure. The MCP surface registers 59 tools covering query and traversal (`search_graph`, `get_node_details`, `get_related_nodes`), semantic matching (`find_similar_nodes`), mutation (`add_nodes`, `update_node`), schema introspection (`get_schema`, `list_node_types`) and session/visualisation control. It offers two transports — legacy SSE and Streamable HTTP — so that both older and current MCP clients can connect.

Two implementation choices in that layer generalise beyond this prototype:

- **The MCP instructions are generated at runtime from the active deployment profile.** The node types, relationship types and domain context that an external agent receives are derived from the profile's schema configuration, so a new profile produces a correctly briefed agent with no code change. Domain adaptation and agent briefing are the same act.
- **REST and MCP share one service layer, with a regression test asserting equivalence between them.** The human view and the agent view of the graph cannot drift apart, which is a governance property as much as a technical one: what an agent can see and change is exactly what the application permits a user to see and change.

The practical effect is that an NSI running the prototype can point an external assistant at its own metadata graph and query it conversationally, with no integration code. The architectural effect is more significant: it is the point at which a prototype stops being an application with an AI feature and becomes a data source that arbitrary agents can reason over.

MCP is also the prerequisite for the next mode of operation. Both prototypes currently run bounded loops driven by a single user request. Autonomous agentic loops — where an agent plans, traverses, evaluates intermediate results, and decides whether to continue, backtrack or conclude — are what several of the metadata user stories actually require. Change impact assessment after a classification version change cannot be answered by one prompt–response exchange, because the queries that need to be issued depend on what earlier queries returned. The sprint architecture does not implement that loop, but it deliberately builds the harness for it.

### 3.4 Architecture Descriptions

Per-prototype architecture descriptions and diagrams are maintained with the prototypes themselves:

- Web Corner: agent loop and tool layer — [webcorner/Report.md](../webcorner/Report.md)
- Metadata Graph: five-component architecture, MCP layer, skills — [metadata/Report.md](../metadata/Report.md)
- News Corner: eleven-module specification — [News_Corner/Draft-architecture.md](../News_Corner/Draft-architecture.md)

---

## 4. Use Cases and Prototypes

### 4.1 News Corner - Statistical Media Consistency (experiment)

#### 4.1.1 Description

An experiment testing whether an LLM can determine how accurately a newspaper article reports on an official National Statistical Institute (NSI) release, together with an architecture specification for the operational system such a capability would support.

The designed system retrieves RSS feeds from newspapers and NSIs, stores metadata and content locally, generates candidate article-release pairs, uses an LLM for semantic comparison, and produces alignment scores. During the sprint the comparison step was tested directly; the surrounding pipeline was specified but not implemented.

#### 4.1.2 Input and Output

- **Input**: RSS feeds from news websites and official releases from NSIs
- **Output**: Alignment indication measuring how closely each news article matches the corresponding official release, across five dimensions (topic match, figures accuracy, reference period, source attribution, framing consistency)

#### 4.1.3 Architecture

The prototype is specified in Python, using standard libraries wherever possible. The architecture is modular with the following components:

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

The specification places the LLM only where semantic interpretation is required — identifying the statistical topic, extracting claims, comparing article to release, producing the alignment assessment — and keeps ingestion, parsing, deduplication, scheduling, storage and score aggregation as deterministic Python. Content is persisted locally and candidate pairs are generated deterministically before any model call, so cost scales with genuinely new comparisons rather than with feed volume.

See [Draft-architecture.md](../News_Corner/Draft-architecture.md) for the full architecture specification.

#### 4.1.4 Evaluation Results

Seven tests were run: six using GEMMA4-26B-MOE hosted on SSPCloud Onyxia, and one supplementary comparison using ChatGPT 5.5 on the case GEMMA4 did not pass.

| Test | Model | Source pair | Languages | Expected | Result | Outcome |
|---|---|---|---|---|---|---|
| 1 | GEMMA4-26B-MOE | CBS vs NL Times | English | Consistent | Consistent | Passed |
| 1B | GEMMA4-26B-MOE | CBS vs NL Times | English | Consistent | Consistent | Passed |
| 2 | GEMMA4-26B-MOE | CBS vs NL Times (different topic) | English | Inconsistent | Inconsistent | Passed |
| 3 | GEMMA4-26B-MOE | CBS vs NL Times (wrong figures/period) | English | Inconsistent | Inconsistent | Passed |
| 4 | GEMMA4-26B-MOE | CBS (Dutch) vs NL Times (English) | Dutch/English | Consistent | Consistent | Passed |
| 5 | GEMMA4-26B-MOE | SURS vs RTVSLO (year mismatch) | Slovenian | Inconsistent | Consistent | Failed |
| 5B | ChatGPT 5.5 | Same case | Slovenian | Inconsistent | Inconsistent | Passed |

The model performed very well given the difficulty of the task. It correctly identified consistent reporting, topic mismatches, incorrect figures and incorrect reference months, and handled a Dutch release against an English article correctly — directly relevant, since NSIs publish in national languages while media coverage is often in English.

The single failure is informative rather than disqualifying. The Slovenian article and release shared a topic and a quarter label but referred to different years (Q3 2022 vs Q3 2025), and the model accepted them as consistent. The supplementary test established that the case was solvable, which locates the weakness precisely: reference-period comparison must be forced to include the year, not just the period label. This is a prompt-design finding, and an actionable one.

See [Report.md](../News_Corner/Report.md) for dimension-level results and the evaluation prompt, and [Experiment-Report.md](../News_Corner/Experiment-Report.md) for the summary write-up.

#### 4.1.5 Evaluation Summary

| Criterion | Assessment |
|---|---|
| Efficiency gain | High, compared to manual work |
| Reusability | High by design — intended so that any NSI can download, configure and run it, though the runnable implementation does not yet exist |
| Data accessibility | Medium/Low, due to paywalls on news articles |
| On-prem compatibility | High — the task requires only a chat completion endpoint, with no tool-calling requirement, so it places the lowest infrastructure demand of the three use cases |
| Low-hanging fruit for NSIs | Medium/Low |
| Evaluation robustness | High by design — the benchmark method (synthetic articles at known alignment levels, scored by human evaluators as a reference) is specified and is the most rigorous evaluation design produced during the sprint; it has not yet been executed at scale |
| Feasibility | High — demonstrated by the tests |
| Lifespan | Medium, because of fast changes in media |
| Cost effectiveness | High, compared to manual work |

---

### 4.2 Web Corner - Agentic Web Scraping

#### 4.2.1 Description

A proof of concept for using LLMs as agentic web scrapers. The system takes a natural language prompt including a starting URL and a specific goal, and returns structured information along with a step-by-step reasoning trace.

Traditional web scraping relies on rigid, rule-based scripts that break whenever a website updates its design. This prototype replaces those scripts with an agent that interprets a natural language request, navigates web pages autonomously, and decides where to click or scroll to locate specific data.

#### 4.2.2 Input and Output

- **Input**: A natural language prompt including a starting URL and a specific goal (e.g., "Find all job listings related to information security on the SCB website. Start at www.scb.se")
- **Output**: A structured response containing the requested information, the relevant URLs found, and a step-by-step reasoning trace

#### 4.2.3 Architecture

The system operates as a decision-and-action loop driven by an LLM with tool-calling capabilities:

1. **User input** — a natural language objective and, optionally, a starting URL
2. **LLM decision engine** — evaluates page context and decides the next action using chain-of-thought reasoning
3. **Tool execution layer** — fetches hyperlinks, extracts content, or operates a headless browser for dynamic interaction
4. **Feedback loop** — returns retrieved content to the LLM, which determines whether the task is complete or requires further navigation

The tools are kept "atomic" (performing only one small task) for higher reliability:
- `fetch_page_urls`: retrieve hyperlinks from a URL, allowing the agent to move from "hub" listing pages to "leaf node" detail pages
- `fetch_page_content`: fetch fully rendered page text, filtering out raw markup
- `interact_with_web`: Playwright-based interaction — `click`, `type` or `scroll` — for dynamic content

The control loop itself is deliberately thin — it forwards tool calls, appends results to the message history, and iterates until the model returns content instead of a tool call. All navigation strategy lives in the system prompt, which encodes three operating modes (exploration, enumeration, extraction) and names two explicit failure states: returning a directory URL instead of the item, and extracting content from a list page.

**Technical configuration.** Python, with Playwright (and `playwright-stealth`) for headless browser automation and an OpenAI-compatible client for tool-calling. A single YAML file defines API key, endpoint, model name and temperature. Token usage is printed after each run, making the cost of a given task directly observable.

**A note on collection practice.** The two fetching tools mask the browser fingerprint — running Playwright through a stealth wrapper and presenting a desktop Chrome user-agent — so that the headless browser is not identifiable as such. This is what allows the prototype to work on sites that block headless clients, but it is an active measure to avoid bot detection rather than a neutral technical choice. An NSI adopting this approach should decide deliberately whether it is compatible with its own policy on automated collection, the target site's terms of use, and the transparency expectations that apply to official statistics. This is an architectural constraint of the agentic-scraping pattern rather than a defect of this implementation, and it is offered as one of the considerations the T12.1 architecture work may want to take up alongside data protection.

#### 4.2.4 Evaluation Results

The proof of concept was evaluated across several diverse use cases, with full run transcripts checked into the repository:

| Use case | Task | Result | Tokens |
|---|---|---|---|
| Comparative analysis | Newest inflation figures from the Swedish and Dutch NSIs, as a table | Both figures with period and source URLs | 14k |
| OJA discovery | CBS vacancies that are field-based rather than office-based, with direct URLs | Five individual vacancies with direct URLs | 13k |
| E-commerce extraction | Product URLs, prices and review sentiment for a product category on a retail site | Fifteen products as structured CSV | 24k |

Each run reached atomic-level results rather than stopping at a listing page, which is the failure mode the system prompt is written to prevent. The OJA run also exercised URL discovery: it was given no starting URL and located the CBS recruitment site itself. Token totals include chain-of-thought reasoning, and place the cost of a completed task at cents against a commercial API and at compute only when self-hosted.

#### 4.2.5 Evaluation Summary

| Criterion | Assessment |
|---|---|
| Efficiency gain | High in generic applicability; low in raw runtime speed. Setup time for a new task is close to zero, but LLM reasoning makes each run slower than a hardcoded script |
| Reusability | High — reusable across diverse sites without writing new code |
| Data accessibility | High — targets are public web pages requiring no credentials. Constrained in practice by robots.txt, terms of use, rate limiting and bot protection rather than by availability; note the fingerprint-masking caveat in §4.2.3 |
| On-prem compatibility | Medium/High — no external dependency beyond an OpenAI-compatible endpoint, but running it well on-premises requires local infrastructure capable of serving a medium-sized tool-calling model |
| Low-hanging fruit for NSIs | Medium/High — small codebase and a single configuration file, but Playwright and its browser dependencies add installation friction, an available tool-calling endpoint is a precondition, and the task prompt is currently edited in `main.py` rather than passed as an argument |
| Evaluation robustness | Low — outputs were assessed by inspection against known page content; no benchmark dataset or ground truth was established during the sprint |
| Feasibility | Medium |
| Lifespan | Medium/High — resilience to site redesign is the central design argument, and the tool interface is model-agnostic |
| Cost effectiveness | Medium/High — the three checked-in runs completed on 13k–24k total tokens including reasoning, so a task costs cents at commercial rates and nothing beyond compute when self-hosted; cost scales with pages traversed, so exhaustive enumeration of a large listing is the case to watch |
| Performance vs. chatbots | Comparable; often superior due to specialized system prompting |

#### 4.2.6 Key Takeaways

- **System prompting** is the most critical component for tuning behaviour from a generic LLM into a specialised scraping agent. Most of the prototype's capability is encoded there rather than in code
- **Reasoning trade-offs**: enabling chain-of-thought reasoning significantly improves output quality but increases inference latency and token cost
- **Tool design**: keeping tools atomic ensures higher reliability during the LLM's tool-calling phase

#### 4.2.7 Current Status

Core fetching, extraction and reasoning are fully functional. Playwright integration is in place and the system can launch headless browsers and render dynamic content, but the interaction between the LLM's decision timing and asynchronous JavaScript execution is still being refined. Automated benchmarking against standard datasets has not been started.

Reviewing the code against this report surfaced one defect and three smaller inconsistencies. The defect — `playwright-stealth` was imported by two tools but missing from `requirements.txt`, so the documented getting-started sequence failed on a fresh clone — has been fixed. The remaining items are recorded in the prototype report: `interact_with_web` does not apply the same fingerprint masking as the other two tools, the `use_extra_body` reasoning flag in `config.yaml` is overridden in code, and the task prompt is edited in `main.py` rather than supplied as an argument.

See [webcorner/README.md](../webcorner/README.md) for getting-started instructions and [webcorner/Report.md](../webcorner/Report.md) for the full write-up.

---

### 4.3 Metadata Management using AI

#### 4.3.1 Description

A prototype exploring the use of an AI-powered knowledge graph as a foundation for metadata management in statistical offices. Users see concepts, variables, classifications, populations and process steps as nodes in an interactive graph; any element can be clicked and asked about in natural language, with contextualised responses generated by an LLM. The underlying graph captures and enforces the relationships defined by standards such as GSIM, SDMX, DDI and Dublin Core.

Beyond exploration and interrogation, the prototype allows users to enrich the graph — adding new concepts and relationships directly — with the LLM mediating between the user's natural language and the formal expression of those concepts in the relevant standards.

**Scope of the sprint contribution.** The underlying graph application is a general-purpose, profile-based platform that existed before the sprint and was not developed as part of it. The sprint contribution was to apply that platform to official statistics: a deployment profile for the European Statistical System (schema, presentation, expert agents, domain knowledge skills), a seed graph of ESS metadata as example data, six user stories describing the statistical use cases, and a documented SSPCloud getting-started path. This distinction matters when reading the evaluation: the reusability and architecture assessments concern the platform, while the feasibility and coverage assessments concern the ESS profile built on it.

**Code**: <https://github.com/AIML4OS/WP12_MetadataGraph>

#### 4.3.2 Input and Output

- **Input**: Natural language queries and/or uploaded documents describing statistical metadata
- **Output**: A navigable, visual knowledge graph and conversational answers grounded in the graph's content

#### 4.3.3 Architecture

Five components with distinct responsibilities:

| Component | Responsibility |
|---|---|
| React frontend | Graph canvas (React Flow), chat, search, editing dialogs; communicates with the backend only over HTTP |
| Node.js build toolchain | Compiles the React workspaces into static files (build time only) |
| Python backend (FastAPI + uvicorn) | Serves static files, the REST API and the MCP endpoints; calls the LLM on behalf of the chat |
| Knowledge graph | Single source of truth for nodes and edges; persisted as JSON per profile and held in memory as a NetworkX graph for traversal and query; schema configurable per deployment profile |
| LLM | Pluggable external service. During the sprint: `gemma4-26b-moe` on the SSPCloud-hosted vLLM endpoint; replaceable with any OpenAI-compatible endpoint including on-premises Ollama or vLLM, or with Anthropic's API |

The application starts and runs with no LLM key configured: the graph API and the MCP server remain fully operational and the chat panel is hidden. This lets an NSI evaluate the graph layer before committing to a model deployment.

**Domain model.** The ESS profile (`stat-metadata`) defines 18 node types and 27 relationship types modelled after GSIM, covering actors and programmes, the dataset chain, semantic building blocks, collection instruments and the technical production layer. Relationship types include the provenance edges needed for lineage (`INPUT_TO`, `PRODUCES_OUTPUT`) and the classification-versioning edges needed for change impact (`HAS_VERSION`, `DERIVED_FROM`, `PREDECESSOR_OF`, `CORRESPONDS_TO`). The seed graph holds 256 nodes and 327 edges. Configuration is fully driven by the profile schema — no code changes are needed to adapt the domain model.

**MCP layer.** Described in §3.3.

**Skills and expert agents.** Rather than relying on a single static system prompt, the chat service supports progressive loading of domain knowledge skills — structured markdown files containing established concepts, standard classifications, methodological conventions and example artefacts for a statistical domain. Skills load in two stages: metadata only (name, description, when-to-use) at startup, giving the system awareness of what exists without occupying context; full content lazily, on the first request that requires it. Each skill declares its own `allowed-tools`, which ties the mechanism to the MCP tool layer: a skill both supplies domain knowledge and constrains which graph operations may be used while it is active. Multiple skills can be active simultaneously, and switching or stacking them mid-session does not require restarting the workflow.

The sprint profile ships two skills as worked examples — Graph Analysis, and GSIM Lineage and Change Impact — and three expert agents through which skills are activated: a Metadata Expert, an ESS Expert, and a deliberately minimal Population Domain Expert included to show how an organisation would define a narrow domain agent of its own.

#### 4.3.4 Use Cases Explored

Six user stories were written for the sprint; four were exercised against the prototype. Each contains a validation scenario and open questions for prototype validation, so they double as a test script for anyone launching it — they define what a user can actually explore, not only the design intent.

| ID | Title | Actor | Exercised |
|---|---|---|---|
| US-01 | Concept extraction from documents and graph matching | Statistical producer / analyst | Yes |
| US-02 | Standard facilitator agent | — | No |
| US-03 | Lineage explanation and change impact assessment | Methodology officer, data steward | Yes |
| US-04 | Concept harmoniser | — | No |
| US-05 | Guided metadata curation using domain knowledge skills | Metadata curator, domain expert | Yes |
| US-06 | Interactive graph exploration for dissemination | Dissemination officer, data steward | Yes |

US-01 uploads a real questionnaire (the Swedish Labour Force Survey questionnaire) and asks the assistant to find question groups structurally similar to a selected pattern and propose how to represent them in the graph. US-03 traverses the derivation chain of a published dataset and, in reverse, identifies artefacts affected by a classification version change (for example NACE Rev. 2 to Rev. 2.1). US-05 exercises the skill and expert agent mechanism during curation. US-06 covers plain-language explanation of nodes for publication purposes.

Full text: [`docs/sprint_documentation/`](https://github.com/AIML4OS/WP12_MetadataGraph/tree/main/docs/sprint_documentation).

#### 4.3.5 Evaluation Criteria

Alongside the shared WP12 framework, the group defined four use-case-specific criteria:

- **Discoverability**: Can users find related metadata faster than in existing registries?
- **Interoperability**: Does the graph model align with GSIM/SDMX well enough to support real metadata exchange workflows?
- **Collaboration**: Can multiple statistical offices contribute to and benefit from a shared metadata landscape?
- **Flexibility**: How easily can the schema be adapted to other statistical domains?

Of these, flexibility is demonstrated by the profile mechanism and interoperability by the GSIM-aligned schema. Discoverability and collaboration were not measured during the sprint and remain open.

#### 4.3.6 Evaluation Summary

| Criterion | Assessment |
|---|---|
| Efficiency gain | High in discoverability and cross-office collaboration; low/none in replacing existing registries for authoritative publication |
| Reusability | High — the profile system supports any metadata domain without code changes, and the ESS profile is a complete worked example of what a new profile must contain |
| Data accessibility | Medium — seed data is built from public ESS sources and ships with the prototype, so the demo is self-contained; applying it to a real NSI environment requires access to internal metadata registers, which was not available during the sprint |
| On-prem compatibility | Medium/High — no component requires an external service, and the app runs without an LLM key at all, but useful AI behaviour requires local infrastructure capable of serving a medium-sized tool-calling model |
| Low-hanging fruit for NSIs | Medium — launching and exploring the prototype in SSPCloud is documented and reproducible, and a new profile is a configuration exercise rather than a development one; populating it with an organisation's real metadata is not low-hanging |
| Evaluation robustness | Low/Medium — the user stories provide validation scenarios and acceptance criteria, a stronger starting point than ad-hoc testing, but no scored benchmark or ground truth was established and results were assessed qualitatively |
| Feasibility | Medium — seed data demonstrates the concept, real-world coverage requires curation |
| Lifespan | Medium/High — standards-aligned schema (GSIM/SDMX) and LLM-agnostic, protocol-based integration reduce lock-in on both the metadata and the model side |
| Cost effectiveness | Medium/High — graph exploration costs nothing; LLM cost is incurred per question rather than per record and is bounded when self-hosted; curation effort to populate a real graph is the dominant cost, not inference |

#### 4.3.7 Current Status

Implemented and demonstrable in the sprint profile:
- Graph seeded with ESS actors and their relationships
- Statistical programmes linked to datasets and data structures
- Variables, concepts, unit types, code lists, questionnaires and classifications modelled and connected
- Natural language queries resolve against the metadata graph
- AI extracts entities and relationships from uploaded documents
- Duplicate detection flags overlapping definitions across offices
- Expert agents with domain knowledge skills
- MCP access for external agents

Available in the platform but not exercised during the sprint:
- Federation support for connecting metadata graphs across organisations
- Autonomous agentic loops over the MCP layer — the harness exists, the loop does not

See [metadata/README.md](../metadata/README.md) and [metadata/Report.md](../metadata/Report.md).

---

## 5. Scope and Limitations

### 5.1 General Limitations

- All prototypes are **proof-of-concept implementations** developed during time-limited sprints and are not production-ready.
- Results are **model-dependent**: switching LLM provider, version, or configuration may produce different outcomes.
- **Evaluation datasets are small**: broader, systematic evaluation is needed before generalising findings.
- **Reproducibility** is limited by the non-deterministic nature of LLM outputs; the same prompt may yield different results across runs.
- The three contributions are at **different levels of maturity**: two are runnable prototypes, one is an experiment with a designed but unimplemented architecture.

### 5.2 Data and Access Limitations

- Some use cases depend on **paywalled or restricted content** (e.g., newspaper articles behind paywalls).
- **Language coverage** is uneven: English performs best, while other European languages show variable quality.
- Access to **internal NSI metadata** and production systems was not available during the sprint; the metadata graph is populated with curated public example data rather than an organisation's real holdings.
- The shared environment permits **only non-sensitive material**, so every use case was exercised on public or otherwise non-sensitive data. Nothing in this deliverable demonstrates that these prototypes may be used on sensitive statistical material; that requires an environment cleared for it and an assessment by the office concerned.
- **Web access** is subject to robots.txt, terms of use, rate limiting and bot protection, which constrain the agentic scraper in practice more than data availability does.

### 5.3 Technical Limitations

- **On-premises deployment** of open-source models with tool-calling support remains challenging for NSIs without local GPU infrastructure sufficient to serve a medium-sized model.
- **Latency and cost** of LLM API calls may limit operational-scale deployment; agentic loops multiply the number of calls per task.
- **Dynamic web content** (JavaScript-heavy sites) is not yet reliably handled by the web scraping prototype.
- The metadata prototype requires **curation effort** to populate the knowledge graph with real-world data.
- **Autonomous agentic loops** are not implemented in any prototype; both prototypes run bounded loops driven by a single user request.

### 5.4 Methodological Limitations

- The **evaluation framework** relies partly on subjective assessments by sprint participants, and its criteria are an indication of what appears to matter rather than a settled framework (see §2).
- **Systematic effectiveness evaluation was outside the preconditions of this work**, for the reasons set out in §2: a three-day sprint, an environment restricted to non-sensitive material, and no existing benchmark datasets. The News Corner benchmark method is specified but not executed at scale; the metadata user stories carry acceptance criteria but no scored benchmark; the web scraper's outputs were judged by inspection.
- No **longitudinal evaluation** has been performed to assess how outputs change as models are updated.
- Consequently, this deliverable evaluates whether the prototypes are reasonable and reusable, not how accurate they are. Statements about output quality are observations from individual cases, and should not be read as measured performance.

---

## 6. Quality of GenAI-based Approaches

This section captures observations and reflections on the quality of results produced by generative AI approaches during the sprint.

### 6.1 Strengths Observed

- LLMs performed well on **structured English-language tasks** such as comparing statistical releases with news articles and extracting information from web pages.
- **Multilingual comparison** (e.g., Dutch source vs. English article) worked correctly in tested cases.
- **Agentic tool-calling** enabled flexible, multi-step workflows that adapted to different website structures without manual code changes.
- **Prompt engineering** had a significant and positive impact on result quality when done carefully.
- An **open-weights model on infrastructure of a kind an NSI could realistically operate** (`gemma4-26b-moe` on SSPCloud) was sufficient for all three use cases at prototype level, without recourse to a commercial API.

### 6.2 Weaknesses and Failure Modes

- Models can **miss subtle mismatches**, such as reference-period year differences in non-English text (Slovenian test case).
- **Chain-of-thought reasoning** improves quality but increases latency and cost, creating a practical trade-off.
- **Open-source models** showed promising but uneven quality compared to commercial alternatives on the hardest cases.
- Model outputs are **non-deterministic**: the same input may produce different quality outputs across runs.
- **Timing between model decisions and asynchronous browser behaviour** is an unsolved practical problem for agentic scraping of JavaScript-heavy sites.

### 6.3 Practical Recommendations

- Use **structured prompts** with explicit step-by-step instructions (e.g., extract reference period before making a judgement).
- Implement **benchmark datasets** to track quality over time and across model changes.
- Consider **model comparison** as part of the evaluation workflow (e.g., running difficult cases through multiple models). The News Corner supplementary test is a good pattern: when a model fails a case, re-run it on a stronger model to distinguish a model limitation from an unsolvable task.
- Keep **tool interfaces atomic** and well-defined to reduce failure points in agentic setups.
- Where a capability may later be needed by other systems, **expose it through a standard protocol** rather than only through an application-specific interface.

---

## 7. Future Considerations

### 7.1 Prototype Development

- Extend News Corner from an experiment into a runnable prototype by implementing the ingestion and storage modules.
- Complete Playwright integration for Web Corner to handle dynamic web content reliably.
- Expand the metadata knowledge graph with real-world ESS data and test federation across organisations.
- Implement autonomous agentic loops over the MCP layer, starting with the change-impact use case (US-03) that most requires them.

### 7.2 Evaluation and Benchmarking

This is the second of the two evaluation questions in §2, and the one the present work could not reach. The prototypes are the means to reach it.

- Use the prototypes as the basis for systematic effectiveness evaluation in environments cleared for sensitive material, where an organisation's own data can supply ground truth that public sources cannot.
- Create shared benchmark datasets for each use case to enable systematic model comparison, against a fixed model version and repeated as models change.
- Execute the specified News Corner benchmark (synthetic articles at known alignment levels, scored by human evaluators) as a template for the other use cases.
- Develop human-in-the-loop evaluation workflows for ongoing quality assessment.
- Compare LLM-based approaches with rule-based baselines and existing tools.

### 7.3 Deployment and Reuse

- Package prototypes for easy deployment by other NSIs (containerisation, configuration templates).
- Test on-premises deployment with open-source models to support NSIs with data-protection constraints, and document the infrastructure actually required to serve a medium-sized tool-calling model.
- Document deployment guides and minimum infrastructure requirements, including lighter-weight guidance for less technical users.

### 7.4 Integration

- Explore integration with existing statistical production systems and metadata registries.
- Investigate how prototype outputs can feed into quality assurance and dissemination workflows.
- Consider alignment with GSBPM and other statistical process models.
- Explore MCP as a general integration pattern for exposing statistical systems to agents, and the governance questions that follow from it.

### 7.5 Research Directions

- Investigate fine-tuning or domain adaptation of open-source models for statistical tasks (T12.3).
- Explore multi-agent architectures for complex statistical workflows.
- Assess the impact of retrieval-augmented generation (RAG) on output quality for metadata and dissemination tasks.

### 7.6 Coverage of the Remaining High-Value Areas

D12.2 covers areas A and E. The sprint architectures extend towards the other three without requiring new foundations:

- **Draft text for the *Analyse* step (B).** The metadata user story US-06 already generates plain-language narrative from graph content for publication purposes, and the skill mechanism is the natural way to encode an organisation's house conventions for such text. The step from explaining a node to drafting an analytical passage grounded in the same metadata is short.
- **Improving and translating production code (C).** Not addressed by either prototype. The `ProductionSolution` node type in the metadata profile links statistical programmes to the pipelines and repositories that implement them, which would give a code-oriented use case the context an LLM needs about what a piece of code is *for* — but this is a starting point, not a partial result.
- **Dissemination chatbots (D).** The closest existing work. The metadata prototype's MCP layer already lets an external assistant answer questions grounded in the graph, which is a dissemination chatbot in all but framing; what is missing is a public-facing interface, an access model governing what an anonymous user may query, and evaluation against the standards that apply to published statistical information.

The common prerequisite in all three cases is the same one identified in §8: an evaluation method robust enough to say whether the output is good enough to use.

---

## 8. Conclusions

The Stockholm sprint set out to advance both prototype development and the material needed for this deliverable. It produced two runnable prototypes and one experiment, satisfying the T12.2 requirement of at least two prototypes across at least two high-value areas — data and metadata handling, and analysis of large documents and web page data. The results support four conclusions.

**First, the sprint moved WP12 from LLMs as text processors to LLMs as actors.** In D12.1 the prototypes used models to interpret content that conventional code had already retrieved. In D12.2 both prototypes let the model decide what to retrieve or traverse next. That shift is what produces the sprint's central practical argument: capability that used to require per-site or per-source code is now carried by a system prompt and a small set of atomic tools, moving the maintenance burden from many fragile scripts to one prompt and one tool layer. For statistical production, where source systems and websites change continuously and maintenance dominates the total cost of a data collection, this is the most consequential property observed.

**Second, an open-weights model on infrastructure an NSI could realistically operate was sufficient for all three use cases at prototype level.** Every use case ran against the same SSPCloud-hosted model. It handled multilingual comparison of statistical releases, autonomous multi-step web navigation, and grounded conversation over a metadata graph. It was outperformed by a commercial model on the single hardest case — a reference-year mismatch expressed implicitly in Slovenian — and the manner of that failure was itself instructive: the gap was closed by requiring explicit extraction of the full reference period before judgement, which is a prompt-design fix rather than a reason to change model. The practical conclusion is that model capability is not the binding constraint at this level of ambition; the ability to serve a medium-sized tool-calling model is. Since the environment used here admits only non-sensitive material, this says nothing about processing sensitive data — but it does establish that the pattern an office would need to reproduce for that purpose is an ordinary one, and that choosing an open model over a proprietary one does not cost capability in these applications.

**Third, the way a capability is exposed matters as much as the capability itself.** The two prototypes use tool-calling in structurally different ways — as a private control loop inside an agent, and as a published interface over a standard protocol — and the difference determines what can be built next. By exposing graph operations through both a REST API for humans and an MCP layer for agents, backed by one service layer and briefed at runtime from the deployment profile, the Metadata Graph stops being an application with an AI feature and becomes a data source that arbitrary agents can reason over. This is a general pattern, not a metadata-specific one, and it is the sprint's most transferable architectural result. It also raises governance questions the project has not yet addressed: when an agent can query and modify a statistical system through a standard protocol, the boundaries of what it may see and change become an explicit design decision rather than an implicit consequence of the user interface.

**Fourth, the prototypes answer the first evaluation question and set up the second.** As §2 sets out, "evaluation" here covers two different questions. The first — is this reasonable, buildable, reusable, operable under realistic conditions, and sound against sensible architectural principles — is the one this deliverable answers, and the answer across all three use cases is yes. The second — how accurate and reliable the output actually is, measured systematically against ground truth — was outside the preconditions of this work: three days, an environment restricted to non-sensitive material, and no benchmark datasets to measure against.

That is not a shortfall to be apologised for so much as a description of where the work now stands, and it identifies what these prototypes are most useful for next. They are working systems, documented and runnable, that can serve as the basis for systematic effectiveness evaluation — carried out in environments where sensitive material may also be used, and where an office's own data supplies the ground truth that public sources cannot. The News Corner experiment already specifies a suitable method: synthetic articles at known alignment levels, scored by human evaluators, with system agreement measured against that reference. Applying that kind of design to each use case, against a fixed model version and repeated as models change, is the natural continuation — and it is work the ESS is better placed to do collectively than each NSI separately.

Taken together, the sprint outputs show that the technical barriers to LLM-based systems in official statistics are now lower than the methodological ones. Building a working agentic prototype took three days. Establishing how well it works, reliably and repeatably, is the next piece of work — and it now has something concrete to be performed on.

---

## Appendices

### Appendix A: Sprint Agenda

See [agenda.md](../agenda.md)

### Appendix B: Background Material

See [background.md](../background.md)

### Appendix C: Use-case Planning

See [use-case-planning.md](../use-case-planning.md)

### Appendix D: News Corner

- Experiment summary: [News_Corner/Experiment-Report.md](../News_Corner/Experiment-Report.md)
- Detailed test report: [News_Corner/Report.md](../News_Corner/Report.md)
- Architecture specification: [News_Corner/Draft-architecture.md](../News_Corner/Draft-architecture.md)

### Appendix E: Web Corner

- Prototype report: [webcorner/Report.md](../webcorner/Report.md)
- Getting started and code: [webcorner/](../webcorner/)

### Appendix F: Metadata Graph

- Prototype report: [metadata/Report.md](../metadata/Report.md)
- Repository: <https://github.com/AIML4OS/WP12_MetadataGraph>
- SSPCloud setup: [docs/SSPCloud-setup.md](https://github.com/AIML4OS/WP12_MetadataGraph/blob/main/docs/SSPCloud-setup.md)
- Sprint user stories: [docs/sprint_documentation/](https://github.com/AIML4OS/WP12_MetadataGraph/tree/main/docs/sprint_documentation)
