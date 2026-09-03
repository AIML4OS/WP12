# Metadata Prototype

## Where the code lives

This prototype is larger than the other sprint outputs and has **its own repository**:

**<https://github.com/AIML4OS/WP12_MetadataGraph>**

| What | Where |
|---|---|
| Getting started in SSPCloud | [`docs/SSPCloud-setup.md`](https://github.com/AIML4OS/WP12_MetadataGraph/blob/main/docs/SSPCloud-setup.md) |
| Sprint user stories | [`docs/sprint_documentation/`](https://github.com/AIML4OS/WP12_MetadataGraph/tree/main/docs/sprint_documentation) |
| The ESS profile used during the sprint | [`config/stat-metadata/`](https://github.com/AIML4OS/WP12_MetadataGraph/tree/main/config/stat-metadata) |
| Profile mechanism | [`docs/PROFILES.md`](https://github.com/AIML4OS/WP12_MetadataGraph/blob/main/docs/PROFILES.md) |

A narrative write-up for D12.2 - architecture, MCP layer, skills, use cases, evaluation - is in
[Report.md](Report.md).

## Description

This project explores the use of an AI-powered knowledge graph as a foundation for metadata
management in statistical offices.

Users interact with statistical metadata (variables, concepts, data structures, code lists, etc.)
through natural language, while the underlying graph captures and enforces the relationships defined
by standards such as GSIM and SDMX.

Compared to traditional metadata registries (rigid, database-driven catalogues), this approach
supports flexible exploration, AI-assisted entity extraction from documents, and collaborative
knowledge building across organizations.

### What was built during the sprint

The underlying graph application is a general-purpose, profile-based platform that **existed before
the sprint**. The sprint contribution was to apply it to official statistics:

- the **`stat-metadata` profile** - schema (18 node types, 27 relationship types modelled after
  GSIM), presentation, expert agents and domain knowledge skills;
- a **seed graph** of ESS metadata as example data (256 nodes, 327 edges);
- **six user stories** describing the statistical use cases, of which US-01, US-03, US-05 and US-06
  were exercised during the sprint;
- an **SSPCloud getting-started path** so others can launch and explore the prototype.

## Input and output

The input is natural language queries and/or uploaded documents describing statistical metadata.

For this proof of concept the domain is the European Statistical System (ESS) - mapping statistical
offices (NSIs), statistical programmes (e.g. Labour Force Survey, Party Preference Survey), datasets,
data structures, variables, concepts, unit types, code lists, questionnaires, classifications and
production solutions.

The output is a navigable, visual knowledge graph and conversational answers grounded in the graph's
content.

## Use cases explored

The user stories are shipped with the prototype and each contains a validation scenario, so they
double as a test script for anyone launching it.

| ID | Title | Actor | Exercised in sprint |
|---|---|---|---|
| US-01 | Concept extraction from documents and graph matching | Statistical producer / analyst | Yes |
| US-02 | Standard facilitator agent | - | No |
| US-03 | Lineage explanation and change impact assessment | Methodology officer, data steward | Yes |
| US-04 | Concept harmoniser | - | No |
| US-05 | Guided metadata curation using domain knowledge skills | Metadata curator, domain expert | Yes |
| US-06 | Interactive graph exploration for dissemination | Dissemination officer, data steward | Yes |

## Architecture

A profile-based knowledge graph platform with five components:

- **React frontend** (React Flow visualisation) - graph canvas, chat, search, editing dialogs;
  communicates with the backend only over HTTP.
- **Node.js build toolchain** - compiles the React workspaces to static files (build time only).
- **Python backend** (FastAPI + uvicorn) - serves the static files, the REST API and the MCP
  endpoints, and calls the LLM on behalf of the chat.
- **Knowledge graph** - the single source of truth, persisted as JSON per profile
  (`config/<profile>/graph.json`) and held in memory as a NetworkX graph for traversal and query.
- **LLM** - pluggable. During the sprint: `gemma4-26b-moe` on the SSPCloud-hosted vLLM endpoint
  (`https://llm.lab.sspcloud.fr/api`). Replaceable with any OpenAI-compatible endpoint, including a
  local Ollama or vLLM server, or with Anthropic's API. The app also runs with **no** LLM key: the
  graph API and MCP server stay operational and the chat panel is hidden.

Configuration is fully driven by the profile schema - no code changes are needed to adapt the domain
model to another organisation.

### MCP layer

Graph operations are exposed on two parallel surfaces backed by the same service layer: a **REST
API** for the browser frontend, and an **MCP tool layer** (59 tools) for LLM agents. Both run on the
same uvicorn server on port 8000, mounted on different paths - MCP at `/mcp`, offering both legacy
SSE and Streamable HTTP transports. The MCP instructions are generated at runtime from the active
profile schema, so a new profile produces a correctly briefed agent with no code change.

Practical consequence: a user can point an external assistant (Claude, ChatGPT, Open WebUI) at their
running instance and query their own metadata graph conversationally. See Step 5 of the
[SSPCloud setup guide](https://github.com/AIML4OS/WP12_MetadataGraph/blob/main/docs/SSPCloud-setup.md).

### Skills and expert agents

Domain knowledge skills are structured markdown files loaded in two stages - metadata at startup,
full content lazily on first use - so a domain skill only occupies LLM context when it is relevant.
Each skill declares its own `allowed-tools`, tying it to the MCP tool layer.

The sprint profile ships two skills (**Graph Analysis**, **GSIM Lineage and Change Impact**) and
three expert agents: **Metadata Expert**, **ESS Expert**, and a deliberately minimal **Population
Domain Expert** included as a simple worked example of how an organisation would define its own.

## Evaluation criteria

- **Discoverability**: can users find related metadata (e.g. "which variables measure labour market
  status?") faster than in existing registries?
- **Interoperability**: does the graph model align with GSIM/SDMX well enough to support real
  metadata exchange workflows?
- **Collaboration**: can multiple statistical offices contribute to and benefit from a shared
  metadata landscape?
- **Flexibility**: how easily can the schema be adapted to other statistical domains beyond the ESS
  seed data?

Of these, flexibility is demonstrated by the profile mechanism and interoperability by the
GSIM-aligned schema. Discoverability and collaboration were not measured during the sprint.

## Evaluation summary

| Criterion | Assessment |
|---|---|
| Efficiency gain | High in discoverability and cross-office collaboration; low/none in replacing existing registries for authoritative publication |
| Reusability | High - profile system supports any metadata domain without code changes, and the ESS profile is a complete worked example |
| Data accessibility | Medium - seed data is built from public ESS sources and ships with the prototype, so the demo is self-contained; applying it to a real NSI environment requires access to internal metadata registers, which was not available during the sprint |
| On-prem compatibility | Medium/high - no component requires an external service, and the app runs without an LLM key at all, but useful AI behaviour requires local infrastructure capable of serving a medium-sized tool-calling model |
| Low-hanging fruit for NSIs | Medium - launching and exploring the prototype in SSPCloud is documented and reproducible, and a new profile is a configuration exercise rather than a development one; populating it with real metadata is not low-hanging |
| Evaluation robustness | Low/medium - user stories provide validation scenarios and acceptance criteria, but no scored benchmark or ground truth was established; results assessed qualitatively |
| Feasibility | Medium - seed data demonstrates the concept, but real-world coverage requires curation effort |
| Lifespan | Medium/high - standards-aligned schema (GSIM/SDMX) and LLM-agnostic, protocol-based integration reduce lock-in |
| Cost effectiveness | Medium/high - graph exploration costs nothing; LLM cost is per question rather than per record and is bounded when self-hosted; curation effort is the dominant cost, not inference |

## Status

Implemented and demonstrable in the sprint profile:

- graph seeded with ESS actors and their relationships;
- statistical programmes linked to datasets and data structures;
- variables, concepts, unit types, code lists, questionnaires and classifications modelled and connected;
- natural language queries resolving against the metadata graph;
- AI extraction of entities and relationships from uploaded documents;
- duplicate detection flagging overlapping definitions across offices;
- expert agents with domain knowledge skills;
- MCP access for external agents.

Available in the platform but not exercised during the sprint:

- federation support for connecting metadata graphs across organizations
  (see [`docs/FEDERATED_GRAPH_DESIGN.md`](https://github.com/AIML4OS/WP12_MetadataGraph/blob/main/docs/FEDERATED_GRAPH_DESIGN.md));
- autonomous agentic loops over the MCP layer - the harness exists, the loop does not.

## Next steps

- Add lighter-weight guidance for less technical users on top of the SSPCloud setup guide.
- Establish a scored benchmark for discoverability against an existing registry.
- Test federation across two organisations' graphs.
- Implement the agentic loop for US-03 (change impact), the use case that most requires it.
