# Agentic Web Scraper (Web Corner)

_Prototype report — WP12 Stockholm sprint, 16–18 June 2026._

## Overview

The agentic web scraper, developed by the Web Corner group, is a proof of concept for modern,
agentic data collection for national statistical institutes. Traditional web scraping relies on
rigid, rule-based scripts that break whenever a website updates its design. This prototype
explores replacing those scripts with an "agentic" web scraper — an automated tool driven by a
Large Language Model (LLM) equipped with interactive tool-calling.

Rather than following fixed instructions, the agent interprets a natural language request,
navigates web pages autonomously, and decides where to click or scroll to locate specific data.
The use cases explored for official statistics were:

- **Comparative analysis** — collecting and comparing statistical figures across national sources
  (for example inflation figures from Statistics Sweden (SCB) and Statistics Netherlands (CBS)).
- **Online job advertisement (OJA) discovery** — scanning websites to match specific job postings
  against official job vacancy descriptions.
- **E-commerce data extraction** — gathering online retail product catalogues, prices and consumer
  reviews for economic tracking.

Worked examples of all three are checked in as [`output_compare_NSI_numbers.md`](output_compare_NSI_numbers.md),
[`output_outside_job_vacancies.md`](output_outside_job_vacancies.md) and
[`output_coolblue.md`](output_coolblue.md). Each run reached atomic-level results — the two national
inflation figures with their period and source URLs, fifteen products with prices and review scores,
and five individual field-interviewer vacancies with their direct URLs — rather than stopping at a
listing page. The job-vacancy run also illustrates the URL-discovery capability: it was given no
starting URL and located the CBS recruitment site itself.

## Architecture

The agentic web scraper operates as a decision-and-action loop driven by an LLM that autonomously
uses specialised tools to reach target information.

```
                 ┌──────────────────────────────────────────────┐
                 │                                              │
   User input ──►│  LLM decision engine (chain-of-thought)      │
   (goal +       │  - evaluates page context                    │
    optional     │  - decides the next action                   │
    start URL)   │  - emits a tool call, or the final answer    │
                 └───────────────┬──────────────────────────────┘
                                 │ tool call
                                 ▼
                 ┌──────────────────────────────────────────────┐
                 │  Tool execution layer                        │
                 │  fetch_page_urls | fetch_page_content |      │
                 │  interact_with_web            (Playwright)   │
                 └───────────────┬──────────────────────────────┘
                                 │ tool result
                                 ▼
                      Feedback loop: result is appended to the
                      message history and returned to the LLM,
                      which decides whether the task is complete
                      or further navigation is required.
```

1. **User input** — a natural language objective and, optionally, a starting URL.
2. **LLM decision engine** — evaluates page context and decides the next action using
   chain-of-thought reasoning.
3. **Tool execution layer** — fetches hyperlinks, extracts content, or operates a headless browser
   for dynamic interaction.
4. **Feedback loop** — returns retrieved content to the LLM, which determines whether the task is
   complete or requires further navigation.

The loop in [`main.py`](main.py) is deliberately thin: it forwards tool calls, appends results to
the message history, and iterates until the model returns content instead of a tool call (bounded
at 100 iterations). All navigation strategy lives in the system prompt, not in the control flow.

## Tool integration and core capabilities

The scraper relies on three primary tools, all implemented in [`tools/`](tools/):

| Tool | Purpose |
|---|---|
| `fetch_page_urls` | Loads the page until the network is idle so dynamic elements render, then extracts and normalises every hyperlink in the resulting DOM, allowing the agent to move from "hub" listing pages to "leaf node" detail pages. |
| `fetch_page_content` | Extracts the rendered visible text of the page body, stripping all markup and collapsing whitespace to reduce token count. |
| `interact_with_web` | Uses Playwright to perform one of three actions — `click`, `type` or `scroll` — to trigger JavaScript-heavy content or fill a form or search field. Records a video of each session to `recordings/`. |

The system prompt encodes three operating modes over these tools — **exploration** (map the site,
distinguish hubs from leaf nodes, follow pagination), **enumeration** (treat a list page as a
waypoint, not a destination), and **extraction** (only fetch content once an atomic leaf node is
reached). Two explicit failure states are named in the prompt: the "overview trap" (returning a
directory URL instead of the item) and the "single-click trap" (extracting content from a list page).

Three implementation details are worth recording, because they affect both reproducibility and how
the prototype should be used:

- **Browser fingerprint masking.** `fetch_page_urls` and `fetch_page_content` run Playwright through
  the `playwright-stealth` wrapper and set a desktop Chrome user-agent and viewport, so the headless
  browser does not identify itself as such. This makes the tools work on sites that block headless
  clients, but it is an active measure to avoid bot detection rather than a neutral technical choice.
  An NSI adopting this prototype should decide deliberately whether that is compatible with its own
  policy on automated collection, the target site's terms of use, and the transparency expectations
  that apply to official statistics. `interact_with_web` does not apply the same masking, so tool
  behaviour is not uniform in this respect.
- **Everything fetched is written to disk.** Both fetch tools log their results to `output/` (one
  file per call), and `interact_with_web` records a video of each browser session to `recordings/`.
  This is useful for tracing what the agent actually saw, but means a run leaves a local copy of the
  retrieved material, which is a data-management consideration rather than a neutral debug feature.
- **The reasoning toggle in the configuration file is currently inert.** `config.yaml` exposes
  `llm.use_extra_body`, but `main.py` overwrites it to `True` before the first request, so extended
  reasoning is always requested regardless of the configured value. Anyone wanting to measure the
  latency and cost of chain-of-thought reasoning against a non-reasoning baseline must change the
  code, not the configuration.

## Technical configuration and implementation

- **Infrastructure:** developed in the SSPCloud (Onyxia) environment. Requires an
  OpenAI-compatible LLM API with tool-calling support.
- **Model used during the sprint:** `gemma4-26b-moe`, served from the SSPCloud-hosted vLLM endpoint
  at `https://llm.lab.sspcloud.fr/api`.
- **Dependencies:** Python, with Playwright and `playwright-stealth` for headless browser automation,
  `omegaconf` for configuration and the `openai` client library for the tool-calling protocol. See
  [`requirements.txt`](requirements.txt). Playwright also needs its browsers installed separately
  (`playwright install && playwright install-deps`), which the README documents.
- **Configuration:** a single YAML file ([`config/config_template.yaml`](config/config_template.yaml))
  defining the API key, endpoint, model name, temperature, and whether extended reasoning
  (`enable_thinking`) is requested from the model.
- **Getting started:** see the [README](README.md); token usage is printed after each run, which
  makes the cost of a given task directly observable.

## Evaluation

| Criterion | Assessment |
|---|---|
| Efficiency gain | High in generic applicability; low in raw runtime speed. Setup time for a new task is close to zero, but LLM reasoning makes each run slower than a hardcoded script. |
| Reusability | High — highly reusable across diverse sites without new code. |
| Data accessibility | High — targets are public web pages requiring no credentials. Constrained in practice by robots.txt, terms of use, rate limiting and bot protection rather than by availability; note that the prototype currently masks its browser fingerprint to work around the last of these. |
| On-prem compatibility | Medium/high — the software has no external dependency beyond an OpenAI-compatible endpoint, but running it well on-premises requires local infrastructure capable of serving a medium-sized tool-calling model. |
| Low-hanging fruit for NSIs | Medium/high — small codebase (one control loop and three tools) and a single configuration file, but Playwright and its browser dependencies add installation friction, and an available tool-calling endpoint is a precondition. Running a task also requires editing the prompt in `main.py`; there is no command-line or file-based input yet. |
| Evaluation robustness | Low — outputs were assessed by inspection against known page content. No benchmark dataset or ground truth was established during the sprint; automated benchmarking remains an open roadmap item. |
| Feasibility | Medium |
| Lifespan | Medium/high — resilience to site redesign is the central design argument, and the tool interface is model-agnostic. |
| Cost effectiveness | Medium/high — the three checked-in runs completed on 13k, 14k and 24k total tokens including reasoning, so a single task costs cents at commercial rates and nothing beyond compute when self-hosted. Cost scales with the number of pages traversed rather than with a fixed overhead, so exhaustive enumeration of a large listing is the case to watch. |
| Performance vs. chatbots | Comparable, and often superior, due to specialised system prompting. |

## Key takeaways

- **System prompting** is the most critical component for tuning behaviour from a generic LLM into
  a specialised scraping agent. Most of the prototype's capability is encoded there rather than in code.
- **Reasoning trade-offs:** enabling chain-of-thought reasoning significantly improves output
  quality but increases inference latency and token cost.
- **Tool design:** keeping tools atomic — each performing one small task — ensures higher
  reliability during the LLM's tool-calling phase.

## Current status

Core fetching, extraction and reasoning are fully functional. Playwright integration is in place
and the system can launch headless browsers and render dynamic content, but the interaction between
the LLM's decision timing and asynchronous JavaScript execution is still being refined to ensure
reliable capture on dynamic sites. Automated benchmarking against standard datasets has not been started.

Smaller open items identified when the code was reviewed against this report: `interact_with_web`
does not apply the same fingerprint masking as the other two tools; the `use_extra_body`
configuration flag is overridden in code; and the task prompt is edited in `main.py` rather than
supplied as an argument. None of these block use of the prototype, but all three affect how easily
someone else can reproduce a run or vary it systematically.

## Conclusions

The sprint results demonstrate a functional proof of concept for autonomous, AI-driven web data
collection within the European Statistical System. Because the input is natural language, the tool
is flexible and accessible to users who are not developers. By replacing fragile, site-specific code
with atomic tools and LLM reasoning, the approach shifts the maintenance burden from per-site scripts
to a single prompt and tool layer — which is the main argument for its relevance to official statistics.

The principal gap is evaluation: the prototype's outputs have been judged qualitatively, and a shared
benchmark with ground truth is needed before the approach can be compared fairly against existing
rule-based scraping workflows.
