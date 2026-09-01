## Description
The goal of this project is to move beyond traditional, rule-based web scraping by creating an "agentic" scraper. Instead of writing specific scripts for every website, we use an LLM with tool calling. Tool-calling allows LLMs to go beyond text and enables the automatic invoking of specialized functions that are fed back to the LLM in a loop. 

This prototype was developed during the WP12 Stockholm sprint (16-18 June 2026) using the [SSPcloud](https://datalab.sspcloud.fr/) environment, though the software only requires access to an LLM with tool-calling capabilities via an OpenAI-compatible API. During the sprint the model used was `gemma4-26b-moe`, served from the SSPCloud-hosted vLLM endpoint.

A narrative write-up of the prototype - architecture, configuration, evaluation and conclusions - is in [Report.md](Report.md).

**Key Capabilities:**
- **URL Discovery:** Finding relevant statistical units or specific pages within a domain. If not provided, the LLM generates a most plausible URL itself or tries to find it on the web.
- **Web-page Traversal:** Navigating through "Hubs" (list pages) to reach "Leaf Nodes" (detail pages).
- **Content Extraction and processing:** The LLM extracts the web-page content using the tools and processes the content to generate the output specified in the user prompt

**User stories explored**
- **Comparative Analysis:** Finding specific information on a topic across multiple sources (e.g. comparing statistics from SCB and CBS).
- **OJA Discovery:** Matching online job advertisements to specific professional descriptions (e.g. finding non-office jobs at CBS).
- **E-commerce Extraction:** Fetching product catalogs and real-time pricing. (e.g. fetching prices and reviews from online stores)

**Implemented Tools:**
- `fetch_page_urls`: Uses Playwright to render the page and extract all valid hyperlinks.
- `fetch_page_content`: Uses Playwright to extract the fully rendered text content.
- `interact_with_web`: Allows the agent to perform actions like `click` or `scroll` to trigger dynamic content loading.

## Getting Started

### Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AIML4OS/WP12.git
   cd stockholm_sprint_2026/webcorner
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure the LLM:**
   - Copy the template configuration: `cp config/config_template.yaml config/config.yaml`
   - Edit `config/config.yaml` with your API key and the appropriate `api_host` and `api_model`.

5. **Run the code:**
   Modify the `user_prompt` in `main.py` to your desired task and execute:
   ```bash
   python main.py
   ```


### Headless Browsing (Playwright)

To handle the modern web with dynamically rendered content, we added a tool `interact_with_web` that allows the use of Playwright

To use Playwright some additional installation is required after setting up the virtual environment:

```bash
playwright install
playwright install-deps
```

### Input and Output
- **Input:** A natural language prompt including a starting URL and a specific goal (e.g., *"Find all job listings related to information security on the SCB website. Start at www.scb.se"*).
- **Output:** A structured response containing the requested information, the relevant URLs found, and a step-by-step reasoning trace.

## Evaluation 

| Criterion | Assessment |
|----------|-------------|
| **Efficiency gain** | High in generic applicability; low in raw runtime speed. Setup time for a new task is close to zero, but LLM reasoning makes each run slower than a hardcoded script. |
| **Reusability** | High - reusable across diverse sites without writing new code. |
| **Data accessibility** | High - targets are public web pages requiring no credentials. Constrained in practice by robots.txt, terms of use, rate limiting and bot protection rather than by availability. |
| **On-prem compatibility** | Medium/high - the software has no external dependency beyond an OpenAI-compatible endpoint, but running it well on-premises requires local infrastructure capable of serving a medium-sized tool-calling model. |
| **Low-hanging fruit for NSIs** | Medium/high - small codebase (one control loop and three tools) and a single configuration file, but Playwright and its browser dependencies add installation friction, and an available tool-calling endpoint is a precondition. |
| **Evaluation robustness** | Low - outputs were assessed by inspection against known page content. No benchmark dataset or ground truth was established during the sprint. |
| **Feasibility** | Medium |
| **Lifespan** | Medium/high - resilience to site redesign is the central design argument, and the tool interface is model-agnostic. |
| **Cost effectiveness** | Medium - a task consumes many round-trips, and chain-of-thought reasoning increases token count further. Cost is bounded when the model is self-hosted; against a commercial API it scales with the number of pages traversed. `main.py` prints token usage after each run, which makes the cost of a given task directly observable. |
| **Performance vs. chatbots** | Comparable; often superior due to specialized system prompting. |

## Key Takeaways
- **System Prompting:** The system prompt is the most critical component for tuning behavior from a generic LLM into a specialized scraping agent.
- **Reasoning Trade-offs:** Enabling "Chain of Thought" reasoning significantly improves output quality but increases inference latency.
- **Tool Design:** Keeping tools "atomic" (performing only one small task) ensures higher reliability during the LLM's tool-calling phase.

### Current Development: Playwright Integration
We have initiated integration with **Playwright** to handle modern, JavaScript-heavy websites. 
- **Progress:** The system can now launch headless browsers and attempt to render dynamic content.
- **Current Status:** "Almost there"—while the plumbing is in place, the interaction between the LLM's decision-making and the timing of asynchronous JS execution is still being refined to ensure reliable data capture.

## Roadmap
- [x] LLM can fetch hyperlinks from URLs
- [x] LLM can fetch page content from URLs
- [x] LLM path to output page is traceable
- [x] LLM adds reasoning to output
- [/] Robust handling of dynamically loaded (JS) content via Playwright
- [ ] Automated benchmarking against standard datasets
