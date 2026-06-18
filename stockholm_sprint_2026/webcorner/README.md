
## Overview
### Input and Output
- **Input:** A natural language prompt including a starting URL and a specific goal (e.g., *"Find all job listings related to information security on the SCB website. Start at www.scb.se"*).
- **Output:** A structured response containing the requested information, the relevant URLs found, and a step-by-step reasoning trace.

### Evaluation Framework
The proof of concept is evaluated across several diverse use cases:
- **Comparative Analysis:** Finding specific articles on a topic across multiple sources.
- **Job Discovery:** Matching job offerings to specific professional descriptions.
- **E-commerce Extraction:** Fetching product catalogs and real-time pricing.

**Evaluation Criteria:**
- **Runtime:** Comparison against traditional rule-based scrapers.
- **Lifespan:** The "plug-and-play" capability of the toolset across different domains.
- **Flexibility:** Ease of re-use for different websites without manual code changes.

## Evaluation Summary

| Criterion | Assessment |
|----------|-------------|
| **Efficiency Gain** | High in generic applicability; Low in raw runtime speed |
| **Reusability** | High |
| **On-prem Compatibility** | Medium/High (Requires local LLM with tool-calling) |
| **Feasibility** | Medium |
| **Lifespan** | Medium/High |
| **Performance vs. Chatbots** | Comparable; often superior due to specialized system prompting |

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
