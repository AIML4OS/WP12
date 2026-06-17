## Description
This is a prototype using tool calling using an LLM for web scraping. The idea is to let the LLM infer domain specific knowledge and arrive at the relevant (sub-)page starting from a base url. Compared to traditional scraping (e.g. rule-based approaches) this allows for a generic toolset where a user prompt is sufficient to traverse a web-page.

### Input and output
The input is a prompt including a starting URL and a natural language goal (e.g. "Find all job listings related to information securit on the SCB website. Start at www.scb.se")

### Output
The output depends on the prompt, but its general case is a structured response to the question in the prompt containing the requested information along with the reasoning used to find it.

## Evaluation
For our proof of concept we will prompt the LLM using a variety of use cases
- Find specific articles on a topic 
- Find job offerings matching a description
- Fetch all products and product prices 

### Evaluation criteria
One tangible criteria could be runtime compared to traditional scrapers
Lifespan, expectation is high due to plug and play capabilities LLMs with the software
Flexibility and re-usability are also relevant but harder to quantify

## Architecture
The software will be generic and support any LLM that can be approached using the API and supports tool-calling.
The tools itself will be simplistic by design and carry out one small task per tool.
Only relevant tools will be provided.
To avoid infinite loops we do not permit re-visits for the proof of concept

## Evaluation summary

| Criterion | Assessment |
|---|---|
| Efficiency gain | High in generic applicability, low/none in runtime gains |
| Reusability | High |
| Data accessibility |  |
| On-prem compatibility | Medium/high (requires on-site LLM that supports tool calling) |
| Low-hanging fruit for NSIs |  |
| Evaluation robustness | We're not using a benchmark dataset... |
| Feasibility | Medium |
| Lifespan | Medium/high |
| Cost effectiveness |  |

### Problems and limitations

### Benchmarking

## TODO

- ~~LLM can fetch hyperlinks from URLs~~
- ~~LLM can fetch page content from URLs~~
- ~~LLM path to output page is traceable~~
- ~~LLM adds reasoning to output~~ 