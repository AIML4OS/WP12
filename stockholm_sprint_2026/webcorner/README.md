## Description
This is a prototype using tool calling using an LLM for web scraping. The idea is to let the LLM infer domain specific knowledge and arrive at the relevant (sub-)page starting from a base url. Compared to traditional scraping (e.g. rule-based approaches) this allows for a generic toolset where a user prompt is sufficient to traverse a web-page.

### Input
The input is a prompt including a starting URL and a natural language goal (e.g. "Find all job listings related to information securit on the SCB website. Start at www.scb.se")

### Output
The output depends on the prompt, but its general case is a structured response to the question in the prompt containing the requested information along with the reasoning used to find it.

## Evaluation
For our proof of concept we will prompt the LLM to find specific articles and job offerings on the CBS and SCB websites. 

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
Reusability: high
On prem compatability: medium/high (requires on-site LLM that supports tool calling)
Feasibility: medium
Lifespan: medium/high
Efficiency gain: high in generic applicability, low/none in runtime gains

### Problems and limitations

- dynamically generated content is problematic. A tool for allowing the LLM to use Playwright...

### Benchmarking

## TODO

- ~~LLM can fetch hyperlinks from URLs~~
- ~~LLM can fetch page content from URLs~~
- ~~LLM path to output page is traceable~~
- ~~LLM adds reasoning to output~~ 