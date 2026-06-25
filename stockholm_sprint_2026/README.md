# WP12 Sprint - Stockholm 2026

This folder contains planning material and outputs from the WP12 sprint on Large Language Models held in **Stockholm, Sweden, 16-18 June 2026**, with participation from project members representing several countries, and with technical support from France through WP3 (SSPCloud/Onyxia infrastructure).

The sprint took place at **Statistics Sweden (SCB)**:

**Solna strandväg 86**
**171 54 Solna**
**Stockholm, Sweden**

## Purpose of the Sprint

The sprint is organised as part of **Work Package 12 (WP12) - Use Case: Large Language Models** within the AIML4OS project.

The aim is to support continued development of WP12 deliverables through focused collaboration on LLM-based prototypes, architectural considerations and practical implementation aspects relevant to official statistics.

The sprint builds on the experience from the WP12 hackathon held in Lisbon in June 2025, where several prototype examples were developed and later used as input for D12.1.

A key ambition for the Stockholm sprint is to make progress on both prototype development and the draft structure and initial content for the next WP12 deliverable, **D12.2**. Work during the sprint should therefore capture technical outputs as well as documentation, reflections and evaluation considerations needed for the deliverable.

## Evaluation Framework

To support sustainability and maintainability of AI systems developed within the project, each use case and prototype is evaluated along the following dimensions:

- **Efficiency gain**
- **Reusability**
- **Data accessibility**
- **On-prem compatibility**
- **Low-hanging fruit (for NSIs)**
- **Evaluation robustness**
- **Feasibility**
- **Lifespan**
- **Cost effectiveness**

## Architecture Perspective

An architecture perspective was introduced at the start of the hackathon and continued during the sprint to guide prototype development and encourage conscious choices around components such as:

- AI models (LLMs)
- APIs and tool-calling interfaces
- Frameworks and orchestration
- Hosting and on-prem considerations

A generic reference model is used as a starting point and extended by each group.

## Use Cases

The sprint focused on the following use cases:

### 1. News Corner - Statistical Media Consistency

Monitoring how official statistics are reported in the media. The system collects RSS feeds from newspapers and NSIs, compares articles with official releases using an LLM, and produces alignment scores across five dimensions (topic match, figures accuracy, reference period, source attribution, framing consistency).

See: [`News_Corner/`](News_Corner/)

### 2. Web Corner - Agentic Web Scraping

Using LLMs as agentic web scrapers that can navigate websites, extract structured information, and classify web content according to statistical concepts. The approach uses tool-calling and chain-of-thought reasoning to handle diverse websites without manual code changes.

See: [`webcorner/`](webcorner/)

### 3. Metadata Management using AI

Exploring how AI can support the management, structuring and use of metadata in official statistics, using a graph-based structure to represent statistical programmes, datasets, variables, data structures, code lists and related metadata objects.

See: [`metadata/`](metadata/)

## Scope and Limitations

- The prototypes developed during the sprint are **proof-of-concept implementations** and are not intended for production use.
- The work relies on **experimental LLM APIs** (SSPCloud/Onyxia-hosted models and commercial APIs) that may change behaviour or availability over time.
- Test datasets are **limited in size and diversity**; broader evaluation is needed before drawing general conclusions about model reliability across languages, topics, and statistical domains.
- **Data accessibility** varies across use cases: some rely on publicly available RSS feeds or websites, while others depend on access to internal NSI data or metadata registries.
- The sprint had a **fixed time frame** (three days), which limited the depth of evaluation and the number of use cases that could be explored.
- **On-premises compatibility** depends on the availability of local LLM endpoints with tool-calling support; not all models or hosting setups meet this requirement.
- Results are **model-dependent** and may differ across LLM providers, versions, and configurations.

## Quality of GenAI-based Approaches

The sprint provided practical experience with using generative AI for several statistical tasks. Key observations on output quality include:

- LLMs performed **well on structured English-language tasks** such as comparing statistical releases with news articles and extracting information from web pages.
- **Multilingual performance** was generally good but showed weaknesses in edge cases, such as detecting reference-period mismatches in Slovenian text.
- **Prompt design** had a significant impact on result quality; carefully structured prompts with explicit instructions (e.g., requiring full reference-period extraction before judgement) produced notably better results.
- The **reasoning trace** provided by agentic setups (chain-of-thought) improved output quality but increased inference latency and cost.
- **Reproducibility** varied: the same prompt could yield different results across runs or model versions, highlighting the importance of evaluation benchmarks.
- Open-source models hosted on SSPCloud (e.g., GEMMA4-26B-MOE) showed **promising but uneven quality** compared to commercial models, particularly for complex or multilingual tasks.

## Future Considerations

- **Benchmark development**: Create shared benchmark datasets for each use case so that model performance can be tracked over time and across LLM versions.
- **On-prem deployment**: Further testing of open-source models in on-premises environments to support NSIs with strict data-protection requirements.
- **Cross-NSI reuse**: Package prototypes so that other NSIs can deploy and configure them with minimal effort.
- **Integration with existing workflows**: Explore how prototype outputs can feed into existing statistical production systems (dissemination, metadata registries, quality assurance).
- **Evaluation methodology**: Develop more robust evaluation approaches, including human-in-the-loop validation and comparison with rule-based baselines.
- **Scaling**: Assess performance and cost implications of running prototypes at operational scale (e.g., monitoring hundreds of RSS feeds or classifying thousands of web pages).
- **D12.2 deliverable**: Use sprint outputs as core input for the next WP12 deliverable.

## Agenda

The draft agenda is available here:

- [agenda.md](agenda.md)

## Folder Structure

Each prototype folder contains:

- A clear description of the **use case**
- Evaluation summary based on the above criteria
- A **mapping to architectural components** used
- Code and/or scripts that can be used to understand and test the prototype, especially within the **SSPCloud (Onyxia)** environment

```plaintext
stockholm_sprint_2026/
├── README.md                      # This file
├── agenda.md                      # Sprint agenda
├── background.md                  # Background on WP12 and AIML4OS
├── use-case-planning.md           # Use-case planning notes
├── remote-participation.md        # Remote participation info
├── meeting_notes/                 # Meeting notes
├── News_Corner/                   # News Corner prototype (media consistency)
│   ├── README.md
│   ├── Report.md
│   ├── Draft-architecture.md
│   └── News-Corner-app.pptx
├── webcorner/                     # Web Corner prototype (agentic web scraping)
│   ├── README.md
│   ├── main.py
│   ├── config/
│   ├── tools/
│   └── output*.md
├── metadata/                      # Metadata management prototype
│   └── README.md
└── D12.2/                         # Draft structure for deliverable D12.2
    └── README.md
```

## First Steps to Start

The projects within this repository require an API key. To retrieve it, you'll need an SSP Cloud account. If it's not already done, create an account on the [SSP Cloud](https://datalab.sspcloud.fr/) platform. Then, to get your API key, go to [the AI chatbot hosted on the platform](https://llm.lab.sspcloud.fr) > settings > account > API KEYS

Add your API key in an environment variable (LLM_API_KEY):
```
export LLM_API_KEY=your_api_key_you_just_retrieved
```

## Note

This folder is under active development. Content may be updated before, during and after the Stockholm sprint.
