# Use-case planning
## WP12 Stockholm Sprint 2026

This document summarises the current use-case planning for the WP12 Stockholm sprint, based on the WP12 planning meeting on 9 June 2026.

## General principles for sprint use cases

The sprint use cases should support both prototype development and deliverable material. The work should therefore capture not only code and technical results, but also documentation that can be reused in upcoming WP12 reporting.

Each use case should, as far as possible, include:

- a short description of the statistical or organisational problem
- input data or examples that can be used during the sprint
- a proposed LLM-based approach
- an alternative approach for comparison, such as manual review, rule-based methods or existing technical solutions
- a way to assess whether the output is useful or sufficiently accurate
- notes on limitations, risks and possible reuse
- documentation of the technical setup and architectural choices

A key lesson from earlier WP12 work is that evaluation needs to be considered already during use-case design. The sprint should therefore focus on use cases where it is possible to create or access some form of ground truth, expected output or comparison baseline.

## Selected use cases

### 1. Web Corner / agentic web scraping

The Web Corner use case will focus on web scraping, statistical web scraping and more agentic approaches to web-based data collection or classification.

The aim is to explore how LLM-based systems can support tasks such as:

- Use LLMs as alternatives to domain-specific knowledge
- identifying relevant information on websites
- classifying websites or web pages according to statistical concepts
- extracting information from web pages
- using tool-calling or agentic workflows to navigate or process web content
- comparing LLM-based outputs with manual, rule-based or existing scraping approaches

The planning meeting decided that Web Corner should be included during the Stockholm sprint. The ambition is to use some form of tool-calling or more agentic setup, rather than only static prompting.

Possible evaluation approaches include:

- comparing results with existing labelled data
- comparing results with an existing web scraping workflow
- using manually created ground truth during the sprint
- comparing LLM-based classification with simpler rule-based approaches

Possible data or inspiration sources mentioned during planning:

- Kaggle website classification dataset: https://www.kaggle.com/datasets/hetulmehta/website-classification
- Kaggle DMOZ URL classification dataset: https://www.kaggle.com/datasets/shawon10/url-classification-dataset-dmoz/data
- dabling: https://github.com/snstatcomp/dabling

Open questions and actions:

- Check with Norway whether the open business register includes domains or other information relevant for web scraping use cases.
- Olav will check whether internal web scraping used for CPI could be used as ground truth.
- Jakob will also check possible ground truth options.
- If no existing ground truth is available, the sprint may create manual ground truth for a limited set of websites.

### 2. Metadata management using AI

The metadata management use case will explore how AI can support the management, structuring and use of metadata in official statistics.

One example to explore is the graph-structured metadata model previously presented by Sweden. The idea is to use a graph-based structure to represent statistical programmes, datasets, variables, data structures, code lists and related metadata objects. LLM-based or agentic approaches could then be used to interact with, enrich, validate or navigate this metadata structure.

A separate branch has been prepared that should work in SSPCloud:

https://github.com/jakobengdahl/CommunityOverview/tree/stockholmsprint

Possible directions during the sprint include:

- exploring how LLMs can query or explain graph-structured metadata
- testing how AI can help create, enrich or validate metadata objects
- using graph data as context for LLM-based interaction
- exploring how metadata relationships can support better discovery and understanding of statistical outputs
- documenting architectural implications of combining LLMs with graph-based metadata structures

Possible evaluation approaches include:

- comparing AI-generated metadata suggestions with expected or manually created metadata
- checking whether the system can correctly answer questions based on the metadata graph
- testing whether generated relationships or explanations are consistent with the graph structure
- documenting errors, missing context and limitations

Open questions and actions:

- Clarify which part of the graph structure should be used as the sprint starting point.
- Ensure that the Stockholm sprint branch works in SSPCloud.
- Decide whether the use case should focus on metadata creation, metadata validation, metadata navigation or metadata-based question answering.

## Other candidate use cases

The following use cases have also been discussed and may be considered if there is participant interest and enough capacity during the sprint.

### Summarising, translating and tagging statistical output reports

This use case would build on earlier work around using LLMs to summarise statistical reports, translate content and generate tags or keywords for dissemination and reuse. It may be relevant for both legacy reports and new statistical outputs.

Possible sprint work could include:

- testing report summarisation across languages
- comparing generated summaries with existing manual summaries
- generating keywords or tags for reports
- exploring whether the approach can be made reusable across NSIs
- documenting the role of prompts, retrieval and evaluation

### Coding and classification with agentic setups

This use case would explore new approaches for coding or classification, potentially using more agentic setups. It may connect to earlier work in other WPs and to national experiences with classification tasks.

Possible sprint work could include:

- testing LLM-based classification workflows
- comparing outputs with existing coded datasets or rule-based approaches
- exploring human review and uncertainty handling
- documenting how agentic workflows could support coding tasks

## Documentation expectations

Each use-case group should aim to produce a short README or equivalent documentation covering:

- use-case description
- team members and mode of work
- data or examples used
- technical setup
- model or tools used
- evaluation approach
- results and observations
- limitations
- possible next steps
- relevance for WP12 deliverables

The documentation should be written during the sprint, not only after the technical work has ended.
