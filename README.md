# AIML4OS - Work Package 12 (WP12): Large Language Models for Official Statistics

> **EU funding:** This project has received funding from the European Union's Horizon Europe programme under grant agreement No 101146355 (AIML4OS). Any opinions expressed are those of the authors and do not necessarily reflect the views of the European Union.

![Co-funded by the European Union](assets/eu_cofunded.png)

This repository contains prototypes and guidance developed within work package 12 (WP12) of AIML4OS. All prototypes here are **demonstrators** intended for experimentation and knowledge sharing. They are not production-ready solutions and should be treated accordingly.

See the [D12.1 report](Deliverables/D12.1/AIML4OS%20D12.1%20Report.md) for more details on scope, limitations and future plans.

---

Welcome to the official repository for **Work Package 12 (WP12)** of the EU-funded project **AIML4OS - Artificial Intelligence and Machine Learning for Official Statistics**.

## Purpose of this Repository

This repository collects all publicly shareable outputs, experiments, and prototypes developed within WP12, which explores the potential of **Large Language Models (LLMs)** to create value for statistical organisations across Europe.

WP12 focuses on:
- Prototyping use cases where LLMs can support or enhance statistical production (e.g. data handling, code translation, chatbot interfaces, and large document analysis).
- Analysing architectural implications and governance aspects of integrating LLM-based tools in public sector workflows.
- Exploring reusability and reproducibility across organisations, including with open-source and cloud-based tools.

This repository is meant to serve as a working space where ongoing development and early deliverables are documented and shared with the broader community involved in the AIML4OS project.

> ⚠️ **Note**: All content in this repository is currently under active development and experimentation. The examples and code provided are intended for demonstration and learning purposes only. They are not production-ready and should not be used in live systems.

---

## Repository Structure

```plaintext
/
├── Deliverables/
│   └── D12.1/                               # D12.1 report (Lisbon hackathon, June 2025)
├── wp12_hackathon/                          # Lisbon hackathon prototypes (input to D12.1)
│   ├── dissemination_summary_prototype/     # Summarisation and tagging from report PDFs
│   ├── from_pdf_to_figure_prototype/        # Structured data extraction from PDFs
│   └── web_corner_prototype/                # Web scraping and LLM content classification
├── stockholm_sprint_2026/                   # Stockholm sprint (June 2026), input to D12.2
│   ├── D12.2/                               # Draft deliverable D12.2
│   ├── News_Corner/                         # Statistical media consistency (experiment)
│   ├── webcorner/                           # Agentic web scraping (prototype)
│   └── metadata/                            # Metadata Graph (prototype, own repository)
└── meeting_notes/                           # WP12 meeting notes
```

Related repository: the Metadata Graph prototype has its own repository at
<https://github.com/AIML4OS/WP12_MetadataGraph>.

---

## License

The source code in this repository is licensed under the [MIT License](LICENSE).

Note that individual prototypes depend on third-party packages with their own licenses. Most dependencies use permissive licenses (MIT, BSD, Apache 2.0). One notable exception is **PyMuPDF** (used by `from_pdf_to_figure_prototype`), which is licensed under AGPL 3.0 or a commercial Artifex license. Users who install and run that prototype must comply with PyMuPDF's license terms.
