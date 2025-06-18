# WP12 Hackathon – Lisbon 2025

This folder contains the outputs from the two-day WP12 hackathon held in **Lisbon, June 2025**, with participation from project members representing **Portugal, the Netherlands, Sweden, Ireland, and Norway**, and with **technical support from France**, who, through WP3, delivers the shared data lab infrastructure (SSPCloud) used in the AIML4OS project.

## 🎯 Objectives of the Hackathon

The goals of the hackathon were to:
- Facilitate cross-country collaboration and knowledge exchange on LLM-based applications
- Explore different technical approaches for using large language models (LLMs) in official statistics
- Create a first working draft of **Deliverable D12.1**, including concrete and testable prototypes

## 🧩 Evaluation Framework

To support sustainability and maintainability of AI systems developed within the project, each use case and prototype was evaluated along the following dimensions:

- **Efficiency gain**
- **Reusability**
- **Data accessibility**
- **On-prem compatibility**
- **Low-hanging fruit (for NSIs)**
- **Evaluation robustness**
- **Feasibility**
- **Lifespan**

## 🏗️ Architecture Perspective

An architecture perspective was introduced at the start of the hackathon to guide prototype development and encourage conscious choices around components such as:

- AI models (LLMs)
- APIs
- Frameworks

A generic reference model was used as a starting point and extended by each group. The shared architecture diagram can be found here:

📎 `ai_architecture_building_blocks.png`

## 📁 Folder Structure

Each prototype folder contains:

- A clear description of the **use case**
- Evaluation summary based on the above criteria
- A **mapping to architectural components** used
- Code and/or scripts that can be used to understand and test the prototype, especially within the **SSPCloud (Onyxia)** environment

These examples are also adaptable for use **outside of SSPCloud**, though minor changes may be required to fit other environments.

```plaintext
wp12_hackathon/
├── dissemination_summary_prototype/   # PDF summarisation and keyword extraction
├── from_pdf_to_figures/               # Structured data extraction from PDFs
├── web_corner_prototype/              # Web scraping + LLM content classification
├── reflections-and-summary.md         # Shared lessons and evaluation across prototypes
└── ai_architecture_building_blocks.png # Visual overview of architecture components
