# Dissemination Summary Prototype

This prototype explores a practical use-case where a statistician produces a report with analysis based on published statistical data. In dissemination workflows, it is often necessary to create a **summary of the report** in both the **local language** (e.g., Portuguese) and **English**, along with a set of **keywords or tags** that describe the subject areas of the report.

The goal of this prototype is to evaluate whether Large Language Models (LLMs) can support and streamline this summarisation process in a way that is **reusable, transferable and compatible** with statistical organisations’ needs and constraints.

---

## 🧠 Use Case Description

### Input
A report document in PDF format, typically written by a subject-matter expert or analyst within a statistical office.

### Process
An AI-powered workflow performs the following steps:
1. **PDF Ingestion and Processing** – Text is extracted and prepared for semantic search.
2. **RAG Pipeline Setup** – The processed content is fed into a Retrieval-Augmented Generation (RAG) system.
3. **LLM Prompting** – A structured prompt generates:
   - A short **summary** of the document in both local language and English.
   - A set of **keywords/tags** describing the thematic content.
4. **Output Delivery** – Results are returned to the user for review or integration into the publication pipeline.

---

## 🔍 Evaluation Parameters

During the WP12 hackathon, this use-case was evaluated using the following criteria:

| Parameter              | Assessment      |
|------------------------|-----------------|
| Reusability            | High            |
| Data accessibility     | High            |
| On-prem compatibility  | High            |
| Low-hanging fruit (for NSIs) | High    |
| Feasibility            | High            |
| Lifespan               | High            |
| Efficiency gain        | Mid             |
| Evaluation robustness  | Low to mid      |

These evaluations reflect the group’s view on the potential of the solution to be adapted and reused in different national statistical institutes (NSIs), especially where document-based dissemination is common.

---

## 🛠️ Technical Components

This prototype includes a simplified reference architecture based on common AI component categories (e.g., LLMs, APIs, frameworks). A separate architectural diagram is provided in this folder showing:

- LLM used: (Llama 3.3)
- Framework: (LangChain)
- API: SSPCloud-hosted models through Ollama+OpenWebUI

Please refer to `architecture.png` in this directory for a visual representation of the chosen setup.

### Prototype B

- Python 3.12+
- PDF Parsing: `PyPDFLoader` (from LangChain Community), and `pypdf`
- Vector Database: `ChromaDB`
- Templates: `LangChain`
- AI Framework: `Ollama` 

## 🧪 Test Data
- A Portuguese example pdf: [Carregue aqui](https://www.ine.pt/ngt_server/attachfileu.jsp?look_parentBoui=731841778&att_display=n&att_download=y)
- A Swedish example pdf: [Ladda ner här](https://www.scb.se/contentassets/d5bc9f56c1f740e092734788191d9b6c/2025-03-10/instruktioner-uba.pdf)