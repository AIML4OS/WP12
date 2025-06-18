# Web Corner Prototype

This prototype explores the potential of **modern web scraping techniques combined with Large Language Models (LLMs)** to support targeted content detection on web pages. The primary use case involves determining whether a given web page contains information related to a specific variable or concept, such as:

> "Does this web page contain a job vacancy page?"

## 🧠 Use Case Description

### Input
- A single URL or a list of URLs provided by the user
- A target **variable/criterion** (e.g., “job vacancy”, “event listing”, “statistical publication”)

### Output
- A binary **yes/no** response for each URL indicating whether the target variable is matched
- Optionally, a short justification or confidence score provided by the LLM

This type of workflow may support NSIs in automating web monitoring tasks for statistical content detection, e.g., for labor market indicators or price collection.

---

## 📌 Evaluation Summary

During the WP12 hackathon, the use-case was evaluated using the following dimensions:

| Parameter              | Assessment |
|------------------------|------------|
| Efficiency gain        | High       |
| Reusability            | High       |
| Data accessibility     | High       |
| On-prem compatible     | Mid        |
| Low-hanging fruit (for NSIs) | Mid  |
| Feasibility            | Mid        |
| Lifespan               | Mid        |
| Evaluation robustness  | Low        |

These assessments reflect the high potential for automation and cross-domain application, with caveats around evaluation precision and long-term stability of web content.

---

## 🔍 Technical Concept

The prototype will involve:
...
---

## 📁 Folder Structure

```plaintext
web_corner_prototype/
├── README.md
└── [to be added: scraping scripts, LLM prompt logic, test URLs, architecture diagrams]
```

## Possible test data
- During the hackathon testdata was used from the [WIN project online hackathon](https://cros.ec.europa.eu/book-page/win-hackathon)
