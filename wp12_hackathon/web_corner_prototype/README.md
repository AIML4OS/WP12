# Web Corner Prototype

### Scope and Limitations

This prototype was developed as a proof-of-concept during the WP12 hackathon. Its purpose is to explore how large language models combined with web scraping can identify domain-specific information on a set of URLs. The implementation is not intended for production use; it relies on experimental LLM APIs that evolve quickly and may change behaviour as models are updated.

#### Environment requirements
* Requires access to the SSP Cloud platform (Onyxia) or an equivalent environment. Instructions in [`main.py`](main.py) assume the environment variables defined in the hackathon.
* Relies on Python packages listed in `requirements.txt`. Model availability in SSP Cloud may change over time; update `test_openai.py` and the model identifiers accordingly.

#### Future directions
* Evaluate the approach against manual or rule-based scraping to quantify gains in accuracy and efficiency.
* Extend the pipeline to capture not only binary classifications but also extract structured features (e.g. email addresses, social-media links).
* Replace hard-coded model names with configurable parameters to simplify updates when newer LLMs become available.

---

This prototype explores the potential of **modern web scraping techniques combined with Large Language Models (LLMs)** to support targeted content detection on web pages. The primary use case involves determining whether a given web page contains information related to a specific variable or concept, such as:

> "Does this web page contain a job vacancy page?"
> "Is this website representing an accommodation site"
> ...

## 🧠 Use Case Description

### Input
- A list of URLs provided by the user
- A target **variable/criterion** (e.g., “job vacancy”, "hotel", “event listing”, “statistical publication”)

### Output
- A binary **yes/no** response for each URL indicating whether the target variable is matched
- Optionally, a short justification or confidence score provided by the LLM (not yet)
- A feature representation (email address, social media, etc.) (not yet)

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

## Installation instructions

Run the command `pip3 install -r requirements.txt` in order to install the required packages.

---

## Execution

Run the command `python main.py` in order to run the main program.

---

## Test data used
- During this LLM hackathon the same test data was re-used from the hackathon of the [WIN project](https://cros.ec.europa.eu/book-page/win-hackathon)

## Related work
- Note the Statistical Scraping Interest Group ([SSIG](https://github.com/SNStatComp/SSIG))
- This [repo](https://github.com/SNStatComp/webtextclassifier) of the WEB-FOSS-NL project builds on this web_corner hackathon.

---

## Troubleshooting

* Sometimes the models available are removed or changed. Run the `test_openai.py` script in order to
  check which models are currently available and also check if they support *chat* mode. If the
  selected model does not support *chat* mode then it cannot be used.
* Make sure the `api_key.txt` file contains the api key string. You can find your api key by following
  these instructions: 
  * Login or create an SSP Cloud account [SSP Cloud](https://datalab.sspcloud.fr/)
  * Go to [the AI chatbot hosted on the platform](https://llm.lab.sspcloud.fr)
  * Navigate to Settings > Account > API keys
* Make sure you have access to the bucket where the input url data file is found, otherwise, use
  a backup file with url data such as the one found under `test/` folder.

---

> **EU funding:** The development of this prototype was co-funded by the European Union's Horizon Europe programme under grant agreement No 101146355.
> **Disclaimer:** The content reflects only the authors' views and the EU is not responsible for any use that may be made of the information it contains.

![Co-funded by the European Union](../../assets/eu_cofunded.png)
