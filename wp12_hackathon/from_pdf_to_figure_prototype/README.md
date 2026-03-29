# From PDF to Figures – Prototype

### Scope and Limitations

This prototype was developed as a proof-of-concept during the WP12 hackathon. It explores how AI tools can automate the extraction of structured data (tables, figures, key financial variables) from annual reports in PDF format. The implementation is in early development and is not intended for production use; PDF layouts vary widely and extraction accuracy depends heavily on document structure.

#### Environment requirements
* Requires Python 3.10+ and the packages listed in `requirements.txt` (PyMuPDF, PyPDF2, pytesseract, Pillow).
* OCR-based extraction via pytesseract requires Tesseract to be installed on the host system.

#### Future directions
* Improve extraction accuracy for complex table layouts and multi-column pages.
* Add LLM-based post-processing to classify and validate extracted values (e.g. revenue, number of employees).
* Support additional output formats (CSV, JSON) for downstream integration.

---

This prototype aims to explore how AI tools can support the extraction and processing of structured quantitative content (e.g., tables, figures, numeric summaries) from annual reports and similar content in PDF format.

The goal is to enable downstream use of this information for analysis, data validation, or integration into other statistical processes.

## 📌 Status

This prototype is currently under early development. More information about the architecture, chosen technologies, and implementation approach will be added here as the work progresses.

## Installation

Run the command `pip3 install -r requirements.txt` in order to install the required packages.

## Execution

Run the command `python app.py` in order to execute the main program.

## 📁 Structure

```plaintext
from_pdf_to_figures/
├── README.md
└── [to be added: code, scripts, architecture diagrams, data samples]
```

## Troubleshooting

---

> **EU funding:** The development of this prototype was co-funded by the European Union's Horizon Europe programme under grant agreement No 101146355.
> **Disclaimer:** The content reflects only the authors' views and the EU is not responsible for any use that may be made of the information it contains.

![EU_flag](../../assets/eu_flag.png)