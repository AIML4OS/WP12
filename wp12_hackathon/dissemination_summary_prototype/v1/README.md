# WP12 v1 — PDF Summarizer

A lean rewrite of the WP12 RAG summarizer. Takes a PDF, returns a tight JSON
summary with numeric verification and source citations. Includes a benchmarking
harness and a NiceGUI web app.

## Files

| File | Purpose |
|---|---|
| `summarizer.py` | Engine + CLI: `setup`, `probe`, `models`, `summarize` |
| `bench.py` | Eval harness over `bench/configs/*.yml` scenarios |
| `app_summarizer.py` | NiceGUI web app on http://localhost:5001 |
| `document_load.py` | PDF loaders + chunk_id — graduation-ready (no consumer knowledge) |
| `endpoints.py` | Config, endpoint probe, LangChain factories — graduation-ready |
| `_bootstrap.py` | Env-var setup; imported first by every other module |

## Quickstart

```powershell
# 1 — create the Python 3.12 environment (first time only)
#     The env lives inside v1/.conda/ and is gitignored.
conda create --prefix .\.conda python=3.12 -y
.\.conda\python.exe -m pip install -r requirements.txt

# 2 — copy and fill in secrets
copy .env.example .env   # then open .env and set SSP_KEY / OPENAI_API_KEY etc.

# 3 — pre-download models for offline use
.\.conda\python.exe summarizer.py setup --stage models

# 4 — run the web app
.\.conda\python.exe app_summarizer.py
#   → http://localhost:5001
```

**Activate in your shell** (optional, for interactive use):
```powershell
conda activate .\.conda
python app_summarizer.py
```

## CLI usage

```bash
# Probe all configured endpoints
python summarizer.py probe

# List models from a specific endpoint
python summarizer.py models --endpoint https://llm.lab.sspcloud.fr/api --role chat

# Summarize a PDF (prints JSON to stdout)
python summarizer.py summarize demo_docs/Aereo.pdf \
  --endpoint https://llm.lab.sspcloud.fr/api \
  --model qwen3-6-35b-moe \
  --mode vector --loader docling --lang pt-pt

# Run the bench harness (all scenarios × all PDFs in bench/)
python bench.py

# Run a specific scenario only
python bench.py --scenario ssp_docling_vector_local

# Compare a previous run
python bench.py --compare bench/runs/20260514T001234Z.jsonl
```

## Cache

`.cache/summaries/<fp>__<params_hash>.json` holds previously-computed summaries.
Delete a file to force regeneration. Caches are **not** auto-invalidated — every
parameter that can change the output (model, prompt, mode, loader, embedding, language,
word limit) is included in the hash, so changing any knob automatically produces a new
cache file.

## Adding / changing endpoints

Edit `endpoints:` in `config.yml`. The bare minimum is just a URL:

```yaml
endpoints:
  - url: http://localhost:11434
```

The probe auto-detects API kind (Ollama native vs OpenAI-compatible). API keys go in
`.env` only — never in `config.yml`.

## Graduation-ready files

`document_load.py` and `endpoints.py` are written so that a future Knowledge Base
project (its own repo) can lift them out with `git mv` + a `pyproject.toml`. They
have no summarizer-specific arguments and never import from `summarizer.py`, `bench.py`,
or `app_summarizer.py`. Run `python tools/check_shareable.py` to verify.
