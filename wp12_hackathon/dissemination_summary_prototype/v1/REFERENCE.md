# v1 Programmer Reference

Quick-reference for every public parameter, model field, function signature,
environment variable, and CLI flag.  Keep this open while coding.

---

## Table of contents

1. [Config schema (`config.yml`)](#1-config-schema)
2. [Environment variables](#2-environment-variables)
3. [`SummarizeOptions` — all fields](#3-summarizeoptions)
4. [`document_load` — public API](#4-document_load-public-api)
5. [`endpoints` — public API](#5-endpoints-public-api)
6. [`summarizer` — public API](#6-summarizer-public-api)
7. [`bench` — public API](#7-bench-public-api)
8. [CLI surfaces](#8-cli-surfaces)
9. [Cache key composition](#9-cache-key-composition)
10. [Import order rule](#10-import-order-rule)

---

## 1  Config schema

Full `config.yml` with every field and its default.

```yaml
cache_dir: .cache          # relative to config.yml, or absolute

defaults:
  out_lang: pt-pt          # BCP-47 tag for summary output language
  max_words: 200           # target word count for the generated summary
  max_keywords: 6          # max items in keywords list
  max_tags: 5              # max items in tags list
  processing_mode: vector  # "vector" (RAG via Chroma) | "plain" (full-text)
  pdf_loader: docling      # "docling" (smart, slow) | "pypdf" (fast fallback)
  embedding: local         # "local" (HuggingFace BAAI/bge-m3) | <endpoint URL>
  llm:
    timeout_sec: 600       # per-request LLM timeout
    max_retries: 2
  temperature: 0.1         # lower = more deterministic JSON output

docling:
  do_ocr: false            # enable OCR pipeline (for scanned / image PDFs)
  ocr_engine: easyocr      # "easyocr" | "rapidocr" | "tesseract"
                           #   easyocr  : multilingual; models ~200 MB, cached in
                           #              .cache/easyocr/ on first use
                           #   rapidocr : ONNX bundled in pip package, zero extra
                           #              download, fastest option
                           #   tesseract: system binary required; set TESSDATA_PREFIX
  table_mode: fast         # "fast" | "accurate" (accurate improves table quality
                           # but adds significant latency on CPU)
  convert_timeout_sec: 900 # abort Docling if conversion takes longer than this

local_embedding:
  model: BAAI/bge-m3       # any sentence-transformers model name

bench:
  pdfs_dir: bench/pdfs
  configs_dir: bench/configs
  runs_dir: bench/runs
  stability_repeat: 2      # how many times to repeat each scenario (for cosine stability)

instruction_prompt: |      # multiline; replaces the entire system instruction
  You are a statistician…

endpoints:                 # list; bare minimum is just "url:"
  - url: https://…
    id: ssp                # optional; derived from hostname if omitted
    name: SSP Cloud        # optional display name
    auth_env: SSP_KEY      # name of env var holding the API key (never put key here)
    auth_help: "…"         # shown in probe output / UI when auth fails
    network_hint: "…"      # shown in probe output to remind about VPN / network
```

---

## 2  Environment variables

Set in `.env` (loaded by `_bootstrap.py` before any heavy import).

| Variable | Default (set by bootstrap) | Purpose |
|---|---|---|
| `HF_HOME` | `.cache/hf` | HuggingFace model cache root |
| `DOCLING_ARTIFACTS_PATH` | `.cache/docling` | Docling layout / table model cache |
| `TORCH_HOME` | `.cache/torch` | PyTorch hub cache |
| `EASYOCR_MODULE_PATH` | `.cache/easyocr` | EasyOCR model download target |
| `DOCLING_DO_OCR` | `false` | Enable OCR when no caller override given |
| `DOCLING_OCR_ENGINE` | `easyocr` | OCR engine when no caller override given |
| `DOCLING_TABLE_MODE` | `fast` | Table mode when no caller override given |
| `DOCLING_CONVERT_TIMEOUT_SEC` | `900` | Converter timeout (seconds) |
| `SSP_KEY` | — | API key for SSP Cloud endpoint |
| `OPENAI_API_KEY` | — | API key for OpenAI endpoint |
| `ANONYMIZED_TELEMETRY` | `False` | Disable ChromaDB telemetry |
| `CHROMA_TELEMETRY_ENABLED` | `False` | Disable ChromaDB telemetry (alt key) |
| `TOKENIZERS_PARALLELISM` | `false` | Silence HF tokenizer fork warning |
| `HF_HUB_DISABLE_SYMLINKS_WARNING` | `1` | Silence HF symlink warning on Windows |

---

## 3  `SummarizeOptions`

Defined in `summarizer.py`. All fields have defaults; only `llm_endpoint` and
`llm_model` **must** be set before calling `summarize()`.

| Field | Type | Default | Notes |
|---|---|---|---|
| `processing_mode` | `"vector"` \| `"plain"` | `"vector"` | `"vector"` builds a per-run in-memory Chroma store and uses RAG; `"plain"` sends the full text to the LLM in one call (fine for short docs) |
| `pdf_loader` | `"docling"` \| `"pypdf"` | `"docling"` | Docling is smarter (tables, headings, hybrid chunker) but slower; PyPDF is a fast fallback |
| `do_ocr` | `bool` | `False` | Enable the Docling OCR pipeline. Only useful for scanned / image-based PDFs. Adds ~40–120 s on CPU |
| `ocr_engine` | `"easyocr"` \| `"rapidocr"` \| `"tesseract"` | `"easyocr"` | Active only when `do_ocr=True`. See §1 for trade-offs |
| `embedding` | `str` | `"local"` | `"local"` → HuggingFace `local_embedding.model`; any endpoint URL → remote embedding via that endpoint |
| `llm_endpoint` | `str` | `""` | **Required.** Endpoint URL or `id` from `config.yml` |
| `llm_model` | `str` | `""` | **Required.** Model name as returned by the endpoint |
| `out_lang` | `str` | `"pt-pt"` | BCP-47 language tag injected into the prompt |
| `max_words` | `int` | `200` | Target word count. The LLM is instructed to stay within this limit |
| `max_keywords` | `int` | `6` | Max items in the `keywords` array |
| `max_tags` | `int` | `5` | Max items in the `tags` array |
| `temperature` | `float` | `0.1` | LLM temperature. `0.0` = fully deterministic (best for bench reproducibility) |
| `use_cache` | `bool` | `True` | Read from / write to `.cache/summaries/`. Set `False` in bench scenarios |
| `display_source_name` | `str \| None` | `None` | Override the filename shown in source-attribution entries (useful when the PDF came from a temp file) |

**Cache note:** The cache key is a SHA-256 hash of all fields above **plus** `config.instruction_prompt`. Changing *any* of them produces a new cache entry automatically.

---

## 4  `document_load` public API

**Graduation-ready.** Never imports from `summarizer`, `bench`, or `app_summarizer`.

```python
from document_load import (
    fingerprint,
    prewarm,
    load_docling_chunks,
    load_docling_text,
    load_pypdf_chunks,
    load_pypdf_text,
)
```

### `fingerprint(pdf_path) -> str`
SHA-256 of the raw PDF bytes, first 16 hex chars. Pure function; no I/O side-effects beyond reading the file.

### `prewarm() -> None`
Eagerly load the Docling converter + HybridChunker singletons (uses env-var defaults). Call on app startup to avoid a cold-start delay on the first request.

### `load_docling_chunks(pdf_path, fp, *, do_ocr, ocr_engine, table_mode, loader_id) -> list[Document]`

| Param | Type | Default | Notes |
|---|---|---|---|
| `pdf_path` | `Path\|str` | — | Path to the PDF file |
| `fp` | `str` | — | Fingerprint string (from `fingerprint()`); embedded in every chunk's `chunk_id` |
| `do_ocr` | `bool\|None` | `None` → env var | Override `DOCLING_DO_OCR` for this call |
| `ocr_engine` | `str\|None` | `None` → env var | Override `DOCLING_OCR_ENGINE` (`"easyocr"`, `"rapidocr"`, `"tesseract"`) |
| `table_mode` | `str\|None` | `None` → env var | Override `DOCLING_TABLE_MODE` (`"fast"` or `"accurate"`) |
| `loader_id` | `str` | `"docling"` | Written into each chunk's `loader` metadata field |

**Chunk metadata schema** (guaranteed on every returned `Document`):

| Key | Type | Notes |
|---|---|---|
| `chunk_id` | `str` | `"<fp>::p<page>::b<block_idx:04d>"` — stable, deterministic |
| `doc_fingerprint` | `str` | Same as `fp` argument |
| `source_pdf` | `str` | Filename only (e.g. `"Aereo.pdf"`) |
| `source_path` | `str` | Absolute path |
| `page` | `int` | 1-based; `0` if unknown |
| `loader` | `str` | `"docling"` or `"pypdf"` |
| `bbox` | `list` | `[x1,y1,x2,y2]` — Docling only, omitted for PyPDF |
| `section` | `str` | Last heading above the chunk — Docling only, when available |

### `load_pypdf_chunks(pdf_path, fp, *, loader_id) -> list[Document]`
Same metadata schema as above. Splits each page with `RecursiveCharacterTextSplitter(chunk_size=1000, overlap=200)`.

### `load_docling_text(pdf_path, *, do_ocr, ocr_engine, table_mode) -> str`
Full plain-text extraction via Docling (for `processing_mode="plain"`).

### `load_pypdf_text(pdf_path) -> str`
Full plain-text extraction via PyPDF.

---

## 5  `endpoints` public API

**Graduation-ready.** Never imports from `summarizer`, `bench`, or `app_summarizer`.

```python
from endpoints import (
    load_config,
    probe,
    list_models,
    make_chat_model,
    make_embeddings,
    setup,
    # models:
    Config, Defaults, DoclingOptions, LocalEmbedding,
    BenchSettings, Endpoint, EndpointStatus, LLMDefaults,
)
```

### `load_config(path=None) -> Config`
Load and validate `config.yml`. Falls back to hard-coded defaults if the file doesn't exist. `path` defaults to `config.yml` next to `endpoints.py`.

### `probe(endpoint, *, timeout=25) -> EndpointStatus`
Auto-detect the API kind and list available models. Probe sequence: Ollama `/api/tags` → OpenAI `/v1/models` → bare Ollama `/api/version`. Non-blocking on unreachable hosts (returns `EndpointStatus(ok=False, error=…)`).

### `list_models(endpoint, role="all") -> list[str]`
Convenience wrapper around `probe()`. `role`: `"chat"` | `"embed"` | `"all"`.

### `make_chat_model(endpoint, model, *, temperature, timeout, max_retries) -> BaseChatModel`
Returns `ChatOllama` for Ollama-native endpoints, `ChatOpenAI` for everything else.

### `make_embeddings(target, model, *, endpoints) -> Embeddings`
`target="local"` → `HuggingFaceEmbeddings(model)`. Any URL → `OllamaEmbeddings` or `OpenAIEmbeddings` depending on the probed kind.

### `setup(stage="all") -> None`
`stage`: `"deps"` (pip install), `"models"` (download HF + Docling models), `"all"`.

### `DoclingOptions` fields (reminder)

| Field | Default | Notes |
|---|---|---|
| `do_ocr` | `False` | |
| `ocr_engine` | `"easyocr"` | `"easyocr"` \| `"rapidocr"` \| `"tesseract"` |
| `table_mode` | `"fast"` | `"fast"` \| `"accurate"` — accurate improves table extraction quality |
| `convert_timeout_sec` | `900` | |

---

## 6  `summarizer` public API

```python
from summarizer import (
    summarize,
    SummarizeOptions,
    cache_path_for,
    safe_json_parse,
    build_sources_from_docs,
    build_source_from_full_text,
    build_numeric_coverage,
    get_shared_prompts,
)
```

### `summarize(pdf_path, options, *, config, cancel_event, on_progress) -> dict`

| Param | Type | Notes |
|---|---|---|
| `pdf_path` | `Path\|str` | PDF to summarise |
| `options` | `SummarizeOptions` | See §3 |
| `config` | `Config\|None` | If `None`, calls `load_config()` |
| `cancel_event` | `threading.Event\|None` | Set to abort mid-run |
| `on_progress` | `Callable[[str], None]\|None` | Called with each progress line |

**Return dict schema** (all keys always present):

| Key | Type | Notes |
|---|---|---|
| `summary` | `str` | The generated summary text |
| `keywords` | `list[str]` | |
| `tags` | `list[str]` | |
| `sources` | `list[dict]` | See source schema below |
| `numeric_claims` | `list[dict]` | `[{"number": str, "source_ids": [str]}]` |
| `unmatched_numbers` | `list[str]` | Numbers in summary not found in any source excerpt |
| `_from_cache` | `bool` | `True` if result was loaded from disk cache |
| `_timing_sec` | `dict[str, float]` | Keys: `chunks`, `vector_store`, `pdf_load`, `llm` |

**Source entry schema:**

| Key | Notes |
|---|---|
| `id` | `"1"`, `"2"`, … |
| `source` | Filename |
| `location` | `"p.12"` or `"full document"` |
| `excerpt` | Up to 280 chars of source text |
| `supports_numbers` | Numbers from the summary found in this excerpt |

### `cache_path_for(fp, options, config) -> Path`
Returns the `.cache/summaries/<fp>__<params_hash>.json` path without reading it. Useful for cache-hit indicators.

### `safe_json_parse(result) -> tuple[dict, bool]`
Parse LLM output as JSON. Returns `(parsed_dict, used_fallback)`. `used_fallback=True` means the direct parse failed and brace-counting was used — the bench harness uses this as the `json_ok` metric.

### `get_shared_prompts(instruction_prompt) -> tuple[ChatPromptTemplate, PromptTemplate]`
Returns `(chat_prompt, retrieval_string_prompt)` — the three-layer prompt pair used internally by `summarize()`. Useful for prompt experimentation outside the full pipeline.

---

## 7  `bench` public API

```python
from bench import (
    bench,
    load_scenarios,
    BenchScenario,
    BenchResult,
)
```

### `load_scenarios(configs_dir) -> list[BenchScenario]`
Reads `*.yml` from `configs_dir`. **Raises** if any scenario has `use_cache: true` (bench must measure the cold pipeline).

### `bench(scenarios, pdfs, *, config, on_progress) -> list[BenchResult]`
Runs every `(scenario × pdf × repeat)` combination, computes metrics, appends to a timestamped JSONL file in `bench/runs/`, and returns the results.

### `BenchResult` fields (metrics)

| Field | Type | Notes |
|---|---|---|
| `numeric_coverage` | `float` | matched / total numbers in summary (1.0 if no numbers) |
| `word_count_ratio` | `float` | `actual_words / max_words` — should be close to 1.0 |
| `json_ok` | `bool` | `True` if the LLM returned valid JSON without needing brace-counting fallback |
| `stability_cosine` | `float\|None` | Cosine similarity between repeat-run summary embeddings; `None` if `repeat=1` |
| `reproducible` | `bool\|None` | Byte-identical across repeats when `temperature=0.0`; `None` if `repeat=1` |
| `timing_sec` | `dict` | Average timing breakdown across repeats |
| `unmatched_numbers_count` | `int` | Count of numbers in summary with no matching source |
| `error` | `str\|None` | Error message if the run failed |

---

## 8  CLI surfaces

### `python summarizer.py`

```
summarizer setup [--stage deps|models|all]
summarizer probe [--endpoint URL]
summarizer models --endpoint URL [--role chat|embed|all]
summarizer summarize PATH
    --endpoint URL          required
    --model NAME            required
    --mode vector|plain     default: config default
    --loader docling|pypdf  default: config default
    --embedding local|URL   default: config default
    --lang CODE             default: config default  (e.g. pt-pt, en)
    --max-words N           default: config default
    --max-keywords N        default: config default
    --max-tags N            default: config default
    --temperature F         default: config default
    --do-ocr                enable OCR pipeline
    --ocr-engine easyocr|rapidocr|tesseract
    --no-cache              bypass read + write of summary cache
    --display-name NAME     override source filename in attribution output
```

### `python bench.py`

```
bench [--configs DIR] [--pdfs DIR] [--scenario NAME ...] [--compare FILE]
```

---

## 9  Cache key composition

`.cache/summaries/<fingerprint>__<params_hash>.json`

- **`fingerprint`** — first 16 hex chars of `sha256(pdf_bytes)`. Changes only when the PDF file itself changes.
- **`params_hash`** — first 16 hex chars of `sha256(json.dumps(key_dict, sort_keys=True))` where `key_dict` contains:

```
embedding, llm_endpoint, llm_model, max_words, max_keywords, max_tags,
out_lang, processing_mode, pdf_loader, do_ocr, ocr_engine (or "off"),
temperature, instruction_prompt
```

Any change to any of those fields produces a **new** cache file automatically. Old files are never deleted — remove them manually or wipe `.cache/summaries/` to free space.

---

## 10  Import order rule

Every module in this package starts with:

```python
import _bootstrap  # noqa: F401 — env setup; must come before heavy imports
```

This **must** be the first non-stdlib import. `_bootstrap` sets `HF_HOME`,
`DOCLING_ARTIFACTS_PATH`, `TORCH_HOME`, `EASYOCR_MODULE_PATH`, and misc flags
before torch / langchain / docling are imported. If those libraries load first
they lock in `~/.cache` defaults and `_bootstrap` has no effect.

**Graduation-readiness check** (run after any refactor):
```bash
python tools/check_shareable.py
```
Asserts `document_load.py` and `endpoints.py` contain no imports of
`summarizer`, `bench`, or `app_summarizer`.
