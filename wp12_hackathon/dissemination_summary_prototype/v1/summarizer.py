"""
v1 PDF Summarizer — pipeline engine + CLI.

Imports ONLY from _bootstrap, document_load, endpoints, and third-party libs.
Never imports from bench.py or app_summarizer.py.

Public API
----------
    summarize(pdf_path, options, *, config, cancel_event, on_progress) -> dict

Output JSON schema (always present, even on errors):
    summary         str
    keywords        list[str]
    tags            list[str]
    sources         list[dict]
    numeric_claims  list[dict]
    unmatched_numbers list[str]
    _from_cache     bool
    _timing_sec     dict[str, float]

CLI
---
    python summarizer.py setup [--stage deps|models|all]
    python summarizer.py probe [--endpoint URL]
    python summarizer.py models [--endpoint URL] [--role chat|embed|all]
    python summarizer.py summarize PATH [options...]
"""
import _bootstrap  # noqa: F401 — env setup; must come before heavy imports

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Literal

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_chroma import Chroma
from langchain_core.documents import Document

import document_load
import endpoints as ep
from document_load import fingerprint, load_docling_chunks, load_docling_text, load_pypdf_chunks, load_pypdf_text
from endpoints import Config, Endpoint, EndpointStatus, load_config, make_chat_model, make_embeddings, probe, list_models
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore", message=".*pin_memory.*no accelerator.*", category=UserWarning)
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Numeric token regex (port from summarizer_unified.py lines 121-132)
# ---------------------------------------------------------------------------
_NUMERIC_TOKEN_RE = re.compile(
    r"(?<!\w)"
    r"(?:[+\u2212-])?"
    r"(?:"
    r"\d{1,3}(?:[.,\u00A0\s]\d{3})+"
    r"|"
    r"\d+"
    r")"
    r"(?:[.,]\d+)?"
    r"(?:\s?%)?"
    r"(?!\w)"
)

# ---------------------------------------------------------------------------
# SummarizeOptions
# ---------------------------------------------------------------------------

class SummarizeOptions(BaseModel):
    """Options for a single summarize() call."""
    processing_mode: Literal["vector", "plain"] = "vector"
    pdf_loader: Literal["docling", "pypdf"] = "docling"
    embedding: str = "local"           # "local" or "<endpoint_url>::<embed_model>"
    llm_endpoint: str = ""             # endpoint URL (must be set before calling summarize)
    llm_model: str = ""                # model name (must be set before calling summarize)
    out_lang: str = "pt-pt"
    max_words: int = 200
    max_keywords: int = 6
    max_tags: int = 5
    temperature: float = 0.1
    use_cache: bool = True
    display_source_name: str | None = None
    # Docling options — defaults come from DOCLING_* env vars (set from config.yml
    # by _bootstrap.py); UI / CLI can override for a specific run.
    do_ocr: bool = Field(
        default_factory=lambda: os.environ.get("DOCLING_DO_OCR", "false").strip().lower() == "true"
    )
    ocr_engine: Literal["easyocr", "rapidocr", "tesseract"] = Field(
        default_factory=lambda: os.environ.get("DOCLING_OCR_ENGINE", "easyocr").strip().lower()  # type: ignore[return-value]
    )
    table_mode: Literal["fast", "accurate"] = Field(
        default_factory=lambda: os.environ.get("DOCLING_TABLE_MODE", "fast").strip().lower()  # type: ignore[return-value]
    )


# ---------------------------------------------------------------------------
# Internal helpers — ported verbatim from summarizer_unified.py
# ---------------------------------------------------------------------------

def _embedding_target_and_model(options: SummarizeOptions, cfg: Config) -> tuple[str, str]:
    """Resolve (target, model_name) for make_embeddings."""
    if options.embedding == "local":
        return "local", cfg.local_embedding.model
    if "::" in options.embedding:
        url, model = options.embedding.split("::", 1)
        return url.strip(), model.strip()
    # CLI / legacy: URL only — no model suffix
    return options.embedding, options.llm_model


def _embedding_label(options: SummarizeOptions, cfg: Config) -> str:
    if options.embedding == "local":
        return cfg.local_embedding.model
    if "::" in options.embedding:
        return options.embedding.split("::", 1)[1]
    return options.embedding


def _extract_numeric_tokens(text: str) -> list[str]:
    """Extract unique numeric tokens from text in reading order."""
    if not text:
        return []
    tokens: list[str] = []
    seen: set[str] = set()
    for m in _NUMERIC_TOKEN_RE.finditer(text):
        tok = m.group(0).strip().rstrip(".,;:")
        if not tok or not any(c.isdigit() for c in tok):
            continue
        if tok in seen:
            continue
        seen.add(tok)
        tokens.append(tok)
    return tokens


def _numeric_token_in_text(text: str, token: str) -> bool:
    """Whitespace-flexible verbatim check: is token present in text?"""
    if not text or not token:
        return False
    norm_text = re.sub(r"\s+", " ", text.replace("\u00A0", " "))
    norm_token = re.sub(r"\s+", " ", token.replace("\u00A0", " "))
    if norm_token in norm_text:
        return True
    if "%" in norm_token:
        if norm_token.replace(" %", "%") in norm_text:
            return True
        if norm_token.replace("%", " %") in norm_text:
            return True
    return False


def _find_first_numeric_span(text: str, token: str) -> tuple[int, int] | None:
    """Find (start, end) offsets of token in text. Returns None if not found."""
    if not text or not token:
        return None
    candidates = [token]
    if "%" in token:
        candidates += [token.replace(" %", "%"), token.replace("%", " %")]
    for cand in candidates:
        parts = [p for p in re.split(r"\s+", cand.strip()) if p]
        if not parts:
            continue
        flex = r"\s+".join(re.escape(p) for p in parts)
        m = re.search(flex, text)
        if m:
            return (m.start(), m.end())
    return None


def _normalize_for_match(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9à-ÿ ]+", "", text)
    return text.strip()


def _load_pdf_page_texts_norm(pdf_path: str) -> list[str]:
    try:
        from langchain_community.document_loaders import PyPDFLoader
        pages = PyPDFLoader(pdf_path).load()
    except Exception:
        return []
    return [_normalize_for_match(getattr(p, "page_content", "")) for p in pages]


def _infer_page_from_excerpt(excerpt: str, page_texts_norm: list[str]) -> int | None:
    if not page_texts_norm:
        return None
    ex = _normalize_for_match(excerpt)
    if len(ex) < 24:
        return None
    probe_text = ex[:150]
    for i, page_txt in enumerate(page_texts_norm, start=1):
        if probe_text in page_txt:
            return i
    return None


def _extract_page_label(meta: dict[str, Any]) -> str:
    """Convert integer page from metadata to 'p.N' string (or 'page n/a')."""
    page = meta.get("page")
    if isinstance(page, int) and page > 0:
        return f"p.{page}"
    return "page n/a"


def _extract_source_name(meta: dict[str, Any], display_source_name: str | None) -> str:
    if display_source_name and str(display_source_name).strip():
        return os.path.basename(str(display_source_name).strip())
    source_name = (
        meta.get("source_pdf")
        or meta.get("source")
        or meta.get("file_path")
        or meta.get("filename")
        or "document"
    )
    source_name = os.path.basename(str(source_name))
    if re.match(r"^tmp[a-z0-9_-]+\.pdf$", source_name.lower()):
        return "uploaded_file.pdf"
    return source_name


def build_sources_from_docs(
    docs: Any,
    *,
    summary_text: str | None = None,
    display_source_name: str | None = None,
    pdf_path: str | None = None,
    max_sources: int = 6,
    max_excerpt_chars: int = 280,
) -> list[dict]:
    """Build source-attribution entries from retrieved chunks."""
    if not isinstance(docs, list):
        return []
    summary_numbers = _extract_numeric_tokens(summary_text or "")
    candidates: list[dict] = []
    seen: set = set()
    page_texts_norm: list[str] | None = None

    for doc in docs:
        page_content = getattr(doc, "page_content", "") if doc is not None else ""
        meta = getattr(doc, "metadata", {}) if doc is not None else {}
        if not isinstance(meta, dict):
            meta = {}
        page_label = _extract_page_label(meta)
        source_name = _extract_source_name(meta, display_source_name)
        excerpt = " ".join(str(page_content).split())
        if len(excerpt) > max_excerpt_chars:
            excerpt = excerpt[:max_excerpt_chars - 1].rstrip() + "…"
        excerpt = excerpt or "(empty excerpt)"
        if page_label == "page n/a" and pdf_path and excerpt != "(empty excerpt)":
            if page_texts_norm is None:
                page_texts_norm = _load_pdf_page_texts_norm(pdf_path)
            inferred = _infer_page_from_excerpt(excerpt, page_texts_norm)
            if inferred is not None:
                page_label = f"p.{inferred}"
        dedupe_key = (source_name, page_label, excerpt[:160])
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        supports = (
            [n for n in summary_numbers if _numeric_token_in_text(page_content, n)]
            if summary_numbers else []
        )
        candidates.append({"source_name": source_name, "page_label": page_label,
                            "excerpt": excerpt, "supports": supports})

    if not candidates:
        return []

    if summary_numbers:
        chosen: list[int] = []
        covered: set[str] = set()
        remaining: set[int] = set(range(len(candidates)))
        while remaining and len(chosen) < max_sources:
            best = min(remaining, key=lambda i: (-(len(set(candidates[i]["supports"]) - covered)), i))
            if not set(candidates[best]["supports"]) - covered:
                break
            chosen.append(best)
            covered |= set(candidates[best]["supports"])
            remaining.discard(best)
        for i in range(len(candidates)):
            if len(chosen) >= max_sources:
                break
            if i not in chosen:
                chosen.append(i)
        chosen.sort()
    else:
        chosen = list(range(min(len(candidates), max_sources)))

    sources: list[dict] = []
    for new_idx, i in enumerate(chosen, start=1):
        c = candidates[i]
        sources.append({
            "id": str(new_idx),
            "source": c["source_name"],
            "location": c["page_label"],
            "excerpt": c["excerpt"],
            "supports_numbers": list(c["supports"]),
        })
    return sources


def build_source_from_full_text(
    pdf_path: str,
    content: str,
    *,
    summary_text: str | None = None,
    display_source_name: str | None = None,
    max_sources: int = 6,
    excerpt_window: int = 240,
) -> list[dict]:
    """Source attribution for plain (non-vector) mode."""
    src_name = (
        os.path.basename(str(display_source_name))
        if display_source_name
        else (os.path.basename(pdf_path) or "document")
    )
    summary_numbers = _extract_numeric_tokens(summary_text or "")
    sources: list[dict] = []
    if summary_numbers and content:
        used_windows: list[tuple[int, int]] = []
        for num in summary_numbers:
            span = _find_first_numeric_span(content, num)
            if span is None:
                continue
            start, end = span
            reused_idx: int | None = None
            for j, (ws, we) in enumerate(used_windows):
                if ws <= start and end <= we:
                    reused_idx = j
                    break
            if reused_idx is not None:
                if num not in sources[reused_idx]["supports_numbers"]:
                    sources[reused_idx]["supports_numbers"].append(num)
                continue
            if len(sources) >= max_sources:
                continue
            ws = max(0, start - excerpt_window // 2)
            we = min(len(content), end + excerpt_window // 2)
            raw_excerpt = " ".join(content[ws:we].split())
            if len(raw_excerpt) > 360:
                raw_excerpt = raw_excerpt[:359].rstrip() + "…"
            prefix = "…" if ws > 0 else ""
            suffix = "…" if we < len(content) else ""
            excerpt = f"{prefix}{raw_excerpt}{suffix}".strip()
            sources.append({"id": str(len(sources) + 1), "source": src_name,
                             "location": "full document", "excerpt": excerpt or "(content unavailable)",
                             "supports_numbers": [num]})
            used_windows.append((ws, we))
    if not sources:
        excerpt = " ".join(str(content).split())
        if len(excerpt) > 360:
            excerpt = excerpt[:359].rstrip() + "…"
        sources.append({"id": "1", "source": src_name, "location": "full document",
                         "excerpt": excerpt or "(content unavailable)", "supports_numbers": []})
    return sources


def build_numeric_coverage(
    summary_text: str,
    sources: list[dict],
) -> tuple[list[dict], list[str]]:
    """Map each summary number to the source ids that support it.

    Returns (numeric_claims, unmatched_numbers).
    numeric_claims: [{"number": str, "source_ids": [str]}]
    unmatched_numbers: [str]  — numbers in summary not found in any source excerpt.
    """
    summary_numbers = _extract_numeric_tokens(summary_text or "")
    claims: list[dict] = []
    unmatched: list[str] = []
    for num in summary_numbers:
        ids = [
            str(s.get("id"))
            for s in sources
            if isinstance(s, dict) and num in (s.get("supports_numbers") or [])
        ]
        claims.append({"number": num, "source_ids": ids})
        if not ids:
            unmatched.append(num)
    return claims, unmatched


def _extract_balanced_json_objects(text: str) -> list[str]:
    """Find all balanced {...} substrings, handling strings and escapes."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        start, depth, in_str, escape = i, 0, False, False
        j = i
        while j < n:
            ch = text[j]
            if escape:
                escape = False
            elif ch == "\\" and in_str:
                escape = True
            elif ch == '"':
                in_str = not in_str
            elif not in_str:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        out.append(text[start: j + 1])
                        i = j + 1
                        break
            j += 1
        else:
            break
        if depth != 0:
            i = start + 1
    return out


def safe_json_parse(result: Any) -> tuple[dict, bool]:
    """Parse LLM output as JSON. Returns (parsed_dict, used_fallback).

    used_fallback=True means the happy-path parse failed and we fell back to
    brace-counting. The bench harness uses this to populate json_ok.
    """
    content = result.content if hasattr(result, "content") else str(result)

    # 1 — direct parse
    try:
        return json.loads(content.strip()), False
    except json.JSONDecodeError:
        pass

    # 2 — markdown code fence
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", content)
    if m:
        try:
            return json.loads(m.group(1)), True
        except json.JSONDecodeError:
            pass

    # 3 — brace-counting scan (largest object first)
    for obj in sorted(_extract_balanced_json_objects(content), key=len, reverse=True):
        try:
            return json.loads(obj), True
        except json.JSONDecodeError:
            continue

    # 4 — last-resort fallback
    logger.warning("JSON parsing failed completely. Raw output: %s", content[:200])
    return {
        "summary": content,
        "keywords": ["parsing_failed"],
        "tags": ["raw_text"],
        "sources": [],
    }, True


def _safe_json_parse_for_chain(result: Any) -> dict:
    """Wrapper for use as a LangChain output_parser (returns dict only)."""
    parsed, _ = safe_json_parse(result)
    return parsed


def get_shared_prompts(instruction_prompt: str) -> tuple[Any, Any]:
    """Build the three-layer prompt pair (chat prompt + retrieval string prompt)."""
    inner_system = (
        "Your responses must be valid JSON only — "
        "no explanations, markdown, or any text outside the JSON object.\n\n"
        "Numerical fidelity is absolute: reproduce every number verbatim from the "
        "source text — same digits, same decimal and thousands separators "
        '(e.g. "1.234,56" stays "1.234,56"), same units, same sign. '
        "Never round, convert, infer, or paraphrase a number. "
        "If you cannot locate the exact value in the source, omit it."
    )
    system_message = f"{inner_system}\n\n{instruction_prompt}"
    human_prompt = (
        "Analyse the content below and return a JSON object with this exact structure:\n"
        "{{\n"
        '    "summary": "... (max {max_words} words, language: {out_lang})",\n'
        '    "keywords": ["up to {max_keywords} keywords"],\n'
        '    "tags": ["up to {max_tags} tags"]\n'
        "}}\n\n"
        "Reminder: any number in your summary must be copied verbatim from the source "
        "(same digits, separators, units). Omit numbers you cannot locate exactly.\n\n"
        "{content}"
    )
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", human_prompt),
    ])
    retrieval_human = human_prompt.replace(
        "{content}", "Context: {context}\n\nUser Question: {input}"
    )
    string_prompt = PromptTemplate(
        template=system_message + retrieval_human,
        input_variables=["context", "input", "max_keywords", "max_tags", "max_words", "out_lang"],
    )
    return chat_prompt, string_prompt


# ---------------------------------------------------------------------------
# Cache key computation
# ---------------------------------------------------------------------------

def _params_hash(options: SummarizeOptions, config: Config) -> str:
    key_dict = {
        "embedding": options.embedding,
        "llm_endpoint": options.llm_endpoint,
        "llm_model": options.llm_model,
        "max_words": options.max_words,
        "max_keywords": options.max_keywords,
        "max_tags": options.max_tags,
        "out_lang": options.out_lang,
        "processing_mode": options.processing_mode,
        "pdf_loader": options.pdf_loader,
        "do_ocr": options.do_ocr,
        "ocr_engine": options.ocr_engine if options.do_ocr else "off",
        "temperature": options.temperature,
        "instruction_prompt": config.instruction_prompt,
    }
    raw = json.dumps(key_dict, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def cache_path_for(fp: str, options: SummarizeOptions, config: Config) -> Path:
    ph = _params_hash(options, config)
    return config.cache_dir / "summaries" / f"{fp}__{ph}.json"


# ---------------------------------------------------------------------------
# Empty-result template
# ---------------------------------------------------------------------------

def _empty_result(*, from_cache: bool = False, timing: dict | None = None, error: str | None = None) -> dict:
    return {
        "summary": error or "No extractable content.",
        "keywords": [],
        "tags": [],
        "sources": [],
        "numeric_claims": [],
        "unmatched_numbers": [],
        "_from_cache": from_cache,
        "_timing_sec": timing or {},
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def summarize(
    pdf_path: Path | str,
    options: SummarizeOptions,
    *,
    config: Config | None = None,
    cancel_event: threading.Event | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> dict:
    """Run the summarizer pipeline on one PDF.

    Uses an in-memory Chroma collection (vector mode) deleted in finally.
    Writes to .cache/summaries/ when use_cache=True.
    Returns a dict with the guaranteed 8-key schema.
    """
    cfg = config or load_config()
    pdf_path = Path(pdf_path)
    timing: dict[str, float] = {}

    def progress(msg: str) -> None:
        print(msg)
        if on_progress:
            on_progress(msg)

    def check_cancel() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Cancelled.")

    # 1 — cache check
    fp = fingerprint(pdf_path)
    if options.use_cache:
        cp = cache_path_for(fp, options, cfg)
        if cp.exists():
            progress("Cache hit — loading previous result.")
            cached = json.loads(cp.read_text(encoding="utf-8"))
            cached["_from_cache"] = True
            cached["_timing_sec"] = {"cache_hit": 0.0}
            return cached

    # Resolve LLM endpoint object
    ep_url = options.llm_endpoint
    endpoint: Endpoint | None = None
    for e in cfg.endpoints:
        if e.url.rstrip("/") == ep_url.rstrip("/") or e.id == ep_url:
            endpoint = e
            break
    if endpoint is None:
        endpoint = Endpoint(url=ep_url) if ep_url else Endpoint(url="http://localhost:11434")

    emb_label = _embedding_label(options, cfg)

    # Build LLM
    llm = make_chat_model(
        endpoint, options.llm_model,
        temperature=options.temperature,
        timeout=cfg.defaults.llm.timeout_sec,
        max_retries=cfg.defaults.llm.max_retries,
    )

    vector_store: Any = None
    try:
        result_dict: dict

        if options.processing_mode == "vector":
            # --- vector mode ---
            progress(f"⏳ Step 1/4 — Loading and chunking PDF ({options.pdf_loader})…")
            t0 = time.perf_counter()
            if options.pdf_loader == "docling":
                chunks = load_docling_chunks(
                    pdf_path, fp,
                    do_ocr=options.do_ocr,
                    ocr_engine=options.ocr_engine,
                    table_mode_str=options.table_mode,
                )
            else:
                chunks = load_pypdf_chunks(pdf_path, fp)
            timing["chunks"] = time.perf_counter() - t0
            progress(f"   ✓ {len(chunks)} chunks in {timing['chunks']:.1f}s")
            check_cancel()

            if not chunks:
                return _empty_result(error="No extractable content from PDF.", timing=timing)

            progress(f"⏳ Step 2/4 — Building vector store (embedding: {emb_label})…")
            t1 = time.perf_counter()
            emb_target, emb_model = _embedding_target_and_model(options, cfg)
            embedding_fn = make_embeddings(
                emb_target, emb_model, endpoints=cfg.endpoints,
            )
            vector_store = Chroma.from_documents(
                collection_name=f"run_{time.time_ns()}",
                documents=chunks,
                embedding=embedding_fn,
            )
            retriever = vector_store.as_retriever(search_kwargs={"k": 10})
            timing["vector_store"] = time.perf_counter() - t1
            progress(f"   ✓ Vector store ready in {timing['vector_store']:.1f}s")
            check_cancel()

            progress("⏳ Step 3/4 — Generating summary with LLM…")
            _, string_prompt = get_shared_prompts(cfg.instruction_prompt)
            doc_chain = create_stuff_documents_chain(
                llm=llm, prompt=string_prompt, output_parser=_safe_json_parse_for_chain
            )
            chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=doc_chain)
            t2 = time.perf_counter()
            chain_result = chain.invoke({
                "input": "Summarize the main content and key information from this document.",
                "max_keywords": options.max_keywords,
                "max_tags": options.max_tags,
                "max_words": options.max_words,
                "out_lang": options.out_lang,
            })
            timing["llm"] = time.perf_counter() - t2
            progress(f"   ✓ Summary generated in {timing['llm']:.1f}s")

            progress("⏳ Step 4/4 — Extracting sources and numeric claims…")
            raw_answer = chain_result.get("answer", chain_result)
            if not isinstance(raw_answer, dict):
                raw_answer = {"summary": str(raw_answer), "keywords": [], "tags": []}
            summary_text = raw_answer.get("summary", "") if isinstance(raw_answer.get("summary"), str) else ""
            raw_answer["sources"] = build_sources_from_docs(
                chain_result.get("context"), summary_text=summary_text,
                display_source_name=options.display_source_name, pdf_path=str(pdf_path),
            )
            claims, unmatched = build_numeric_coverage(summary_text, raw_answer["sources"])
            raw_answer["numeric_claims"] = claims
            raw_answer["unmatched_numbers"] = unmatched
            result_dict = raw_answer

        else:
            # --- plain mode ---
            progress(f"⏳ Step 1/3 — Loading PDF ({options.pdf_loader})…")
            t0 = time.perf_counter()
            if options.pdf_loader == "docling":
                pdf_text = load_docling_text(
                    pdf_path,
                    do_ocr=options.do_ocr,
                    ocr_engine=options.ocr_engine,
                    table_mode_str=options.table_mode,
                )
            else:
                pdf_text = load_pypdf_text(pdf_path)
            timing["pdf_load"] = time.perf_counter() - t0
            progress(f"   ✓ PDF loaded in {timing['pdf_load']:.1f}s")
            check_cancel()

            progress("⏳ Step 2/3 — Generating summary with LLM…")
            chat_prompt, _ = get_shared_prompts(cfg.instruction_prompt)
            chain = chat_prompt | llm | _safe_json_parse_for_chain
            t1 = time.perf_counter()
            chain_result = chain.invoke({
                "content": "\n\n=== Document Section ===\n\n" + pdf_text,
                "max_keywords": options.max_keywords,
                "max_tags": options.max_tags,
                "max_words": options.max_words,
                "out_lang": options.out_lang,
            })
            timing["llm"] = time.perf_counter() - t1
            progress(f"   ✓ Summary generated in {timing['llm']:.1f}s")

            progress("⏳ Step 3/3 — Extracting sources and numeric claims…")
            if not isinstance(chain_result, dict):
                chain_result = {"summary": str(chain_result), "keywords": [], "tags": []}
            summary_text = chain_result.get("summary", "") if isinstance(chain_result.get("summary"), str) else ""
            chain_result["sources"] = build_source_from_full_text(
                str(pdf_path), pdf_text, summary_text=summary_text,
                display_source_name=options.display_source_name,
            )
            claims, unmatched = build_numeric_coverage(summary_text, chain_result["sources"])
            chain_result["numeric_claims"] = claims
            chain_result["unmatched_numbers"] = unmatched
            result_dict = chain_result

    except RuntimeError as exc:
        if "Cancelled." in str(exc):
            return _empty_result(error="Cancelled.", timing=timing)
        raise
    finally:
        if vector_store is not None:
            try:
                vector_store.delete_collection()
            except Exception:
                pass

    # Ensure all required keys exist
    for key in ("summary", "keywords", "tags", "sources", "numeric_claims", "unmatched_numbers"):
        result_dict.setdefault(key, [] if key != "summary" else "")

    result_dict["_from_cache"] = False
    result_dict["_timing_sec"] = timing

    # Write cache
    if options.use_cache:
        cp = cache_path_for(fp, options, cfg)
        cp.parent.mkdir(parents=True, exist_ok=True)
        tmp = cp.with_suffix(".tmp")
        tmp.write_text(json.dumps(result_dict, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(cp)

    return result_dict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_probe_table(cfg: Config, filter_url: str | None = None) -> None:
    targets = [e for e in cfg.endpoints if not filter_url or e.url.rstrip("/") == filter_url.rstrip("/") or e.id == filter_url]
    if not targets:
        if filter_url:
            targets = [Endpoint(url=filter_url)]
        else:
            print("No endpoints configured.")
            return
    for e in targets:
        status = probe(e)
        ok_str = "✓" if status.ok else "✗"
        chat_str = ", ".join(status.chat_models[:5]) or "—"
        embed_str = ", ".join(status.embed_models[:3]) or "—"
        err_str = f"  ERROR: {status.error}" if status.error else ""
        print(f"{ok_str} [{e.id}] {e.name} ({e.url})")
        if status.ok:
            print(f"      kind:   {status.kind}")
            print(f"      chat:   {chat_str}{'…' if len(status.chat_models) > 5 else ''}")
            print(f"      embed:  {embed_str}")
        else:
            print(f"     {err_str}")


def _build_options_from_args(args: Any, cfg: Config) -> SummarizeOptions:
    d = cfg.defaults
    # do_ocr / ocr_engine: CLI flags win; otherwise SummarizeOptions reads
    # DOCLING_* env vars that _bootstrap.py populated from config.yml.
    kwargs: dict = {}
    if getattr(args, "do_ocr", False):
        kwargs["do_ocr"] = True
    if getattr(args, "ocr_engine", None):
        kwargs["ocr_engine"] = args.ocr_engine
    return SummarizeOptions(
        processing_mode=getattr(args, "mode", d.processing_mode),
        pdf_loader=getattr(args, "loader", d.pdf_loader),
        embedding=getattr(args, "embedding", d.embedding),
        llm_endpoint=getattr(args, "endpoint", "") or "",
        llm_model=getattr(args, "model", "") or "",
        out_lang=getattr(args, "lang", d.out_lang),
        max_words=getattr(args, "max_words", d.max_words),
        max_keywords=getattr(args, "max_keywords", d.max_keywords),
        max_tags=getattr(args, "max_tags", d.max_tags),
        temperature=getattr(args, "temperature", d.temperature),
        use_cache=not getattr(args, "no_cache", False),
        **kwargs,
    )


def _cli_main() -> None:
    parser = argparse.ArgumentParser(
        prog="summarizer",
        description="WP12 v1 PDF Summarizer",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # setup
    p_setup = sub.add_parser("setup", help="Install requirements and/or download models.")
    p_setup.add_argument("--stage", choices=["deps", "models", "all"], default="all")

    # probe
    p_probe = sub.add_parser("probe", help="Probe configured endpoints.")
    p_probe.add_argument("--endpoint", metavar="URL", default=None,
                         help="Probe a single URL instead of all config endpoints.")

    # models
    p_models = sub.add_parser("models", help="List models from an endpoint.")
    p_models.add_argument("--endpoint", metavar="URL", required=True)
    p_models.add_argument("--role", choices=["chat", "embed", "all"], default="all")

    # summarize
    p_sum = sub.add_parser("summarize", help="Summarize a PDF.")
    p_sum.add_argument("pdf", metavar="PATH")
    p_sum.add_argument("--endpoint", metavar="URL", default=None)
    p_sum.add_argument("--model", metavar="NAME", default=None)
    p_sum.add_argument("--mode", choices=["vector", "plain"], default=None)
    p_sum.add_argument("--loader", choices=["docling", "pypdf"], default=None)
    p_sum.add_argument("--embedding", metavar="local|URL", default=None)
    p_sum.add_argument("--lang", metavar="CODE", default=None)
    p_sum.add_argument("--max-words", type=int, default=None, dest="max_words")
    p_sum.add_argument("--max-keywords", type=int, default=None, dest="max_keywords")
    p_sum.add_argument("--max-tags", type=int, default=None, dest="max_tags")
    p_sum.add_argument("--temperature", type=float, default=None)
    p_sum.add_argument("--no-cache", action="store_true", dest="no_cache")
    p_sum.add_argument("--display-name", metavar="NAME", default=None, dest="display_source_name")
    p_sum.add_argument("--do-ocr", action="store_true", dest="do_ocr",
                       help="Enable OCR (for scanned PDFs). Default: off.")
    p_sum.add_argument("--ocr-engine", choices=["easyocr", "rapidocr", "tesseract"],
                       default=None, dest="ocr_engine")

    args = parser.parse_args()
    cfg = load_config()

    if args.command == "setup":
        from endpoints import setup
        setup(args.stage)

    elif args.command == "probe":
        _print_probe_table(cfg, filter_url=args.endpoint)

    elif args.command == "models":
        e_url = args.endpoint
        target: Endpoint | None = None
        for e in cfg.endpoints:
            if e.url.rstrip("/") == e_url.rstrip("/") or e.id == e_url:
                target = e
                break
        if target is None:
            target = Endpoint(url=e_url)
        models = list_models(target, role=args.role)
        if models:
            for m in models:
                print(m)
        else:
            print(f"No {args.role} models found at {e_url}.")

    elif args.command == "summarize":
        pdf = Path(args.pdf)
        if not pdf.exists():
            print(f"Error: file not found: {pdf}", file=sys.stderr)
            sys.exit(1)
        opts = _build_options_from_args(args, cfg)
        if args.display_source_name:
            opts = opts.model_copy(update={"display_source_name": args.display_source_name})
        # Fill in defaults from first reachable endpoint if not specified
        if not opts.llm_endpoint or not opts.llm_model:
            print("Error: --endpoint and --model are required.", file=sys.stderr)
            sys.exit(1)
        result = summarize(pdf, opts, config=cfg)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli_main()
