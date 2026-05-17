"""
v1 NiceGUI app — Summarizer UI.

Imports only from _bootstrap, summarizer, endpoints, document_load, and nicegui.
No KB, no bench (bench is a dev-only CLI tool).
"""
import _bootstrap  # noqa: F401 — env setup; must come before heavy imports

import asyncio
import base64
import html
import json
import logging
import os
import queue
import re
import sys
import tempfile
import threading
import time
import warnings
from pathlib import Path
from typing import Any

from nicegui import app, ui, run
from nicegui.events import UploadEventArguments

import document_load
from document_load import fingerprint, prewarm
from endpoints import (
    Config, Endpoint, EndpointStatus,
    load_config, probe, list_models,
)
from summarizer import (
    SummarizeOptions, summarize, cache_path_for,
    build_sources_from_docs, build_numeric_coverage,
)

warnings.filterwarnings(
    "ignore", message=".*pin_memory.*no accelerator.*", category=UserWarning)
logging.getLogger("pypdf._reader").setLevel(logging.ERROR)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------------------------------------------------------------------------
# Config (loaded once at startup)
# ---------------------------------------------------------------------------
_CFG: Config = load_config()

# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def _esc(s: str) -> str:
    return html.escape(str(s))


def _sources_block_html(sources: list) -> str:
    if not isinstance(sources, list) or not sources:
        return '<p class="text-sm text-gray-400 italic">No source attribution available.</p>'
    parts = []
    for idx, src in enumerate(sources, 1):
        if not isinstance(src, dict):
            continue
        excerpt = (
            f'<div class="text-base leading-relaxed mt-1 text-gray-800">'
            f'{_esc(str(src.get("excerpt", "")).strip())}</div>'
        )
        parts.append(
            f'<div class="mb-6">'
            f'<div class="text-sm text-gray-400 font-mono">'
            f'{idx}. {_esc(str(src.get("source", "doc")))} ({_esc(str(src.get("location", "n/a")))})'
            f'</div>{excerpt}</div>'
        )
    return (
        '<div>' + "".join(parts) + "</div>"
        if parts
        else '<p class="text-base text-gray-400 italic">No source attribution available.</p>'
    )


def _sources_block_plain(sources: list) -> str:
    if not isinstance(sources, list) or not sources:
        return "No source attribution available."
    parts = []
    for idx, src in enumerate(sources, 1):
        if not isinstance(src, dict):
            continue
        body = f"   {str(src.get('excerpt', ''))}"
        parts.append(
            f"{idx}. {src.get('source', 'doc')} ({src.get('location', 'n/a')})\n{body}")
    return "\n\n".join(parts)


def _numeric_groups(claims: list, sources: list) -> tuple[list[tuple[list[str], list[str]]], list[str]]:
    """Group numeric claims by their shared page locations.

    Returns (groups, unmatched) where each group is ([number, ...], [page_loc, ...]).
    """
    source_id_to_loc = {
        str(s.get("id", "")): str(s.get("location", "n/a"))
        for s in (sources or []) if isinstance(s, dict)
    }
    groups: dict[tuple, list[str]] = {}
    group_order: list[tuple] = []
    unmatched: list[str] = []
    for c in (claims or []):
        if not isinstance(c, dict):
            continue
        ids = tuple(sorted(str(i) for i in (c.get("source_ids") or [])))
        num = str(c.get("number", ""))
        if not ids:
            unmatched.append(num)
            continue
        if ids not in groups:
            groups[ids] = []
            group_order.append(ids)
        groups[ids].append(num)
    result: list[tuple[list[str], list[str]]] = []
    for key in group_order:
        locs_seen: set = set()
        locs: list[str] = []
        for sid in key:
            loc = source_id_to_loc.get(sid, "n/a")
            if loc not in locs_seen:
                locs_seen.add(loc)
                locs.append(loc)
        result.append((groups[key], locs))
    return result, unmatched


def _numeric_claims_block_plain(claims: list, unmatched: list, sources: list | None = None) -> str:
    groups, u_extra = _numeric_groups(claims, sources or [])
    all_unmatched = list(unmatched or []) + \
        [n for n in u_extra if n not in (unmatched or [])]
    lines = []
    for nums, locs in groups:
        lines.append(f"{', '.join(nums)} : {', '.join(locs)}")
    if all_unmatched:
        lines.append("Unmatched: " + ", ".join(str(n) for n in all_unmatched))
    return "\n".join(lines) or "No numeric values detected in the summary."


def _numeric_claims_block_html(claims: list, sources: list) -> str:
    """Grouped HTML for the Numeric verification card: numbers sharing the same page(s) appear together.

    Numbers are highlighted in blue; separator commas are dimmed so the eye can
    tell where one token ends and the next begins.
    """
    groups, unmatched = _numeric_groups(claims, sources)
    if not groups and not unmatched:
        return '<p class="text-sm text-gray-400 italic font-mono">No numeric values detected.</p>'

    parts: list[str] = []
    for nums, locs in groups:
        num_tokens: list[str] = []
        for i, n in enumerate(nums):
            num_tokens.append(f'<span class="text-gray-800">{_esc(n)}</span>')
            if i < len(nums) - 1:
                num_tokens.append(
                    '<span class="text-slate-300 select-none">,\u202f</span>')
        page_str = _esc(", ".join(locs))
        parts.append(
            '<div class="flex flex-wrap items-baseline gap-x-0.5 py-0.5 font-mono text-sm leading-relaxed">'
            + "".join(num_tokens)
            + '<span class="text-slate-400 mx-1.5">:</span>'
            + f'<span class="text-slate-500 text-xs">{page_str}</span>'
            + "</div>"
        )

    if unmatched:
        u_tokens: list[str] = []
        for i, n in enumerate(unmatched):
            u_tokens.append(
                f'<span class="text-red-600 font-semibold">{_esc(n)}</span>')
            if i < len(unmatched) - 1:
                u_tokens.append(
                    '<span class="text-slate-300 select-none">,\u202f</span>')
        parts.append(
            '<div class="flex flex-wrap items-baseline gap-x-0.5 py-0.5 mt-1 font-mono text-sm leading-relaxed">'
            + '<span class="text-red-400 text-xs mr-1.5">unmatched :</span>'
            + "".join(u_tokens)
            + "</div>"
        )

    return "<div>" + "".join(parts) + "</div>"


def _build_export_plaintext(summary, sources, keywords, tags, numeric_claims=None, unmatched_numbers=None) -> str:
    kw = ", ".join(str(k) for k in keywords) if keywords else "—"
    tg = ", ".join(str(t) for t in tags) if tags else "—"
    return (
        f"Summary\n-------\n{summary}\n\n"
        f"Information sources\n-------------------\n{_sources_block_plain(sources)}\n\n"
        f"Numeric verification\n--------------------\n"
        f"{_numeric_claims_block_plain(numeric_claims or [], unmatched_numbers or [], sources or [])}\n\n"
        f"Keywords\n--------\n{kw}\n\nTags\n----\n{tg}\n"
    )


def _build_export_markdown(summary, sources, keywords, tags, numeric_claims=None, unmatched_numbers=None) -> str:
    kw = ", ".join(str(k) for k in keywords) if keywords else "—"
    tg = ", ".join(str(t) for t in tags) if tags else "—"
    lines = ["## Summary", "", (summary or "—").strip(),
             "", "## Information sources", ""]
    for idx, src in enumerate(sources or [], 1):
        if not isinstance(src, dict):
            continue
        lines.append(
            f"{idx}. **{src.get('source', 'doc')}** ({src.get('location', 'n/a')})")
        lines += ["", f"   {str(src.get('excerpt', '')).strip()}", ""]
    lines += ["## Numeric verification", ""]
    lines.append(_numeric_claims_block_plain(
        numeric_claims or [], unmatched_numbers or [], sources or []))
    lines += ["", f"## Keywords\n\n{kw}\n\n## Tags\n\n{tg}"]
    return "\n".join(lines).rstrip() + "\n"


def _build_export_json(summary, sources, keywords, tags, numeric_claims=None, unmatched_numbers=None) -> str:
    return json.dumps({
        "summary": summary, "sources": sources or [], "keywords": list(keywords or []),
        "tags": list(tags or []), "numeric_claims": list(numeric_claims or []),
        "unmatched_numbers": list(unmatched_numbers or []),
    }, ensure_ascii=False, indent=2) + "\n"


def _build_export_html(summary, sources, keywords, tags, numeric_claims=None, unmatched_numbers=None) -> str:
    h = _esc
    parts = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<title>Summary export</title></head><body>",
        f"<h2>Summary</h2><p>{h(summary).replace(chr(10), '<br>')}</p>"
        "<h2>Information sources</h2>",
    ]
    if sources:
        parts.append("<ol>")
        for src in sources:
            if not isinstance(src, dict):
                continue
            parts.append(
                f"<li><strong>{h(str(src.get('source', 'doc')))}</strong> "
                f"({h(str(src.get('location', 'n/a')))})<br>"
                f"{h(str(src.get('excerpt', '')).strip())}</li>"
            )
        parts.append("</ol>")
    parts.append("<h2>Numeric verification</h2><ul>")
    _exp_groups, _exp_unmatched = _numeric_groups(
        numeric_claims or [], sources or [])
    for _nums, _locs in _exp_groups:
        parts.append(
            f"<li><strong>{h(', '.join(_nums))}</strong> : {h(', '.join(_locs))}</li>"
        )
    if _exp_unmatched:
        parts.append(
            f"<li><em>Unmatched: {h(', '.join(_exp_unmatched))}</em></li>")
    parts.append("</ul>")
    kw = "".join(f"<li>{h(str(k))}</li>" for k in (keywords or []))
    tg = "".join(f"<li>{h(str(t))}</li>" for t in (tags or []))
    parts += [f"<h2>Keywords</h2><ul>{kw}</ul>",
              f"<h2>Tags</h2><ul>{tg}</ul>", "</body></html>"]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Stdout tee for progress capture
# ---------------------------------------------------------------------------

class _StdoutTee:
    def __init__(self, real, q: queue.SimpleQueue):
        self._real, self._q, self._buf = real, q, ""

    def write(self, text: str) -> int:
        self._real.write(text)
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            stripped = line.strip()
            if stripped:
                self._q.put(stripped)
        return len(text)

    def flush(self):
        self._real.flush()

    def __getattr__(self, name):
        return getattr(self._real, name)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_size(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "Unknown size"
    units = ["B", "KB", "MB", "GB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def _sanitize_filename(s: str, max_len: int = 72) -> str:
    raw = re.sub(r'[\s<>:\"/\\|?*]+', "_", (s or "").strip())
    raw = re.sub(r"_+", "_", raw).strip("_")
    return (raw[:max_len].rstrip("_") or "na")


def _pdf_thumbnail(file_path: str | None = None, pdf_bytes: bytes | None = None) -> str | None:
    try:
        import fitz  # type: ignore[import-untyped]
        doc = fitz.open(stream=pdf_bytes,
                        filetype="pdf") if pdf_bytes else fitz.open(file_path)
        pix = doc[0].get_pixmap(matrix=fitz.Matrix(0.45, 0.45), alpha=False)
        png = pix.tobytes("png")
        doc.close()
        return f"data:image/png;base64,{base64.b64encode(png).decode('ascii')}"
    except Exception:
        return None


def _list_demo_pdfs() -> list[str]:
    demo_dir = Path(__file__).resolve().parent / "demo_docs"
    if not demo_dir.is_dir():
        return []
    return sorted(f.name for f in demo_dir.glob("*.pdf"))


def _format_summarize_error(exc: BaseException, *, model: str = "", endpoint: str = "") -> str:
    """Turn LLM API errors into a short, actionable message."""
    msg = str(exc)
    low = msg.lower()
    if "404" in msg or "notfound" in low or "not found" in low:
        if "model group" in low or "embed" in low:
            extra = (
                "If you use vector mode with a remote embedding from SSP, pick an "
                "embedding model (e.g. qwen3-embedding-8b), not the chat model."
            )
        else:
            extra = (
                "Restart the app after updating v1. If it persists, pick another chat model "
                "or check SSP Cloud for enabled model IDs."
            )
        hint = f"LLM/embedding request failed for “{model}”." if model else "Request failed for the selected model."
        return f"{hint}\n\n{extra}\n\nRaw: {msg[:400]}"
    if endpoint:
        return f"{msg}\n\nEndpoint: {endpoint}" + (f" · Model: {model}" if model else "")
    return msg


# ---------------------------------------------------------------------------
# Probe state — populated asynchronously at startup
# ---------------------------------------------------------------------------
PROBE_RESULTS: dict[str, EndpointStatus] = {}


def _probe_all_endpoints(cfg: Config) -> None:
    """Run endpoint probes (blocking) — called via run.io_bound."""
    for ep in cfg.endpoints:
        PROBE_RESULTS[ep.id] = probe(ep)


# ---------------------------------------------------------------------------
# App startup
# ---------------------------------------------------------------------------

@app.on_startup
async def _on_startup() -> None:
    await run.io_bound(_probe_all_endpoints, _CFG)
    await run.io_bound(prewarm)


# ---------------------------------------------------------------------------
# Timing formatter
# ---------------------------------------------------------------------------
_TIMING_ORDER = ("chunks", "vector_store", "pdf_load", "llm")
_TIMING_LABELS = {"chunks": "Chunks",
                  "vector_store": "Index", "pdf_load": "PDF", "llm": "LLM"}


def _format_status_done(total_s: float, timing_sec: dict | None) -> str:
    line1 = f"Done · {total_s:.1f}s total"
    if not timing_sec:
        return line1
    parts = [
        f"{_TIMING_LABELS[k]} {timing_sec[k]:.1f}s" for k in _TIMING_ORDER if k in timing_sec]
    return line1 + ("\n" + " · ".join(parts) if parts else "")


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

_PAGE_CSS = """
<style>
  body { zoom: 100%; }

  /* Upload zone: hide Q-Uploader chrome; stretch component to fill the dashed card. */
  .pdf-upload-styled .q-uploader__header,
  .pdf-upload-styled .q-uploader__list { display: none !important; }
  .pdf-upload-drop-layer.q-uploader,
  .pdf-upload-drop-layer .q-uploader {
    width: 100% !important; min-height: 100% !important; height: 100% !important;
    box-shadow: none !important; background: rgba(255,255,255,.35) !important;
    border-radius: .875rem !important; border: none !important;
  }

  /* Radio labels: ::first-line = title (large/bold), rest = description (small/muted). */
  .summarizer-radio-stack .q-radio { align-items: flex-start; padding: .6rem 0; min-height: unset; }
  .summarizer-radio-stack .q-radio:not(:last-child) { border-bottom: 1px solid rgb(241 245 249); }
  .summarizer-radio-stack .q-radio__label {
    white-space: pre-line; font-size: .8125rem; line-height: 1.55;
    color: rgb(100 116 139); letter-spacing: .01em;
  }
  .summarizer-radio-stack .q-radio__label::first-line {
    font-size: .9375rem; font-weight: 600; color: rgb(30 41 59); letter-spacing: -.01em;
  }

  /* Summarize button: pulsing scanner icon while processing. */
  .processing-spinner-matrix .q-icon {
    color: rgba(255,255,255,.95);
    animation: scanPulse 1.1s ease-in-out infinite;
    transform-origin: center center;
  }
  @keyframes scanPulse {
    0%, 100% { transform: scale(.92); opacity: .66; }
    50%       { transform: scale(1.06); opacity: 1;  }
  }
</style>
"""

_FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 -960 960 960">'
    '<path fill="#1976d2" d="M280-280h400v-80H280v80Zm0-160h400v-80H280v80Z'
    'm0-160h160v-80H280v80ZM200-120q-33 0-56.5-23.5T120-200v-560q0-33 23.5'
    '-56.5T200-840h560q33 0 56.5 23.5T840-760v560q0 33-23.5 56.5T760-120H2'
    '00Zm0-80h560v-560H200v560Zm0-560v560-560Z"/></svg>'
)

_GITHUB_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="15" height="15"'
    ' style="fill:#64748b;vertical-align:middle">'
    '<path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255'
    ".825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345"
    "-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23"
    " 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335"
    "-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315"
    " 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23"
    ".66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475"
    " 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57"
    'A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/></svg>'
)


def _main_page() -> None:  # noqa: C901
    cfg = _CFG

    ui.add_head_html(_PAGE_CSS.strip())
    ui.query("body").style("background:#f5f5f5")

    # ── Shared UI state ───────────────────────────────────────────────────────
    state: dict[str, Any] = {
        "processing": False,
        "completed": False,
        "summarizer_editor_open": False,
        "parameters_editor_open": False,
        "file_picker_open": True,
        "cancel_event": None,
    }
    file_info: dict[str, Any] = {
        "name": None, "size_bytes": None, "source": None, "path": None, "bytes": None,
    }
    export_meta: dict[str, Any] = {
        "ready": False, "summary": "", "sources": [], "keywords": [], "tags": [],
        "numeric_claims": [], "unmatched_numbers": [], "elapsed_s": None, "timing_breakdown": None,
    }
    refs: dict[str, Any] = {}

    # ── Option builders ───────────────────────────────────────────────────────
    def _endpoint_options() -> dict[str, str]:
        opts: dict[str, str] = {}
        for ep in cfg.endpoints:
            status = PROBE_RESULTS.get(ep.id)
            available = bool(status and status.ok)
            opts[ep.id] = ep.name if available else f"⊘ {ep.name}"
        return opts

    def _patch_endpoint_disabled(select_el: Any) -> None:
        """Mark Quasar option entries as disabled where the label starts with ⊘.

        NiceGUI re-encodes options as [{value: idx, label: str}] internally, so
        we cannot use option-disable props. Instead we augment the already-built
        _props['options'] list so Quasar renders those rows grayed-out and
        unclickable without NiceGUI's validation layer seeing anything unusual.
        """
        for opt in select_el._props.get("options", []):
            opt["disable"] = str(opt.get("label", "")).startswith("⊘")
        select_el.update()

    def _chat_models_for(ep_id: str) -> list[str]:
        status = PROBE_RESULTS.get(ep_id)
        return status.chat_models if status and status.ok else []

    def _embed_options() -> dict[str, str]:
        """Collect embedding models from ALL probed endpoints, independent of the LLM endpoint."""
        opts: dict[str, str] = {
            "local": f"Local · {cfg.local_embedding.model}"}
        for ep in cfg.endpoints:
            status = PROBE_RESULTS.get(ep.id)
            if status and status.ok:
                for m in status.embed_models:
                    opts[f"{status.endpoint.url}::{m}"] = f"{status.endpoint.name} · {m}"
        return opts

    def _default_emb(opts: dict[str, str]) -> str:
        """Prefer the first remote embedding; fall back to local."""
        return next((k for k in opts if k != "local"), "local")

    def _cache_hit_for_current() -> bool:
        if not file_info.get("path"):
            return False
        try:
            fp = fingerprint(file_info["path"])
            opts = _build_current_options(include_endpoint=True)
            if not opts:
                return False
            return cache_path_for(fp, opts, cfg).exists()
        except Exception:
            return False

    def _build_current_options(*, include_endpoint: bool = False) -> SummarizeOptions | None:
        try:
            ep_id = refs.get(
                "endpoint_select") and refs["endpoint_select"].value
            model_val = refs.get("model_select") and refs["model_select"].value
            emb_val = refs.get(
                "embedding_select") and refs["embedding_select"].value or "local"
            if not ep_id or not model_val:
                return None
            ep_obj = next((e for e in cfg.endpoints if e.id == ep_id), None)
            ep_url = ep_obj.url if ep_obj else ep_id
            # Keep "url::embed_model" so vector mode does not pass the chat model to embeddings.
            emb_setting = "local" if not emb_val or emb_val == "local" else str(
                emb_val)
            d = cfg.defaults
            mode_val = refs.get(
                "mode_radio") and refs["mode_radio"].value or d.processing_mode
            loader_val = refs.get(
                "loader_radio") and refs["loader_radio"].value or d.pdf_loader
            lang_val = refs.get(
                "lang_input") and refs["lang_input"].value or d.out_lang
            mw_val = int(refs.get("max_words_input")
                         and refs["max_words_input"].value or d.max_words)
            mkw_val = int(refs.get("max_kw_input")
                          and refs["max_kw_input"].value or d.max_keywords)
            mt_val = int(refs.get("max_tags_input")
                         and refs["max_tags_input"].value or d.max_tags)
            temp_val = float(refs.get("temp_input")
                             and refs["temp_input"].value or d.temperature)
            ocr_on = bool(refs.get("do_ocr_toggle")
                          and refs["do_ocr_toggle"].value)
            ocr_eng = (refs.get("ocr_engine_select")
                       and refs["ocr_engine_select"].value) or cfg.docling.ocr_engine
            table_mode_val = (refs.get("table_mode_select")
                              and refs["table_mode_select"].value) or cfg.docling.table_mode
            skip_cache = bool(refs.get("skip_cache_toggle")
                              and refs["skip_cache_toggle"].value)
            return SummarizeOptions(
                processing_mode=mode_val, pdf_loader=loader_val,
                embedding=emb_setting, llm_endpoint=ep_url, llm_model=str(
                    model_val),
                out_lang=lang_val, max_words=mw_val, max_keywords=mkw_val, max_tags=mt_val,
                temperature=temp_val, use_cache=not skip_cache, do_ocr=ocr_on, ocr_engine=ocr_eng,
                table_mode=table_mode_val,
            )
        except Exception:
            return None

    def _refresh_summarize_button() -> None:
        btn = refs.get("summarize_btn")
        if btn is None:
            return
        has_file = bool(file_info.get("path"))
        has_ep = bool(refs.get("endpoint_select")
                      and refs["endpoint_select"].value)
        has_model = bool(refs.get("model_select")
                         and refs["model_select"].value)
        can_run = has_file and has_ep and has_model and not state[
            "completed"] and not state["processing"]
        skip_cache = bool(refs.get("skip_cache_toggle")
                          and refs["skip_cache_toggle"].value)
        if not has_file:
            btn.set_visibility(False)
            btn.classes(remove="processing-spinner-matrix")
        elif state["processing"]:
            btn.set_text("Please wait…")
            btn.set_visibility(True)
            btn.classes(add="processing-spinner-matrix")
            btn.disable()
        elif not skip_cache and _cache_hit_for_current():
            btn.set_text("Summarize (cached · instant)")
            btn.set_visibility(can_run)
            btn.classes(remove="processing-spinner-matrix")
            btn.enable()
        else:
            btn.set_text("Summarize")
            btn.set_visibility(can_run)
            btn.classes(remove="processing-spinner-matrix")
            btn.enable()
        btn.update()

    # ── Page layout ───────────────────────────────────────────────────────────
    with ui.column().classes("items-center p-8 pb-2 gap-8 text-xl max-w-7xl w-full mx-auto"):

        # ── Header ────────────────────────────────────────────────────────────
        with ui.row().classes("items-center gap-3 justify-center flex-wrap text-center"):
            ui.icon("sym_o_article").classes(
                "text-4xl text-primary shrink-0 opacity-90")
            ui.label("PDF Summarizer with LLMs — WP12 · Statistics Portugal").classes(
                "text-3xl font-medium text-slate-900 tracking-tight"
            )

        # ── Card 1: File ──────────────────────────────────────────────────────
        with ui.card().classes("w-full p-6 pb-3 shadow-lg"):
            with ui.row().classes("items-center gap-3 mb-2"):
                ui.icon("sym_o_picture_as_pdf").classes(
                    "text-3xl text-primary shrink-0 opacity-90")
                ui.label("Add a PDF").classes(
                    "text-2xl font-medium text-slate-800 tracking-tight")

            # File picker (shown when no file is selected)
            with ui.column().classes("w-full gap-4").bind_visibility_from(
                state, "file_picker_open"
            ):
                # Drop zone
                with ui.column().classes(
                    "w-full rounded-2xl border-2 border-dashed border-slate-300 "
                    "bg-gradient-to-b from-slate-50 to-white "
                    "shadow-inner hover:border-primary hover:from-blue-50/60 hover:to-white "
                    "transition-colors duration-200 overflow-hidden gap-0 p-4 relative"
                ):
                    refs["pdf_upload"] = ui.upload(
                        on_upload=lambda e: _handle_upload(e),
                        auto_upload=True,
                    ).props("accept=.pdf max-files=1 color=primary flat").classes(
                        "absolute inset-0 w-full h-full pdf-upload-styled pdf-upload-drop-layer"
                    )
                    with ui.column().classes("relative z-10 w-full gap-4 pointer-events-none"):
                        with ui.row().classes("w-full gap-4 items-center"):
                            ui.icon("sym_o_picture_as_pdf").classes(
                                "text-5xl text-red-700/85 shrink-0 hidden sm:flex"
                            )
                            ui.label("Browse or drop a PDF").classes(
                                "text-base font-medium text-slate-700"
                            )
                        ui.button(
                            "Browse PDF files",
                            icon="sym_o_folder_open",
                            on_click=lambda: refs["pdf_upload"].run_method(
                                "pickFiles"),
                        ).props("unelevated color=primary no-caps").classes(
                            "w-full py-3 text-lg shadow-sm pointer-events-auto"
                        )

                # Demo file buttons
                demo_pdfs = _list_demo_pdfs()
                if demo_pdfs:
                    _demo_dir = Path(__file__).resolve().parent / "demo_docs"
                    with ui.row().classes("w-full items-center gap-2 mt-1 flex-wrap"):
                        ui.label("…or use our example PDF:").classes(
                            "text-sm text-gray-600")
                        for _pdf_name in demo_pdfs:
                            with ui.row().classes("items-center gap-0"):
                                ui.button(
                                    _pdf_name,
                                    icon="sym_o_picture_as_pdf",
                                    on_click=lambda _, p=_pdf_name: _select_demo(
                                        p),
                                ).props("flat dense no-caps color=primary").classes("px-2 py-1")
                                ui.button(
                                    icon="download",
                                    on_click=lambda _, path=str(_demo_dir / _pdf_name), name=_pdf_name: (
                                        ui.download.file(path, name)
                                    ),
                                ).props("flat dense round color=primary").tooltip(f"Download {_pdf_name}")

            # Selected file display (shown when a file is selected)
            with ui.column().classes("w-full gap-4").bind_visibility_from(
                state, "file_picker_open", value=False
            ):
                ui.label("File to process").classes(
                    "text-lg font-medium text-slate-800 tracking-tight mb-1"
                )
                with ui.row().classes(
                    "w-full gap-4 items-start p-4 rounded-xl bg-slate-50 border border-slate-100"
                ):
                    refs["thumbnail"] = ui.image().classes(
                        "w-36 flex-shrink-0 rounded-lg border border-slate-200 shadow-sm bg-white object-cover"
                    )
                    refs["thumbnail"].set_visibility(False)
                    with ui.column().classes("flex-1 min-w-0 gap-2"):
                        with ui.row().classes("w-full items-start justify-between gap-3"):
                            refs["file_name_label"] = ui.label("—").classes(
                                "flex-1 min-w-0 text-xl font-semibold text-gray-900 leading-snug break-words"
                            )
                            refs["clear_selection_btn"] = ui.button(
                                "Clear selection",
                                on_click=lambda: _clear_file(),
                            ).props(
                                "outline dense no-caps color=primary unelevated icon=sym_o_refresh"
                            ).classes("flex-shrink-0 shadow-sm")
                        refs["file_meta_label"] = ui.label("").classes(
                            "text-sm text-gray-500 tabular-nums")

        # ── Card 2: Summarizer configuration ──────────────────────────────────
        with ui.card().classes("w-full p-6 shadow-lg"):
            with ui.row().classes("items-center gap-3 mb-2"):
                ui.icon("sym_o_auto_awesome").classes(
                    "text-3xl text-primary shrink-0 opacity-90")
                ui.label("Summarizer configuration").classes(
                    "text-2xl font-medium text-slate-800 tracking-tight"
                )

            # Summary panel (collapsed view)
            with ui.column().classes("w-full gap-3").bind_visibility_from(
                state, "summarizer_editor_open", value=False
            ):
                with ui.row().classes(
                    "w-full flex-wrap items-start justify-between gap-3 "
                    "rounded-xl bg-slate-50/90 px-4 py-3 border border-slate-100"
                ):
                    with ui.column().classes("flex-1 min-w-0 gap-1"):
                        refs["sum_ep_label"] = ui.label("").classes(
                            "text-sm text-slate-800 leading-snug"
                        )
                        refs["sum_model_label"] = ui.label("").classes(
                            "text-xs text-slate-600 leading-snug ml-3 pl-3 border-l border-slate-200"
                        )
                        refs["sum_mode_label"] = ui.label("").classes(
                            "text-sm text-slate-800 leading-snug"
                        )
                        refs["sum_loader_label"] = ui.label("").classes(
                            "text-sm text-slate-800 leading-snug"
                        )
                    refs["summarizer_customize_btn"] = ui.button(
                        "Customize",
                        icon="sym_o_tune",
                        on_click=lambda: _toggle_summarizer_editor(),
                    ).props("outline dense no-caps color=primary unelevated").classes("flex-shrink-0")

                # Prompt preview
                with ui.row().classes("items-start gap-2 px-1"):
                    ui.icon("sym_o_psychology").classes(
                        "text-base text-primary opacity-70 mt-0.5 shrink-0")
                    _sum_preview = cfg.instruction_prompt.replace("\n", " ")
                    if len(_sum_preview) > 160:
                        _sum_preview = _sum_preview[:157].rstrip() + "…"
                    refs["sum_prompt_preview"] = ui.label(_sum_preview).classes(
                        "text-xs text-slate-400 leading-relaxed italic"
                    )

            # Editor panel (expanded view)
            with ui.column().classes("w-full gap-6").bind_visibility_from(
                state, "summarizer_editor_open"
            ):
                ui.label(
                    "Choose the chat endpoint, model, and temperature (LLM only — not Docling). "
                    "Then set PDF parsing and retrieval options below."
                ).classes("text-xs sm:text-sm text-slate-500 leading-relaxed max-w-3xl")

                # Language model subsection
                with ui.column().classes("w-full gap-2"):
                    ui.label("Language model").classes(
                        "text-xs font-semibold uppercase tracking-wide text-slate-500"
                    )
                    with ui.row().classes(
                        "w-full items-stretch rounded-xl bg-slate-50/90 border border-slate-100 overflow-hidden"
                    ):
                        ep_opts = _endpoint_options()
                        default_ep = list(ep_opts.keys())[
                            0] if ep_opts else None

                        # Endpoint — equal third
                        with ui.column().classes("flex-1 min-w-0 px-3 py-3 gap-3"):
                            refs["endpoint_select"] = ui.select(
                                ep_opts, value=default_ep, label="Endpoint",
                                on_change=lambda e: _on_endpoint_change(
                                    e.value),
                            ).classes("w-full")
                            _patch_endpoint_disabled(refs["endpoint_select"])
                            with ui.row().classes("w-full gap-2 flex-wrap items-center"):
                                refs["reprobe_btn"] = ui.button(
                                    "Re-probe",
                                    icon="sym_o_network_ping",
                                    on_click=lambda: asyncio.ensure_future(
                                        _reprobe_all()),
                                ).props("outline dense no-caps color=primary").classes("shadow-sm")
                                ui.label(
                                    "Query endpoints to refresh model list."
                                ).classes("text-xs text-slate-500 flex-1 min-w-0")
                            ui.timer(
                                0.35,
                                lambda: asyncio.ensure_future(_reprobe_all()),
                                once=True,
                                immediate=False,
                            )

                        # Model — equal third
                        with ui.column().classes(
                            "flex-1 min-w-0 px-3 py-3 gap-3 border-l border-slate-200/80"
                        ):
                            initial_models = _chat_models_for(
                                default_ep) if default_ep else []
                            refs["model_select"] = ui.select(
                                initial_models or ["—"],
                                value=initial_models[0] if initial_models else "—",
                                label="Model",
                                on_change=lambda _: (
                                    _refresh_summarizer_summary(), _refresh_summarize_button()),
                            ).classes("w-full")

                        # Temperature (LLM only) — equal third
                        with ui.column().classes(
                            "flex-1 min-w-0 px-3 py-3 gap-0 border-l border-slate-200/80 bg-indigo-50/40 justify-center"
                        ):
                            with ui.row().classes("w-full items-center justify-between mb-1"):
                                ui.label("Temperature").classes(
                                    "text-xs text-slate-500")
                                refs["temp_badge"] = ui.label(f"{cfg.defaults.temperature:.2f}").classes(
                                    "text-xs font-mono font-semibold px-2 py-0.5 rounded-full "
                                    "bg-indigo-100 text-indigo-700 tabular-nums"
                                )
                            refs["temp_input"] = ui.slider(
                                value=cfg.defaults.temperature, min=0.0, max=1.0, step=0.05,
                                on_change=lambda e: (
                                    refs["temp_badge"].set_text(
                                        f"{e.value:.2f}"),
                                    _refresh_summarize_button(),
                                ),
                            ).props("color=primary").classes("w-full")
                            with ui.row().classes("w-full justify-between"):
                                ui.label("Precise").classes(
                                    "text-[10px] text-slate-400")
                                ui.label("Creative").classes(
                                    "text-[10px] text-slate-400")

                # Summarization mode subsection
                with ui.column().classes("w-full gap-2"):
                    ui.label("Summarization mode").classes(
                        "text-xs font-semibold uppercase tracking-wide text-slate-500"
                    )
                    with ui.row().classes(
                        "w-full items-stretch rounded-xl bg-slate-50/90 border border-slate-100 overflow-hidden"
                    ):
                        # Left ~60% — radio options
                        with ui.column().classes(
                            "summarizer-radio-stack flex-[3] min-w-0 px-3"
                        ):
                            refs["mode_radio"] = ui.radio(
                                {
                                    "plain": (
                                        "Plain text\n"
                                        "Summarize from full extracted text in one pass "
                                        "(no ChromaDB retrieval step)."
                                    ),
                                    "vector": (
                                        "Vector retrieval\n"
                                        "Chunk the document, embed and index in ChromaDB, retrieve "
                                        "relevant passages, then summarize. Best for longer PDFs."
                                    ),
                                },
                                value=cfg.defaults.processing_mode,
                                on_change=lambda _: (
                                    _refresh_summarizer_summary(), _refresh_summarize_button()),
                            ).props("vertical")

                        # Right ~40% — embedding model, visible only for vector mode
                        with ui.column().classes(
                            "flex-[2] min-w-0 gap-3 px-4 py-3 border-l border-slate-200/80 "
                            "bg-indigo-50/40 justify-center"
                        ).bind_visibility_from(refs["mode_radio"], "value", value="vector"):
                            ui.label("Embedding model").classes(
                                "text-xs font-semibold uppercase tracking-wide text-slate-500"
                            )
                            emb_opts = _embed_options()
                            refs["embedding_select"] = ui.select(
                                emb_opts, value=_default_emb(emb_opts),
                                on_change=lambda _: _refresh_summarize_button(),
                            ).classes("w-full").props("dense")
                            ui.label(
                                'Use "Re-probe endpoints" to discover remote models.'
                            ).classes("text-[11px] text-slate-400 leading-snug")

                # PDF loader subsection
                with ui.column().classes("w-full gap-2"):
                    ui.label("PDF loader").classes(
                        "text-xs font-semibold uppercase tracking-wide text-slate-500"
                    )
                    with ui.row().classes(
                        "w-full items-stretch rounded-xl bg-slate-50/90 border border-slate-100 overflow-hidden"
                    ):
                        # Left ~60% — radio options
                        with ui.column().classes(
                            "summarizer-radio-stack flex-[3] min-w-0 px-3"
                        ):
                            refs["loader_radio"] = ui.radio(
                                {
                                    "docling": (
                                        "Docling\n"
                                        "Structured, layout-aware extraction. Vector mode uses "
                                        "Docling's hybrid chunking."
                                    ),
                                    "pypdf": (
                                        "PyPDF\n"
                                        "Lightweight page-level text. In vector mode, chunks use "
                                        "RecursiveCharacterTextSplitter."
                                    ),
                                },
                                value=cfg.defaults.pdf_loader,
                                on_change=lambda _: (
                                    _refresh_summarizer_summary(), _refresh_summarize_button()),
                            ).props("vertical")

                        # Right ~40% — Docling settings, visible only when Docling is selected
                        with ui.column().classes(
                            "flex-[2] min-w-0 gap-3 px-4 py-3 border-l border-slate-200/80 "
                            "bg-indigo-50/40"
                        ).bind_visibility_from(refs["loader_radio"], "value", value="docling"):
                            ui.label("Docling settings").classes(
                                "text-xs font-semibold uppercase tracking-wide text-slate-500"
                            )

                            # OCR row: toggle on left, engine select on right
                            _ENGINE_LABELS = {"rapidocr": "RapidOCR", "easyocr": "EasyOCR", "tesseract": "Tesseract"}
                            with ui.row().classes("w-full items-start gap-4"):
                                # Left: toggle + help text
                                with ui.column().classes("flex-1 gap-1"):
                                    refs["do_ocr_toggle"] = ui.switch(
                                        "Enable OCR",
                                        value=cfg.docling.do_ocr,
                                        on_change=lambda e: _on_ocr_toggle(e.value),
                                    ).classes("text-sm")
                                    ui.label("for scanned / image-based PDFs").classes(
                                        "text-[11px] text-slate-400 leading-snug"
                                    )

                                # Right when OCR ON: engine label + select + hints
                                with ui.column().classes("flex-1 gap-1").bind_visibility_from(
                                    refs["do_ocr_toggle"], "value"
                                ) as ocr_engine_row:
                                    refs["ocr_engine_row"] = ocr_engine_row
                                    ui.label("Engine").classes(
                                        "text-xs text-gray-500 font-semibold"
                                    )
                                    refs["ocr_engine_select"] = ui.select(
                                        {
                                            "rapidocr": "RapidOCR",
                                            "easyocr":  "EasyOCR",
                                        },
                                        value=cfg.docling.ocr_engine,
                                        on_change=lambda _: _refresh_summarize_button(),
                                    ).classes("w-full").props("dense")
                                    ui.label(
                                        "RapidOCR – faster, no extra download, good for Latin scripts\n"
                                        "EasyOCR – better multilingual coverage (~200 MB first-run download)"
                                    ).classes("text-[11px] text-slate-400 leading-snug whitespace-pre-line")

                                # Right when OCR OFF: static engine name (select already exists above)
                                with ui.column().classes("flex-1 gap-1").bind_visibility_from(
                                    refs["do_ocr_toggle"], "value", backward=lambda v: not v
                                ):
                                    ui.label("Engine").classes(
                                        "text-xs text-gray-500 font-semibold"
                                    )
                                    ui.label("").classes(
                                        "text-[11px] text-slate-400 leading-snug"
                                    ).bind_text_from(
                                        refs["ocr_engine_select"], "value",
                                        backward=lambda v: _ENGINE_LABELS.get(v or "rapidocr", v or "rapidocr"),
                                    )

                            ui.separator().classes("opacity-30 my-0")

                            # Table mode row: label on left, select on right
                            with ui.row().classes("w-full items-center gap-3"):
                                ui.label("Table mode").classes(
                                    "text-xs text-gray-500 font-semibold shrink-0"
                                )
                                refs["table_mode_select"] = ui.select(
                                    {
                                        "fast":     "Fast (default)",
                                        "accurate": "Accurate",
                                    },
                                    value=cfg.docling.table_mode,
                                    on_change=lambda _: _refresh_summarize_button(),
                                ).classes("flex-1").props("dense")
                            ui.label("Accurate improves table extraction quality but is slower").classes(
                                "text-[11px] text-slate-400 -mt-1"
                            )

                # Cache options
                with ui.column().classes("w-full gap-2"):
                    ui.label("Cache").classes(
                        "text-xs font-semibold uppercase tracking-wide text-slate-500"
                    )
                    with ui.row().classes(
                        "w-full items-center gap-4 rounded-xl bg-slate-50/90 border border-slate-100 px-4 py-3"
                    ):
                        refs["skip_cache_toggle"] = ui.switch(
                            "Skip cache",
                            value=False,
                            on_change=lambda _: _refresh_summarize_button(),
                        ).classes("text-sm")
                        ui.label(
                            "Force a fresh run even when a cached result exists for this document and settings."
                        ).classes("text-xs text-slate-400 leading-snug flex-1")

                # Instruction prompt (read-only display — edit in config.yml)
                with ui.row().classes("items-center gap-2"):
                    ui.icon("sym_o_psychology").classes(
                        "text-lg text-primary opacity-80")
                    ui.label("Instruction prompt").classes(
                        "text-xs font-semibold uppercase tracking-wide text-slate-500"
                    )
                ui.label(
                    "The instruction sent to the LLM on every run. Edit in config.yml to change."
                ).classes("text-xs text-slate-400 leading-relaxed -mt-1")
                with ui.element("div").classes(
                    "w-full rounded-lg border border-slate-200 bg-slate-50/60 px-4 py-3"
                ):
                    ui.label(cfg.instruction_prompt).classes(
                        "text-sm text-slate-400 leading-7 whitespace-pre-wrap"
                    )

                ui.separator().classes("w-full opacity-60")
                ui.button(
                    "Done",
                    icon="sym_o_expand_less",
                    on_click=lambda: _toggle_summarizer_editor(),
                ).props("flat dense no-caps color=primary").classes("self-start")

        # ── Card 3: Parameters ─────────────────────────────────────────────────
        with ui.card().classes("w-full p-6 shadow-lg"):
            with ui.row().classes("items-center gap-3 mb-2"):
                ui.icon("sym_o_tune").classes(
                    "text-3xl text-primary shrink-0 opacity-90")
                ui.label("Parameters").classes(
                    "text-2xl font-medium text-slate-800 tracking-tight")

            # Summary panel (pills)
            with ui.column().classes("w-full gap-3").bind_visibility_from(
                state, "parameters_editor_open", value=False
            ):
                _pill_val = (
                    "inline-flex items-center max-w-full px-2.5 py-1 rounded-full "
                    "bg-white border border-slate-200/90 shadow-sm "
                    "text-sm font-medium text-slate-900 tabular-nums tracking-tight break-all"
                )
                _pill_cap = "text-[10px] font-medium uppercase tracking-wide text-slate-400"
                with ui.row().classes(
                    "w-full flex-wrap items-start justify-between gap-3 "
                    "rounded-xl bg-slate-50/90 px-4 py-3 border border-slate-100"
                ):
                    with ui.column().classes("flex-1 min-w-0"):
                        with ui.row().classes("w-full flex-wrap gap-x-3 gap-y-2 items-start"):
                            with ui.column().classes("gap-1 shrink-0"):
                                ui.label("Max words").classes(_pill_cap)
                                refs["param_pill_words"] = ui.label(
                                    "").classes(_pill_val)
                            with ui.column().classes("gap-1 shrink-0"):
                                ui.label("Keywords").classes(_pill_cap)
                                refs["param_pill_kw"] = ui.label(
                                    "").classes(_pill_val)
                            with ui.column().classes("gap-1 shrink-0"):
                                ui.label("Tags").classes(_pill_cap)
                                refs["param_pill_tags"] = ui.label(
                                    "").classes(_pill_val)
                            with ui.column().classes("gap-1 min-w-0 flex-1 max-w-full"):
                                ui.label("Language").classes(_pill_cap)
                                refs["param_pill_lang"] = ui.label(
                                    "").classes(_pill_val + " font-mono")
                    refs["params_customize_btn"] = ui.button(
                        "Customize",
                        icon="sym_o_tune",
                        on_click=lambda: _toggle_params_editor(),
                    ).props("outline dense no-caps color=primary unelevated").classes("flex-shrink-0")

            # Editor panel
            with ui.column().classes("w-full gap-4").bind_visibility_from(
                state, "parameters_editor_open"
            ):
                ui.label(
                    "Set limits for summary length, keyword/tag counts, and output language. "
                    "Use any BCP 47 / locale-style code (examples: pt-pt, en, fr, ja)."
                ).classes("text-xs sm:text-sm text-slate-500 leading-relaxed")

                d = cfg.defaults

                def _slider_col(label, ref_key, badge_key, val, lo, hi, step=1):
                    with ui.column().classes("flex-1 min-w-[10rem] gap-0"):
                        with ui.row().classes("w-full items-center justify-between mb-1"):
                            ui.label(label).classes("text-xs text-slate-500")
                            refs[badge_key] = ui.label(str(int(val))).classes(
                                "text-xs font-mono font-semibold px-2 py-0.5 rounded-full "
                                "bg-indigo-100 text-indigo-700 tabular-nums"
                            )
                        refs[ref_key] = ui.slider(
                            value=val, min=lo, max=hi, step=step,
                            on_change=lambda e, bk=badge_key: (
                                refs[bk].set_text(str(int(e.value))),
                                _refresh_params_summary(),
                                _refresh_summarize_button(),
                            ),
                        ).props("color=primary").classes("w-full")
                        with ui.row().classes("w-full justify-between"):
                            ui.label(str(lo)).classes(
                                "text-[10px] text-slate-400")
                            ui.label(str(hi)).classes(
                                "text-[10px] text-slate-400")

                # One row — sliders + Max Words + language
                with ui.row().classes("w-full gap-6 items-start"):
                    _slider_col("Max Keywords", "max_kw_input",
                                "kw_badge",   d.max_keywords, 0, 20)
                    _slider_col("Max Tags",     "max_tags_input",
                                "tags_badge", d.max_tags,     0, 20)

                    # Max Words + Language
                    with ui.column().classes("flex-1 min-w-[7rem] gap-0 justify-end"):
                        refs["max_words_input"] = ui.number(
                            "Max Words", value=d.max_words, min=50, max=2000, step=50,
                            on_change=lambda _: (
                                _refresh_params_summary(), _refresh_summarize_button()),
                        ).classes("w-full")
                    with ui.column().classes("flex-1 min-w-[8rem] gap-0 justify-end"):
                        refs["lang_input"] = ui.input(
                            label="Output language", value=d.out_lang,
                            placeholder="e.g. pt-pt, en, fr, de-DE…",
                            on_change=lambda _: (
                                _refresh_params_summary(), _refresh_summarize_button()),
                        ).props("clearable").classes("w-full")

                ui.separator().classes("w-full opacity-60")
                ui.button(
                    "Done",
                    icon="sym_o_expand_less",
                    on_click=lambda: _toggle_params_editor(),
                ).props("flat dense no-caps color=primary").classes("self-start")

        # ── Actions ────────────────────────────────────────────────────────────
        with ui.column().classes("w-full gap-2"):
            refs["status_label"] = ui.label("Select a document to begin.").classes(
                "text-lg text-gray-600 whitespace-pre-line leading-snug"
            )
            refs["restart_btn"] = ui.button(
                "New summary",
                on_click=lambda: _do_restart(),
            ).props("outline icon=sym_o_restart_alt").classes("w-full text-lg py-2 rounded")
            refs["restart_btn"].set_visibility(False)

            with ui.row().classes("w-full gap-2 items-stretch"):
                refs["summarize_btn"] = ui.button(
                    "Summarize",
                    on_click=lambda: asyncio.ensure_future(_do_summarize()),
                ).props("icon=sym_o_document_scanner").classes(
                    "flex-1 bg-blue-600 text-white text-lg py-2 rounded hover:bg-blue-700"
                )
                refs["summarize_btn"].set_visibility(False)

                refs["abort_btn"] = ui.button(
                    "Abort",
                    on_click=lambda: _do_abort(),
                ).props("flat icon=sym_o_cancel color=grey-6").classes(
                    "text-sm text-gray-400 px-3 rounded hover:text-red-500 hover:bg-red-50"
                )
                refs["abort_btn"].set_visibility(False)

        # ── Results ────────────────────────────────────────────────────────────
        with ui.column().classes("w-full gap-4") as results_section:
            refs["results_card"] = results_section
            results_section.set_visibility(False)

            with ui.card().classes("w-full p-6 bg-slate-50/90 border border-slate-100"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("sym_o_summarize").classes(
                        "text-2xl text-primary shrink-0 opacity-90")
                    ui.label("Summary").classes("text-xl font-medium")
                refs["summary_label"] = ui.label(
                    "").classes("w-full whitespace-pre-wrap")

            with ui.card().classes("w-full p-6 bg-slate-50/90 border border-slate-100"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("sym_o_format_quote").classes(
                        "text-2xl text-primary shrink-0 opacity-90")
                    ui.label("Information sources").classes(
                        "text-xl font-medium")
                refs["sources_html"] = ui.html("").classes("w-full")

            with ui.card().classes("w-full p-6 bg-slate-50/90 border border-slate-100"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("sym_o_123").classes(
                        "text-2xl text-primary shrink-0 opacity-90")
                    ui.label("Numeric verification").classes(
                        "text-xl font-medium")
                refs["numeric_html"] = ui.html("").classes("w-full")

            with ui.card().classes("w-full p-6 bg-slate-50/90 border border-slate-100"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("sym_o_key").classes(
                        "text-2xl text-primary shrink-0 opacity-90")
                    ui.label("Keywords").classes("text-xl font-medium")
                refs["keywords_label"] = ui.label("").classes("w-full")

            with ui.card().classes("w-full p-6 bg-slate-50/90 border border-slate-100"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("sym_o_sell").classes(
                        "text-2xl text-primary shrink-0 opacity-90")
                    ui.label("Tags").classes("text-xl font-medium")
                refs["tags_label"] = ui.label("").classes("w-full")

            # Download card
            with ui.card().classes("w-full p-4 bg-white border border-slate-100 shadow-sm") as dl_card:
                refs["dl_row"] = dl_card
                with ui.row().classes("w-full items-center gap-3 flex-wrap"):
                    ui.icon("sym_o_download").classes(
                        "text-2xl text-primary shrink-0 opacity-90")
                    with ui.column().classes("gap-1 flex-1 min-w-[12rem]"):
                        ui.label("Download export").classes(
                            "text-base font-medium text-slate-800")
                        ui.label("Plain text, Markdown, HTML, or JSON — all sections above.").classes(
                            "text-xs text-slate-500 leading-snug"
                        )
                    refs["export_fmt"] = ui.select(
                        {
                            "plaintext": "Plain text (.txt)",
                            "markdown": "Markdown (.md)",
                            "html": "HTML (.html)",
                            "json": "JSON (.json)",
                        },
                        value="plaintext",
                    ).props("dense outlined").classes("w-full sm:w-auto min-w-[13rem]")
                ui.button(
                    "Download export",
                    icon="sym_o_download",
                    on_click=lambda: _do_download_export(),
                ).props("color=primary no-caps").classes("w-full mt-2")

        # Footer
        with ui.row().classes(
            "items-center justify-center gap-1.5 mt-2 opacity-50 hover:opacity-80 transition-opacity"
        ):
            ui.html(_GITHUB_SVG)
            ui.link(
                "AIML4OS/WP12 · dissemination_summary_prototype",
                "https://github.com/AIML4OS/WP12/tree/main/wp12_hackathon/dissemination_summary_prototype",
                new_tab=True,
            ).classes("text-xs text-slate-500 no-underline hover:underline")

    # ── Logic callbacks ────────────────────────────────────────────────────────

    def _handle_upload(e: UploadEventArguments) -> None:
        e.content.seek(0, os.SEEK_END)
        size = e.content.tell()
        e.content.seek(0)
        pdf_bytes = e.content.read()
        e.content.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            file_info["path"] = tmp.name
        file_info.update({"name": e.name, "size_bytes": size,
                         "source": "Uploaded", "bytes": pdf_bytes})
        state["completed"] = False
        _show_selected_file()
        ui.notify(f"Uploaded: {e.name}", type="positive")

    def _show_selected_file() -> None:
        state["file_picker_open"] = False
        refs["file_name_label"].set_text(file_info["name"] or "Unknown")
        refs["file_meta_label"].set_text(
            f"{_format_size(file_info.get('size_bytes'))} · {file_info.get('source', '')}"
        )
        thumb = _pdf_thumbnail(file_path=file_info.get(
            "path"), pdf_bytes=file_info.get("bytes"))
        if thumb:
            refs["thumbnail"].set_source(thumb)
            refs["thumbnail"].set_visibility(True)
        else:
            refs["thumbnail"].set_visibility(False)
        _refresh_summarize_button()
        _refresh_params_summary()

    def _select_demo(pdf_name: str) -> None:
        demo_path = Path(__file__).resolve().parent / "demo_docs" / pdf_name
        file_info.update({
            "name": pdf_name,
            "path": str(demo_path),
            "size_bytes": demo_path.stat().st_size if demo_path.exists() else None,
            "source": "Demo",
            "bytes": None,
        })
        state["completed"] = False
        _show_selected_file()
        ui.notify(f"Selected: {pdf_name}", type="positive")

    def _clear_file() -> None:
        file_info.update({"name": None, "path": None,
                         "size_bytes": None, "source": None, "bytes": None})
        state["file_picker_open"] = True
        state["completed"] = False
        refs["results_card"].set_visibility(False)
        refs["restart_btn"].set_visibility(False)
        refs["summarize_btn"].set_visibility(False)
        refs["status_label"].set_text("Select a document to begin.")

    def _toggle_summarizer_editor() -> None:
        state["summarizer_editor_open"] = not state["summarizer_editor_open"]
        refs["summarizer_customize_btn"].set_text(
            "Done" if state["summarizer_editor_open"] else "Customize"
        )
        if not state["summarizer_editor_open"]:
            _refresh_summarizer_summary()
        _refresh_summarize_button()

    def _toggle_params_editor() -> None:
        state["parameters_editor_open"] = not state["parameters_editor_open"]
        refs["params_customize_btn"].set_text(
            "Done" if state["parameters_editor_open"] else "Customize"
        )
        if not state["parameters_editor_open"]:
            _refresh_params_summary()
        _refresh_summarize_button()

    def _on_endpoint_change(ep_id: str) -> None:
        models = _chat_models_for(ep_id)
        refs["model_select"].set_options(
            models or ["—"], value=models[0] if models else "—")
        _emb_opts = _embed_options()
        cur_emb = refs["embedding_select"].value
        # Preserve the current selection only if it is already a remote embedding;
        # if it is "local" (the fallback when no probes had run yet) let _default_emb
        # upgrade it to the first available remote.
        keep_emb = cur_emb if (cur_emb and cur_emb != "local" and cur_emb in _emb_opts) else _default_emb(_emb_opts)
        refs["embedding_select"].set_options(_emb_opts, value=keep_emb)
        _refresh_summarizer_summary()
        _refresh_summarize_button()

    def _refresh_summarizer_summary() -> None:
        ep_id = refs.get("endpoint_select") and refs["endpoint_select"].value
        model = refs.get("model_select") and refs["model_select"].value
        ep = next((e for e in cfg.endpoints if e.id == ep_id), None)
        status = PROBE_RESULTS.get(ep_id or "") if ep_id else None
        ok = "✓" if (status and status.ok) else "—"
        ep_name = ep.name if ep else (ep_id or "—")
        mode = (refs.get("mode_radio")
                and refs["mode_radio"].value) or cfg.defaults.processing_mode
        loader = (refs.get("loader_radio")
                  and refs["loader_radio"].value) or cfg.defaults.pdf_loader
        mode_txt = "Plain text" if mode == "plain" else "Vector retrieval"
        loader_txt = "Docling" if loader == "docling" else "PyPDF"
        refs["sum_ep_label"].set_text(f"LLM: {ok} {ep_name}")
        refs["sum_model_label"].set_text(f"Model: {model or '—'}")
        refs["sum_mode_label"].set_text(f"Mode: {mode_txt}")
        refs["sum_loader_label"].set_text(f"Loader: {loader_txt}")

    async def _reprobe_all() -> None:
        if refs.get("reprobe_btn"):
            refs["reprobe_btn"].disable()
            refs["reprobe_btn"].set_text("Probing…")
        await run.io_bound(_probe_all_endpoints, cfg)
        ep_opts = _endpoint_options()
        cur_ep = refs["endpoint_select"].value if refs.get(
            "endpoint_select") else None
        cur_still_available = cur_ep in ep_opts and not ep_opts[cur_ep].startswith(
            "⊘")
        if not cur_still_available:
            cur_ep = next(
                (k for k, v in ep_opts.items() if not v.startswith("⊘")),
                next(iter(ep_opts), None),
            )
        if refs.get("endpoint_select"):
            refs["endpoint_select"].set_options(ep_opts, value=cur_ep)
            _patch_endpoint_disabled(refs["endpoint_select"])
        if cur_ep:
            models = _chat_models_for(cur_ep)
            if refs.get("model_select"):
                refs["model_select"].set_options(
                    models or ["—"], value=models[0] if models else "—")
            if refs.get("embedding_select"):
                _emb_opts = _embed_options()
                cur_emb = refs["embedding_select"].value
                keep_emb = cur_emb if (cur_emb and cur_emb != "local" and cur_emb in _emb_opts) else _default_emb(_emb_opts)
                refs["embedding_select"].set_options(_emb_opts, value=keep_emb)
        _refresh_summarizer_summary()
        _refresh_summarize_button()
        if refs.get("reprobe_btn"):
            refs["reprobe_btn"].enable()
            refs["reprobe_btn"].set_text("Re-probe endpoints")

    def _on_ocr_toggle(enabled: bool) -> None:
        _refresh_params_summary()
        _refresh_summarize_button()

    def _refresh_params_summary() -> None:
        try:
            d = cfg.defaults
            mw = int(refs.get("max_words_input")
                     and refs["max_words_input"].value or d.max_words)
            mkw = int(refs.get("max_kw_input")
                      and refs["max_kw_input"].value or d.max_keywords)
            mt = int(refs.get("max_tags_input")
                     and refs["max_tags_input"].value or d.max_tags)
            lang = ((refs.get("lang_input")
                    and refs["lang_input"].value) or d.out_lang or "pt-pt").strip()
            refs["param_pill_words"].set_text(str(mw))
            refs["param_pill_kw"].set_text(str(mkw))
            refs["param_pill_tags"].set_text(str(mt))
            refs["param_pill_lang"].set_text(lang or "pt-pt")
        except Exception:
            pass

    def _do_abort() -> None:
        ev = state.get("cancel_event")
        if ev is not None:
            ev.set()
        refs["status_label"].set_text("Aborting…")
        refs["abort_btn"].set_visibility(False)

    def _do_restart() -> None:
        state["completed"] = False
        refs["results_card"].set_visibility(False)
        refs["restart_btn"].set_visibility(False)
        refs["status_label"].set_text("Ready.")
        _refresh_summarize_button()

    async def _do_summarize() -> None:
        if not file_info.get("path"):
            ui.notify("Please select a PDF first.", type="warning")
            return
        opts = _build_current_options(include_endpoint=True)
        if opts is None:
            ui.notify("Please select an endpoint and model.", type="warning")
            return

        cfg_copy = cfg

        state["processing"] = True
        state["completed"] = False
        cancel_ev = threading.Event()
        state["cancel_event"] = cancel_ev

        if state.get("summarizer_editor_open"):
            state["summarizer_editor_open"] = False
            if refs.get("summarizer_customize_btn"):
                refs["summarizer_customize_btn"].set_text("Customize")
            _refresh_summarizer_summary()
        if state.get("parameters_editor_open"):
            state["parameters_editor_open"] = False
            if refs.get("params_customize_btn"):
                refs["params_customize_btn"].set_text("Customize")
            _refresh_params_summary()

        for _btn_key in ("clear_selection_btn", "summarizer_customize_btn", "params_customize_btn"):
            if refs.get(_btn_key):
                refs[_btn_key].disable()

        _refresh_summarize_button()
        refs["abort_btn"].set_visibility(True)
        refs["restart_btn"].set_visibility(False)
        refs["results_card"].set_visibility(False)
        refs["status_label"].set_text("Processing…")

        _progress_q: queue.SimpleQueue = queue.SimpleQueue()
        _real_stdout = sys.stdout

        def _on_progress(msg: str) -> None:
            _progress_q.put(msg)

        t_start = time.perf_counter()

        def _run() -> dict:
            sys.stdout = _StdoutTee(_real_stdout, _progress_q)
            try:
                return summarize(
                    Path(file_info["path"]), opts,
                    config=cfg_copy, cancel_event=cancel_ev, on_progress=_on_progress,
                )
            finally:
                sys.stdout = _real_stdout

        async def _poll_loop() -> None:
            while state["processing"]:
                while not _progress_q.empty():
                    try:
                        refs["status_label"].set_text(_progress_q.get_nowait())
                    except Exception:
                        pass
                await asyncio.sleep(0.25)

        poll_task = asyncio.ensure_future(_poll_loop())
        try:
            result = await run.io_bound(_run)
        except Exception as exc:
            err_text = _format_summarize_error(
                exc,
                model=opts.llm_model,
                endpoint=opts.llm_endpoint,
            )
            result = {
                "summary": f"Error: {err_text}", "keywords": [], "tags": [],
                "sources": [], "numeric_claims": [], "unmatched_numbers": [],
                "_from_cache": False, "_timing_sec": {},
            }
        finally:
            state["processing"] = False
            poll_task.cancel()
            sys.stdout = _real_stdout
            for _btn_key in ("clear_selection_btn", "summarizer_customize_btn", "params_customize_btn"):
                if refs.get(_btn_key):
                    refs[_btn_key].enable()

        elapsed = time.perf_counter() - t_start
        timing = result.get("_timing_sec", {})
        from_cache = result.get("_from_cache", False)

        if cancel_ev.is_set():
            refs["status_label"].set_text("Aborted.")
            refs["abort_btn"].set_visibility(False)
            _refresh_summarize_button()
            return

        summary_text = str(result.get("summary") or "")
        keywords = result.get("keywords") or []
        tags = result.get("tags") or []
        sources = result.get("sources") or []
        numeric_claims = result.get("numeric_claims") or []
        unmatched = result.get("unmatched_numbers") or []

        refs["summary_label"].set_text(summary_text)
        refs["sources_html"].set_content(_sources_block_html(sources))

        refs["numeric_html"].set_content(
            _numeric_claims_block_html(numeric_claims, sources))
        refs["keywords_label"].set_text(
            "Keywords: " + (", ".join(str(k) for k in keywords) or "—"))
        refs["tags_label"].set_text(
            "Tags: " + (", ".join(str(t) for t in tags) or "—"))

        export_meta.update({
            "ready": True, "summary": summary_text, "sources": sources,
            "keywords": keywords, "tags": tags, "numeric_claims": numeric_claims,
            "unmatched_numbers": unmatched, "elapsed_s": elapsed, "timing_breakdown": timing,
        })

        cache_prefix = "Cached · " if from_cache else ""
        refs["status_label"].set_text(
            cache_prefix + _format_status_done(elapsed, timing))
        refs["results_card"].set_visibility(True)
        refs["abort_btn"].set_visibility(False)
        state["completed"] = True
        refs["restart_btn"].set_visibility(True)
        _refresh_summarize_button()

    def _do_download_export() -> None:
        if not export_meta.get("ready"):
            ui.notify("No result to download yet.", type="warning")
            return
        fmt = refs["export_fmt"].value or "plaintext"
        ext_map = {"plaintext": ".txt", "markdown": ".md",
                   "html": ".html", "json": ".json"}
        ext = ext_map.get(fmt, ".txt")
        stem = _sanitize_filename(export_meta.get("summary", "summary")[:60])
        filename = f"summary_{stem}{ext}"
        builders = {
            "plaintext": _build_export_plaintext,
            "markdown": _build_export_markdown,
            "html": _build_export_html,
            "json": _build_export_json,
        }
        content = builders.get(fmt, _build_export_plaintext)(
            export_meta["summary"], export_meta["sources"],
            export_meta["keywords"], export_meta["tags"],
            export_meta["numeric_claims"], export_meta["unmatched_numbers"],
        )
        ui.download(content.encode("utf-8"), filename=filename)

    # ── Initial state sync ─────────────────────────────────────────────────────
    _refresh_summarizer_summary()
    _refresh_params_summary()
    _refresh_summarize_button()


# ── Run ───────────────────────────────────────────────────────────────────────

@ui.page("/")
def index() -> None:
    _main_page()


if __name__ in ("__main__", "__mp_main__"):
    ui.run(
        host="0.0.0.0",
        port=5001,
        reload=True,
        title="WP12 · PDF Summarizer",
        favicon=_FAVICON_SVG,
    )
