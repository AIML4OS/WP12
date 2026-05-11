import os
import re
import json
import html
import asyncio
import tempfile
import base64
import time
from nicegui import ui, run
from summarizer_unified import (
    PDFSummarizer,
    DEFAULT_LLM_MODELS_FALLBACK,
    fetch_llm_models_for_provider,
    probe_ollama_runtime,
    probe_specific_ollama_runtime,
)
from nicegui.events import UploadEventArguments

uploaded_file = {}
selected_demo_file = {'path': None}
selected_file_info = {'name': None, 'size_bytes': None, 'source': None}
summary_label = None
sources_label = None
keywords_label = None
tags_label = None
results_container = None
download_button = None
file_picker_container = None
selected_file_container = None
selected_file_name_label = None
selected_file_size_label = None
selected_file_source_label = None
selected_file_thumbnail = None
pdf_upload_component = None
summarize_actions_container = None
summarize_button = None
restart_button = None
clear_selection_button = None
change_file_hint_label = None
summarizer_customize_button = None
parameters_customize_button = None
summarizer_config_summary_panel_ref = None
summarizer_config_editor_panel_ref = None
parameters_summary_panel_ref = None
parameters_editor_panel_ref = None
file_card_container = None
summarizer_config_card_container = None
parameters_card_container = None

UI_STATE = {
    "processing": False,
    "process_completed": False,
    "summarizer_editor_open": False,
    "parameters_editor_open": False,
    "file_picker_open": True,
}

# Last successful run: download stem (no extension), structured export + timing.
LAST_SUMMARY_META = {
    "download_stem": "summary_full",
    "export_ready": False,
    "export_summary": "",
    "export_sources": [],
    "export_keywords": [],
    "export_tags": [],
    "export_numeric_claims": [],
    "export_unmatched_numbers": [],
    "elapsed_s": None,
    "timing_breakdown": None,
}

_EXPORT_FORMAT_EXT = {
    "plaintext": ".txt",
    "markdown": ".md",
    "html": ".html",
    "json": ".json",
}

# Order matches summarizer_unified.process_pdf phase keys.
_TIMING_DISPLAY_ORDER = ("chunks", "vector_store", "pdf_load", "llm")
_TIMING_LABELS = {
    "chunks": "Chunks",
    "vector_store": "Index",
    "pdf_load": "PDF",
    "llm": "LLM",
}


def format_done_status_line(total_s: float, timing_sec: dict | None) -> str:
    """Two-line status: total wall time (worker) + per-phase breakdown from summarizer."""
    line1 = f"Done · {total_s:.1f}s total"
    if not timing_sec:
        return line1
    parts = []
    for key in _TIMING_DISPLAY_ORDER:
        if key in timing_sec:
            parts.append(f"{_TIMING_LABELS[key]} {timing_sec[key]:.1f}s")
    if not parts:
        return line1
    return line1 + "\n" + " · ".join(parts)


def _sanitize_filename_token(s: str, max_len: int = 72) -> str:
    raw = (s or "").strip()
    raw = re.sub(r'[\s<>:\"/\\|?*]+', "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    if len(raw) > max_len:
        raw = raw[:max_len].rstrip("_")
    return raw or "na"


def build_summary_download_stem(
    *,
    max_words: int,
    max_keywords: int,
    max_tags: int,
    out_lang: str,
    provider: str,
    processing_mode: str,
    embedding_choice: str | None,
    loader: str,
    llm_model: str,
) -> str:
    """Dense ASCII filename stem (no extension) for exports."""
    lang = _sanitize_filename_token(out_lang.replace(" ", ""), max_len=28)
    prov = _sanitize_filename_token(provider, max_len=12)
    load = _sanitize_filename_token(loader, max_len=12)
    model = _sanitize_filename_token(llm_model, max_len=56)
    if processing_mode == "plain":
        mode_emb = "plain"
    else:
        emb = "loc" if (embedding_choice or "") == "local" else "rem"
        mode_emb = f"vect-{emb}"
    stem = (
        f"summary_w{max_words}_kw{max_keywords}_tg{max_tags}_{lang}_{prov}_"
        f"{mode_emb}_{load}_{model}"
    )
    if len(stem) > 175:
        stem = stem[:175].rstrip("_-.")
    return stem


def _new_supports(src: dict, already_shown: set) -> list:
    """Return the supports_numbers entries on `src` not yet labelled elsewhere.

    Supports are deduplicated across the displayed source list so each summary
    number is only labelled once - on the first source that actually contains
    it - rather than repeating "verifies: 2025" on every chunk that mentions
    the year. The underlying per-source supports_numbers data is preserved on
    the source dict itself for any consumer that wants the raw mapping.
    """
    supports = src.get("supports_numbers") if isinstance(src, dict) else None
    if not supports:
        return []
    return [str(n) for n in supports if str(n) not in already_shown]


def _format_supports_line(new_numbers: list) -> str:
    if not new_numbers:
        return ""
    return "verifies: " + ", ".join(new_numbers)


def _sources_block_html(sources: list) -> str:
    """Render sources as styled HTML for the UI panel."""
    if not isinstance(sources, list) or not sources:
        return '<p class="text-sm text-gray-400 italic">No source attribution available.</p>'
    parts = []
    shown_numbers: set = set()
    for idx, src in enumerate(sources, start=1):
        if not isinstance(src, dict):
            continue
        src_name = html.escape(str(src.get("source", "document")))
        src_loc = html.escape(str(src.get("location", "n/a")))
        src_excerpt = html.escape(str(src.get("excerpt", "")).strip())
        new_numbers = _new_supports(src, shown_numbers)
        shown_numbers.update(new_numbers)

        header = (
            f'<div class="text-sm text-gray-400 font-mono">'
            f'{idx}. {src_name} ({src_loc})'
            f'</div>'
        )

        if new_numbers:
            years_html = html.escape(", ".join(new_numbers))
            verifies_line = (
                f'<div class="text-base mt-0.5">'
                f'<span class="font-semibold" style="color: var(--q-primary)">verifies:</span>'
                f'<span class="text-gray-600 ml-1">{years_html}</span>'
                f'</div>'
            )
        else:
            verifies_line = ""

        excerpt_html = (
            f'<div class="text-base leading-relaxed mt-1 text-gray-800">{src_excerpt}</div>'
            if src_excerpt
            else ""
        )

        parts.append(
            f'<div class="mb-6">{header}{verifies_line}{excerpt_html}</div>'
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
    shown_numbers: set = set()
    for idx, src in enumerate(sources, start=1):
        if not isinstance(src, dict):
            continue
        src_name = str(src.get("source", "document"))
        src_loc = str(src.get("location", "n/a"))
        src_excerpt = str(src.get("excerpt", "")).strip()
        new_numbers = _new_supports(src, shown_numbers)
        shown_numbers.update(new_numbers)
        supports_line = _format_supports_line(new_numbers)
        body = f"   {supports_line}\n   {src_excerpt}" if supports_line else src_excerpt
        parts.append(f"{idx}. {src_name} ({src_loc})\n{body}")
    return "\n\n".join(parts) if parts else "No source attribution available."


def _numeric_claims_block_plain(
    numeric_claims: list,
    unmatched_numbers: list,
) -> str:
    """Render a 'Numeric verification' block for the plaintext export."""
    if not isinstance(numeric_claims, list):
        numeric_claims = []
    if not isinstance(unmatched_numbers, list):
        unmatched_numbers = []
    if not numeric_claims and not unmatched_numbers:
        return "No numeric values detected in the summary."
    lines = []
    for claim in numeric_claims:
        if not isinstance(claim, dict):
            continue
        num = str(claim.get("number", "")).strip()
        ids = claim.get("source_ids") or []
        if ids:
            lines.append(f"- {num}: source(s) {', '.join(str(i) for i in ids)}")
        else:
            lines.append(f"- {num}: NO MATCHING SOURCE EXCERPT")
    if unmatched_numbers:
        lines.append("")
        lines.append(
            "Warning: the following summary numbers could not be matched "
            "verbatim to any retrieved source excerpt: "
            + ", ".join(str(n) for n in unmatched_numbers)
        )
    return "\n".join(lines) if lines else "No numeric values detected in the summary."


def _build_export_plaintext(
    summary: str,
    sources: list,
    keywords: list,
    tags: list,
    numeric_claims: list | None = None,
    unmatched_numbers: list | None = None,
) -> str:
    kw = ", ".join(str(k) for k in keywords) if keywords else "—"
    tg = ", ".join(str(t) for t in tags) if tags else "—"
    src_block = _sources_block_plain(sources)
    num_block = _numeric_claims_block_plain(
        numeric_claims or [], unmatched_numbers or []
    )
    return (
        f"Summary\n-------\n{summary}\n\n"
        f"Information sources\n-------------------\n{src_block}\n\n"
        f"Numeric verification\n--------------------\n{num_block}\n\n"
        f"Keywords\n--------\n{kw}\n\n"
        f"Tags\n----\n{tg}\n"
    )


def _build_export_markdown(
    summary: str,
    sources: list,
    keywords: list,
    tags: list,
    numeric_claims: list | None = None,
    unmatched_numbers: list | None = None,
) -> str:
    kw = ", ".join(str(k) for k in keywords) if keywords else "—"
    tg = ", ".join(str(t) for t in tags) if tags else "—"
    lines = [
        "## Summary",
        "",
        summary.strip() or "—",
        "",
        "## Information sources",
        "",
    ]
    if isinstance(sources, list) and sources:
        shown_numbers: set = set()
        for idx, src in enumerate(sources, start=1):
            if not isinstance(src, dict):
                continue
            src_name = str(src.get("source", "document"))
            src_loc = str(src.get("location", "n/a"))
            src_excerpt = str(src.get("excerpt", "")).strip()
            new_numbers = _new_supports(src, shown_numbers)
            shown_numbers.update(new_numbers)
            supports_line = _format_supports_line(new_numbers)
            lines.append(f"{idx}. **{src_name}** ({src_loc})")
            lines.append("")
            if supports_line:
                lines.append(f"   *{supports_line}*")
                lines.append("")
            lines.append(f"   {src_excerpt}")
            lines.append("")
    else:
        lines.append("*No source attribution available.*")
        lines.append("")

    lines.extend(["## Numeric verification", ""])
    claims = numeric_claims if isinstance(numeric_claims, list) else []
    unmatched = unmatched_numbers if isinstance(unmatched_numbers, list) else []
    if claims:
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            num = str(claim.get("number", "")).strip()
            ids = claim.get("source_ids") or []
            if ids:
                lines.append(
                    f"- **{num}** — source(s) {', '.join(str(i) for i in ids)}"
                )
            else:
                lines.append(f"- **{num}** — *no matching source excerpt*")
    else:
        lines.append("*No numeric values detected in the summary.*")
    if unmatched:
        lines.append("")
        lines.append(
            "> Warning: the following summary numbers could not be matched "
            "verbatim to any retrieved source excerpt: "
            + ", ".join(str(n) for n in unmatched)
        )
    lines.append("")

    lines.extend(
        [
            "## Keywords",
            "",
            kw,
            "",
            "## Tags",
            "",
            tg,
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _build_export_html(
    summary: str,
    sources: list,
    keywords: list,
    tags: list,
    numeric_claims: list | None = None,
    unmatched_numbers: list | None = None,
) -> str:
    h = html.escape
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        "<title>Summary export</title>",
        "</head>",
        "<body>",
        "<h2>Summary</h2>",
        "<p>" + h(summary).replace("\n", "<br>") + "</p>",
        "<h2>Information sources</h2>",
    ]
    if isinstance(sources, list) and sources:
        parts.append("<ol>")
        shown_numbers: set = set()
        for src in sources:
            if not isinstance(src, dict):
                continue
            src_name = h(str(src.get("source", "document")))
            src_loc = h(str(src.get("location", "n/a")))
            src_excerpt = h(str(src.get("excerpt", "")).strip())
            br_excerpt = src_excerpt.replace("\n", "<br>")
            new_numbers = _new_supports(src, shown_numbers)
            shown_numbers.update(new_numbers)
            supports_line = _format_supports_line(new_numbers)
            supports_html = (
                f"<em>{h(supports_line)}</em><br>" if supports_line else ""
            )
            parts.append(
                "<li><strong>"
                f"{src_name}</strong> ({src_loc})<br>"
                f"{supports_html}"
                f"{br_excerpt}</li>"
            )
        parts.append("</ol>")
    else:
        parts.append("<p><em>No source attribution available.</em></p>")

    parts.append("<h2>Numeric verification</h2>")
    claims = numeric_claims if isinstance(numeric_claims, list) else []
    unmatched = unmatched_numbers if isinstance(unmatched_numbers, list) else []
    if claims:
        parts.append("<ul>")
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            num = h(str(claim.get("number", "")).strip())
            ids = claim.get("source_ids") or []
            if ids:
                ids_str = ", ".join(h(str(i)) for i in ids)
                parts.append(f"<li><strong>{num}</strong> — source(s) {ids_str}</li>")
            else:
                parts.append(
                    f"<li><strong>{num}</strong> — <em>no matching source excerpt</em></li>"
                )
        parts.append("</ul>")
    else:
        parts.append("<p><em>No numeric values detected in the summary.</em></p>")
    if unmatched:
        unmatched_str = ", ".join(h(str(n)) for n in unmatched)
        parts.append(
            "<p><strong>Warning:</strong> the following summary numbers could "
            "not be matched verbatim to any retrieved source excerpt: "
            f"{unmatched_str}</p>"
        )

    kw_items = "".join(f"<li>{h(str(k))}</li>" for k in keywords) if keywords else ""
    tg_items = "".join(f"<li>{h(str(t))}</li>" for t in tags) if tags else ""
    parts.extend(
        [
            "<h2>Keywords</h2>",
            f"<ul>{kw_items}</ul>" if kw_items else "<p>—</p>",
            "<h2>Tags</h2>",
            f"<ul>{tg_items}</ul>" if tg_items else "<p>—</p>",
            "</body>",
            "</html>",
        ]
    )
    return "\n".join(parts)


def _build_export_json(
    summary: str,
    sources: list,
    keywords: list,
    tags: list,
    numeric_claims: list | None = None,
    unmatched_numbers: list | None = None,
) -> str:
    raw_sources = sources if isinstance(sources, list) else []
    clean_sources = [s for s in raw_sources if isinstance(s, dict)]
    raw_claims = numeric_claims if isinstance(numeric_claims, list) else []
    clean_claims = [c for c in raw_claims if isinstance(c, dict)]
    payload = {
        "summary": summary,
        "sources": clean_sources,
        "keywords": list(keywords) if isinstance(keywords, list) else [],
        "tags": list(tags) if isinstance(tags, list) else [],
        "numeric_claims": clean_claims,
        "unmatched_numbers": (
            list(unmatched_numbers) if isinstance(unmatched_numbers, list) else []
        ),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def build_download_export_content(fmt: str) -> str:
    """Serialize LAST_SUMMARY_META snapshot for the chosen format."""
    summary = str(LAST_SUMMARY_META.get("export_summary") or "")
    sources = LAST_SUMMARY_META.get("export_sources") or []
    keywords = LAST_SUMMARY_META.get("export_keywords") or []
    tags = LAST_SUMMARY_META.get("export_tags") or []
    numeric_claims = LAST_SUMMARY_META.get("export_numeric_claims") or []
    unmatched_numbers = LAST_SUMMARY_META.get("export_unmatched_numbers") or []
    if fmt == "plaintext":
        return _build_export_plaintext(
            summary, sources, keywords, tags, numeric_claims, unmatched_numbers
        )
    if fmt == "markdown":
        return _build_export_markdown(
            summary, sources, keywords, tags, numeric_claims, unmatched_numbers
        )
    if fmt == "html":
        return _build_export_html(
            summary, sources, keywords, tags, numeric_claims, unmatched_numbers
        )
    if fmt == "json":
        return _build_export_json(
            summary, sources, keywords, tags, numeric_claims, unmatched_numbers
        )
    return _build_export_plaintext(
        summary, sources, keywords, tags, numeric_claims, unmatched_numbers
    )


SOURCE_DISPLAY = {
    'Uploaded file': 'Uploaded',
    'Demo file': 'Example PDF',
}


def has_pdf_source_selected():
    if uploaded_file.get('file') is not None:
        return True
    demo_path = selected_demo_file.get('path')
    return bool(demo_path and os.path.exists(demo_path))


def update_summarize_actions_visibility():
    if summarize_actions_container is None:
        return
    actions_visible = has_pdf_source_selected()
    summarize_actions_container.set_visibility(actions_visible)
    summarize_actions_container.update()
    if restart_button is not None:
        restart_button.set_visibility(
            actions_visible
            and not UI_STATE.get("processing", False)
            and UI_STATE.get("process_completed", False)
        )
        restart_button.update()

    if summarize_button is not None:
        can_run = (
            actions_visible
            and not UI_STATE.get("process_completed", False)
            and not UI_STATE.get("summarizer_editor_open", False)
            and not UI_STATE.get("parameters_editor_open", False)
            and not UI_STATE.get("file_picker_open", False)
        )
        if UI_STATE.get("processing", False):
            summarize_button.set_text("Please Wait")
            summarize_button.props('icon=sym_o_document_scanner')
            summarize_button.classes(add='processing-spinner-matrix')
            summarize_button.set_visibility(actions_visible)
            disabler = getattr(summarize_button, "disable", None)
            if callable(disabler):
                disabler()
        else:
            summarize_button.set_text("Summarize")
            summarize_button.props('icon=sym_o_document_scanner')
            summarize_button.classes(remove='processing-spinner-matrix')
            enabler = getattr(summarize_button, "enable", None)
            if callable(enabler):
                enabler()
            summarize_button.set_visibility(can_run)
        summarize_button.update()


def invalidate_result() -> None:
    """Mark any previous result as stale so the Summarize button reappears."""
    if UI_STATE.get("process_completed", False) and not UI_STATE.get("processing", False):
        UI_STATE["process_completed"] = False
        update_summarize_actions_visibility()


def set_processing_ui_locked(locked: bool) -> None:
    """Keep page visible; close editors and lock action controls while processing."""
    UI_STATE["processing"] = locked

    if locked:
        if (
            summarizer_config_editor_panel_ref is not None
            and summarizer_config_summary_panel_ref is not None
        ):
            summarizer_config_editor_panel_ref.set_visibility(False)
            summarizer_config_summary_panel_ref.set_visibility(True)
            summarizer_config_editor_panel_ref.update()
            summarizer_config_summary_panel_ref.update()
            UI_STATE["summarizer_editor_open"] = False
        if parameters_editor_panel_ref is not None and parameters_summary_panel_ref is not None:
            parameters_editor_panel_ref.set_visibility(False)
            parameters_summary_panel_ref.set_visibility(True)
            parameters_editor_panel_ref.update()
            parameters_summary_panel_ref.update()
            UI_STATE["parameters_editor_open"] = False

    controls_visible = not locked
    if clear_selection_button is not None:
        clear_selection_button.set_visibility(controls_visible)
        clear_selection_button.update()
    if change_file_hint_label is not None:
        change_file_hint_label.set_visibility(controls_visible)
        change_file_hint_label.update()
    if summarizer_customize_button is not None:
        summarizer_customize_button.set_visibility(controls_visible)
        summarizer_customize_button.update()
    if parameters_customize_button is not None:
        parameters_customize_button.set_visibility(controls_visible)
        parameters_customize_button.update()

    update_summarize_actions_visibility()


LLM_PROVIDER_LABEL = {
    "ssp": "Remote Ollama (SSPCloud)",
    "ollama": "Local Ollama",
    "ollama_ine": "Remote Ollama (Statistics Portugal)",
    "openai": "Remote OpenAI",
}

# Updated by probe_ollama_runtime (worker thread / cpu_bound).
OLLAMA_PROBE_STATE: dict = {
    "ok": False,
    "base_url": None,
    "models": [],
    "error": None,
}
OLLAMA_INE_PROBE_STATE: dict = {
    "ok": False,
    "base_url": "https://ollama.ine.pt",
    "models": [],
    "error": None,
}

PROVIDER_SELECT_OPTIONS_NO_OLLAMA = {
    "ssp": "Remote SSP (SSPCloud)",
    "openai": "Remote OpenAI",
}

PROVIDER_SELECT_OPTIONS_WITH_OLLAMA = {
    **PROVIDER_SELECT_OPTIONS_NO_OLLAMA,
    "ollama": "Local Ollama",
    "ollama_ine": "Remote Ollama (Statistics Portugal)",
}


def apply_provider_select_options(
    select_el, *, include_local_ollama: bool, include_ine_ollama: bool
) -> None:
    """
    Refresh provider dropdown options so 'Local Ollama' appears after a successful probe.

    Prefer ChoiceElement.set_options() so value indices stay aligned with the client (#1073).
    """
    opts = dict(
        PROVIDER_SELECT_OPTIONS_NO_OLLAMA
    )
    if include_local_ollama:
        opts["ollama"] = PROVIDER_SELECT_OPTIONS_WITH_OLLAMA["ollama"]
    if include_ine_ollama:
        opts["ollama_ine"] = PROVIDER_SELECT_OPTIONS_WITH_OLLAMA["ollama_ine"]
    prev = select_el.value
    if prev in ("ollama", "ollama_ine") and prev not in opts:
        prev = 'ssp'
    setter = getattr(select_el, "set_options", None)
    if callable(setter):
        setter(opts, value=prev)
    else:
        select_el.options = opts
        select_el.value = prev
        select_el.update()


def current_ollama_base_url() -> str:
    """Resolved URL for local Ollama after probe, else env/default."""
    bu = (OLLAMA_PROBE_STATE.get("base_url") or "").strip()
    if bu:
        return bu
    return (os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").strip()


def current_ollama_ine_base_url() -> str:
    bu = (OLLAMA_INE_PROBE_STATE.get("base_url") or "").strip()
    if bu:
        return bu
    return "https://ollama.ine.pt"


def run_summarization(
    file_path,
    source_display_name,
    max_keywords,
    max_tags,
    max_words,
    out_lang,
    use_vector_store,
    document_loader,
    use_remote_embedding,
    llm_provider,
    llm_model,
    ollama_base_url,
):
    # Re-read .env on every call so key changes take effect without a restart.
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'),
        override=True,
    )
    lp = (llm_provider or "ssp").lower().strip()
    lc = {"temperature": 0.1}
    if lp == "ssp":
        lc["api_key"] = os.getenv("SSP_KEY")
        lc["model"] = llm_model
    elif lp == "openai":
        lc["api_key"] = os.getenv("OPENAI_API_KEY")
        lc["model"] = llm_model
    elif lp == "ollama":
        lc["model"] = llm_model
        base = (ollama_base_url or "").strip() or os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        lc["base_url"] = base
    elif lp == "ollama_ine":
        lc["model"] = llm_model
        base = (ollama_base_url or "").strip() or "https://ollama.ine.pt"
        lc["base_url"] = base
        lp = "ollama"
    else:
        lc["api_key"] = os.getenv("SSP_KEY")
        lc["model"] = llm_model
        lp = "ssp"

    summarizer = PDFSummarizer(llm_provider=lp, llm_config=lc)
    return summarizer.process_pdf(
        file_path,
        use_vector_store=use_vector_store,
        document_loader=document_loader,
        embedding_model="BAAI/bge-m3",
        use_remote_embedding=use_remote_embedding,
        max_keywords=max_keywords,
        max_tags=max_tags,
        max_words=max_words,
        out_lang=out_lang,
        display_source_name=source_display_name,
    )


def format_size(num_bytes):
    if num_bytes is None:
        return "Unknown size"
    units = ["B", "KB", "MB", "GB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def build_pdf_thumbnail_data_uri(file_path=None, pdf_bytes=None):
    try:
        fitz = __import__('fitz')
    except Exception:
        return None
    try:
        if pdf_bytes is not None:
            pdf_doc = fitz.open(stream=pdf_bytes, filetype='pdf')
        else:
            pdf_doc = fitz.open(file_path)
        page = pdf_doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(0.45, 0.45), alpha=False)
        png_bytes = pix.tobytes("png")
        pdf_doc.close()
        return f"data:image/png;base64,{base64.b64encode(png_bytes).decode('ascii')}"
    except Exception:
        return None


def show_file_selection():
    if (
        file_picker_container is None
        or selected_file_container is None
        or selected_file_name_label is None
    ):
        return
    name = selected_file_info.get('name', 'Unknown file')
    source = selected_file_info.get('source', '')
    size_text = format_size(selected_file_info.get('size_bytes'))
    source_text = SOURCE_DISPLAY.get(source, source or '—')

    selected_file_name_label.set_text(name)
    if selected_file_size_label is not None:
        selected_file_size_label.set_text(size_text)
        selected_file_size_label.update()
    if selected_file_source_label is not None:
        selected_file_source_label.set_text(source_text)
        selected_file_source_label.update()

    selected_file_name_label.update()
    file_picker_container.set_visibility(False)
    selected_file_container.set_visibility(True)
    file_picker_container.update()
    selected_file_container.update()
    UI_STATE["file_picker_open"] = False
    update_summarize_actions_visibility()


def clear_selected_file():
    uploaded_file['file'] = None
    selected_demo_file['path'] = None
    selected_file_info['name'] = None
    selected_file_info['size_bytes'] = None
    selected_file_info['source'] = None
    if selected_file_thumbnail is not None:
        selected_file_thumbnail.set_visibility(False)
        selected_file_thumbnail.update()
    if file_picker_container is not None and selected_file_container is not None:
        selected_file_container.set_visibility(False)
        file_picker_container.set_visibility(True)
        selected_file_container.update()
        file_picker_container.update()
    UI_STATE["file_picker_open"] = True
    if pdf_upload_component is not None:
        pdf_upload_component.reset()
        pdf_upload_component.update()
    update_summarize_actions_visibility()
    ui.notify('File selection cleared.', type='info')


def handle_upload(e: UploadEventArguments):
    uploaded_file['file'] = e
    selected_demo_file['path'] = None
    e.content.seek(0, os.SEEK_END)
    size_bytes = e.content.tell()
    e.content.seek(0)
    selected_file_info['name'] = e.name
    selected_file_info['size_bytes'] = size_bytes
    selected_file_info['source'] = 'Uploaded file'
    thumbnail = build_pdf_thumbnail_data_uri(pdf_bytes=e.content.read())
    e.content.seek(0)
    if selected_file_thumbnail is not None:
        if thumbnail:
            selected_file_thumbnail.set_source(thumbnail)
            selected_file_thumbnail.set_visibility(True)
        else:
            selected_file_thumbnail.set_visibility(False)
        selected_file_thumbnail.update()
    show_file_selection()
    ui.notify(f'✅ Uploaded: {e.name}', type='positive')


def list_demo_pdfs():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    demo_docs_dir = os.path.join(base_dir, 'demo_docs')
    if not os.path.isdir(demo_docs_dir):
        return []
    pdfs = [
        file_name for file_name in os.listdir(demo_docs_dir)
        if file_name.lower().endswith('.pdf')
    ]
    return sorted(pdfs)


def select_demo_pdf(file_name: str):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'demo_docs', file_name)
    selected_demo_file['path'] = file_path
    uploaded_file['file'] = None
    selected_file_info['name'] = file_name
    selected_file_info['size_bytes'] = os.path.getsize(
        file_path) if os.path.exists(file_path) else None
    selected_file_info['source'] = 'Demo file'
    thumbnail = build_pdf_thumbnail_data_uri(file_path=file_path)
    if selected_file_thumbnail is not None:
        if thumbnail:
            selected_file_thumbnail.set_source(thumbnail)
            selected_file_thumbnail.set_visibility(True)
        else:
            selected_file_thumbnail.set_visibility(False)
        selected_file_thumbnail.update()
    show_file_selection()
    ui.notify(f'✅ Selected demo PDF: {file_name}', type='positive')


async def handle_summarize(
    max_words_input,
    max_keywords_input,
    max_tags_input,
    out_lang_input,
    processing_mode_radio,
    loader_radio,
    embedding_radio,
    provider_select,
    llm_model_select,
    status_label,
):
    global summary_label, sources_label, keywords_label, tags_label, results_container, download_button

    UI_STATE["process_completed"] = False
    set_processing_ui_locked(True)
    if status_label is not None:
        status_label.set_text('Processing...')
        status_label.update()
    if results_container and download_button:
        results_container.set_visibility(False)
        download_button.set_visibility(False)
        results_container.update()
        download_button.update()
    temp_file_path = None
    source_file_path = None

    try:
        if 'file' in uploaded_file and uploaded_file['file'] is not None:
            uploaded_file['file'].content.seek(0)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                temp_file_path = tmp_file.name
                tmp_file.write(uploaded_file['file'].content.read())
            source_file_path = temp_file_path
        elif selected_demo_file.get('path'):
            source_file_path = selected_demo_file.get('path')
            if not os.path.exists(source_file_path):
                ui.notify('Please choose a demo PDF file first.',
                          type='warning')
                if status_label is not None:
                    status_label.set_text('Ready.')
                    status_label.update()
                return
        else:
            ui.notify(
                'Please upload a PDF file or select one demo PDF first.', type='warning')
            if status_label is not None:
                status_label.set_text('Ready.')
                status_label.update()
            return

        use_vector = processing_mode_radio.value == 'vector'
        use_remote = use_vector and embedding_radio.value == 'remote'
        llm_model = llm_model_select.value
        if not llm_model:
            ui.notify(
                'Choose an LLM model (refresh the list if empty).', type='warning')
            if status_label is not None:
                status_label.set_text('Ready.')
                status_label.update()
            return

        if provider_select.value == 'ollama' and not OLLAMA_PROBE_STATE.get('ok'):
            ui.notify(
                'Local Ollama is not available. Start Ollama or click “Check local Ollama”, '
                'or pick another provider.',
                type='warning',
            )
            if status_label is not None:
                status_label.set_text('Ready.')
                status_label.update()
            return

        if provider_select.value == 'ollama_ine' and not OLLAMA_INE_PROBE_STATE.get('ok'):
            ui.notify(
                'Remote Statistics Portugal Ollama is not available from this network. '
                'Connect to the corporate network or pick another provider.',
                type='warning',
            )
            if status_label is not None:
                status_label.set_text('Ready.')
                status_label.update()
            return

        out_lang = (out_lang_input.value or '').strip() or 'pt-pt'
        proc_mode = processing_mode_radio.value
        embed_choice = embedding_radio.value if use_vector else None

        t0 = time.perf_counter()
        result = await run.io_bound(
            run_summarization,
            source_file_path,
            selected_file_info.get('name') or '',
            int(max_keywords_input.value),
            int(max_tags_input.value),
            int(max_words_input.value),
            out_lang,
            use_vector,
            loader_radio.value,
            use_remote,
            provider_select.value,
            llm_model,
            (
                current_ollama_base_url()
                if provider_select.value == 'ollama'
                else (
                    current_ollama_ine_base_url()
                    if provider_select.value == 'ollama_ine'
                    else ''
                )
            ),
        )
        elapsed_s = time.perf_counter() - t0

        timing_sec = result.get("_timing_sec") if isinstance(
            result, dict) else None
        summary_text = result.get("summary", "No summary.") if isinstance(
            result, dict) else str(result)
        summary_label.set_text(summary_text)
        sources = result.get("sources", []) if isinstance(result, dict) else []
        unmatched_numbers = (
            result.get("unmatched_numbers", []) if isinstance(result, dict) else []
        )
        numeric_claims = (
            result.get("numeric_claims", []) if isinstance(result, dict) else []
        )
        sources_html = _sources_block_html(sources)
        if isinstance(unmatched_numbers, list) and unmatched_numbers:
            escaped_nums = html.escape(", ".join(str(n) for n in unmatched_numbers))
            sources_html += (
                '<p class="text-xs text-red-500 mt-2">'
                f'Warning: numbers in the summary without a matching source excerpt: {escaped_nums}'
                '</p>'
            )
        if sources_label is not None:
            sources_label.set_content(sources_html)
        keywords = (result.get("keywords", []) if isinstance(result, dict) else [])[
            : int(max_keywords_input.value)
        ]
        tags = (result.get("tags", []) if isinstance(result, dict) else [])[
            : int(max_tags_input.value)
        ]
        keywords_label.set_text(", ".join(keywords))
        tags_label.set_text(", ".join(tags))
        LAST_SUMMARY_META["elapsed_s"] = elapsed_s
        LAST_SUMMARY_META["timing_breakdown"] = timing_sec
        LAST_SUMMARY_META["download_stem"] = build_summary_download_stem(
            max_words=int(max_words_input.value),
            max_keywords=int(max_keywords_input.value),
            max_tags=int(max_tags_input.value),
            out_lang=out_lang,
            provider=str(provider_select.value),
            processing_mode=str(proc_mode),
            embedding_choice=embed_choice,
            loader=str(loader_radio.value),
            llm_model=str(llm_model),
        )
        LAST_SUMMARY_META["export_summary"] = summary_text
        LAST_SUMMARY_META["export_sources"] = (
            list(sources) if isinstance(sources, list) else []
        )
        LAST_SUMMARY_META["export_keywords"] = list(keywords)
        LAST_SUMMARY_META["export_tags"] = list(tags)
        LAST_SUMMARY_META["export_numeric_claims"] = (
            list(numeric_claims) if isinstance(numeric_claims, list) else []
        )
        LAST_SUMMARY_META["export_unmatched_numbers"] = (
            list(unmatched_numbers) if isinstance(unmatched_numbers, list) else []
        )
        LAST_SUMMARY_META["export_ready"] = True
        if status_label is not None:
            status_label.set_text(
                format_done_status_line(elapsed_s, timing_sec))
            status_label.update()
        if results_container and download_button:
            results_container.set_visibility(True)
            download_button.set_visibility(True)
        UI_STATE["process_completed"] = True
        summary_label.update()
        if sources_label is not None:
            sources_label.update()
        keywords_label.update()
        tags_label.update()
        if results_container and download_button:
            results_container.update()
            download_button.update()

    except Exception as e:
        error_msg = str(e)
        auth_hint_ssp = "Please update `SSP_KEY` in `.env` with a valid token and restart the app."
        auth_hint_openai = "Please set a valid `OPENAI_API_KEY` in `.env`."
        if "Authentication with SSP LLM failed (401)" in error_msg or "SSP LLM returned 401" in error_msg:
            error_msg = f"{error_msg} {auth_hint_ssp}"
        elif "401" in error_msg and "OpenAI" in error_msg:
            error_msg = f"{error_msg} {auth_hint_openai}"
        elif "Ollama API is disabled" in error_msg or ("503" in error_msg and "sspcloud" in error_msg.lower()):
            error_msg = (
                "Remote embedding failed: the SSP Cloud Ollama API endpoint is disabled. "
                "Switch to Local embedding in the Summarizer configuration, or check SSP Cloud access."
            )
        LAST_SUMMARY_META["export_ready"] = False
        summary_label.set_text(f"❌ Error: {error_msg}")
        if sources_label is not None:
            sources_label.set_content('<p class="text-sm text-red-400 italic">Error</p>')
        keywords_label.set_text("Error")
        tags_label.set_text("Error")
        summary_label.update()
        if sources_label is not None:
            sources_label.update()
        keywords_label.update()
        tags_label.update()
        UI_STATE["process_completed"] = True
        if status_label is not None:
            status_label.set_text('Failed')
            status_label.update()
        ui.notify(f"Summarization error: {error_msg}", type='negative')

    finally:
        # Always unlock the UI even if we return early due to validation.
        set_processing_ui_locked(False)
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@ui.page('/')
def main_page():
    global summary_label, sources_label, keywords_label, tags_label, results_container, download_button
    global file_picker_container, selected_file_container
    global selected_file_name_label, selected_file_size_label, selected_file_source_label, selected_file_thumbnail
    global pdf_upload_component, summarize_actions_container, summarize_button, restart_button
    global clear_selection_button, change_file_hint_label, summarizer_customize_button, parameters_customize_button
    global summarizer_config_summary_panel_ref, summarizer_config_editor_panel_ref
    global parameters_summary_panel_ref, parameters_editor_panel_ref
    global file_card_container, summarizer_config_card_container, parameters_card_container

    ui.add_head_html(
        '''
<style>
  body { zoom: 100%; }
  /*
    Visible QUploader (opacity must stay > 0 or many browsers ignore clicks).
    We hide only the file queue row — filename/size belong in “File to process”.
  */
  /* Header hidden: we use a full-width Python button that calls pickFiles() instead
     (transparent overlays often break native click handling). */
  .pdf-upload-styled .q-uploader__header {
    display: none !important;
  }
  .pdf-upload-styled .q-uploader__list {
    display: none !important;
  }
  /*
    QUploader only handles drag once the pointer is over .q-uploader. A short
    strip under the Browse button meant drops on the rest of the dashed card
    never reached it — stretch the uploader to the full card via the layer
    class and pointer-events (see layout below).
  */
  .pdf-upload-drop-layer.q-uploader,
  .pdf-upload-drop-layer .q-uploader {
    width: 100% !important;
    min-height: 100% !important;
    height: 100% !important;
    box-shadow: none !important;
    background: rgba(255, 255, 255, 0.35) !important;
    border-radius: 0.875rem !important;
    border: none !important;
  }

  /* Summarizer configuration: title line vs description line in radio labels */
  .summarizer-radio-stack .q-radio {
    align-items: flex-start;
    padding: 0.6rem 0;
    min-height: unset;
  }
  .summarizer-radio-stack .q-radio:not(:last-child) {
    border-bottom: 1px solid rgb(241 245 249);
  }
  .summarizer-radio-stack .q-radio__label {
    white-space: pre-line;
    font-size: 0.8125rem;
    line-height: 1.55;
    color: rgb(100 116 139);
    letter-spacing: 0.01em;
  }
  .summarizer-radio-stack .q-radio__label::first-line {
    font-size: 0.9375rem;
    font-weight: 600;
    color: rgb(30 41 59);
    letter-spacing: -0.01em;
  }

  /* Processing button: option 2 (white pulse scanner icon). */
  .processing-spinner-matrix .q-icon {
    color: rgba(255, 255, 255, 0.95);
    animation: scanPulse 1.1s ease-in-out infinite;
    transform-origin: center center;
  }
  @keyframes scanPulse {
    0%, 100% { transform: scale(0.92); opacity: 0.66; }
    50% { transform: scale(1.06); opacity: 1; }
  }

</style>
        '''.strip()
    )

    with ui.column().classes('items-center p-10 gap-8 text-xl max-w-7xl w-full mx-auto'):

        with ui.row().classes('items-center gap-3 justify-center flex-wrap text-center'):
            ui.icon('sym_o_article').classes(
                'text-4xl text-primary shrink-0 opacity-90')
            ui.label('PDF Summarizer with LLMs — WP12 · Statistics Portugal').classes(
                'text-3xl font-medium text-slate-900 tracking-tight'
            )

        with ui.card().classes('w-full p-6 shadow-lg') as file_card_container:
            with ui.column().classes('w-full gap-4') as file_picker_container:
                with ui.row().classes('items-center gap-3 mb-4'):
                    ui.icon('sym_o_picture_as_pdf').classes(
                        'text-3xl text-primary shrink-0 opacity-90'
                    )
                    ui.label('Add a PDF').classes(
                        'text-2xl font-medium text-slate-800 tracking-tight'
                    )

                with ui.column().classes(
                    'w-full rounded-2xl border-2 border-dashed '
                    'border-slate-300 bg-gradient-to-b from-slate-50 to-white '
                    'shadow-inner hover:border-primary hover:from-blue-50/60 hover:to-white '
                    'transition-colors duration-200 overflow-hidden gap-4 p-5 relative'
                ):
                    pdf_upload_component = ui.upload(
                        on_upload=handle_upload,
                        auto_upload=True,
                    ).props(
                        'accept=".pdf" max-files=1 color=primary flat'
                    ).classes(
                        'absolute inset-0 w-full h-full pdf-upload-styled '
                        'pdf-upload-drop-layer'
                    )

                    with ui.column().classes(
                        'relative z-10 w-full gap-4 pointer-events-none'
                    ):
                        with ui.row().classes('w-full gap-4 items-center'):
                            ui.icon('sym_o_picture_as_pdf').classes(
                                'text-5xl text-red-700/85 shrink-0 hidden sm:flex'
                            )
                            with ui.column().classes('flex-1 min-w-0 gap-1'):
                                ui.label('Browse or drop a PDF').classes(
                                    'text-base font-medium text-slate-700'
                                )
                                ui.label(
                                    'Choose a file with the button, or drag a PDF anywhere '
                                    'in this dashed area.'
                                ).classes('text-sm text-gray-500')

                        ui.button(
                            'Browse PDF files',
                            icon='sym_o_folder_open',
                            on_click=lambda: pdf_upload_component.run_method(
                                'pickFiles'),
                        ).props('unelevated color=primary no-caps').classes(
                            'w-full py-3 text-lg shadow-sm pointer-events-auto'
                        )

                demo_files = list_demo_pdfs()
                if demo_files:
                    with ui.row().classes('w-full items-center gap-2 mt-1 flex-wrap'):
                        ui.label('…or use our example PDF:').classes(
                            'text-sm text-gray-600'
                        )
                        for file_name in demo_files:
                            ui.button(
                                f'📎 {file_name}',
                                on_click=lambda _, name=file_name: select_demo_pdf(
                                    name),
                            ).props('flat dense no-caps color=primary').classes('px-2 py-1')
                else:
                    ui.label('No demo PDFs found in demo_docs.').classes(
                        'text-sm text-gray-600')

            with ui.column().classes('w-full gap-4') as selected_file_container:
                ui.label('File to process').classes(
                    'text-lg font-medium text-slate-800 tracking-tight mb-1'
                )

                with ui.row().classes(
                    'w-full gap-4 items-start p-4 rounded-xl bg-slate-50 border border-slate-100'
                ):
                    selected_file_thumbnail = ui.image().classes(
                        'w-36 flex-shrink-0 rounded-lg border border-slate-200 shadow-sm bg-white object-cover'
                    )
                    selected_file_thumbnail.set_visibility(False)

                    with ui.column().classes('flex-1 min-w-0 gap-2'):
                        with ui.row().classes('w-full items-start justify-between gap-3'):
                            selected_file_name_label = ui.label('—').classes(
                                'flex-1 min-w-0 text-xl font-semibold text-gray-900 leading-snug break-words'
                            )
                            clear_selection_button = ui.button(
                                'Clear selection',
                                on_click=clear_selected_file,
                            ).props(
                                'outline dense no-caps color=primary unelevated icon=sym_o_refresh'
                            ).classes(
                                'flex-shrink-0 shadow-sm hover:shadow-md transition-shadow'
                            )
                        with ui.row().classes(
                            'w-full items-baseline justify-between gap-3 flex-wrap'
                        ):
                            with ui.row().classes('items-baseline gap-2 flex-wrap min-w-0'):
                                selected_file_size_label = ui.label('').classes(
                                    'text-sm text-gray-500 tabular-nums'
                                )
                                ui.label('·').classes(
                                    'text-sm text-gray-300 select-none')
                                selected_file_source_label = ui.label('').classes(
                                    'text-xs text-gray-400'
                                )
                            change_file_hint_label = ui.label('Change file anytime with Clear selection.').classes(
                                'text-xs text-gray-400 text-right max-w-[14rem] leading-tight flex-shrink-0'
                            )

            selected_file_container.set_visibility(False)

        with ui.card().classes('w-full p-6 shadow-lg') as summarizer_config_card_container:
            with ui.row().classes('items-center gap-3 mb-4'):
                ui.icon('sym_o_auto_awesome').classes(
                    'text-3xl text-primary shrink-0 opacity-90'
                )
                ui.label('Summarizer configuration').classes(
                    'text-2xl font-medium text-slate-800 tracking-tight'
                )

            summarizer_config_summary_panel = ui.column().classes('w-full gap-3')
            with summarizer_config_summary_panel:
                with ui.row().classes(
                    'w-full flex-wrap items-start justify-between gap-3 '
                    'rounded-xl bg-slate-50/90 px-4 py-3 border border-slate-100'
                ):
                    with ui.column().classes('flex-1 min-w-0'):
                        with ui.column().classes('gap-1'):
                            summarizer_summary_llm_provider_label = ui.label('').classes(
                                'text-sm text-slate-800 leading-snug break-words'
                            )
                            summarizer_summary_llm_model_label = ui.label('').classes(
                                'text-xs text-slate-600 leading-snug break-words '
                                'ml-3 pl-3 border-l border-slate-200'
                            )
                            summarizer_summary_mode_label = ui.label('').classes(
                                'text-sm text-slate-800 leading-snug break-words'
                            )
                            summarizer_summary_embeddings_row = ui.column().classes(
                                'gap-0 ml-3 md:ml-4 pl-3 border-l border-indigo-100'
                            )
                            with summarizer_summary_embeddings_row:
                                summarizer_summary_embeddings_label = ui.label('').classes(
                                    'text-xs text-slate-600 leading-snug break-words'
                                )
                            summarizer_summary_loader_label = ui.label('').classes(
                                'text-sm text-slate-800 leading-snug break-words'
                            )
                    summarizer_customize_button = ui.button(
                        'Customize',
                        icon='sym_o_tune',
                        on_click=lambda: open_summarizer_editor(),
                    ).props('outline dense no-caps color=primary unelevated').classes(
                        'flex-shrink-0'
                    )

            summarizer_config_editor_panel = ui.column().classes('w-full gap-6')
            summarizer_config_summary_panel_ref = summarizer_config_summary_panel
            summarizer_config_editor_panel_ref = summarizer_config_editor_panel
            with summarizer_config_editor_panel:
                ui.label(
                    'Choose the chat model, then PDF parsing and retrieval options below.'
                ).classes('text-xs sm:text-sm text-slate-500 leading-relaxed max-w-3xl')

                with ui.column().classes('w-full gap-2'):
                    ui.label('Language model').classes(
                        'text-xs font-semibold uppercase tracking-wide text-slate-500'
                    )
                    with ui.column().classes(
                        'w-full rounded-xl bg-slate-50/90 px-3 py-3 border '
                        'border-slate-100 gap-3'
                    ):
                        ui.label(
                            'SSP/OpenAI plus available Ollama runtimes (local and Statistics Portugal remote).'
                        ).classes('text-xs text-slate-500 leading-snug')

                        provider_select = ui.select(
                            PROVIDER_SELECT_OPTIONS_NO_OLLAMA,
                            value='ssp',
                            label='Provider',
                        ).classes('w-full')

                        ollama_status_label = ui.label(
                            'Checking local Ollama availability…'
                        ).classes('text-xs text-slate-600 leading-snug')
                        ollama_ine_status_label = ui.label(
                            'Checking Statistics Portugal remote Ollama availability…'
                        ).classes('text-xs text-slate-600 leading-snug')

                        _fb_ssp = DEFAULT_LLM_MODELS_FALLBACK['ssp']
                        llm_model_select = ui.select(
                            _fb_ssp,
                            value=_fb_ssp[0],
                            label='Model',
                        ).classes('w-full')

                        provider_models_cache = {
                            'ssp': list(_fb_ssp),
                            'openai': list(DEFAULT_LLM_MODELS_FALLBACK['openai']),
                            'ollama': list(DEFAULT_LLM_MODELS_FALLBACK['ollama']),
                            'ollama_ine': list(DEFAULT_LLM_MODELS_FALLBACK['ollama_ine']),
                        }

                        def _current_ollama_base_for_provider(prov: str) -> str:
                            if prov == 'ollama':
                                return current_ollama_base_url()
                            if prov == 'ollama_ine':
                                return current_ollama_ine_base_url()
                            return ''

                        def _sync_provider_options_and_model_choice():
                            apply_provider_select_options(
                                provider_select,
                                include_local_ollama=bool(
                                    OLLAMA_PROBE_STATE.get('ok')),
                                include_ine_ollama=bool(
                                    OLLAMA_INE_PROBE_STATE.get('ok')),
                            )
                            pv = provider_select.value
                            models = provider_models_cache.get(pv) or DEFAULT_LLM_MODELS_FALLBACK.get(
                                pv, DEFAULT_LLM_MODELS_FALLBACK['ssp']
                            )
                            llm_model_select.options = models
                            llm_model_select.value = models[0] if models else None
                            llm_model_select.update()

                        async def refresh_llm_models_for_provider(prov: str, *, notify: bool = True):
                            base_url = _current_ollama_base_for_provider(prov)

                            # Re-read .env so key changes take effect without a restart.
                            from dotenv import load_dotenv as _load_dotenv
                            _load_dotenv(
                                os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'),
                                override=True,
                            )

                            ak = None
                            if prov == 'openai':
                                ak = os.getenv('OPENAI_API_KEY')
                            elif prov == 'ssp':
                                ak = os.getenv('SSP_KEY')

                            try:
                                models = await run.io_bound(
                                    fetch_llm_models_for_provider,
                                    prov,
                                    api_key=ak,
                                    ollama_base_url=base_url,
                                )
                                if not models:
                                    models = DEFAULT_LLM_MODELS_FALLBACK.get(
                                        prov, DEFAULT_LLM_MODELS_FALLBACK['ssp']
                                    )
                                provider_models_cache[prov] = list(models)
                                if provider_select.value == prov:
                                    llm_model_select.options = models
                                    llm_model_select.value = models[0]
                                    llm_model_select.update()
                                if notify:
                                    ui.notify(
                                        f'Loaded {len(models)} models for {LLM_PROVIDER_LABEL.get(prov, prov)}.', type='positive')
                            except Exception as e:
                                fb = DEFAULT_LLM_MODELS_FALLBACK.get(
                                    prov, DEFAULT_LLM_MODELS_FALLBACK['ssp']
                                )
                                provider_models_cache[prov] = list(fb)
                                if provider_select.value == prov:
                                    llm_model_select.options = fb
                                    llm_model_select.value = fb[0]
                                    llm_model_select.update()
                                if notify:
                                    ui.notify(str(e), type='negative')

                        async def refresh_llm_models_clicked():
                            await refresh_llm_models_for_provider(provider_select.value, notify=True)

                        async def run_ollama_probe_ui(user_hint=None, *, notify: bool = True):
                            ollama_status_label.set_text(
                                'Checking local Ollama…')
                            ollama_status_label.classes(
                                'text-xs text-slate-600 leading-snug')

                            try:
                                result = await run.io_bound(probe_ollama_runtime, user_hint)
                            except Exception as ex:
                                result = {
                                    'ok': False,
                                    'base_url': None,
                                    'models': [],
                                    'error': str(ex),
                                }

                            OLLAMA_PROBE_STATE.clear()
                            OLLAMA_PROBE_STATE.update(result)

                            if result['ok']:
                                models = result.get('models') or []
                                n = len(models)
                                ollama_status_label.set_text(
                                    f'Ollama is running at {result["base_url"]} — {n} model(s) listed.'
                                )
                                ollama_status_label.classes(
                                    'text-xs text-emerald-800 leading-snug')
                                if models:
                                    provider_models_cache['ollama'] = list(
                                        models)
                                if notify:
                                    ui.notify(
                                        f'Local Ollama OK ({n} models).', type='positive')
                            else:
                                msg = result.get(
                                    'error') or 'Ollama not available.'
                                ollama_status_label.set_text(msg)
                                ollama_status_label.classes(
                                    'text-xs text-red-700 leading-snug')
                                provider_models_cache['ollama'] = list(
                                    DEFAULT_LLM_MODELS_FALLBACK['ollama']
                                )
                                if notify:
                                    ui.notify(
                                        'Local Ollama is not running or not reachable; option disabled.',
                                        type='warning',
                                    )

                            _sync_provider_options_and_model_choice()
                            ollama_status_label.update()
                            refresh_summarizer_summary()

                        async def run_ollama_ine_probe_ui(*, notify: bool = True):
                            ollama_ine_status_label.set_text(
                                'Checking Statistics Portugal remote Ollama…')
                            ollama_ine_status_label.classes(
                                'text-xs text-slate-600 leading-snug')
                            try:
                                result = await run.io_bound(
                                    probe_specific_ollama_runtime, 'https://ollama.ine.pt'
                                )
                            except Exception as ex:
                                result = {
                                    'ok': False,
                                    'base_url': 'https://ollama.ine.pt',
                                    'models': [],
                                    'error': str(ex),
                                }

                            OLLAMA_INE_PROBE_STATE.clear()
                            OLLAMA_INE_PROBE_STATE.update(result)

                            if result['ok']:
                                models = result.get('models') or []
                                n = len(models)
                                ollama_ine_status_label.set_text(
                                    f'Statistics Portugal Ollama is reachable at {result.get("base_url") or "https://ollama.ine.pt"} — {n} model(s) listed.'
                                )
                                ollama_ine_status_label.classes(
                                    'text-xs text-emerald-800 leading-snug')
                                if models:
                                    provider_models_cache['ollama_ine'] = list(
                                        models)
                                if notify:
                                    ui.notify(
                                        f'Statistics Portugal remote Ollama OK ({n} models).', type='positive')
                            else:
                                msg = result.get(
                                    'error') or 'Statistics Portugal remote Ollama not available.'
                                ollama_ine_status_label.set_text(msg)
                                ollama_ine_status_label.classes(
                                    'text-xs text-red-700 leading-snug')
                                provider_models_cache['ollama_ine'] = list(
                                    DEFAULT_LLM_MODELS_FALLBACK['ollama_ine']
                                )
                                if notify:
                                    ui.notify(
                                        'Statistics Portugal remote Ollama not reachable from this network; option disabled.',
                                        type='warning',
                                    )

                            _sync_provider_options_and_model_choice()
                            ollama_ine_status_label.update()
                            refresh_summarizer_summary()

                        async def refresh_all_llm_availability_and_models():
                            await run_ollama_probe_ui(notify=False)
                            await run_ollama_ine_probe_ui(notify=False)
                            await refresh_llm_models_for_provider('ssp', notify=False)
                            await refresh_llm_models_for_provider('openai', notify=False)
                            if OLLAMA_PROBE_STATE.get('ok'):
                                await refresh_llm_models_for_provider('ollama', notify=False)
                            if OLLAMA_INE_PROBE_STATE.get('ok'):
                                await refresh_llm_models_for_provider('ollama_ine', notify=False)
                            _sync_provider_options_and_model_choice()
                            refresh_summarizer_summary()

                        with ui.row().classes('w-full gap-2 flex-wrap items-center'):
                            ui.button(
                                'Check local Ollama',
                                icon='sym_o_network_ping',
                                on_click=lambda: asyncio.create_task(
                                    run_ollama_probe_ui()),
                            ).props('outline dense no-caps color=primary').classes('shadow-sm')
                            ui.button(
                                'Check Statistics Portugal Ollama',
                                icon='sym_o_network_ping',
                                on_click=lambda: asyncio.create_task(
                                    run_ollama_ine_probe_ui()),
                            ).props('outline dense no-caps color=primary').classes('shadow-sm')
                            ui.button(
                                'Refresh models',
                                icon='sym_o_refresh',
                                on_click=refresh_llm_models_clicked,
                            ).props('outline dense no-caps color=primary').classes('shadow-sm')
                            ui.label(
                                '“Check” probes /api/version and /api/tags. “Refresh models” '
                                'uses the currently selected provider.'
                            ).classes('text-xs text-slate-500 flex-1 min-w-[12rem]')

                        ui.timer(
                            0.35,
                            lambda: asyncio.create_task(
                                refresh_all_llm_availability_and_models()),
                            once=True,
                            immediate=False,
                        )

                with ui.column().classes('w-full gap-2'):
                    ui.label('Summarization mode').classes(
                        'text-xs font-semibold uppercase tracking-wide text-slate-500'
                    )
                    with ui.column().classes(
                        'summarizer-radio-stack w-full rounded-xl bg-slate-50/90 px-3 border '
                        'border-slate-100'
                    ):
                        processing_mode = ui.radio(
                            {
                                'plain': (
                                    'Plain text\n'
                                    'Summarize from full extracted text in one pass (no ChromaDB '
                                    'retrieval step).'
                                ),
                                'vector': (
                                    'Vector retrieval\n'
                                    'Chunk the document, embed and index in ChromaDB, retrieve '
                                    'relevant passages, then summarize. Best for longer PDFs.'
                                ),
                            },
                            value='vector',
                        ).props('vertical')

                    embedding_section = ui.column().classes(
                        'w-full gap-2 ml-3 md:ml-5 pl-4 border-l-2 border-indigo-200/90'
                    )
                    with embedding_section:
                        ui.label('Embeddings (vector retrieval)').classes(
                            'text-xs font-semibold uppercase tracking-wide text-indigo-900/80'
                        )
                        ui.label('Chunks are embedded and stored in ChromaDB.').classes(
                            'text-xs text-slate-500 leading-snug -mt-1 mb-1'
                        )
                        with ui.column().classes(
                            'summarizer-radio-stack w-full rounded-xl bg-indigo-50/50 px-3 py-1 '
                            'border border-indigo-100/90'
                        ):
                            embedding_radio = ui.radio(
                                {
                                    'local': (
                                        'Local embedding\n'
                                        'BAAI/bge-m3 via Hugging Face on this machine.'
                                    ),
                                    'remote': (
                                        'Remote embedding\n'
                                        'qwen3-embedding-8b via the SSP embedding endpoint.'
                                    ),
                                },
                                value='local',
                            ).props('vertical')

                    def sync_embedding_visibility():
                        embedding_section.set_visibility(
                            processing_mode.value == 'vector')
                        embedding_section.update()

                    embedding_section.set_visibility(True)
                    sync_embedding_visibility()

                with ui.column().classes('w-full gap-2'):
                    ui.label('PDF loader').classes(
                        'text-xs font-semibold uppercase tracking-wide text-slate-500'
                    )
                    with ui.column().classes(
                        'summarizer-radio-stack w-full rounded-xl bg-slate-50/90 px-3 border '
                        'border-slate-100'
                    ):
                        loader_radio = ui.radio(
                            {
                                'docling': (
                                    'Docling\n'
                                    'Structured, layout-aware extraction. Vector mode uses '
                                    "Docling's hybrid chunking."
                                ),
                                'pypdf': (
                                    'PyPDF\n'
                                    'Lightweight page-level text. In vector mode, chunks use '
                                    'RecursiveCharacterTextSplitter.'
                                ),
                            },
                            value='docling',
                        ).props('vertical')

                def refresh_summarizer_summary():
                    pv = provider_select.value
                    summarizer_summary_llm_provider_label.set_text(
                        f'LLM: {LLM_PROVIDER_LABEL.get(pv, pv)}'
                    )
                    mv = llm_model_select.value
                    summarizer_summary_llm_model_label.set_text(
                        f'Model: {mv}' if mv else 'Model: —'
                    )
                    summarizer_summary_llm_provider_label.update()
                    summarizer_summary_llm_model_label.update()
                    mode = processing_mode.value
                    loader = loader_radio.value
                    mode_txt = 'Plain text' if mode == 'plain' else 'Vector retrieval'
                    loader_txt = 'Docling' if loader == 'docling' else 'PyPDF'
                    summarizer_summary_mode_label.set_text(f'Mode: {mode_txt}')
                    summarizer_summary_loader_label.set_text(
                        f'Loader: {loader_txt}')
                    summarizer_summary_mode_label.update()
                    summarizer_summary_loader_label.update()
                    if mode == 'vector':
                        emb_txt = (
                            'Local (BAAI/bge-m3)'
                            if embedding_radio.value == 'local'
                            else 'Remote (qwen3-embedding-8b)'
                        )
                        summarizer_summary_embeddings_label.set_text(
                            f'Embeddings: {emb_txt}')
                        summarizer_summary_embeddings_label.update()
                        summarizer_summary_embeddings_row.set_visibility(True)
                    else:
                        summarizer_summary_embeddings_row.set_visibility(False)
                    summarizer_summary_embeddings_row.update()

                def _sync_mode_and_summary():
                    sync_embedding_visibility()
                    refresh_summarizer_summary()

                processing_mode.on_value_change(
                    lambda _: (_sync_mode_and_summary(), invalidate_result()))

                def _apply_llm_fallback_models():
                    prov = provider_select.value
                    probe_state = (
                        OLLAMA_PROBE_STATE if prov == 'ollama' else
                        OLLAMA_INE_PROBE_STATE if prov == 'ollama_ine' else
                        None
                    )
                    if probe_state and probe_state.get('ok'):
                        models = probe_state.get('models') or list(DEFAULT_LLM_MODELS_FALLBACK[prov])
                    else:
                        models = list(DEFAULT_LLM_MODELS_FALLBACK.get(prov, DEFAULT_LLM_MODELS_FALLBACK['ssp']))
                    llm_model_select.options = models
                    llm_model_select.value = models[0] if models else None
                    llm_model_select.update()

                def _on_llm_provider_changed():
                    _apply_llm_fallback_models()
                    refresh_summarizer_summary()
                    invalidate_result()

                provider_select.on_value_change(
                    lambda _: _on_llm_provider_changed())

                llm_model_select.on_value_change(
                    lambda _: (refresh_summarizer_summary(), invalidate_result()))

                loader_radio.on_value_change(
                    lambda _: (refresh_summarizer_summary(), invalidate_result()))

                embedding_radio.on_value_change(
                    lambda _: (refresh_summarizer_summary(), invalidate_result()))

                def open_summarizer_editor():
                    summarizer_config_summary_panel.set_visibility(False)
                    summarizer_config_editor_panel.set_visibility(True)
                    summarizer_config_summary_panel.update()
                    summarizer_config_editor_panel.update()
                    UI_STATE["summarizer_editor_open"] = True
                    update_summarize_actions_visibility()

                def close_summarizer_editor():
                    refresh_summarizer_summary()
                    summarizer_config_editor_panel.set_visibility(False)
                    summarizer_config_summary_panel.set_visibility(True)
                    summarizer_config_editor_panel.update()
                    summarizer_config_summary_panel.update()
                    UI_STATE["summarizer_editor_open"] = False
                    update_summarize_actions_visibility()

                ui.separator().classes('w-full opacity-60')
                ui.button(
                    'Done',
                    icon='sym_o_expand_less',
                    on_click=lambda: close_summarizer_editor(),
                ).props('flat dense no-caps color=primary').classes('self-start')

            summarizer_config_editor_panel.set_visibility(False)
            refresh_summarizer_summary()

        with ui.card().classes('w-full p-6 shadow-lg') as parameters_card_container:
            with ui.row().classes('items-center gap-3 mb-4'):
                ui.icon('sym_o_tune').classes(
                    'text-3xl text-primary shrink-0 opacity-90'
                )
                ui.label('Parameters').classes(
                    'text-2xl font-medium text-slate-800 tracking-tight'
                )

            parameters_summary_panel = ui.column().classes('w-full gap-3')
            with parameters_summary_panel:
                with ui.row().classes(
                    'w-full flex-wrap items-start justify-between gap-3 '
                    'rounded-xl bg-slate-50/90 px-4 py-3 border border-slate-100'
                ):
                    with ui.column().classes('flex-1 min-w-0'):
                        _pill_val = (
                            'inline-flex items-center max-w-full px-2.5 py-1 rounded-full '
                            'bg-white border border-slate-200/90 shadow-sm '
                            'text-sm font-medium text-slate-900 tabular-nums tracking-tight '
                            'break-all'
                        )
                        _pill_cap = (
                            'text-[10px] font-medium uppercase tracking-wide text-slate-400'
                        )
                        with ui.row().classes(
                            'w-full flex-wrap gap-x-3 gap-y-2 items-start'
                        ):
                            with ui.column().classes('gap-1 shrink-0'):
                                ui.label('Max words').classes(_pill_cap)
                                parameters_pill_max_words = ui.label(
                                    '').classes(_pill_val)
                            with ui.column().classes('gap-1 shrink-0'):
                                ui.label('Keywords').classes(_pill_cap)
                                parameters_pill_keywords = ui.label(
                                    '').classes(_pill_val)
                            with ui.column().classes('gap-1 shrink-0'):
                                ui.label('Tags').classes(_pill_cap)
                                parameters_pill_tags = ui.label(
                                    '').classes(_pill_val)
                            with ui.column().classes('gap-1 min-w-0 flex-1 max-w-full'):
                                ui.label('Language').classes(_pill_cap)
                                parameters_pill_lang = ui.label('').classes(
                                    _pill_val + ' font-mono'
                                )
                    parameters_customize_button = ui.button(
                        'Customize',
                        icon='sym_o_tune',
                        on_click=lambda: open_parameters_editor(),
                    ).props('outline dense no-caps color=primary unelevated').classes(
                        'flex-shrink-0'
                    )

            parameters_editor_panel = ui.column().classes('w-full gap-4')
            parameters_summary_panel_ref = parameters_summary_panel
            parameters_editor_panel_ref = parameters_editor_panel
            with parameters_editor_panel:
                ui.label(
                    'Set limits for summary length, keyword/tag counts, and output language. '
                    'Use any BCP 47 / locale-style code the model should target (examples: '
                    'pt-pt, en, fr, ja).'
                ).classes('text-xs sm:text-sm text-slate-500 leading-relaxed')

                with ui.element('div').classes(
                    'grid grid-cols-2 lg:grid-cols-4 gap-4 w-full items-start'
                ):
                    max_words_input = ui.number(
                        'Max Words', value=200, min=50, max=1000
                    ).classes('w-full min-w-0')
                    max_keywords_input = ui.number(
                        'Max Keywords', value=6, min=1, max=20
                    ).classes('w-full min-w-0')
                    max_tags_input = ui.number(
                        'Max Tags', value=5, min=1, max=15
                    ).classes('w-full min-w-0')
                    out_lang_input = ui.input(
                        label='Output language',
                        value='pt-pt',
                        placeholder='Language code (e.g. pt-pt, en, fr, de-DE, zh-Hans…)',
                    ).classes('w-full min-w-0').props('clearable')

                ui.separator().classes('w-full opacity-60')
                ui.button(
                    'Done',
                    icon='sym_o_expand_less',
                    on_click=lambda: close_parameters_editor(),
                ).props('flat dense no-caps color=primary').classes('self-start')

            def refresh_parameters_summary():
                try:
                    mw = int(max_words_input.value)
                except (TypeError, ValueError):
                    mw = 200
                try:
                    mk = int(max_keywords_input.value)
                except (TypeError, ValueError):
                    mk = 6
                try:
                    mt = int(max_tags_input.value)
                except (TypeError, ValueError):
                    mt = 5
                lang_code = (out_lang_input.value or '').strip() or 'pt-pt'
                parameters_pill_max_words.set_text(str(mw))
                parameters_pill_keywords.set_text(str(mk))
                parameters_pill_tags.set_text(str(mt))
                parameters_pill_lang.set_text(lang_code)
                parameters_pill_max_words.update()
                parameters_pill_keywords.update()
                parameters_pill_tags.update()
                parameters_pill_lang.update()

            def open_parameters_editor():
                parameters_summary_panel.set_visibility(False)
                parameters_editor_panel.set_visibility(True)
                parameters_summary_panel.update()
                parameters_editor_panel.update()
                UI_STATE["parameters_editor_open"] = True
                update_summarize_actions_visibility()

            def close_parameters_editor():
                refresh_parameters_summary()
                parameters_editor_panel.set_visibility(False)
                parameters_summary_panel.set_visibility(True)
                parameters_editor_panel.update()
                parameters_summary_panel.update()
                UI_STATE["parameters_editor_open"] = False
                update_summarize_actions_visibility()

            parameters_editor_panel.set_visibility(False)
            refresh_parameters_summary()

            max_words_input.on_value_change(
                lambda _: (refresh_parameters_summary(), invalidate_result()))
            max_keywords_input.on_value_change(
                lambda _: (refresh_parameters_summary(), invalidate_result()))
            max_tags_input.on_value_change(
                lambda _: (refresh_parameters_summary(), invalidate_result()))
            out_lang_input.on_value_change(
                lambda _: (refresh_parameters_summary(), invalidate_result()))

        with ui.column().classes('w-full gap-2') as summarize_actions_container:
            status_label = ui.label('Ready.').classes(
                'text-lg text-gray-600 whitespace-pre-line leading-snug'
            )

            def restart_app_state():
                UI_STATE["process_completed"] = False
                UI_STATE["processing"] = False
                UI_STATE["summarizer_editor_open"] = False
                UI_STATE["parameters_editor_open"] = False

                # Reset selection + results first.
                clear_selected_file()
                if results_container is not None:
                    results_container.set_visibility(False)
                    results_container.update()
                if download_button is not None:
                    download_button.set_visibility(False)
                    download_button.update()
                if summary_label is not None:
                    summary_label.set_text('Summary will appear here.')
                    summary_label.update()
                if sources_label is not None:
                    sources_label.set_content(
                        '<p class="text-sm text-gray-400 italic">Information sources will appear here.</p>')
                    sources_label.update()
                if keywords_label is not None:
                    keywords_label.set_text('Keywords will appear here.')
                    keywords_label.update()
                if tags_label is not None:
                    tags_label.set_text('Tags will appear here.')
                    tags_label.update()
                LAST_SUMMARY_META["export_ready"] = False
                status_label.set_text('Ready.')
                status_label.update()

                # Reset editors to "summary" mode.
                summarizer_config_editor_panel.set_visibility(False)
                summarizer_config_summary_panel.set_visibility(True)
                summarizer_config_editor_panel.update()
                summarizer_config_summary_panel.update()
                refresh_summarizer_summary()

                parameters_editor_panel.set_visibility(False)
                parameters_summary_panel.set_visibility(True)
                parameters_editor_panel.update()
                parameters_summary_panel.update()
                refresh_parameters_summary()

                set_processing_ui_locked(False)
                UI_STATE["file_picker_open"] = True
                update_summarize_actions_visibility()

            restart_button = ui.button(
                'Restart',
                on_click=restart_app_state,
            ).props('outline icon=sym_o_restart_alt').classes('w-full text-lg py-2 rounded')
            restart_button.set_visibility(False)

            summarize_button = ui.button('Summarize', on_click=lambda: handle_summarize(
                max_words_input,
                max_keywords_input,
                max_tags_input,
                out_lang_input,
                processing_mode,
                loader_radio,
                embedding_radio,
                provider_select,
                llm_model_select,
                status_label,
            )).props('icon=sym_o_document_scanner').classes('w-full bg-blue-600 text-white text-lg py-2 rounded hover:bg-blue-700')

        update_summarize_actions_visibility()

        ui.separator().classes('w-full my-4')

        with ui.column().classes('w-full gap-4') as results_container:
            with ui.card().classes('w-full p-6 bg-slate-50/90 border border-slate-100'):
                with ui.row().classes('items-center gap-2'):
                    ui.icon('sym_o_summarize').classes(
                        'text-2xl text-primary shrink-0 opacity-90')
                    ui.label('Summary').classes('text-xl font-medium')
                summary_label = ui.label('Summary will appear here.').classes(
                    'w-full whitespace-pre-wrap')

            with ui.card().classes('w-full p-6 bg-slate-50/90 border border-slate-100'):
                with ui.row().classes('items-center gap-2'):
                    ui.icon('sym_o_format_quote').classes(
                        'text-2xl text-primary shrink-0 opacity-90')
                    ui.label('Information sources').classes(
                        'text-xl font-medium')
                sources_label = ui.html('<p class="text-sm text-gray-400 italic">Information sources will appear here.</p>').classes(
                    'w-full'
                )

            with ui.card().classes('w-full p-6 bg-slate-50/90 border border-slate-100'):
                with ui.row().classes('items-center gap-2'):
                    ui.icon('sym_o_key').classes(
                        'text-2xl text-primary shrink-0 opacity-90')
                    ui.label('Keywords').classes('text-xl font-medium')
                keywords_label = ui.label(
                    'Keywords will appear here.').classes('w-full')

            with ui.card().classes('w-full p-6 bg-slate-50/90 border border-slate-100'):
                with ui.row().classes('items-center gap-2'):
                    ui.icon('sym_o_sell').classes(
                        'text-2xl text-primary shrink-0 opacity-90')
                    ui.label('Tags').classes('text-xl font-medium')
                tags_label = ui.label(
                    'Tags will appear here.').classes('w-full')

            with ui.card().classes('w-full p-4 bg-white border border-slate-100 shadow-sm'):
                with ui.row().classes('w-full items-center gap-3 flex-wrap'):
                    ui.icon('sym_o_download').classes(
                        'text-2xl text-primary shrink-0 opacity-90'
                    )
                    with ui.column().classes('gap-1 flex-1 min-w-[12rem]'):
                        ui.label('Download export').classes(
                            'text-base font-medium text-slate-800'
                        )
                        ui.label(
                            'Plain text, Markdown, HTML, or JSON — same sections as above.'
                        ).classes('text-xs text-slate-500 leading-snug')
                    export_format_select = ui.select(
                        {
                            'plaintext': 'Plain text (.txt)',
                            'markdown': 'Markdown (.md)',
                            'html': 'HTML (.html)',
                            'json': 'JSON (.json)',
                        },
                        value='plaintext',
                    ).props('dense outlined').classes(
                        'w-full sm:w-auto min-w-[13rem]'
                    )

        def download_summary_export():
            if not LAST_SUMMARY_META.get('export_ready'):
                ui.notify(
                    'Run summarization successfully before downloading.',
                    type='warning',
                )
                return
            fmt = export_format_select.value or 'plaintext'
            ext = _EXPORT_FORMAT_EXT.get(fmt, '.txt')
            stem = LAST_SUMMARY_META.get('download_stem') or 'summary_full'
            filename = f'{stem}{ext}'
            body = build_download_export_content(fmt)
            ui.download.content(body, filename)

        download_button = ui.button(
            'Download export',
            on_click=download_summary_export,
        ).props('color=primary icon=sym_o_download').classes('w-full')

        results_container.set_visibility(False)
        download_button.set_visibility(False)


if __name__ in {'__main__', '__mp_main__'}:
    root_path = os.getenv('NICEGUI_ROOT_PATH', '').strip()
    run_kwargs = {
        'host': '0.0.0.0',
        'port': 5001,
        'reload': True,  # reload the app when the code changes
        'title': 'PDF Summarizer',
    }
    if root_path:
        run_kwargs['root_path'] = root_path
    ui.run(**run_kwargs)
