import os
import re
import asyncio
import tempfile
import base64
import time
from nicegui import ui, app, run
from starlette.datastructures import UploadFile
from summarizer_unified import (
    PDFSummarizer,
    DEFAULT_LLM_MODELS_FALLBACK,
    fetch_llm_models_for_provider,
    probe_ollama_runtime,
)
from nicegui.events import UploadEventArguments

uploaded_file = {}
selected_demo_file = {'path': None}
selected_file_info = {'name': None, 'size_bytes': None, 'source': None}
summary_label = None
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
summary_text = ""

# Last successful run: download filename pattern + timing for status display.
LAST_SUMMARY_META = {
    "download_filename": "summary_full.txt",
    "elapsed_s": None,
    "timing_breakdown": None,
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
    line1 = f"✅ Done · {total_s:.1f}s total"
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


def build_summary_download_filename(
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
    """Dense ASCII stem reflecting extraction params + summarizer configuration."""
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
    return f"{stem}.txt"


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
    summarize_actions_container.set_visibility(has_pdf_source_selected())
    summarize_actions_container.update()


LLM_PROVIDER_LABEL = {
    "ssp": "Remote Ollama (SSPCloud)",
    "ollama": "Local Ollama",
    "openai": "Remote OpenAI",
}

# Updated by probe_ollama_runtime (worker thread / cpu_bound).
OLLAMA_PROBE_STATE: dict = {
    "ok": False,
    "base_url": None,
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
}


def apply_provider_select_options(select_el, *, include_ollama: bool) -> None:
    """
    Refresh provider dropdown options so 'Local Ollama' appears after a successful probe.

    Prefer ChoiceElement.set_options() so value indices stay aligned with the client (#1073).
    """
    opts = dict(
        PROVIDER_SELECT_OPTIONS_WITH_OLLAMA
        if include_ollama
        else PROVIDER_SELECT_OPTIONS_NO_OLLAMA
    )
    prev = select_el.value
    if not include_ollama and prev == 'ollama':
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


def cpu_bound_probe_ollama(preferred_base_url):
    """
    Top-level wrapper for NiceGUI ``run.cpu_bound`` (nested closures are not picklable).
    """
    return probe_ollama_runtime(preferred_base_url)


def cpu_bound_fetch_llm_models(prov: str, api_key, ollama_base_url: str):
    """Top-level wrapper for NiceGUI ``run.cpu_bound``."""
    return fetch_llm_models_for_provider(
        prov,
        api_key=api_key,
        ollama_base_url=ollama_base_url,
    )


def run_summarization(
    file_path,
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
    if not (file_picker_container and selected_file_container and selected_file_name_label):
        return
    name = selected_file_info.get('name', 'Unknown file')
    source = selected_file_info.get('source', '')
    size_text = format_size(selected_file_info.get('size_bytes'))
    source_text = SOURCE_DISPLAY.get(source, source or '—')

    selected_file_name_label.set_text(name)
    if selected_file_size_label:
        selected_file_size_label.set_text(size_text)
        selected_file_size_label.update()
    if selected_file_source_label:
        selected_file_source_label.set_text(source_text)
        selected_file_source_label.update()

    selected_file_name_label.update()
    file_picker_container.set_visibility(False)
    selected_file_container.set_visibility(True)
    file_picker_container.update()
    selected_file_container.update()
    update_summarize_actions_visibility()


def clear_selected_file():
    uploaded_file['file'] = None
    selected_demo_file['path'] = None
    selected_file_info['name'] = None
    selected_file_info['size_bytes'] = None
    selected_file_info['source'] = None
    if selected_file_thumbnail:
        selected_file_thumbnail.set_visibility(False)
        selected_file_thumbnail.update()
    if file_picker_container and selected_file_container:
        selected_file_container.set_visibility(False)
        file_picker_container.set_visibility(True)
        selected_file_container.update()
        file_picker_container.update()
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
    if selected_file_thumbnail:
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
    selected_file_info['size_bytes'] = os.path.getsize(file_path) if os.path.exists(file_path) else None
    selected_file_info['source'] = 'Demo file'
    thumbnail = build_pdf_thumbnail_data_uri(file_path=file_path)
    if selected_file_thumbnail:
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
    global summary_label, keywords_label, tags_label, results_container, download_button, summary_text

    status_label.set_text('⏳ Processing...')
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
                ui.notify('Please choose a demo PDF file first.', type='warning')
                status_label.set_text('Ready.')
                status_label.update()
                return
        else:
            ui.notify('Please upload a PDF file or select one demo PDF first.', type='warning')
            status_label.set_text('Ready.')
            status_label.update()
            return

        use_vector = processing_mode_radio.value == 'vector'
        use_remote = use_vector and embedding_radio.value == 'remote'
        llm_model = llm_model_select.value
        if not llm_model:
            ui.notify('Choose an LLM model (refresh the list if empty).', type='warning')
            status_label.set_text('Ready.')
            status_label.update()
            return

        if provider_select.value == 'ollama' and not OLLAMA_PROBE_STATE.get('ok'):
            ui.notify(
                'Local Ollama is not available. Start Ollama or click “Check local Ollama”, '
                'or pick another provider.',
                type='warning',
            )
            status_label.set_text('Ready.')
            status_label.update()
            return

        out_lang = (out_lang_input.value or '').strip() or 'pt-pt'
        proc_mode = processing_mode_radio.value
        embed_choice = embedding_radio.value if use_vector else None

        t0 = time.perf_counter()
        result = await run.cpu_bound(
            run_summarization,
            source_file_path,
            int(max_keywords_input.value),
            int(max_tags_input.value),
            int(max_words_input.value),
            out_lang,
            use_vector,
            loader_radio.value,
            use_remote,
            provider_select.value,
            llm_model,
            current_ollama_base_url() if provider_select.value == 'ollama' else '',
        )
        elapsed_s = time.perf_counter() - t0

        timing_sec = result.get("_timing_sec") if isinstance(result, dict) else None
        summary_text = result.get("summary", "No summary.") if isinstance(result, dict) else str(result)
        summary_label.set_text(summary_text)
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
        LAST_SUMMARY_META["download_filename"] = build_summary_download_filename(
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
        status_label.set_text(format_done_status_line(elapsed_s, timing_sec))
        if results_container and download_button:
            results_container.set_visibility(True)
            download_button.set_visibility(True)
        summary_label.update()
        keywords_label.update()
        tags_label.update()
        status_label.update()
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
        summary_label.set_text(f"❌ Error: {error_msg}")
        keywords_label.set_text("Error")
        tags_label.set_text("Error")
        status_label.set_text('❌ Failed')
        summary_label.update()
        keywords_label.update()
        tags_label.update()
        status_label.update()
        ui.notify(f"Summarization error: {error_msg}", type='negative')

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@ui.page('/')
def main_page():
    global summary_label, keywords_label, tags_label, results_container, download_button
    global file_picker_container, selected_file_container
    global selected_file_name_label, selected_file_size_label, selected_file_source_label, selected_file_thumbnail
    global pdf_upload_component, summarize_actions_container

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
  .pdf-upload-styled .q-uploader {
    width: 100% !important;
    min-height: 112px !important;
    box-shadow: none !important;
    background: rgba(255, 255, 255, 0.35) !important;
    border-radius: 0.875rem !important;
    border: 1px dashed rgba(148, 163, 184, 0.9) !important;
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
</style>
        '''.strip()
    )

    with ui.column().classes('items-center p-10 gap-8 text-xl max-w-7xl w-full mx-auto'):

        with ui.row().classes('items-center gap-3 justify-center flex-wrap text-center'):
            ui.icon('sym_o_article').classes('text-4xl text-primary shrink-0 opacity-90')
            ui.label('PDF Summarizer with LLMs — WP12 · Statistics Portugal').classes(
                'text-3xl font-medium text-slate-900 tracking-tight'
            )

        with ui.card().classes('w-full p-6 shadow-lg'):
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
                    'transition-colors duration-200 overflow-hidden gap-4 p-5'
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
                                'Choose a file with the button, or drag a PDF into the '
                                'dashed area below.'
                            ).classes('text-sm text-gray-500')

                    ui.button(
                        'Browse PDF files',
                        icon='sym_o_folder_open',
                        on_click=lambda: pdf_upload_component.run_method('pickFiles'),
                    ).props('unelevated color=primary no-caps').classes(
                        'w-full py-3 text-lg shadow-sm'
                    )

                    pdf_upload_component = ui.upload(
                        on_upload=handle_upload,
                        auto_upload=True,
                    ).props(
                        'accept=".pdf" max-files=1 color=primary flat'
                    ).classes('w-full pdf-upload-styled')

                demo_files = list_demo_pdfs()
                if demo_files:
                    with ui.row().classes('w-full items-center gap-2 mt-1 flex-wrap'):
                        ui.label('…or use our example PDF:').classes(
                            'text-sm text-gray-600'
                        )
                        for file_name in demo_files:
                            ui.button(
                                f'📎 {file_name}',
                                on_click=lambda _, name=file_name: select_demo_pdf(name),
                            ).props('flat dense no-caps color=primary').classes('px-2 py-1')
                else:
                    ui.label('No demo PDFs found in demo_docs.').classes('text-sm text-gray-600')

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
                            ui.button(
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
                                ui.label('·').classes('text-sm text-gray-300 select-none')
                                selected_file_source_label = ui.label('').classes(
                                    'text-xs text-gray-400'
                                )
                            ui.label('Change file anytime with Clear selection.').classes(
                                'text-xs text-gray-400 text-right max-w-[14rem] leading-tight flex-shrink-0'
                            )

            selected_file_container.set_visibility(False)

        with ui.card().classes('w-full p-6 shadow-lg'):
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
                    ui.button(
                        'Customize',
                        icon='sym_o_tune',
                        on_click=lambda: open_summarizer_editor(),
                    ).props('outline dense no-caps color=primary unelevated').classes(
                        'flex-shrink-0'
                    )

            summarizer_config_editor_panel = ui.column().classes('w-full gap-6')
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
                            'Remote SSP and OpenAI are always listed. Local Ollama is added only '
                            'when available.'
                        ).classes('text-xs text-slate-500 leading-snug')

                        provider_select = ui.select(
                            PROVIDER_SELECT_OPTIONS_NO_OLLAMA,
                            value='ssp',
                            label='Provider',
                        ).classes('w-full')

                        ollama_status_label = ui.label(
                            'Checking whether local Ollama is running…'
                        ).classes('text-xs text-slate-600 leading-snug')

                        _fb_ssp = DEFAULT_LLM_MODELS_FALLBACK['ssp']
                        llm_model_select = ui.select(
                            _fb_ssp,
                            value=_fb_ssp[0],
                            label='Model',
                        ).classes('w-full')

                        async def refresh_llm_models_clicked():
                            prov = provider_select.value
                            base_url = (
                                current_ollama_base_url() if prov == 'ollama' else ''
                            )

                            ak = None
                            if prov == 'openai':
                                ak = os.getenv('OPENAI_API_KEY')
                            elif prov == 'ssp':
                                ak = os.getenv('SSP_KEY')

                            try:
                                models = await run.cpu_bound(
                                    cpu_bound_fetch_llm_models,
                                    prov,
                                    ak,
                                    base_url,
                                )
                                if not models:
                                    models = DEFAULT_LLM_MODELS_FALLBACK.get(
                                        prov, DEFAULT_LLM_MODELS_FALLBACK['ssp']
                                    )
                                llm_model_select.options = models
                                llm_model_select.value = models[0]
                                llm_model_select.update()
                                ui.notify(f'Loaded {len(models)} models.', type='positive')
                            except Exception as e:
                                ui.notify(str(e), type='negative')
                                fb = DEFAULT_LLM_MODELS_FALLBACK.get(
                                    prov, DEFAULT_LLM_MODELS_FALLBACK['ssp']
                                )
                                llm_model_select.options = fb
                                llm_model_select.value = fb[0]
                                llm_model_select.update()

                        async def run_ollama_probe_ui(user_hint=None):
                            ollama_status_label.set_text('Checking local Ollama…')
                            ollama_status_label.classes('text-xs text-slate-600 leading-snug')

                            hint = user_hint

                            try:
                                result = await run.cpu_bound(cpu_bound_probe_ollama, hint)
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
                                apply_provider_select_options(provider_select, include_ollama=True)
                                models = result.get('models') or []
                                n = len(models)
                                ollama_status_label.set_text(
                                    f'Ollama is running at {result["base_url"]} — {n} model(s) listed.'
                                )
                                ollama_status_label.classes('text-xs text-emerald-800 leading-snug')
                                if provider_select.value == 'ollama' and models:
                                    llm_model_select.options = models
                                    llm_model_select.value = models[0]
                                    llm_model_select.update()
                                ui.notify(f'Local Ollama OK ({n} models).', type='positive')
                            else:
                                apply_provider_select_options(provider_select, include_ollama=False)
                                msg = result.get('error') or 'Ollama not available.'
                                ollama_status_label.set_text(msg)
                                ollama_status_label.classes('text-xs text-red-700 leading-snug')
                                ui.notify(
                                    'Local Ollama is not running or not reachable; option disabled.',
                                    type='warning',
                                )

                            ollama_status_label.update()
                            refresh_summarizer_summary()

                        with ui.row().classes('w-full gap-2 flex-wrap items-center'):
                            ui.button(
                                'Check local Ollama',
                                icon='sym_o_network_ping',
                                on_click=lambda: asyncio.create_task(run_ollama_probe_ui()),
                            ).props('outline dense no-caps color=primary').classes('shadow-sm')
                            ui.button(
                                'Refresh models',
                                icon='sym_o_refresh',
                                on_click=refresh_llm_models_clicked,
                            ).props('outline dense no-caps color=primary').classes('shadow-sm')
                            ui.label(
                                '“Check” probes /api/version and /api/tags and discovers the base URL. '
                                '“Refresh models” uses the current provider (SSP / OpenAI / Ollama).'
                            ).classes('text-xs text-slate-500 flex-1 min-w-[12rem]')

                        ui.timer(
                            0.35,
                            lambda: asyncio.create_task(run_ollama_probe_ui()),
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
                            value='plain',
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
                                        'qwen3-embedding:8b via the SSP Ollama embedding endpoint.'
                                    ),
                                },
                                value='local',
                            ).props('vertical')

                    def sync_embedding_visibility():
                        embedding_section.set_visibility(processing_mode.value == 'vector')
                        embedding_section.update()

                    embedding_section.set_visibility(False)
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
                    summarizer_summary_loader_label.set_text(f'Loader: {loader_txt}')
                    summarizer_summary_mode_label.update()
                    summarizer_summary_loader_label.update()
                    if mode == 'vector':
                        emb_txt = (
                            'Local (BAAI/bge-m3)'
                            if embedding_radio.value == 'local'
                            else 'Remote (qwen3-embedding:8b)'
                        )
                        summarizer_summary_embeddings_label.set_text(f'Embeddings: {emb_txt}')
                        summarizer_summary_embeddings_label.update()
                        summarizer_summary_embeddings_row.set_visibility(True)
                    else:
                        summarizer_summary_embeddings_row.set_visibility(False)
                    summarizer_summary_embeddings_row.update()

                def _sync_mode_and_summary():
                    sync_embedding_visibility()
                    refresh_summarizer_summary()

                processing_mode.on_value_change(lambda _: _sync_mode_and_summary())

                def _apply_llm_fallback_models():
                    prov = provider_select.value
                    if prov == 'ollama' and OLLAMA_PROBE_STATE.get('ok'):
                        models = OLLAMA_PROBE_STATE.get('models') or []
                        if not models:
                            models = list(DEFAULT_LLM_MODELS_FALLBACK['ollama'])
                        llm_model_select.options = models
                        llm_model_select.value = models[0]
                        llm_model_select.update()
                        return
                    if prov == 'ollama':
                        fb = list(DEFAULT_LLM_MODELS_FALLBACK['ollama'])
                        llm_model_select.options = fb
                        llm_model_select.value = fb[0]
                        llm_model_select.update()
                        return
                    fb = DEFAULT_LLM_MODELS_FALLBACK.get(prov, DEFAULT_LLM_MODELS_FALLBACK['ssp'])
                    llm_model_select.options = fb
                    llm_model_select.value = fb[0]
                    llm_model_select.update()

                def _on_llm_provider_changed():
                    _apply_llm_fallback_models()
                    refresh_summarizer_summary()

                provider_select.on_value_change(lambda _: _on_llm_provider_changed())

                llm_model_select.on_value_change(lambda _: refresh_summarizer_summary())

                loader_radio.on_value_change(lambda _: refresh_summarizer_summary())

                embedding_radio.on_value_change(lambda _: refresh_summarizer_summary())

                def open_summarizer_editor():
                    summarizer_config_summary_panel.set_visibility(False)
                    summarizer_config_editor_panel.set_visibility(True)
                    summarizer_config_summary_panel.update()
                    summarizer_config_editor_panel.update()

                def close_summarizer_editor():
                    refresh_summarizer_summary()
                    summarizer_config_editor_panel.set_visibility(False)
                    summarizer_config_summary_panel.set_visibility(True)
                    summarizer_config_editor_panel.update()
                    summarizer_config_summary_panel.update()

                ui.separator().classes('w-full opacity-60')
                ui.button(
                    'Done',
                    icon='sym_o_expand_less',
                    on_click=lambda: close_summarizer_editor(),
                ).props('flat dense no-caps color=primary').classes('self-start')

            summarizer_config_editor_panel.set_visibility(False)
            refresh_summarizer_summary()

        with ui.card().classes('w-full p-6 shadow-lg'):
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
                                parameters_pill_max_words = ui.label('').classes(_pill_val)
                            with ui.column().classes('gap-1 shrink-0'):
                                ui.label('Keywords').classes(_pill_cap)
                                parameters_pill_keywords = ui.label('').classes(_pill_val)
                            with ui.column().classes('gap-1 shrink-0'):
                                ui.label('Tags').classes(_pill_cap)
                                parameters_pill_tags = ui.label('').classes(_pill_val)
                            with ui.column().classes('gap-1 min-w-0 flex-1 max-w-full'):
                                ui.label('Language').classes(_pill_cap)
                                parameters_pill_lang = ui.label('').classes(
                                    _pill_val + ' font-mono'
                                )
                    ui.button(
                        'Customize',
                        icon='sym_o_tune',
                        on_click=lambda: open_parameters_editor(),
                    ).props('outline dense no-caps color=primary unelevated').classes(
                        'flex-shrink-0'
                    )

            parameters_editor_panel = ui.column().classes('w-full gap-4')
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

            def close_parameters_editor():
                refresh_parameters_summary()
                parameters_editor_panel.set_visibility(False)
                parameters_summary_panel.set_visibility(True)
                parameters_editor_panel.update()
                parameters_summary_panel.update()

            parameters_editor_panel.set_visibility(False)
            refresh_parameters_summary()

            max_words_input.on_value_change(lambda _: refresh_parameters_summary())
            max_keywords_input.on_value_change(lambda _: refresh_parameters_summary())
            max_tags_input.on_value_change(lambda _: refresh_parameters_summary())
            out_lang_input.on_value_change(lambda _: refresh_parameters_summary())

        with ui.column().classes('w-full gap-2') as summarize_actions_container:
            status_label = ui.label('Ready.').classes(
                'text-lg text-gray-600 whitespace-pre-line leading-snug'
            )
            ui.button('🔍 Summarize', on_click=lambda: handle_summarize(
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
            )).classes('w-full bg-blue-600 text-white text-lg py-2 rounded hover:bg-blue-700')

        update_summarize_actions_visibility()

        ui.separator().classes('w-full my-4')

        with ui.column().classes('w-full gap-4') as results_container:
            with ui.card().classes('w-full p-6 bg-gray-50'):
                ui.label('📄 Summary').classes('text-xl font-medium')
                summary_label = ui.label('Summary will appear here.').classes('w-full whitespace-pre-wrap')

            with ui.card().classes('w-full p-6 bg-gray-50'):
                ui.label('🔑 Keywords').classes('text-xl font-medium')
                keywords_label = ui.label('Keywords will appear here.').classes('w-full')

            with ui.card().classes('w-full p-6 bg-gray-50'):
                ui.label('🏷️ Tags').classes('text-xl font-medium')
                tags_label = ui.label('Tags will appear here.').classes('w-full')

        def download_summary_export():
            ui.download.content(
                f"""📄 Summary:
        {summary_label.text}

        🔑 Keywords:
        {keywords_label.text}

        🏷️ Tags:
        {tags_label.text}
        """,
                LAST_SUMMARY_META["download_filename"],
            )

        download_button = ui.button(
            '⬇️ Download Summary',
            on_click=download_summary_export,
        ).props('color=primary').classes('w-full')

        results_container.set_visibility(False)
        download_button.set_visibility(False)

if __name__ in {'__main__', '__mp_main__'}:
    root_path = os.getenv('NICEGUI_ROOT_PATH', '').strip()
    run_kwargs = {
        'host': '0.0.0.0',
        'port': 5001,
        'reload': True, # reload the app when the code changes
        'title': 'PDF Summarizer',
    }
    if root_path:
        run_kwargs['root_path'] = root_path
    ui.run(**run_kwargs)