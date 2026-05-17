"""
GRADUATION-READY: PDF loading, chunking, and chunk_id assignment.

This file is designed to be liftable into a shared package when a second
consumer (e.g. a Knowledge Base project) materialises.  Do not add
summarizer-specific concerns here — no SummarizeOptions, no prompts, no
summary cache awareness, no LLM calls.

Public API
----------
    fingerprint(pdf_path) -> str
    prewarm() -> None
    load_docling_chunks(pdf_path, fp, *, do_ocr, ocr_engine) -> list[Document]
    load_pypdf_chunks(pdf_path, fp) -> list[Document]
    load_docling_text(pdf_path, *, do_ocr, ocr_engine) -> str
    load_pypdf_text(pdf_path) -> str

Configuration
-------------
    All Docling settings default to DOCLING_* environment variables
    (set from config.yml by _bootstrap.py).  Explicit kwargs override
    the env var for that single call — used by the UI for runtime changes.
      DOCLING_DO_OCR              "true" | "false"   (default: "false")
      DOCLING_OCR_ENGINE          "easyocr" | "rapidocr" | "tesseract"
      DOCLING_TABLE_MODE          "fast" | "accurate"
      DOCLING_CONVERT_TIMEOUT_SEC seconds (default: 900)

Chunk metadata schema (guaranteed on every returned Document):
    chunk_id        "<fingerprint>::p<page>::b<block_idx>"
    doc_fingerprint "<sha256_first16>"
    source_pdf      "Aereo.pdf"
    source_path     "/abs/path/Aereo.pdf"
    page            int (1-based) or 0 if unknown
    loader          "docling" | "pypdf"
    bbox            [x1,y1,x2,y2] (Docling only, omitted for pypdf)
    section         "3.2 Architecture" (Docling only, when available)
"""
import _bootstrap  # noqa: F401 — env setup; must come before heavy imports

import contextlib
import hashlib
import logging
import os
import re
import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeout
from pathlib import Path
from typing import Any

from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore", message=".*pin_memory.*no accelerator.*", category=UserWarning)
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singletons — loaded once per process, reused across calls.
# _DOCLING_CONVERTERS is keyed by (do_ocr, ocr_engine, table_mode) so that
# different OCR configurations each get their own converter instance.
# ---------------------------------------------------------------------------
_DOCLING_CONVERTERS: dict[tuple, Any] = {}
_HYBRID_CHUNKER: dict[str, Any] = {}


@contextlib.contextmanager
def _suppress_noisy_logging():
    """Temporarily silence docling/chromadb/torch INFO chatter."""
    noisy = [
        "docling.document_converter",
        "docling.models.factories",
        "docling.utils.accelerator_utils",
        "docling.pipeline.base_pipeline",
        "torch.utils.data.dataloader",
        "chromadb.telemetry.product.posthog",
        "chromadb.telemetry",
    ]
    saved = {n: logging.getLogger(n).level for n in noisy}
    try:
        for n in noisy:
            logging.getLogger(n).setLevel(logging.ERROR)
        yield
    finally:
        for n, lvl in saved.items():
            logging.getLogger(n).setLevel(lvl)


def _get_docling_converter(
    *,
    do_ocr: bool | None = None,
    ocr_engine: str | None = None,
    table_mode_str: str | None = None,
) -> Any:
    """Return a cached DocumentConverter, keyed by (do_ocr, ocr_engine, table_mode).

    Parameters fall back to environment variables so callers can use env-only
    configuration (e.g. from .env) without passing explicit arguments.

    OCR engine choices (only relevant when do_ocr=True):
      easyocr   — multilingual; models cached in EASYOCR_MODULE_PATH on first use
      rapidocr  — ONNX models bundled with pip package, zero extra download
      tesseract — system binary; requires TESSDATA_PREFIX env var
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.document_converter import DocumentConverter, PdfFormatOption

    _do_ocr = do_ocr if do_ocr is not None else (
        os.environ.get("DOCLING_DO_OCR", "false").strip().lower() == "true"
    )
    _engine = (ocr_engine or os.environ.get("DOCLING_OCR_ENGINE", "easyocr")).strip().lower()
    _table_raw = (table_mode_str or os.environ.get("DOCLING_TABLE_MODE", "fast")).strip().lower()
    _table_mode = TableFormerMode.ACCURATE if _table_raw == "accurate" else TableFormerMode.FAST

    cache_key = (_do_ocr, _engine, _table_raw)
    if cache_key in _DOCLING_CONVERTERS:
        return _DOCLING_CONVERTERS[cache_key]

    artifacts_raw = os.environ.get("DOCLING_ARTIFACTS_PATH", "").strip() or None
    artifacts_path = str(Path(artifacts_raw)) if artifacts_raw else None

    if artifacts_path:
        import importlib.metadata
        from docling.utils.model_downloader import download_models
        ap = Path(artifacts_path)
        marker = ap / ".docling_version"
        try:
            installed = importlib.metadata.version("docling")
        except importlib.metadata.PackageNotFoundError:
            installed = "unknown"
        cached_ver = marker.read_text().strip() if marker.exists() else ""
        if not ap.exists() or cached_ver != installed:
            print(f"Docling: downloading models (v{installed}) → {artifacts_path}")
            ap.mkdir(parents=True, exist_ok=True)
            download_models(output_dir=ap, progress=True)
            marker.write_text(installed)
            print(f"  ✓ Docling models ready (v{installed})")

    pipeline_options = PdfPipelineOptions(
        do_ocr=_do_ocr,
        do_table_structure=True,
        artifacts_path=artifacts_path,
    )
    pipeline_options.table_structure_options.mode = _table_mode

    if _do_ocr:
        if _engine == "rapidocr":
            from docling.datamodel.pipeline_options import RapidOcrOptions
            pipeline_options.ocr_options = RapidOcrOptions()
        elif _engine == "tesseract":
            from docling.datamodel.pipeline_options import TesseractOcrOptions
            pipeline_options.ocr_options = TesseractOcrOptions()
        else:
            # easyocr is the docling default; models are stored in EASYOCR_MODULE_PATH
            from docling.datamodel.pipeline_options import EasyOcrOptions
            pipeline_options.ocr_options = EasyOcrOptions()

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    _DOCLING_CONVERTERS[cache_key] = converter

    mode_tag = f"offline ({artifacts_raw})" if artifacts_path else "online"
    ocr_tag = f"OCR={_engine}" if _do_ocr else "OCR=off"
    print(f"Docling converter ready — {mode_tag} | {ocr_tag} | tables={_table_raw}")
    return converter


def _get_hybrid_chunker(tokenizer: str = "BAAI/bge-m3") -> Any:
    """Return a cached HybridChunker (loads tokenizer from HF cache once per process)."""
    if tokenizer not in _HYBRID_CHUNKER:
        from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
        _HYBRID_CHUNKER[tokenizer] = HybridChunker(tokenizer=tokenizer)  # type: ignore[arg-type]
    return _HYBRID_CHUNKER[tokenizer]


def _run_with_timeout(fn, timeout_sec: float, error_msg: str):
    """Run a blocking callable in a thread; abort after timeout_sec."""
    if timeout_sec <= 0:
        return fn()
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        return pool.submit(fn).result(timeout=timeout_sec)
    except _FuturesTimeout as exc:
        raise RuntimeError(error_msg) from exc
    finally:
        pool.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fingerprint(pdf_path: Path | str) -> str:
    """Return the first 16 hex chars of sha256(pdf_bytes). Pure function over bytes."""
    data = Path(pdf_path).read_bytes()
    return hashlib.sha256(data).hexdigest()[:16]


def prewarm() -> None:
    """Eagerly load Docling converter + HybridChunker singletons. Idempotent.

    Uses whatever DOCLING_DO_OCR / DOCLING_OCR_ENGINE / DOCLING_TABLE_MODE are
    set in the environment (i.e. from .env via _bootstrap).
    """
    _get_docling_converter()
    artifacts_raw = os.environ.get("DOCLING_ARTIFACTS_PATH", "").strip() or None
    mode = f"offline ({artifacts_raw})" if artifacts_raw else "online"
    _get_hybrid_chunker("BAAI/bge-m3")
    print(f"  ✓ Docling pre-warm complete ({mode})")


def _extract_page_from_dl_meta(dl_meta: dict[str, Any]) -> int:
    """Extract 1-based page number from a Docling dl_meta provenance dict.

    Docling stores provenance as::

        dl_meta.doc_items[].prov[].page_no   (1-based integer)

    Returns 0 when provenance is absent.
    """
    try:
        for item in dl_meta.get("doc_items") or []:
            for prov in item.get("prov") or []:
                page_no = prov.get("page_no")
                if isinstance(page_no, (int, float)) and page_no > 0:
                    return int(page_no)
    except Exception:
        pass
    return 0


def _extract_page_int(metadata: dict[str, Any]) -> int:
    """Extract a 1-based page integer from heterogeneous LangChain metadata.

    Returns 0 when no page can be determined.
    Docling metadata is deeply nested; this does a recursive scan.
    """
    def _to_int(v: Any) -> int | None:
        if isinstance(v, bool):
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(v)
        if isinstance(v, str):
            m = re.search(r"\d+", v)
            return int(m.group(0)) if m else None
        return None

    # Flat keys first — most loaders use these.
    for key in ("page", "page_number", "page_num", "page_no", "pagenum", "page_idx"):
        if key in metadata:
            val = _to_int(metadata[key])
            if val is not None:
                # PyPDF stores 0-based "page"; "page_number" is usually 1-based.
                return val + 1 if key in ("page", "page_idx") else val

    # Deep scan for Docling-style nested provenance.
    def _scan(node: Any) -> int | None:
        if isinstance(node, dict):
            for k, v in node.items():
                if "page" in str(k).lower():
                    r = _to_int(v)
                    if r is not None:
                        return r
                r = _scan(v)
                if r is not None:
                    return r
        elif isinstance(node, list):
            for item in node:
                r = _scan(item)
                if r is not None:
                    return r
        return None

    deep = _scan(metadata)
    return deep if deep is not None else 0


def _make_chunk_id(fp: str, page: int, block_idx: int) -> str:
    return f"{fp}::p{page}::b{block_idx:04d}"


def load_docling_chunks(
    pdf_path: Path | str,
    fp: str,
    *,
    loader_id: str = "docling",
    do_ocr: bool | None = None,
    ocr_engine: str | None = None,
    table_mode_str: str | None = None,
) -> list[Document]:
    """Load a PDF with Docling and return LangChain Documents with full provenance.

    Each Document carries the guaranteed chunk metadata schema defined in this
    module's docstring.  filter_complex_metadata is applied so Chroma never
    chokes on nested types.

    do_ocr / ocr_engine / table_mode_str override the corresponding
    DOCLING_* env vars for this call only.
    """
    pdf_path = Path(pdf_path)
    timeout = float(os.environ.get("DOCLING_CONVERT_TIMEOUT_SEC", "900") or 900)

    def _load():
        from langchain_docling.loader import ExportType, DoclingLoader
        with _suppress_noisy_logging():
            loader = DoclingLoader(
                file_path=str(pdf_path),
                export_type=ExportType.DOC_CHUNKS,
                converter=_get_docling_converter(
                    do_ocr=do_ocr, ocr_engine=ocr_engine,
                    table_mode_str=table_mode_str,
                ),
                chunker=_get_hybrid_chunker("BAAI/bge-m3"),
            )
            # Return raw docs — page/heading extraction needs dl_meta before
            # filter_complex_metadata removes nested dicts.
            return loader.load()

    raw = _run_with_timeout(
        _load, timeout,
        f"Docling chunking exceeded {timeout:.0f}s. Try --loader pypdf or raise DOCLING_CONVERT_TIMEOUT_SEC."
    )

    docs: list[Document] = []
    for idx, doc in enumerate(raw):
        meta = dict(doc.metadata) if isinstance(doc.metadata, dict) else {}

        # Extract page number from dl_meta provenance BEFORE filtering strips it.
        dl_meta = meta.get("dl_meta") if isinstance(meta.get("dl_meta"), dict) else {}
        page = _extract_page_from_dl_meta(dl_meta) or _extract_page_int(meta)

        chunk_id = _make_chunk_id(fp, page, idx)
        # Build clean provenance metadata (flat primitives only — Chroma-safe).
        clean_meta: dict[str, Any] = {
            "chunk_id": chunk_id,
            "doc_fingerprint": fp,
            "source_pdf": pdf_path.name,
            "source_path": str(pdf_path.resolve()),
            "page": page,
            "loader": loader_id,
        }
        # Optionally carry bbox and section when Docling provides them.
        if "bbox" in meta:
            clean_meta["bbox"] = meta["bbox"]
        headings = dl_meta.get("headings") or []
        if headings:
            clean_meta["section"] = headings[-1] if isinstance(headings[-1], str) else str(headings[-1])
        docs.append(Document(page_content=doc.page_content, metadata=clean_meta))

    return docs


def load_pypdf_chunks(
    pdf_path: Path | str,
    fp: str,
    *,
    loader_id: str = "pypdf",
) -> list[Document]:
    """Load a PDF with PyPDF and return LangChain Documents with full provenance.

    Uses RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200) to
    split page text into manageable chunks.
    """
    pdf_path = Path(pdf_path)
    pages = PyPDFLoader(str(pdf_path)).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )
    docs: list[Document] = []
    block_idx = 0
    for page_doc in pages:
        page_meta = page_doc.metadata if isinstance(page_doc.metadata, dict) else {}
        raw_page = page_meta.get("page")
        page = (raw_page + 1) if isinstance(raw_page, int) else 0
        for chunk_text in splitter.split_text(page_doc.page_content):
            chunk_id = _make_chunk_id(fp, page, block_idx)
            docs.append(Document(
                page_content=chunk_text,
                metadata={
                    "chunk_id": chunk_id,
                    "doc_fingerprint": fp,
                    "source_pdf": pdf_path.name,
                    "source_path": str(pdf_path.resolve()),
                    "page": page,
                    "loader": loader_id,
                },
            ))
            block_idx += 1
    return docs


def load_docling_text(
    pdf_path: Path | str,
    *,
    do_ocr: bool | None = None,
    ocr_engine: str | None = None,
    table_mode_str: str | None = None,
) -> str:
    """Return the full plain-text content of a PDF via Docling.

    do_ocr / ocr_engine / table_mode_str override the corresponding
    DOCLING_* env vars for this call only.
    """
    pdf_path = Path(pdf_path)
    timeout = float(os.environ.get("DOCLING_CONVERT_TIMEOUT_SEC", "900") or 900)

    def _convert():
        with _suppress_noisy_logging():
            converter = _get_docling_converter(
                do_ocr=do_ocr, ocr_engine=ocr_engine,
                table_mode_str=table_mode_str,
            )
            result = converter.convert(str(pdf_path))
        return "\n\n".join(
            item.text.strip()
            for item in result.document.texts
            if item.text and item.text.strip()
        )

    return _run_with_timeout(
        _convert, timeout,
        f"Docling conversion exceeded {timeout:.0f}s. Try --loader pypdf."
    )


def load_pypdf_text(pdf_path: Path | str) -> str:
    """Return the full plain-text content of a PDF via PyPDF."""
    pages = PyPDFLoader(str(pdf_path)).load()
    return "\n".join(p.page_content for p in pages)
