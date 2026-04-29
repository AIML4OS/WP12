# this is a unified summarizer that has parameters for the different summarizers

# Load environment variables FIRST, before any other imports
from dotenv import load_dotenv
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Load .env from the script's directory, not the current working directory.
# override=True ensures stale shell-exported SSP_KEY values do not shadow .env.
load_dotenv(os.path.join(script_dir, '.env'), override=True)


import warnings
import time
import re
from typing import List, Dict, Optional, Union, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnablePassthrough
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from langchain_docling import DoclingLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_docling.loader import ExportType
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
#from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
import torch
import logging
import json
from langchain_community.vectorstores.utils import filter_complex_metadata
from openai import AuthenticationError

from langchain_core.embeddings import Embeddings
import requests
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _run_callable_with_timeout(fn, timeout_sec: float, timeout_message: str):
    """
    Run a blocking callable in a worker thread; abort waiting after timeout_sec.

    Note: the worker may keep running in the background after timeout (Docling has no cancel API).
    """
    if timeout_sec <= 0:
        return fn()
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        fut = pool.submit(fn)
        return fut.result(timeout=timeout_sec)
    except FuturesTimeout as exc:
        raise RuntimeError(timeout_message) from exc
    finally:
        pool.shutdown(wait=False)


def _attach_timing_breakdown(payload: Any, timing: Dict[str, float]) -> Any:
    """Merge per-phase timings into the result dict for the UI (values are seconds)."""
    if isinstance(payload, dict):
        merged = dict(payload)
        merged.setdefault("sources", [])
        merged["_timing_sec"] = timing
        return merged
    return {
        "summary": str(payload),
        "keywords": [],
        "tags": [],
        "sources": [],
        "_timing_sec": timing,
    }


# Suppress PyTorch MPS warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*MPS.*")

# Configure logging - reduce verbosity
logging.basicConfig(level=logging.WARNING)  # Changed from INFO to WARNING
logger = logging.getLogger(__name__)


# Fallback lists when model discovery fails or keys are missing (UI still offers a choice).
DEFAULT_LLM_MODELS_FALLBACK: Dict[str, List[str]] = {
    "ssp": ["qwen3-6-35b-moe"],
    "ollama": ["llama3.1:8b"],
    "openai": ["gpt-4o-mini", "gpt-4o"],
}


def _ollama_http_get(url: str, timeout: float) -> requests.Response:
    """
    GET without trusting HTTP(S)_PROXY so localhost / 127.0.0.1 still works when env proxies
    would otherwise break browser-identical URLs like http://localhost:11434.
    """
    host_part = url.split("://", 1)[-1].split("/", 1)[0].lower()
    localish = host_part.startswith("127.") or host_part.startswith("localhost")
    if localish:
        session = requests.Session()
        session.trust_env = False
        return session.get(url, timeout=timeout)
    return requests.get(url, timeout=timeout)


def _ollama_chat_model_names_from_tags_body(body: Any) -> List[str]:
    """
    Parse GET /api/tags JSON. Each row may include ``name``, ``model``, and ``details``.

    Omits typical embedding-only pulls from the chat dropdown (e.g. ``nomic-embed-text``).
    """
    if not isinstance(body, dict):
        return []
    names: List[str] = []
    for row in body.get("models") or []:
        if not isinstance(row, dict):
            continue
        label = (row.get("name") or row.get("model") or "").strip()
        if not label:
            continue
        low = label.lower()
        details = row.get("details") if isinstance(row.get("details"), dict) else {}
        family = (details.get("family") or "").lower()
        if "embed" in low or family == "nomic-bert":
            continue
        names.append(label)
    return sorted(set(names))


def probe_ollama_runtime(preferred_base_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Find a running Ollama API and list tags.

    1. Prefer GET ``/api/tags`` — lists installed models (may be empty).
    2. If tags fails, GET ``/api/version`` — JSON like ``{"version":"0.13.4"}`` confirms the
       daemon (same endpoint browsers/tools often hit).

    Tries ``preferred_base_url`` (if set), ``OLLAMA_BASE_URL``, then 127.0.0.1 and localhost.

    Returns dict keys: ``ok`` (bool), ``base_url`` (Optional[str]), ``models`` (List[str]),
    ``error`` (Optional[str]).
    """
    candidates: List[str] = []
    pb = (preferred_base_url or "").strip()
    if pb:
        candidates.append(pb.rstrip("/"))
    env_b = os.getenv("OLLAMA_BASE_URL")
    if env_b and env_b.strip():
        eb = env_b.strip().rstrip("/")
        if eb not in candidates:
            candidates.append(eb)
    for b in ("http://127.0.0.1:11434", "http://localhost:11434"):
        if b not in candidates:
            candidates.append(b)

    last_err: Optional[str] = None
    for base in candidates:
        tags_note = ""

        try:
            rt = _ollama_http_get(f"{base}/api/tags", timeout=25)
            if rt.status_code == 200:
                body = rt.json()
                models = _ollama_chat_model_names_from_tags_body(body)
                return {
                    "ok": True,
                    "base_url": base,
                    "models": models,
                    "error": None,
                }
            tags_note = f"/api/tags HTTP {rt.status_code}"
        except requests.exceptions.RequestException as e:
            tags_note = f"/api/tags: {e}"
        except Exception as e:
            tags_note = f"/api/tags: {e}"

        try:
            rv = _ollama_http_get(f"{base}/api/version", timeout=8)
            if rv.status_code == 200:
                data = rv.json()
                if isinstance(data, dict) and data.get("version"):
                    return {
                        "ok": True,
                        "base_url": base,
                        "models": [],
                        "error": None,
                    }
            ver_note = f"/api/version HTTP {rv.status_code}"
        except requests.exceptions.RequestException as e:
            ver_note = f"/api/version: {e}"
        except Exception as e:
            ver_note = f"/api/version: {e}"

        last_err = f"{base}: {tags_note}; {ver_note}"

    return {
        "ok": False,
        "base_url": None,
        "models": [],
        "error": last_err
        or "Ollama not reachable — install from https://ollama.com and run `ollama serve` (or ensure it is running).",
    }


def _parse_openai_compatible_models_payload(payload: Any) -> List[str]:
    """Extract model ids from OpenAI-style GET /v1/models JSON."""
    if not isinstance(payload, dict):
        return []
    rows = payload.get("data")
    if rows is None:
        rows = payload.get("models")
    if not isinstance(rows, list):
        return []
    ids: List[str] = []
    for row in rows:
        if isinstance(row, str):
            ids.append(row)
        elif isinstance(row, dict):
            mid = row.get("id") or row.get("name")
            if mid:
                ids.append(str(mid))
    return ids


def fetch_llm_models_for_provider(
    provider: str,
    *,
    api_key: Optional[str] = None,
    ollama_base_url: str = "http://localhost:11434",
) -> List[str]:
    """
    Discover model names for the chosen LLM backend.

    - ollama: GET {base}/api/tags
    - ssp: OpenAI-compatible /v1/models on the SSP host (tries a few URLs)
    - openai: GET https://api.openai.com/v1/models
    """
    p = (provider or "").lower().strip()
    if p == "ollama":
        hint = (ollama_base_url or "").strip() or None
        pr = probe_ollama_runtime(hint)
        if not pr["ok"]:
            raise RuntimeError(pr["error"] or "Ollama unreachable")
        out = pr["models"]
        return out if out else list(DEFAULT_LLM_MODELS_FALLBACK["ollama"])

    if p == "openai":
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set (env or .env).")
        r = requests.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=45,
        )
        if r.status_code == 401:
            raise RuntimeError(
                "OpenAI returned 401. Check OPENAI_API_KEY in `.env`."
            )
        r.raise_for_status()
        ids = _parse_openai_compatible_models_payload(r.json())
        preferred = [
            i
            for i in ids
            if any(
                x in i
                for x in ("gpt-4", "gpt-3.5-turbo", "gpt-5", "o1", "o3", "gpt-4o")
            )
        ]
        out = sorted(set(preferred)) if preferred else sorted(set(ids))[:80]
        return out if out else list(DEFAULT_LLM_MODELS_FALLBACK["openai"])

    if p == "ssp":
        key = api_key or os.getenv("SSP_KEY")
        if not key:
            raise RuntimeError("SSP_KEY is not set (env or .env).")
        candidates = [
            "https://llm.lab.sspcloud.fr/api/v1/models",
            "https://llm.lab.sspcloud.fr/v1/models",
        ]
        last_exc: Optional[Exception] = None
        for url in candidates:
            try:
                r = requests.get(
                    url,
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=45,
                )
                if r.status_code == 401:
                    raise RuntimeError(
                        "SSP LLM returned 401. Check SSP_KEY in `.env`."
                    )
                r.raise_for_status()
                ids = _parse_openai_compatible_models_payload(r.json())
                out = sorted(set(ids))
                if out:
                    return out
            except Exception as e:
                last_exc = e
                continue
        if last_exc:
            raise RuntimeError(
                f"Could not list SSP models ({last_exc}). "
                "Using fallback list; you can still pick a model name manually if needed."
            ) from last_exc
        return list(DEFAULT_LLM_MODELS_FALLBACK["ssp"])

    raise ValueError(f"Unknown LLM provider: {provider}")


def timed_execution(message_template: str = None):
    """
    Decorator that times function execution and prints a custom message with timing.
    
    Args:
        message_template (str): Template message to format with function parameters. 
                               If None, uses the function name.
                               Can use parameter names in curly braces like {pdf_path}
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Format message template with actual parameters if provided
            if message_template:
                try:
                    # For instance methods, args[0] is self, so we need to get the actual parameters
                    if args and hasattr(args[0], '__class__'):  # Instance method
                        # Get function signature to map parameters
                        import inspect
                        sig = inspect.signature(func)
                        bound_args = sig.bind(*args, **kwargs)
                        bound_args.apply_defaults()
                        
                        # Create a dict with parameter names and values
                        param_dict = dict(bound_args.arguments)
                        # Remove 'self' from the dict
                        if 'self' in param_dict:
                            del param_dict['self']
                        
                        display_message = message_template.format(**param_dict)
                    else:  # Regular function
                        display_message = message_template.format(*args, **kwargs)
                except (KeyError, IndexError) as e:
                    # Fallback if formatting fails
                    display_message = f"Executing {func.__name__} (format error: {e})"
            else:
                display_message = f"Executing {func.__name__}"
            
            print(f"{display_message}")
            
            # Time the execution
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Print timing
            print(f"⏱️ Execution time: {execution_time:.2f} seconds")
            return result
        return wrapper
    return decorator


def execute_with_timing(func, message: str = None, *args, **kwargs):
    """
    Execute a function with timing and custom message.
    
    Args:
        func: Function to execute
        message (str): Custom message to print before execution
        *args, **kwargs: Arguments to pass to the function
    
    Returns:
        The result of the function execution
    """
    # Use custom message or function name
    display_message = message or f"Executing {func.__name__}"
    print(f"{display_message}")
    
    # Time the execution
    start_time = time.time()
    result = func(*args, **kwargs)
    execution_time = time.time() - start_time
    
    # Print timing
    print(f"⏱️ Execution time: {execution_time:.2f} seconds")
    return result


class PDFSummarizer:
    def __init__(
        self,
        llm_provider: str = "ssp",
        llm_config: Dict = None
    ):
        # Initialize LLM with configuration
        llm_config = llm_config or {}
        self.llm = self._create_llm(llm_provider, **llm_config)

    def _safe_json_parse(self, result):
        """Parse JSON or return text if parsing fails."""
        import re
        
        content = result.content if hasattr(result, 'content') else str(result)
        
        # First try direct JSON parsing
        try:
            return json.loads(content.strip())
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(1)
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON object anywhere in the text
        json_match = re.search(r'\{.*?\}', content, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(0)
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # If all fails, return fallback
        print(f"⚠️ JSON parsing failed completely")
        print(f"🔍 Raw LLM output: {content[:200]}...")
        return {
            "summary": content,
            "keywords": ["parsing_failed"],
            "tags": ["raw_text"],
            "sources": [],
        }

    def _build_sources_from_docs(
        self,
        docs: Any,
        *,
        display_source_name: Optional[str] = None,
        pdf_path: Optional[str] = None,
        max_sources: int = 6,
        max_excerpt_chars: int = 280,
    ) -> List[Dict[str, str]]:
        """Build compact source-attribution entries from retrieved chunks."""
        if not isinstance(docs, list):
            return []
        sources: List[Dict[str, str]] = []
        seen = set()
        page_texts_norm: Optional[List[str]] = None
        for doc in docs:
            page_content = getattr(doc, "page_content", "") if doc is not None else ""
            meta = getattr(doc, "metadata", {}) if doc is not None else {}
            if not isinstance(meta, dict):
                meta = {}

            page_label = self._extract_page_label(meta)
            source_name = self._extract_source_name(meta, display_source_name)

            excerpt = " ".join(str(page_content).split())
            if len(excerpt) > max_excerpt_chars:
                excerpt = excerpt[: max_excerpt_chars - 1].rstrip() + "…"
            excerpt = excerpt or "(empty excerpt)"

            if page_label == "page n/a" and pdf_path and excerpt != "(empty excerpt)":
                if page_texts_norm is None:
                    page_texts_norm = self._load_pdf_page_texts_norm(pdf_path)
                inferred_page = self._infer_page_from_excerpt(excerpt, page_texts_norm)
                if inferred_page is not None:
                    page_label = f"p.{inferred_page}"

            dedupe_key = (source_name, page_label, excerpt[:160])
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            i = len(sources) + 1
            sources.append(
                {
                    "id": str(i),
                    "source": source_name,
                    "location": page_label,
                    "excerpt": excerpt,
                }
            )
            if len(sources) >= max_sources:
                break
        return sources

    def _normalize_for_match(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^a-z0-9à-ÿ ]+", "", text)
        return text.strip()

    def _load_pdf_page_texts_norm(self, pdf_path: str) -> List[str]:
        """Load page texts (normalized) to support fallback page inference."""
        try:
            pages = PyPDFLoader(pdf_path).load()
        except Exception:
            return []
        out: List[str] = []
        for page in pages:
            out.append(self._normalize_for_match(getattr(page, "page_content", "")))
        return out

    def _infer_page_from_excerpt(self, excerpt: str, page_texts_norm: List[str]) -> Optional[int]:
        """Infer 1-based page index by matching a normalized excerpt against page text."""
        if not page_texts_norm:
            return None
        ex = self._normalize_for_match(excerpt)
        if not ex:
            return None
        probe = ex[:150] if len(ex) > 150 else ex
        if len(probe) < 24:
            return None
        for i, page_txt in enumerate(page_texts_norm, start=1):
            if probe in page_txt:
                return i
        return None

    def _extract_page_label(self, metadata: Dict[str, Any]) -> str:
        """Extract a human-readable page from heterogeneous metadata schemas."""
        def _extract_int(value: Any) -> Optional[int]:
            if isinstance(value, bool):
                return None
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value)
            if isinstance(value, str):
                m = re.search(r"\d+", value)
                if m:
                    return int(m.group(0))
            return None

        # Common flat keys first.
        flat_keys = ("page", "page_number", "page_num", "page_no", "pagenum", "page_idx")
        for key in flat_keys:
            if key in metadata:
                val = _extract_int(metadata.get(key))
                if val is not None:
                    # Most loaders are 0-based for page; convert to 1-based.
                    return f"p.{val + 1}" if key in ("page", "page_idx") else f"p.{val}"

        # Deep recursive scan (e.g., Docling-style nested metadata/provenance).
        def _scan(node: Any) -> Optional[int]:
            if isinstance(node, dict):
                for k, v in node.items():
                    lk = str(k).lower()
                    if "page" in lk:
                        out = _extract_int(v)
                        if out is not None:
                            return out
                    nested = _scan(v)
                    if nested is not None:
                        return nested
            elif isinstance(node, list):
                for item in node:
                    nested = _scan(item)
                    if nested is not None:
                        return nested
            return None

        deep_page = _scan(metadata)
        if deep_page is not None:
            # Deep scans are usually 1-based.
            return f"p.{deep_page}"
        return "page n/a"

    def _extract_source_name(
        self,
        metadata: Dict[str, Any],
        display_source_name: Optional[str] = None,
    ) -> str:
        """Resolve a stable source name and hide temp upload filenames."""
        if display_source_name and str(display_source_name).strip():
            return os.path.basename(str(display_source_name).strip())

        source_name = (
            metadata.get("source")
            or metadata.get("file_path")
            or metadata.get("filename")
            or "document"
        )
        source_name = os.path.basename(str(source_name))

        # Hide transient temp files if we can.
        if re.match(r"^tmp[a-z0-9_-]+\.pdf$", source_name.lower()):
            return "uploaded_file.pdf"
        return source_name

    def _build_source_from_full_text(
        self,
        pdf_path: str,
        content: str,
        display_source_name: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Provide a coarse source reference when retrieval mode is not used."""
        excerpt = " ".join(str(content).split())
        if len(excerpt) > 360:
            excerpt = excerpt[:359].rstrip() + "…"
        return [
            {
                "id": "1",
                "source": (
                    os.path.basename(str(display_source_name))
                    if display_source_name
                    else (os.path.basename(pdf_path) or "document")
                ),
                "location": "full document",
                "excerpt": excerpt or "(content unavailable)",
            }
        ]

    def _create_llm(self, provider: str, **kwargs) -> BaseChatModel:
        """
        Create LLM instance based on provider.
        """
        llm_timeout = _env_float("LLM_REQUEST_TIMEOUT_SEC", 600.0)
        llm_retries = max(0, min(10, _env_int("LLM_MAX_RETRIES", 2)))

        if provider.lower() == "openai":
            model_name = kwargs.get("model_name") or kwargs.get("model", "gpt-4o-mini")
            api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
            return ChatOpenAI(
                model_name=model_name,
                temperature=kwargs.get("temperature", 0.7),
                api_key=api_key,
                request_timeout=llm_timeout,
                max_retries=llm_retries,
            )
        elif provider.lower() == "ollama":
            return ChatOllama(
                model=kwargs.get("model", "llama3.1:8b"),
                temperature=kwargs.get("temperature", 0.7),
                base_url=kwargs.get("base_url", "http://localhost:11434"),
                sync_client_kwargs={"timeout": llm_timeout},
            )
        elif provider.lower() == "ssp":
            #model = "mistral-small3.1:latest",
            #model = "llama3.3:70b",

            return ChatOpenAI(
                api_key = kwargs.get("api_key"),  # replace with your key
                base_url="https://llm.lab.sspcloud.fr/api",
                #model=kwargs.get("model", "mistral-small3.2:latest"),
                model=kwargs.get("model", "qwen3-6-35b-moe"),
                temperature=kwargs.get("temperature", 0.7),
                request_timeout=llm_timeout,
                max_retries=llm_retries,
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    @staticmethod
    def _raise_with_auth_guidance(exc: Exception) -> None:
        """
        Raise a friendlier authentication error with remediation steps.
        """
        message = str(exc)
        if "401" in message or "token is invalid" in message.lower() or "session has expired" in message.lower():
            raise RuntimeError(
                "Authentication with SSP LLM failed (401). "
                "Your `SSP_KEY` is expired or invalid. Please set a valid key in `.env` and restart the app."
            ) from exc
        raise exc

    @timed_execution("Processing PDF: {pdf_path}, use_vector_store: {use_vector_store}, document_loader: {document_loader}, embedding_model: {embedding_model}, use_remote_embedding: {use_remote_embedding}")
    def process_pdf(self, pdf_path: str, use_vector_store:bool = False, document_loader:str = "docling", embedding_model:str = "BAAI/bge-m3", use_remote_embedding:bool = False, max_keywords: int = 6, max_tags: int = 5, out_lang: str = "pt-pt", max_words: int = 200, display_source_name: Optional[str] = None):
        """
        Process a PDF file and return a summary.
        """

        # - 1 - load the pdf with docling or pypdf
        # - 2 - split the pdf into chunks
        # - 3 - load the chunks into a vector store or just the text        
        # - 4 - summarize the chunks with the llm (querying the vector store) or (just the text)
        # - 5 - return the summary

        timing: Dict[str, float] = {}

        if use_vector_store:
            t0 = time.perf_counter()
            if document_loader == "docling":
                chunks = self._load_pdf_with_docling(pdf_path)
            elif document_loader == "pypdf":
                chunks = self._load_pdf_with_pypdf(pdf_path)
            else:
                raise ValueError(f"Unsupported document loader: {document_loader}")
            timing["chunks"] = time.perf_counter() - t0

            t1 = time.perf_counter()
            vector_store = self._get_vector_store(chunks, embedding_model, use_remote_embedding)
            retriever = vector_store.as_retriever(search_kwargs={"k": 10})  # Increased from 5 to 10
            retrieval_chain = self._create_retrieval_chain(retriever)
            timing["vector_store"] = time.perf_counter() - t1

            try:
                t2 = time.perf_counter()
                result = retrieval_chain.invoke({
                    "input": "Summarize the main content and statistics from this document about air transport activity",
                    "max_keywords": max_keywords,
                    "max_tags": max_tags,
                    "max_words": max_words,
                    "out_lang": out_lang
                })
                timing["llm"] = time.perf_counter() - t2
            except AuthenticationError as exc:
                self._raise_with_auth_guidance(exc)
            except Exception as exc:
                # Some wrappers may rethrow auth as generic exceptions.
                self._raise_with_auth_guidance(exc)
            out = result.get("answer", result)
            if isinstance(out, dict):
                out["sources"] = self._build_sources_from_docs(
                    result.get("context"),
                    display_source_name=display_source_name,
                    pdf_path=pdf_path,
                )
            return _attach_timing_breakdown(out, timing)

        else:
            context = "\n\n=== Document Section ===\n\n"
            t0 = time.perf_counter()
            if document_loader == "docling":
                pdf_text = self._load_pdf_txt_with_docling(pdf_path)
                context = context + pdf_text
            elif document_loader == "pypdf":
                pdf_text = self._load_pdf_txt_with_pypdf(pdf_path)
                context = context + pdf_text
            else:
                raise ValueError(f"Unsupported document loader: {document_loader}")
            timing["pdf_load"] = time.perf_counter() - t0

            chain = self._create_summary_chain()

            try:
                t1 = time.perf_counter()
                result = chain.invoke({
                    "content": context,
                    "max_keywords": max_keywords,
                    "max_tags": max_tags,
                    "max_words": max_words,
                    "out_lang": out_lang
                })
                timing["llm"] = time.perf_counter() - t1
            except AuthenticationError as exc:
                self._raise_with_auth_guidance(exc)
            except Exception as exc:
                # Some wrappers may rethrow auth as generic exceptions.
                self._raise_with_auth_guidance(exc)

            if isinstance(result, dict):
                result["sources"] = self._build_source_from_full_text(
                    pdf_path,
                    pdf_text,
                    display_source_name=display_source_name,
                )
            return _attach_timing_breakdown(result, timing)


    def _create_summary_chain(self):
        """
        Create a runnable sequence for direct text summarization (non-vector store mode).
        """
        chat_prompt, _ = self._get_shared_prompts()
        return chat_prompt | self.llm | self._safe_json_parse


    def _docling_plain_text_sync(self, pdf_path: str) -> str:
        """Full Docling conversion + text extraction (runs in worker thread when timed out)."""
        with self._suppress_logging():
            docling_converter = DocumentConverter()
            result = docling_converter.convert(pdf_path)

        text_content = []
        for text_item in result.document.texts:
            if text_item.text and text_item.text.strip():
                text_content.append(text_item.text.strip())
        return "\n\n".join(text_content)

    def _load_pdf_txt_with_docling(self, pdf_path: str) -> str:
        """
        Load PDF using docling's advanced parsing capabilities.
        Bounded by DOCLING_CONVERT_TIMEOUT_SEC (default 900s) so the UI cannot hang forever.
        """
        timeout = _env_float("DOCLING_CONVERT_TIMEOUT_SEC", 900.0)
        msg = (
            f"Docling conversion exceeded {timeout:.0f}s while reading the PDF. "
            "Try PDF loader PyPDF, a smaller document, or raise DOCLING_CONVERT_TIMEOUT_SEC."
        )
        return _run_callable_with_timeout(
            lambda: self._docling_plain_text_sync(pdf_path),
            timeout,
            msg,
        )

    def _load_pdf_txt_with_pypdf(self, pdf_path: str) -> str:
        """
        Fallback method to load PDF using PyPDFLoader.
        """
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        return "\n".join(page.page_content for page in pages)

    def _docling_chunks_sync(self, pdf_path: str, sentence_transformer_model: str = "BAAI/bge-m3") -> List[str]:
        export_type = ExportType.DOC_CHUNKS
        with self._suppress_logging():
            loader = DoclingLoader(
                file_path=pdf_path,
                export_type=export_type,
                chunker=HybridChunker(tokenizer=sentence_transformer_model),
            )

            docs = loader.load()
            if export_type == ExportType.DOC_CHUNKS:
                splits = docs
                splits = filter_complex_metadata(splits)

            elif export_type == ExportType.MARKDOWN:
                from langchain_text_splitters import MarkdownHeaderTextSplitter

                splitter = MarkdownHeaderTextSplitter(
                    headers_to_split_on=[
                        ("#", "Header_1"),
                        ("##", "Header_2"),
                        ("###", "Header_3"),
                    ],
                )
                splits = [split for doc in docs for split in splitter.split_text(doc.page_content)]
            else:
                raise ValueError(f"Unexpected export type: {export_type}")

            return splits

    def _load_pdf_with_docling(self, pdf_path: str, sentence_transformer_model: str = "BAAI/bge-m3") -> List[str]:
        """
        Load PDF using docling's advanced parsing capabilities.
        Same timeout as plain Docling path (DOCLING_CONVERT_TIMEOUT_SEC).
        """
        timeout = _env_float("DOCLING_CONVERT_TIMEOUT_SEC", 900.0)
        msg = (
            f"Docling chunking exceeded {timeout:.0f}s. "
            "Try PDF loader PyPDF, or raise DOCLING_CONVERT_TIMEOUT_SEC."
        )
        return _run_callable_with_timeout(
            lambda: self._docling_chunks_sync(pdf_path, sentence_transformer_model),
            timeout,
            msg,
        )

    def _get_vector_store(self, splits: List[str], embedding_model: str = "BAAI/bge-m3", use_remote_embedding: bool = True) -> str:
        """
        Create vector store with local or remote embeddings.
        
        Args:
            splits: Document chunks
            embedding_model: Model name for embeddings
            use_remote_embedding: If True, use remote embedding service
        """
        if use_remote_embedding:
            remote_model = "qwen3-embedding:8b"

            print(f"🔧 Remote embedding with model: {remote_model} at SSP")
            
            #Configure for SSP's embedding service
            embedding = OllamaEmbeddings(
                model=remote_model,
                base_url="http://llm.lab.sspcloud.fr/ollama",
                client_kwargs={'headers': {
                    "Authorization": f"Bearer {os.getenv('SSP_KEY')}"
                }}
            )

        else:
            # Use local embedding
            print(f"🔧 Local embedding with model: {embedding_model}")

            embedding = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                encode_kwargs={"normalize_embeddings": True}  # recommended for BGE
            )

        # Use an isolated in-memory collection per run to avoid cross-document pollution.
        vector_store = Chroma.from_documents(
            collection_name=f"run_{time.time_ns()}",
            documents=splits,
            embedding=embedding,
        )
        return vector_store

    def _load_pdf_with_pypdf(self, pdf_path: str) -> str:
        """
        Load PDF using pypdf with better chunking strategy.
        """
        from langchain_core.documents import Document
        
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        # Use RecursiveCharacterTextSplitter for better chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Larger chunks
            chunk_overlap=200,  # Some overlap for context
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Split the pages into better chunks and create Document objects
        splits = []
        for page_idx, page in enumerate(pages):
            page_splits = text_splitter.split_text(page.page_content)
            for split in page_splits:
                # Create Document objects with metadata
                metadata = dict(page.metadata) if isinstance(page.metadata, dict) else {}
                raw_page = metadata.get("page")
                if isinstance(raw_page, int):
                    metadata["page_number"] = raw_page + 1
                else:
                    metadata["page_number"] = page_idx + 1
                doc = Document(
                    page_content=split,
                    metadata=metadata
                )
                splits.append(doc)
        
        return splits

    # this is a helper function to suppress the logging of docling, it is used to avoid the verbose output of docling   
    def _suppress_logging(self):
        """
        Context manager to temporarily suppress logging.
        """
        import contextlib
        
        @contextlib.contextmanager
        def suppress_logging():
            # Store original levels
            original_levels = {}
            loggers_to_suppress = [
                "docling.document_converter",
                "docling.models.factories", 
                "docling.utils.accelerator_utils",
                "docling.pipeline.base_pipeline",
                "torch.utils.data.dataloader",
                "chromadb.telemetry.product.posthog",
                "chromadb.telemetry"
            ]
            
            try:
                # Set all to ERROR level (suppress INFO and WARNING, keep ERROR)
                for logger_name in loggers_to_suppress:
                    logger_obj = logging.getLogger(logger_name)
                    original_levels[logger_name] = logger_obj.level
                    logger_obj.setLevel(logging.ERROR)
                yield
            finally:
                # Restore original levels
                for logger_name, original_level in original_levels.items():
                    logging.getLogger(logger_name).setLevel(original_level)
        
        return suppress_logging()

    def _create_retrieval_chain(self, retriever):
        """
        Create a retrieval chain that integrates document retrieval with generation.
        """
        _, string_prompt = self._get_shared_prompts()
        
        # Create the document chain with safe JSON output
        document_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=string_prompt,
            output_parser=self._safe_json_parse
        )
        
        # Create the retrieval chain
        retrieval_chain = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=document_chain
        )
        
        return retrieval_chain

    def _get_shared_prompts(self):
        """
        Get shared prompts that ensure consistent output format across all modes.
        """
        # Instruction prompt (this is the one we can tweak, it could be refactored as a parameter)
        instruction_prompt = """
        You are a statistician, specialized in reporting, your goal is to summarize the content of documents.
        You don't add information and you don't make analysis, you just summarize what is already there.
        You pay special attention not to be biased in your summarizations, stay neutral and follow the spirit and tone present in the document.
        When outputting in Portuguese, use the European Portuguese.
        """
        
        # Inner system prompt (this is fixed)
        inner_system_prompt = """
        You are a professional document analyzer. Your task is to:
        1. Provide a clear and concise summary
        2. Extract relevant keywords and tags
        3. Maintain objectivity and accuracy
        4. Focus on the main points and key findings

        CRITICAL: You must respond ONLY with valid JSON. No explanations, no markdown, no additional text - just pure JSON.
        """

        system_message = f"{inner_system_prompt}\n\n{instruction_prompt}\n\n"
        
        # Single human prompt template to avoid duplication
        human_prompt = """You must analyze the content and return ONLY valid JSON. No explanations, no markdown, no additional text.

Required JSON structure:
{{
    "summary": "comprehensive summary (max {max_words} words, language: {out_lang})",
    "keywords": ["list of {max_keywords} keywords"],
    "tags": ["list of {max_tags} tags"]
}}

Example valid response:
{{
    "summary": "The document discusses air transport statistics showing passenger growth.",
    "keywords": ["aviation", "statistics", "passengers"],
    "tags": ["transport", "data"]
}}

Now analyze this content and respond with JSON only:

{content}"""
        
        # Chat prompt template (for direct text mode)
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_prompt)
        ])
        
        # String prompt template (for retrieval chain) - uses same human prompt with different variable names
        retrieval_human_prompt = human_prompt.replace("{content}", "Context: {context}\n\nUser Question: {input}")
        string_prompt = PromptTemplate(
            template=system_message + retrieval_human_prompt,
            input_variables=["context", "input", "max_keywords", "max_tags", "max_words", "out_lang"]
        )
        
        return chat_prompt, string_prompt


def main():
        
    # config for the llm
    config = {
        "api_key": os.getenv('SSP_KEY'),
        "temperature": 0.1  # Lower temperature for more deterministic JSON output
    }
    # model = "mistral-small3.1:latest", # a bit faster
    # model = "llama3.3:70b",   # a bit better but slower


    # 1 - create and configure the summarizer
    summarizer = PDFSummarizer(llm_config=config)
    
    # our input file - make it relative to the script's directory
    file_path = os.path.join(script_dir,"prototype_a", "Aereo.pdf")

    # result = summarizer.process_pdf(
    #     file_path,
    #     document_loader="pypdf",
    #     use_vector_store=True,
    #     #     embedding_model="thenlper/gte-small", # alternative small fast and weak model for local embeddings
    # )
    # print(json.dumps(result, indent=2, ensure_ascii=False))
    # print("\n" + "="*80 + "\n")

    result = summarizer.process_pdf(
        file_path,
        use_vector_store=True,
        use_remote_embedding=True
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("\n" + "="*80 + "\n")


    # example of use
    # result = summarizer.process_pdf(
    #     pdf_path="Aereo.pdf",
    #     use_vector_store=False,
    #     document_loader="docling",
    #     max_keywords=15,
    #     max_tags=8,
    #     out_lang='en',   # 'pt-pt'
    #     max_words=200
    # )



if __name__ == "__main__":
    main() 