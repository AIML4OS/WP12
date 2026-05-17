"""
GRADUATION-READY: endpoint discovery, config models, and LangChain factory dispatch.

This file is designed to be liftable into a shared package when a second
consumer (e.g. a Knowledge Base project) materialises.  Do not add
summarizer-specific concerns here — no SummarizeOptions, no prompt logic,
no summary cache awareness.

Public API
----------
    load_config(path) -> Config
    probe(endpoint, timeout) -> EndpointStatus
    list_models(endpoint, role) -> list[str]
    make_chat_model(endpoint, model, **kwargs) -> BaseChatModel
    make_embeddings(target, model, endpoints) -> Embeddings
    setup(stage)   -- downloads deps / models for offline use
"""
import _bootstrap  # noqa: F401 — env setup; must come before heavy imports

import logging
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Literal

import requests
import torch
import yaml
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field, model_validator

warnings.filterwarnings("ignore", message=".*pin_memory.*no accelerator.*", category=UserWarning)
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class LLMDefaults(BaseModel):
    timeout_sec: float = 600.0
    max_retries: int = 2


class Defaults(BaseModel):
    out_lang: str = "pt-pt"
    max_words: int = 200
    max_keywords: int = 6
    max_tags: int = 5
    processing_mode: Literal["vector", "plain"] = "vector"
    pdf_loader: Literal["docling", "pypdf"] = "docling"
    embedding: str = "local"
    llm: LLMDefaults = Field(default_factory=LLMDefaults)
    temperature: float = 0.1


class DoclingOptions(BaseModel):
    do_ocr: bool = False
    # ocr_engine is only used when do_ocr=True.
    #   easyocr   multilingual; models cached in .cache/easyocr/ on first use (~200 MB)
    #   rapidocr  ONNX models bundled in pip package — zero extra download, fastest
    #   tesseract system binary required; set TESSDATA_PREFIX env var
    ocr_engine: Literal["easyocr", "rapidocr", "tesseract"] = "easyocr"
    table_mode: Literal["fast", "accurate"] = "fast"
    convert_timeout_sec: int = 900


class LocalEmbedding(BaseModel):
    model: str = "BAAI/bge-m3"


class BenchSettings(BaseModel):
    pdfs_dir: Path = Path("bench/pdfs")
    configs_dir: Path = Path("bench/configs")
    runs_dir: Path = Path("bench/runs")
    stability_repeat: int = 2


class Endpoint(BaseModel):
    url: str
    id: str = ""
    name: str = ""
    auth_env: str | None = None
    auth_help: str | None = None
    network_hint: str | None = None

    @model_validator(mode="after")
    def _fill_id_and_name(self) -> "Endpoint":
        if not self.id:
            from urllib.parse import urlparse
            host = urlparse(self.url).hostname or self.url
            object.__setattr__(self, "id", host)
        if not self.name:
            object.__setattr__(self, "name", self.id)
        return self

    def api_key(self) -> str | None:
        """Resolve the API key from the environment variable named by auth_env."""
        if not self.auth_env:
            return None
        return os.environ.get(self.auth_env) or None


class EndpointStatus(BaseModel):
    endpoint: Endpoint
    ok: bool
    kind: Literal["ollama_native", "openai_compatible", "unknown"] = "unknown"
    chat_models: list[str] = Field(default_factory=list)
    embed_models: list[str] = Field(default_factory=list)
    auth_required: bool = False
    error: str | None = None


class Config(BaseModel):
    cache_dir: Path = Path(".cache")
    defaults: Defaults = Field(default_factory=Defaults)
    docling: DoclingOptions = Field(default_factory=DoclingOptions)
    local_embedding: LocalEmbedding = Field(default_factory=LocalEmbedding)
    bench: BenchSettings = Field(default_factory=BenchSettings)
    endpoints: list[Endpoint] = Field(default_factory=list)
    instruction_prompt: str = (
        "You are a statistician specializing in reporting; your goal is to summarize "
        "the content of documents.\n"
        "You must not add information or make analysis; only summarize what is already present.\n"
        "Pay special attention to avoiding bias in your summaries; remain neutral. "
        "Follow the spirit and tone of the documents.\n"
        "Use European Portuguese for Portuguese output and American English for English output.\n"
        "Strive for eloquence while remaining accessible in the target language."
    )

    @model_validator(mode="after")
    def _check_unique_ids(self) -> "Config":
        seen: set[str] = set()
        for ep in self.endpoints:
            if ep.id in seen:
                raise ValueError(
                    f"Duplicate endpoint id '{ep.id}'. Each endpoint must have a unique id."
                )
            seen.add(ep.id)
        return self


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yml"


def load_config(path: Path | None = None) -> Config:
    """Load and validate config.yml, merging over hard-coded defaults.

    If path is None, looks for config.yml next to this file.
    If the file does not exist the defaults are used as-is.
    """
    p = Path(path) if path else _DEFAULT_CONFIG_PATH
    if not p.exists():
        return Config()
    with open(p, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    # Resolve relative paths anchored at the config file's directory.
    config_dir = p.parent
    if "cache_dir" in raw:
        raw["cache_dir"] = (config_dir / raw["cache_dir"]).resolve()
    if "bench" in raw:
        for key in ("pdfs_dir", "configs_dir", "runs_dir"):
            if key in raw["bench"]:
                raw["bench"][key] = (config_dir / raw["bench"][key]).resolve()
    return Config.model_validate(raw)


# ---------------------------------------------------------------------------
# Endpoint probing helpers
# ---------------------------------------------------------------------------

def _http_get(url: str, timeout: float, headers: dict | None = None) -> requests.Response:
    """GET that bypasses HTTP_PROXY for localhost URLs (avoids corporate proxy issues)."""
    host = url.split("://", 1)[-1].split("/", 1)[0].lower()
    is_local = host.startswith("127.") or host.startswith("localhost")
    if is_local:
        session = requests.Session()
        session.trust_env = False
        return session.get(url, timeout=timeout, headers=headers or {})
    return requests.get(url, timeout=timeout, headers=headers or {})


def _is_embed_model(name: str, family: str = "") -> bool:
    """Heuristic: is this model embedding-only (not a chat model)?"""
    low = name.lower()
    fam = family.lower()
    if fam in ("nomic-bert", "bert"):
        return True
    return any(tok in low for tok in ("embed", "e5-", "bge-", "gte-", "text-embedding-"))


def _split_ollama_body(body: object) -> tuple[list[str], list[str]]:
    """Parse /api/tags JSON → (chat_models, embed_models)."""
    if not isinstance(body, dict):
        return [], []
    chat: list[str] = []
    embed: list[str] = []
    for row in body.get("models") or []:
        if not isinstance(row, dict):
            continue
        label = (row.get("name") or row.get("model") or "").strip()
        if not label:
            continue
        family = (row.get("details") or {}).get("family") or ""
        if _is_embed_model(label, family):
            embed.append(label)
        else:
            chat.append(label)
    return sorted(set(chat)), sorted(set(embed))


def _split_openai_body(body: object) -> tuple[list[str], list[str]]:
    """Parse /v1/models JSON → (chat_models, embed_models)."""
    if not isinstance(body, dict):
        return [], []
    rows = body.get("data") or body.get("models") or []
    chat: list[str] = []
    embed: list[str] = []
    for row in rows:
        mid = row if isinstance(row, str) else (row.get("id") or row.get("name") or "")
        if not mid:
            continue
        if _is_embed_model(mid):
            embed.append(mid)
        else:
            chat.append(mid)
    return sorted(set(chat)), sorted(set(embed))


def probe(endpoint: Endpoint, *, timeout: int = 25) -> EndpointStatus:
    """Auto-detect API kind and list models from a single endpoint URL.

    Probe sequence:
      1. GET /api/tags         → Ollama native
      2. GET /v1/models        → OpenAI-compatible (with auth if auth_env set)
      3. GET /api/version      → bare Ollama (no models listed, confirms alive)
    """
    base = endpoint.url.rstrip("/")
    key = endpoint.api_key()
    auth_header = {"Authorization": f"Bearer {key}"} if key else {}

    # 1 — Ollama native /api/tags
    try:
        r = _http_get(f"{base}/api/tags", timeout=timeout)
        if r.status_code == 200:
            chat, embed = _split_ollama_body(r.json())
            return EndpointStatus(
                endpoint=endpoint, ok=True, kind="ollama_native",
                chat_models=chat, embed_models=embed,
            )
    except Exception:
        pass

    # 2 — OpenAI-compatible /v1/models
    try:
        r = requests.get(f"{base}/v1/models", headers=auth_header or {"Authorization": "Bearer nokeyneeded"}, timeout=timeout)
        if r.status_code == 401:
            return EndpointStatus(endpoint=endpoint, ok=False, kind="openai_compatible",
                                  auth_required=True,
                                  error=f"401 Unauthorized. {endpoint.auth_help or 'Set auth_env in config.'}")
        if r.status_code == 200:
            chat, embed = _split_openai_body(r.json())
            return EndpointStatus(
                endpoint=endpoint, ok=True, kind="openai_compatible",
                chat_models=chat, embed_models=embed,
            )
    except Exception:
        pass

    # 3 — bare Ollama /api/version
    try:
        r = _http_get(f"{base}/api/version", timeout=8)
        if r.status_code == 200 and isinstance(r.json(), dict) and r.json().get("version"):
            return EndpointStatus(
                endpoint=endpoint, ok=True, kind="ollama_native",
                chat_models=[], embed_models=[],
            )
    except Exception:
        pass

    return EndpointStatus(
        endpoint=endpoint, ok=False,
        error=f"Could not reach {base}. Check URL, network, and whether the service is running.",
    )


def list_models(
    endpoint: Endpoint,
    role: Literal["chat", "embed", "all"] = "all",
) -> list[str]:
    """Return model names from an endpoint, filtered by role."""
    status = probe(endpoint)
    if not status.ok:
        return []
    if role == "chat":
        return status.chat_models
    if role == "embed":
        return status.embed_models
    return sorted(set(status.chat_models + status.embed_models))


# ---------------------------------------------------------------------------
# LangChain factory dispatch
# ---------------------------------------------------------------------------

_HF_EMBEDDING_CACHE: dict[tuple[str, str], HuggingFaceEmbeddings] = {}


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_hf_embedding(model_name: str) -> HuggingFaceEmbeddings:
    """Cached HuggingFaceEmbeddings (loaded once per model per process)."""
    dev = _device()
    key = (model_name, dev)
    if key not in _HF_EMBEDDING_CACHE:
        hf_home = os.environ.get("HF_HOME", "")
        cache_folder = os.path.join(hf_home, "sentence_transformers") if hf_home else None
        _HF_EMBEDDING_CACHE[key] = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": dev},
            encode_kwargs={"normalize_embeddings": True},
            cache_folder=cache_folder,
        )
    return _HF_EMBEDDING_CACHE[key]


def make_chat_model(
    endpoint: Endpoint,
    model: str,
    *,
    temperature: float = 0.1,
    timeout: float = 600.0,
    max_retries: int = 2,
) -> BaseChatModel:
    """Return the right LangChain chat model for the given endpoint."""
    status = probe(endpoint)
    base = endpoint.url.rstrip("/")
    key = endpoint.api_key()

    if status.kind == "ollama_native":
        return ChatOllama(
            model=model,
            temperature=temperature,
            base_url=base,
            sync_client_kwargs={"timeout": timeout},
        )
    # OpenAI-compatible — base URL must match how the OpenAI Python client joins paths.
    # SSP Cloud (summarizer_unified.py): ChatOpenAI(..., base_url="https://llm.lab.sspcloud.fr/api")
    # without a trailing "/v1"; the client requests .../api/v1/chat/completions. Appending "/v1"
    # here can yield .../api/v1/v1/... on some client versions.
    chat_base = base
    if not base.endswith("/v1"):
        if base.endswith("/api"):
            chat_base = base
        else:
            chat_base = f"{base}/v1"
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=key or "nokeyneeded",
        base_url=chat_base,
        request_timeout=timeout,  # type: ignore[call-arg]
        max_retries=max_retries,
    )


def make_embeddings(
    target: str,
    model: str,
    *,
    endpoints: list[Endpoint] | None = None,
) -> Embeddings:
    """Return a LangChain Embeddings instance.

    target: "local" → HuggingFaceEmbeddings with the given model name.
            <url>   → look up the endpoint and use OllamaEmbeddings or
                      OpenAIEmbeddings depending on its kind.
    model:  the model name to request.
    """
    if target == "local":
        return _get_hf_embedding(model)

    # Resolve the endpoint by URL.
    ep: Endpoint | None = None
    if endpoints:
        for e in endpoints:
            if e.url.rstrip("/") == target.rstrip("/"):
                ep = e
                break
    if ep is None:
        ep = Endpoint(url=target)

    status = probe(ep)
    base = ep.url.rstrip("/")
    key = ep.api_key()

    if status.kind == "ollama_native":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=model, base_url=base)

    return OpenAIEmbeddings(
        model=model,
        openai_api_key=key or "nokeyneeded",
        openai_api_base=f"{base}/v1" if not base.endswith("/v1") else base,
    )


# ---------------------------------------------------------------------------
# Setup helper
# ---------------------------------------------------------------------------

def setup(stage: Literal["deps", "models", "all"] = "all") -> None:
    """Install requirements and/or pre-download models for offline use."""
    if stage in ("deps", "all"):
        print("Installing requirements...")
        req = Path(__file__).resolve().parent / "requirements.txt"
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req)])
        print("  ✓ Requirements installed.")

    if stage in ("models", "all"):
        cfg = load_config()
        model_name = cfg.local_embedding.model
        print(f"Pre-downloading local embedding model: {model_name}")
        _get_hf_embedding(model_name)
        print(f"  ✓ Embedding model ready.")

        print("Pre-downloading Docling models...")
        import document_load
        document_load.prewarm()
        print("  ✓ Docling models ready.")
