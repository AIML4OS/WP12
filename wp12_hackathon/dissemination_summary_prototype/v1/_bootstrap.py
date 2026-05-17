"""
Env-var setup for cache directories.

Imported FIRST by every module in this package so that HF_HOME,
DOCLING_ARTIFACTS_PATH, and TORCH_HOME are set before torch / langchain_* /
docling are loaded anywhere.  Those libraries read their cache paths from
environment variables at *import time*; if they are imported before this
module runs, they lock in ~/.cache defaults and never see our overrides.

Usage (first non-stdlib import in every file):
    import _bootstrap  # noqa: F401  -- env setup; must come before heavy imports
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Force stdout/stderr to UTF-8 on Windows (default is cp1252 which can't
# encode arrows, check-marks and other Unicode used in progress messages).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

_SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(_SCRIPT_DIR / ".env", override=True)

# ---------------------------------------------------------------------------
# Cache layout
#
# Model caches (HF, Docling, Torch, EasyOCR) live in ~/.cache/<subdir>.
# They are large, shared across projects, and must not live in deep project
# trees (Windows MAX_PATH = 260 chars; HF Hub staging adds ~191 chars of
# suffix inside DOCLING_ARTIFACTS_PATH alone).
#
# Project-specific runtime data (summaries) stays inside the project tree.
# ---------------------------------------------------------------------------
_user_cache = Path.home() / ".cache"
_project_cache = _SCRIPT_DIR / ".cache"

for _env_var, _subdir in (
    ("HF_HOME",                "hf"),
    ("DOCLING_ARTIFACTS_PATH", "docling"),
    ("TORCH_HOME",             "torch"),
    ("EASYOCR_MODULE_PATH",    "easyocr"),
):
    _p = (_user_cache / _subdir).resolve()
    _p.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(_env_var, str(_p))

(Path(os.environ["HF_HOME"]) / "sentence_transformers").mkdir(parents=True, exist_ok=True)
(_project_cache / "summaries").mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Misc library flags
# ---------------------------------------------------------------------------
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "False")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ---------------------------------------------------------------------------
# Static Docling config from config.yml → env vars
#
# document_load.py reads DOCLING_* env vars as its defaults; the UI / CLI
# can still override individual values via explicit function kwargs.
# We use raw yaml here (not endpoints.load_config) to stay import-free.
# ---------------------------------------------------------------------------
_config_path = _SCRIPT_DIR / "config.yml"
if _config_path.exists():
    try:
        import yaml
        _raw = yaml.safe_load(_config_path.read_text(encoding="utf-8")) or {}
        _docling_cfg = _raw.get("docling", {})
        for _k, _v in {
            "DOCLING_DO_OCR":             str(_docling_cfg.get("do_ocr",             "false")).lower(),
            "DOCLING_OCR_ENGINE":         str(_docling_cfg.get("ocr_engine",         "easyocr")).lower(),
            "DOCLING_TABLE_MODE":         str(_docling_cfg.get("table_mode",         "fast")).lower(),
            "DOCLING_CONVERT_TIMEOUT_SEC":str(_docling_cfg.get("convert_timeout_sec","900")),
        }.items():
            os.environ.setdefault(_k, _v)
    except Exception:
        pass  # config.yml absent or malformed — library defaults take over
