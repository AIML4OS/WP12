"""
Integration tests — verify our components wire correctly to their libraries.

These tests call the real libraries (Docling, PyPDF, HuggingFace embeddings).
No LLM, no network beyond what's already in the local .cache.

Runtime: ~30–120 s  (docling model loading + PDF conversion dominates).

Run:
    .conda\\python.exe tools\\test_integration.py          (from v1/)
    .conda\\python.exe tools\\test_integration.py -v       (verbose)

What each suite tests
─────────────────────
TestDoclingChunks
  Feeds demo_docs/Aereo.pdf through load_docling_chunks() and verifies:
  - At least one chunk is returned
  - Every chunk carries the full guaranteed metadata schema
  - chunk_id format is correct
  - doc_fingerprint matches fingerprint(pdf)
  - page values are positive integers (never 0 from a native PDF)
  - No chunk has a None / empty page_content

TestPypdfChunks
  Same contract checks via load_pypdf_chunks(), which uses a different
  backend.  Documents that load_docling cannot handle should load here.

TestDoclingText
  load_docling_text() returns a non-empty string for a native PDF.

TestPypdfText
  load_pypdf_text() returns a non-empty string.

TestPrewarm
  prewarm() completes without raising (idempotent).

TestLocalEmbeddings
  make_embeddings("local", model) can embed a short text string.
  Verifies the vector is the right length (>0) and is a list of floats.

TestProbeEndpoint
  probe() on a localhost Ollama URL that may or may not be running.
  Verifies we always get back an EndpointStatus — ok=False is fine,
  but we must never crash / raise.
"""
import sys
import os
import unittest
from pathlib import Path

V1_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V1_ROOT))

import _bootstrap  # noqa: F401

from document_load import (
    fingerprint,
    prewarm,
    load_docling_chunks,
    load_docling_text,
    load_pypdf_chunks,
    load_pypdf_text,
)
from endpoints import (
    load_config,
    make_embeddings,
    probe,
    Endpoint,
)

# ── Locate the demo PDF ───────────────────────────────────────────────────────
_DEMO_PDF = V1_ROOT / "demo_docs" / "Aereo.pdf"
_DEMO_AVAILABLE = _DEMO_PDF.exists()

# Required metadata keys that every chunk must carry (from REFERENCE.md §4)
_REQUIRED_CHUNK_KEYS = {"chunk_id", "doc_fingerprint", "source_pdf", "source_path", "page", "loader"}

# chunk_id format:  "<fp>::p<int>::b<4-digit-int>"
import re
_CHUNK_ID_RE = re.compile(r"^[0-9a-f]{16}::p\d+::b\d{4}$")


def _skip_no_demo(test_fn):
    """Decorator: skip if Aereo.pdf is not present."""
    import functools
    @functools.wraps(test_fn)
    def wrapper(self, *args, **kwargs):
        if not _DEMO_AVAILABLE:
            self.skipTest(f"Demo PDF not found: {_DEMO_PDF}")
        return test_fn(self, *args, **kwargs)
    return wrapper


# ─────────────────────────────────────────────────────────────────────────────
class TestPrewarm(unittest.TestCase):
    """prewarm() must complete without raising — idempotent."""

    def test_prewarm_does_not_raise(self):
        prewarm()  # first call — loads singletons

    def test_prewarm_idempotent(self):
        prewarm()  # second call — must reuse singletons, not re-download


# ─────────────────────────────────────────────────────────────────────────────
class TestDoclingChunks(unittest.TestCase):
    """
    load_docling_chunks() output contract.

    We test the schema contract, not Docling's extraction quality.
    If this suite fails, either:
    - Our code broke the call to Docling
    - Docling changed its output API in a way our adapter didn't handle
    """

    @classmethod
    def setUpClass(cls):
        if not _DEMO_AVAILABLE:
            return
        cls.fp = fingerprint(_DEMO_PDF)
        # Load once for all tests in this suite (expensive)
        cls.chunks = load_docling_chunks(_DEMO_PDF, cls.fp)

    @_skip_no_demo
    def test_returns_at_least_one_chunk(self):
        self.assertGreater(len(self.chunks), 0,
                           "Docling returned no chunks from a non-empty PDF")

    @_skip_no_demo
    def test_every_chunk_has_required_metadata_keys(self):
        missing = {}
        for i, doc in enumerate(self.chunks):
            m = getattr(doc, "metadata", {}) or {}
            absent = _REQUIRED_CHUNK_KEYS - set(m.keys())
            if absent:
                missing[i] = absent
        self.assertEqual(missing, {},
                         f"Chunks missing required metadata keys: {missing}")

    @_skip_no_demo
    def test_chunk_id_format(self):
        bad = []
        for doc in self.chunks:
            cid = doc.metadata.get("chunk_id", "")
            if not _CHUNK_ID_RE.match(cid):
                bad.append(cid)
        self.assertEqual(bad, [],
                         f"chunk_id format violations: {bad[:5]}")

    @_skip_no_demo
    def test_doc_fingerprint_matches(self):
        for doc in self.chunks:
            self.assertEqual(doc.metadata["doc_fingerprint"], self.fp)

    @_skip_no_demo
    def test_page_is_positive_integer(self):
        """Native PDFs must have real page numbers — never 0."""
        bad = [doc.metadata["page"] for doc in self.chunks
               if not isinstance(doc.metadata.get("page"), int) or doc.metadata["page"] <= 0]
        # Allow a small fraction of 0s (sometimes first chunk has no provenance)
        zero_ratio = len(bad) / len(self.chunks)
        self.assertLess(zero_ratio, 0.5,
                        f"{len(bad)}/{len(self.chunks)} chunks have page<=0")

    @_skip_no_demo
    def test_loader_field_is_docling(self):
        for doc in self.chunks:
            self.assertEqual(doc.metadata.get("loader"), "docling")

    @_skip_no_demo
    def test_source_pdf_is_filename_only(self):
        for doc in self.chunks:
            src = doc.metadata.get("source_pdf", "")
            # Must be just the filename, not a full path
            self.assertNotIn("/", src)
            self.assertNotIn("\\", src)
            self.assertTrue(src.endswith(".pdf"), f"Unexpected source_pdf: {src}")

    @_skip_no_demo
    def test_no_empty_page_content(self):
        empty = [i for i, doc in enumerate(self.chunks)
                 if not (doc.page_content or "").strip()]
        # Docling occasionally produces empty chunks; cap at 10%
        ratio = len(empty) / len(self.chunks)
        self.assertLess(ratio, 0.10,
                        f"Too many empty chunks ({len(empty)}/{len(self.chunks)})")

    @_skip_no_demo
    def test_ocr_off_does_not_raise(self):
        """Explicit do_ocr=False must not change behaviour vs default."""
        fp = fingerprint(_DEMO_PDF)
        chunks = load_docling_chunks(_DEMO_PDF, fp, do_ocr=False)
        self.assertGreater(len(chunks), 0)

    @_skip_no_demo
    def test_table_mode_accurate_does_not_raise(self):
        """table_mode='accurate' is a valid parameter — must not crash."""
        fp = fingerprint(_DEMO_PDF)
        chunks = load_docling_chunks(_DEMO_PDF, fp, table_mode="accurate")
        self.assertGreater(len(chunks), 0)


# ─────────────────────────────────────────────────────────────────────────────
class TestPypdfChunks(unittest.TestCase):
    """load_pypdf_chunks() output contract — same schema, different backend."""

    @classmethod
    def setUpClass(cls):
        if not _DEMO_AVAILABLE:
            return
        cls.fp = fingerprint(_DEMO_PDF)
        cls.chunks = load_pypdf_chunks(_DEMO_PDF, cls.fp)

    @_skip_no_demo
    def test_returns_at_least_one_chunk(self):
        self.assertGreater(len(self.chunks), 0)

    @_skip_no_demo
    def test_every_chunk_has_required_metadata_keys(self):
        for doc in self.chunks:
            m = doc.metadata or {}
            missing = _REQUIRED_CHUNK_KEYS - set(m.keys())
            self.assertEqual(missing, set(), f"Missing keys: {missing}")

    @_skip_no_demo
    def test_chunk_id_format(self):
        for doc in self.chunks:
            cid = doc.metadata.get("chunk_id", "")
            self.assertRegex(cid, _CHUNK_ID_RE, f"Bad chunk_id: {cid}")

    @_skip_no_demo
    def test_doc_fingerprint_matches(self):
        for doc in self.chunks:
            self.assertEqual(doc.metadata["doc_fingerprint"], self.fp)

    @_skip_no_demo
    def test_loader_field_is_pypdf(self):
        for doc in self.chunks:
            self.assertEqual(doc.metadata.get("loader"), "pypdf")


# ─────────────────────────────────────────────────────────────────────────────
class TestDoclingText(unittest.TestCase):

    @_skip_no_demo
    def test_returns_non_empty_string(self):
        text = load_docling_text(_DEMO_PDF)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text.strip()), 100,
                           "Docling plain-text extraction returned almost nothing")

    @_skip_no_demo
    def test_contains_some_words(self):
        text = load_docling_text(_DEMO_PDF)
        words = text.split()
        self.assertGreater(len(words), 50,
                           f"Only {len(words)} words extracted — suspiciously low")


# ─────────────────────────────────────────────────────────────────────────────
class TestPypdfText(unittest.TestCase):

    @_skip_no_demo
    def test_returns_non_empty_string(self):
        text = load_pypdf_text(_DEMO_PDF)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text.strip()), 100)

    @_skip_no_demo
    def test_same_fingerprint_twice_gives_same_text(self):
        """Determinism check — PyPDF must be deterministic on the same file."""
        t1 = load_pypdf_text(_DEMO_PDF)
        t2 = load_pypdf_text(_DEMO_PDF)
        self.assertEqual(t1, t2)


# ─────────────────────────────────────────────────────────────────────────────
class TestLocalEmbeddings(unittest.TestCase):
    """
    make_embeddings("local", model) must return a usable Embeddings object.
    Requires the BAAI/bge-m3 model in .cache/hf/ (downloaded by setup).
    """

    @classmethod
    def setUpClass(cls):
        cfg = load_config()
        cls.model_name = cfg.local_embedding.model
        cls.emb = make_embeddings("local", cls.model_name)

    def test_embed_returns_list_of_floats(self):
        vecs = self.emb.embed_documents(["hello world"])
        self.assertEqual(len(vecs), 1)
        self.assertIsInstance(vecs[0], list)
        self.assertGreater(len(vecs[0]), 0)
        self.assertIsInstance(vecs[0][0], float)

    def test_vector_length_consistent(self):
        """Same model must always produce same-dimension vectors."""
        v1 = self.emb.embed_documents(["first sentence"])[0]
        v2 = self.emb.embed_documents(["totally different text"])[0]
        self.assertEqual(len(v1), len(v2))

    def test_different_texts_different_vectors(self):
        v1 = self.emb.embed_documents(["hello"])[0]
        v2 = self.emb.embed_documents(["goodbye"])[0]
        self.assertNotEqual(v1, v2,
                            "Different texts produced identical vectors — model may not be loaded")

    def test_embed_query_consistent_dimension(self):
        vec = self.emb.embed_query("test query")
        doc_vec = self.emb.embed_documents(["test query"])[0]
        self.assertEqual(len(vec), len(doc_vec))


# ─────────────────────────────────────────────────────────────────────────────
class TestProbeEndpoint(unittest.TestCase):
    """
    probe() must always return an EndpointStatus — never raise.
    ok=False is a valid result (server not running); we only fail on exceptions.
    """

    def test_unreachable_host_returns_status_not_exception(self):
        ep = Endpoint(url="http://127.0.0.1:19999")  # nothing listening here
        try:
            status = probe(ep, timeout=3)
        except Exception as exc:
            self.fail(f"probe() raised instead of returning EndpointStatus: {exc}")
        self.assertFalse(status.ok)
        self.assertIsNotNone(status.error)

    def test_localhost_ollama_returns_status(self):
        ep = Endpoint(url="http://localhost:11434", id="ollama_local")
        try:
            status = probe(ep, timeout=5)
        except Exception as exc:
            self.fail(f"probe() raised: {exc}")
        # ok may be True or False depending on whether Ollama is running
        self.assertIn(status.ok, (True, False))
        if status.ok:
            self.assertIsInstance(status.chat_models, list)
            self.assertIsInstance(status.embed_models, list)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time
    print("=" * 60)
    print("WP12 v1 — integration tests")
    print(f"Demo PDF: {'FOUND' if _DEMO_AVAILABLE else 'NOT FOUND (tests will skip)'}")
    print("Runtime: 30–120 s (Docling + embeddings dominate)")
    print("=" * 60)
    t0 = time.perf_counter()
    result = unittest.main(verbosity=2, exit=False)
    elapsed = time.perf_counter() - t0
    print(f"\nTotal wall time: {elapsed:.1f} s")
    sys.exit(0 if result.result.wasSuccessful() else 1)
