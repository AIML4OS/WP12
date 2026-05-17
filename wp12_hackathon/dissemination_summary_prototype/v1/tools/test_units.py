"""
Offline unit tests for v1 pure-function logic.

No LLM, no network, no file I/O beyond temp files.
Runtime: ~15 s  (dominated by torch/langchain import startup, not the tests).

Run:
    .conda\\python.exe tools\\test_units.py          (from v1/)
    .conda\\python.exe tools\\test_units.py -v       (verbose — shows each test name)
"""
import sys
import os
import json
import hashlib
import tempfile
import types
import unittest
from pathlib import Path

# ── ensure v1/ root is on sys.path ───────────────────────────────────────────
V1_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V1_ROOT))

import _bootstrap  # noqa: F401 — must run before heavy imports

from summarizer import (
    _extract_numeric_tokens,
    _numeric_token_in_text,
    _find_first_numeric_span,
    safe_json_parse,
    build_numeric_coverage,
    cache_path_for,
    SummarizeOptions,
    _params_hash,
)
from document_load import (
    fingerprint,
    _make_chunk_id,
    _extract_page_int,
)
from endpoints import load_config, Config, DoclingOptions, Endpoint


# ── helpers ──────────────────────────────────────────────────────────────────

def _fake_result(content: str):
    """Minimal stand-in for a LangChain AIMessage."""
    r = types.SimpleNamespace()
    r.content = content
    return r


def _make_pdf(content: bytes = b"%PDF-1.4 fake") -> Path:
    """Write a temp PDF-like file and return its Path."""
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    f.write(content)
    f.flush()
    f.close()
    return Path(f.name)


# ─────────────────────────────────────────────────────────────────────────────
class TestFingerprint(unittest.TestCase):
    def test_same_content_same_hash(self):
        p1 = _make_pdf(b"hello pdf")
        p2 = _make_pdf(b"hello pdf")
        try:
            self.assertEqual(fingerprint(p1), fingerprint(p2))
        finally:
            p1.unlink(missing_ok=True)
            p2.unlink(missing_ok=True)

    def test_different_content_different_hash(self):
        p1 = _make_pdf(b"pdf version A")
        p2 = _make_pdf(b"pdf version B")
        try:
            self.assertNotEqual(fingerprint(p1), fingerprint(p2))
        finally:
            p1.unlink(missing_ok=True)
            p2.unlink(missing_ok=True)

    def test_length_is_16_hex_chars(self):
        p = _make_pdf()
        try:
            fp = fingerprint(p)
            self.assertEqual(len(fp), 16)
            int(fp, 16)  # raises ValueError if not valid hex
        finally:
            p.unlink(missing_ok=True)

    def test_matches_manual_sha256(self):
        data = b"unit test content"
        p = _make_pdf(data)
        try:
            expected = hashlib.sha256(data).hexdigest()[:16]
            self.assertEqual(fingerprint(p), expected)
        finally:
            p.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
class TestMakeChunkId(unittest.TestCase):
    def test_format(self):
        cid = _make_chunk_id("abc123", 5, 7)
        self.assertEqual(cid, "abc123::p5::b0007")

    def test_block_zero_padded_to_4(self):
        self.assertTrue(_make_chunk_id("fp", 1, 0).endswith("::b0000"))
        self.assertTrue(_make_chunk_id("fp", 1, 9999).endswith("::b9999"))

    def test_page_not_padded(self):
        cid = _make_chunk_id("fp", 42, 0)
        self.assertIn("::p42::", cid)


# ─────────────────────────────────────────────────────────────────────────────
class TestExtractPageInt(unittest.TestCase):
    def test_flat_page_key_zero_based(self):
        # "page" is 0-based in PyPDF — function should add 1
        self.assertEqual(_extract_page_int({"page": 0}), 1)
        self.assertEqual(_extract_page_int({"page": 4}), 5)

    def test_page_number_one_based(self):
        self.assertEqual(_extract_page_int({"page_number": 3}), 3)

    def test_missing_returns_zero(self):
        self.assertEqual(_extract_page_int({}), 0)
        self.assertEqual(_extract_page_int({"unrelated": "key"}), 0)

    def test_string_page_value(self):
        # Some loaders return "3" as a string
        result = _extract_page_int({"page_number": "7"})
        self.assertEqual(result, 7)


# ─────────────────────────────────────────────────────────────────────────────
class TestExtractNumericTokens(unittest.TestCase):
    def test_plain_integers(self):
        toks = _extract_numeric_tokens("There are 42 items and 7 groups.")
        self.assertIn("42", toks)
        self.assertIn("7", toks)

    def test_decimal_comma_separator(self):
        # European format: 1.234,56
        toks = _extract_numeric_tokens("Total: 1.234,56 euros")
        self.assertTrue(any("1.234,56" in t or "1.234" in t for t in toks))

    def test_percentage(self):
        toks = _extract_numeric_tokens("Growth of 12% was observed.")
        self.assertTrue(any("12" in t for t in toks))

    def test_percentage_with_space(self):
        toks = _extract_numeric_tokens("Growth of 12 % was observed.")
        self.assertTrue(any("12" in t for t in toks))

    def test_no_numbers_returns_empty(self):
        self.assertEqual(_extract_numeric_tokens("No numbers here."), [])
        self.assertEqual(_extract_numeric_tokens(""), [])

    def test_deduplication(self):
        # Same number appearing twice → listed once
        toks = _extract_numeric_tokens("42 items plus 42 more.")
        self.assertEqual(toks.count("42"), 1)

    def test_order_preserved(self):
        toks = _extract_numeric_tokens("First 10 then 20 then 30.")
        nums = [t for t in toks if t in ("10", "20", "30")]
        self.assertEqual(nums, ["10", "20", "30"])


# ─────────────────────────────────────────────────────────────────────────────
class TestNumericTokenInText(unittest.TestCase):
    def test_exact_match(self):
        self.assertTrue(_numeric_token_in_text("value is 42.5 units", "42.5"))

    def test_not_present(self):
        self.assertFalse(_numeric_token_in_text("value is 43 units", "42"))

    def test_empty_inputs(self):
        self.assertFalse(_numeric_token_in_text("", "42"))
        self.assertFalse(_numeric_token_in_text("42", ""))

    def test_whitespace_normalisation(self):
        # "12 %" should match "12%" and vice versa
        self.assertTrue(_numeric_token_in_text("rate of 12%", "12 %"))
        self.assertTrue(_numeric_token_in_text("rate of 12 %", "12%"))


# ─────────────────────────────────────────────────────────────────────────────
class TestFindFirstNumericSpan(unittest.TestCase):
    def test_finds_position(self):
        text = "Total is 99 units"
        span = _find_first_numeric_span(text, "99")
        self.assertIsNotNone(span)
        start, end = span
        self.assertEqual(text[start:end], "99")

    def test_returns_none_when_absent(self):
        self.assertIsNone(_find_first_numeric_span("no match here", "42"))

    def test_empty_inputs(self):
        self.assertIsNone(_find_first_numeric_span("", "42"))
        self.assertIsNone(_find_first_numeric_span("42", ""))


# ─────────────────────────────────────────────────────────────────────────────
class TestSafeJsonParse(unittest.TestCase):
    def test_valid_json_no_fallback(self):
        payload = json.dumps({"summary": "hello", "keywords": [], "tags": []})
        parsed, used_fallback = safe_json_parse(_fake_result(payload))
        self.assertFalse(used_fallback)
        self.assertEqual(parsed["summary"], "hello")

    def test_fenced_json_uses_fallback(self):
        payload = '```json\n{"summary": "fenced", "keywords": [], "tags": []}\n```'
        parsed, used_fallback = safe_json_parse(_fake_result(payload))
        self.assertTrue(used_fallback)
        self.assertEqual(parsed["summary"], "fenced")

    def test_json_buried_in_prose_brace_count(self):
        payload = 'Sure, here you go: {"summary": "buried", "keywords": [], "tags": []}'
        parsed, used_fallback = safe_json_parse(_fake_result(payload))
        self.assertTrue(used_fallback)
        self.assertEqual(parsed["summary"], "buried")

    def test_total_garbage_returns_raw_string(self):
        parsed, used_fallback = safe_json_parse(_fake_result("not json at all"))
        self.assertTrue(used_fallback)
        self.assertIn("summary", parsed)  # fallback schema always has "summary"

    def test_string_input_without_content_attr(self):
        # If something without .content is passed, str() is used
        parsed, _ = safe_json_parse('{"summary": "direct str", "keywords": [], "tags": []}')
        self.assertEqual(parsed["summary"], "direct str")


# ─────────────────────────────────────────────────────────────────────────────
class TestBuildNumericCoverage(unittest.TestCase):
    def _make_source(self, sid, supports):
        return {"id": sid, "source": "doc.pdf", "location": "p.1",
                "excerpt": "...", "supports_numbers": supports}

    def test_all_matched(self):
        # _extract_numeric_tokens extracts "42%" with the percent sign intact,
        # so the source must list "42%" (not "42") in supports_numbers.
        sources = [self._make_source("1", ["42%", "100"])]
        claims, unmatched = build_numeric_coverage("Growth was 42% to 100 units.", sources)
        nums = {c["number"] for c in claims}
        self.assertIn("42%", nums)
        self.assertIn("100", nums)
        self.assertEqual(unmatched, [])

    def test_unmatched_detected(self):
        sources = [self._make_source("1", ["42"])]
        _, unmatched = build_numeric_coverage("Values 42 and 99 were noted.", sources)
        self.assertIn("99", unmatched)

    def test_empty_summary_no_claims(self):
        claims, unmatched = build_numeric_coverage("", [])
        self.assertEqual(claims, [])
        self.assertEqual(unmatched, [])

    def test_no_numbers_in_summary(self):
        sources = [self._make_source("1", [])]
        claims, unmatched = build_numeric_coverage("No numbers here.", sources)
        self.assertEqual(claims, [])
        self.assertEqual(unmatched, [])


# ─────────────────────────────────────────────────────────────────────────────
class TestCachePathFor(unittest.TestCase):
    def _base_opts(self, **overrides) -> SummarizeOptions:
        defaults = dict(
            llm_endpoint="http://localhost:11434",
            llm_model="llama3.1:8b",
            max_words=200, max_keywords=6, max_tags=5,
            out_lang="pt-pt", processing_mode="vector",
            pdf_loader="docling", temperature=0.1,
            use_cache=True,
        )
        defaults.update(overrides)
        return SummarizeOptions(**defaults)

    def _cfg(self) -> Config:
        return Config()

    def test_same_options_same_path(self):
        opts1 = self._base_opts()
        opts2 = self._base_opts()
        cfg = self._cfg()
        self.assertEqual(cache_path_for("fp16chars00000001", opts1, cfg),
                         cache_path_for("fp16chars00000001", opts2, cfg))

    def test_different_model_different_path(self):
        opts_a = self._base_opts(llm_model="llama3.1:8b")
        opts_b = self._base_opts(llm_model="mistral:7b")
        cfg = self._cfg()
        self.assertNotEqual(cache_path_for("fp", opts_a, cfg),
                            cache_path_for("fp", opts_b, cfg))

    def test_different_max_words_different_path(self):
        opts_a = self._base_opts(max_words=200)
        opts_b = self._base_opts(max_words=400)
        cfg = self._cfg()
        self.assertNotEqual(cache_path_for("fp", opts_a, cfg),
                            cache_path_for("fp", opts_b, cfg))

    def test_different_fingerprint_different_path(self):
        opts = self._base_opts()
        cfg = self._cfg()
        self.assertNotEqual(cache_path_for("aaaaaaaaaaaaaaaa", opts, cfg),
                            cache_path_for("bbbbbbbbbbbbbbbb", opts, cfg))

    def test_path_format(self):
        opts = self._base_opts()
        cfg = self._cfg()
        p = cache_path_for("abcd1234abcd1234", opts, cfg)
        self.assertTrue(p.name.startswith("abcd1234abcd1234__"))
        self.assertTrue(p.suffix == ".json")
        self.assertIn("summaries", str(p))

    def test_ocr_on_vs_off_different_path(self):
        opts_off = self._base_opts(do_ocr=False)
        opts_on  = self._base_opts(do_ocr=True, ocr_engine="easyocr")
        cfg = self._cfg()
        self.assertNotEqual(cache_path_for("fp", opts_off, cfg),
                            cache_path_for("fp", opts_on, cfg))

    def test_different_ocr_engines_different_path(self):
        opts_easy   = self._base_opts(do_ocr=True, ocr_engine="easyocr")
        opts_rapid  = self._base_opts(do_ocr=True, ocr_engine="rapidocr")
        cfg = self._cfg()
        self.assertNotEqual(cache_path_for("fp", opts_easy, cfg),
                            cache_path_for("fp", opts_rapid, cfg))


# ─────────────────────────────────────────────────────────────────────────────
class TestDoclingOptions(unittest.TestCase):
    def test_defaults(self):
        d = DoclingOptions()
        self.assertFalse(d.do_ocr)
        self.assertEqual(d.ocr_engine, "easyocr")
        self.assertEqual(d.table_mode, "fast")
        self.assertEqual(d.convert_timeout_sec, 900)

    def test_invalid_ocr_engine_rejected(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            DoclingOptions(ocr_engine="tesseractXXX")

    def test_invalid_table_mode_rejected(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            DoclingOptions(table_mode="turbo")


# ─────────────────────────────────────────────────────────────────────────────
class TestEndpointModel(unittest.TestCase):
    def test_id_derived_from_host_when_omitted(self):
        e = Endpoint(url="http://localhost:11434")
        self.assertEqual(e.id, "localhost")

    def test_explicit_id_preserved(self):
        e = Endpoint(url="http://localhost:11434", id="ollama_local")
        self.assertEqual(e.id, "ollama_local")

    def test_name_falls_back_to_id(self):
        e = Endpoint(url="http://localhost:11434", id="ollama_local")
        self.assertEqual(e.name, "ollama_local")

    def test_api_key_from_env(self, monkeypatch=None):
        import os
        os.environ["_TEST_KEY_XYZ"] = "secret123"
        try:
            e = Endpoint(url="http://example.com", auth_env="_TEST_KEY_XYZ")
            self.assertEqual(e.api_key(), "secret123")
        finally:
            del os.environ["_TEST_KEY_XYZ"]

    def test_api_key_none_when_env_missing(self):
        e = Endpoint(url="http://example.com", auth_env="__DEFINITELY_NOT_SET__")
        self.assertIsNone(e.api_key())

    def test_duplicate_endpoint_ids_rejected(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            Config(endpoints=[
                Endpoint(url="http://a.com", id="same"),
                Endpoint(url="http://b.com", id="same"),
            ])


# ─────────────────────────────────────────────────────────────────────────────
class TestLoadConfig(unittest.TestCase):
    def test_no_file_returns_defaults(self):
        cfg = load_config(Path("/nonexistent/path/config.yml"))
        self.assertEqual(cfg.defaults.max_words, 200)
        self.assertEqual(cfg.defaults.processing_mode, "vector")
        self.assertEqual(cfg.docling.ocr_engine, "easyocr")
        self.assertEqual(cfg.docling.table_mode, "fast")

    def test_yaml_overrides_defaults(self):
        import yaml
        data = {"defaults": {"max_words": 500, "out_lang": "en"},
                "docling": {"table_mode": "accurate", "ocr_engine": "rapidocr"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(data, f)
            tmp = Path(f.name)
        try:
            cfg = load_config(tmp)
            self.assertEqual(cfg.defaults.max_words, 500)
            self.assertEqual(cfg.defaults.out_lang, "en")
            self.assertEqual(cfg.docling.table_mode, "accurate")
            self.assertEqual(cfg.docling.ocr_engine, "rapidocr")
        finally:
            tmp.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("WP12 v1 — offline unit tests")
    print("(~15 s startup = torch/langchain loading, not tests)")
    print("=" * 60)
    unittest.main(verbosity=2)
