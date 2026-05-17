"""
v1 Bench Harness — automated quality + latency evaluation for the summarizer.

Imports from _bootstrap, summarizer, endpoints. Never imports from kb or app_summarizer.

Public API
----------
    load_scenarios(configs_dir) -> list[BenchScenario]
    bench(scenarios, pdfs, *, config, on_progress) -> list[BenchResult]

CLI
---
    python bench.py [--configs DIR] [--pdfs DIR] [--scenario NAME] [--compare FILE]
"""
import _bootstrap  # noqa: F401 — env setup; must come before heavy imports

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import yaml
from pydantic import BaseModel, Field

from endpoints import Config, load_config
from summarizer import SummarizeOptions, summarize, cache_path_for, fingerprint, safe_json_parse


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class BenchScenario(BaseModel):
    name: str
    options: SummarizeOptions
    repeat: int = 1


class BenchResult(BaseModel):
    scenario: str
    pdf: str
    pdf_fingerprint: str
    timestamp: str
    timing_sec: dict[str, float] = Field(default_factory=dict)
    numeric_coverage: float = 0.0       # matched / total numbers in summary (1.0 if no numbers)
    word_count_ratio: float = 0.0       # actual_words / max_words
    json_ok: bool = True                # safe_json_parse did NOT need fallback
    stability_cosine: float | None = None  # cosine similarity between repeat runs (None if repeat==1)
    reproducible: bool | None = None    # byte-identical across repeats when temperature==0
    summary_excerpt: str = ""           # first 200 chars for inspection
    unmatched_numbers_count: int = 0
    error: str | None = None


# ---------------------------------------------------------------------------
# Scenario loading
# ---------------------------------------------------------------------------

def load_scenarios(configs_dir: Path) -> list[BenchScenario]:
    """Read every YAML in configs_dir. Rejects scenarios with use_cache=True."""
    configs_dir = Path(configs_dir)
    if not configs_dir.exists():
        return []
    scenarios: list[BenchScenario] = []
    for yml in sorted(configs_dir.glob("*.yml")):
        with open(yml, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        # Each YAML may define one scenario or a list.
        items = raw if isinstance(raw, list) else [raw]
        for item in items:
            opts_raw = item.get("options", item)  # allow flat format
            if opts_raw.get("use_cache"):
                raise ValueError(
                    f"Scenario '{item.get('name', yml.stem)}' has use_cache=true — "
                    "the bench must measure the cold pipeline. Set use_cache: false."
                )
            opts_raw["use_cache"] = False
            opts = SummarizeOptions.model_validate(opts_raw)
            scenarios.append(BenchScenario(
                name=item.get("name", yml.stem),
                options=opts,
                repeat=item.get("repeat", 1),
            ))
    return scenarios


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _word_count(text: str) -> int:
    return len(str(text).split())


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _embed_text(text: str, cfg: Config) -> list[float]:
    """Embed a text string using the local HF model (no network)."""
    from endpoints import _get_hf_embedding
    model = _get_hf_embedding(cfg.local_embedding.model)
    vecs = model.embed_documents([text])
    return vecs[0] if vecs else []


# ---------------------------------------------------------------------------
# Main bench function
# ---------------------------------------------------------------------------

def bench(
    scenarios: list[BenchScenario],
    pdfs: list[Path],
    *,
    config: Config | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> list[BenchResult]:
    """Run every (pdf × scenario × repeat), compute metrics, append JSONL, return results."""
    cfg = config or load_config()
    runs_dir = cfg.bench.runs_dir
    runs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_file = runs_dir / f"{timestamp}.jsonl"

    results: list[BenchResult] = []

    for pdf_path in pdfs:
        fp = fingerprint(pdf_path)
        for scenario in scenarios:
            opts = scenario.options.model_copy(update={"use_cache": False})
            run_summaries: list[str] = []
            run_timings: list[dict] = []
            last_result: dict = {}
            last_error: str | None = None

            for rep in range(max(1, scenario.repeat)):
                label = f"{scenario.name} × {pdf_path.name} (rep {rep + 1}/{scenario.repeat})"
                if on_progress:
                    on_progress(f"Bench: {label}")
                print(f"\n── {label} ──")
                try:
                    r = summarize(pdf_path, opts, config=cfg)
                    last_result = r
                    run_summaries.append(r.get("summary", "") or "")
                    run_timings.append(r.get("_timing_sec", {}))
                except Exception as exc:
                    last_error = str(exc)
                    print(f"   ERROR: {exc}")
                    break

            # Compute metrics
            summary_text = last_result.get("summary", "") or ""
            unmatched = last_result.get("unmatched_numbers") or []
            numeric_claims = last_result.get("numeric_claims") or []
            total_nums = len(numeric_claims)
            matched_nums = total_nums - len(unmatched)
            numeric_coverage = (matched_nums / total_nums) if total_nums > 0 else 1.0

            actual_words = _word_count(summary_text)
            word_count_ratio = (actual_words / opts.max_words) if opts.max_words > 0 else 0.0

            # JSON quality — re-parse to check if fallback was used
            json_ok = True
            if summary_text:
                _, used_fallback = safe_json_parse(type("R", (), {"content": json.dumps(last_result)})())
                json_ok = not used_fallback

            # Stability (cosine between repeat run summaries)
            stability_cosine: float | None = None
            if len(run_summaries) >= 2 and run_summaries[0] and run_summaries[1]:
                try:
                    v1 = _embed_text(run_summaries[0], cfg)
                    v2 = _embed_text(run_summaries[1], cfg)
                    stability_cosine = round(_cosine(v1, v2), 4)
                except Exception:
                    pass

            # Reproducibility (byte-identical when temperature==0)
            reproducible: bool | None = None
            if opts.temperature == 0.0 and len(run_summaries) >= 2:
                reproducible = all(s == run_summaries[0] for s in run_summaries)

            avg_timing: dict[str, float] = {}
            if run_timings:
                all_keys = set().union(*run_timings)
                for k in all_keys:
                    vals = [t[k] for t in run_timings if k in t]
                    avg_timing[k] = round(sum(vals) / len(vals), 3)

            bres = BenchResult(
                scenario=scenario.name,
                pdf=pdf_path.name,
                pdf_fingerprint=fp,
                timestamp=timestamp,
                timing_sec=avg_timing,
                numeric_coverage=round(numeric_coverage, 4),
                word_count_ratio=round(word_count_ratio, 4),
                json_ok=json_ok,
                stability_cosine=stability_cosine,
                reproducible=reproducible,
                summary_excerpt=summary_text[:200],
                unmatched_numbers_count=len(unmatched),
                error=last_error,
            )
            results.append(bres)
            # Append to JSONL
            with open(run_file, "a", encoding="utf-8") as fh:
                fh.write(bres.model_dump_json() + "\n")

    print(f"\nBench run complete → {run_file}")
    return results


# ---------------------------------------------------------------------------
# Comparison table printer
# ---------------------------------------------------------------------------

def _print_comparison_table(results: list[BenchResult]) -> None:
    if not results:
        print("No results.")
        return
    col_w = 28
    header = (
        f"{'Scenario':<{col_w}} {'PDF':<20} "
        f"{'Total(s)':>8} {'LLM(s)':>7} {'NumCov':>7} {'Words%':>7} "
        f"{'JSON':>5} {'Stab':>6} {'Repro':>6}"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        total = r.timing_sec.get("total") or sum(r.timing_sec.values())
        llm = r.timing_sec.get("llm", 0.0)
        stab = f"{r.stability_cosine:.3f}" if r.stability_cosine is not None else "  n/a"
        repro = ("yes" if r.reproducible else "no") if r.reproducible is not None else "  n/a"
        err = f"  ERROR: {r.error[:40]}" if r.error else ""
        print(
            f"{r.scenario[:col_w]:<{col_w}} {r.pdf[:20]:<20} "
            f"{total:>8.1f} {llm:>7.1f} {r.numeric_coverage:>7.2%} "
            f"{r.word_count_ratio:>7.2%} {'✓' if r.json_ok else '✗':>5} "
            f"{stab:>6} {repro:>6}{err}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_main() -> None:
    parser = argparse.ArgumentParser(
        prog="bench",
        description="WP12 v1 Bench Harness — run summarizer scenarios over PDFs.",
    )
    parser.add_argument("--configs", metavar="DIR", default=None,
                        help="Directory of scenario YAML files (default: bench/configs/).")
    parser.add_argument("--pdfs", metavar="DIR", default=None,
                        help="Directory of reference PDFs (default: bench/pdfs/).")
    parser.add_argument("--scenario", metavar="NAME", action="append", dest="scenarios",
                        help="Run only this scenario (repeatable). Default: all.")
    parser.add_argument("--compare", metavar="FILE",
                        help="Print a comparison table from an existing JSONL run file.")
    args = parser.parse_args()

    cfg = load_config()

    if args.compare:
        comp_path = Path(args.compare)
        if not comp_path.exists():
            print(f"Error: {args.compare} not found.", file=sys.stderr)
            sys.exit(1)
        loaded: list[BenchResult] = []
        for line in comp_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                loaded.append(BenchResult.model_validate_json(line))
        _print_comparison_table(loaded)
        return

    configs_dir = Path(args.configs) if args.configs else cfg.bench.configs_dir
    pdfs_dir = Path(args.pdfs) if args.pdfs else cfg.bench.pdfs_dir

    scenarios = load_scenarios(configs_dir)
    if not scenarios:
        print(f"No scenario YAML files found in {configs_dir}.")
        sys.exit(1)
    if args.scenarios:
        scenarios = [s for s in scenarios if s.name in args.scenarios]
        if not scenarios:
            print(f"No scenarios matched: {args.scenarios}", file=sys.stderr)
            sys.exit(1)

    pdfs = sorted(pdfs_dir.glob("*.pdf")) if pdfs_dir.exists() else []
    if not pdfs:
        print(f"No PDFs found in {pdfs_dir}.")
        sys.exit(1)

    print(f"Bench: {len(scenarios)} scenario(s) × {len(pdfs)} PDF(s)")
    results = bench(scenarios, pdfs, config=cfg)
    _print_comparison_table(results)


if __name__ == "__main__":
    _cli_main()
