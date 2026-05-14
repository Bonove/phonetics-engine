#!/usr/bin/env python3
"""
Latency benchmark: phonetics engine vs. Levenshtein baseline.

Measures per-query latency at realistic corpus sizes.
Index build time is reported separately from query time.

Usage:
    uv run python eval/benchmark.py             # default sizes
    uv run python eval/benchmark.py --queries 200
"""

from __future__ import annotations

import argparse
import random
import statistics
import time
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from phonetics_engine.decision import classify
from phonetics_engine.enums import MatchField
from phonetics_engine.loader import EmployeeRecord
from phonetics_engine.matcher import build_employee_index
from phonetics_engine.models import Thresholds

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_EVAL_DIR = Path(__file__).parent

CORPUS_SIZES = [100, 500, 1_000, 5_000, 10_000]
DEFAULT_N_QUERIES = 100
TOP_K = 5
THRESHOLDS = Thresholds(min_match=0.60, high_confidence=0.65, ambiguity_margin=0.08)
LEV_THRESHOLD = 0.67
LANGUAGE = "nl"

# Seed corpus — real Dutch surnames to keep phonemisation realistic
_SEED_FILE = _EVAL_DIR / "surnames.txt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_seed() -> list[str]:
    names = [l.strip() for l in _SEED_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not names:
        raise RuntimeError(f"Seed corpus empty: {_SEED_FILE}")
    return names


def _build_corpus(seed: list[str], size: int) -> list[str]:
    """Repeat/tile the seed list to reach `size`, then deduplicate by index."""
    rng = random.Random(42)
    tiled = (seed * (size // len(seed) + 1))[:size]
    # Add a numeric suffix to avoid duplicates when seed is smaller than size
    return [f"{n}{i // len(seed)}" if i >= len(seed) else n for i, n in enumerate(tiled)]


def _make_records(names: list[str]) -> list[EmployeeRecord]:
    return [
        EmployeeRecord(id=n, first_name="", infix=None, last_name=n, full_name=n) for n in names
    ]


def _sample_queries(corpus: list[str], n: int, rng: random.Random) -> list[str]:
    return [rng.choice(corpus) for _ in range(n)]


# ---------------------------------------------------------------------------
# Benchmark runs
# ---------------------------------------------------------------------------


def bench_engine(corpus: list[str], queries: list[str]) -> tuple[float, float]:
    """Returns (build_ms, mean_query_ms)."""
    records = _make_records(corpus)

    t0 = time.perf_counter()
    index = build_employee_index(records, match_fields=[MatchField.LAST_NAME], language=LANGUAGE)
    build_ms = (time.perf_counter() - t0) * 1000

    # Warmup
    for q in queries[:5]:
        scored = index.search(q, TOP_K)
        classify(q, scored, THRESHOLDS, TOP_K)

    times = []
    for q in queries:
        t0 = time.perf_counter()
        scored = index.search(q, TOP_K)
        classify(q, scored, THRESHOLDS, TOP_K)
        times.append((time.perf_counter() - t0) * 1000)

    return build_ms, statistics.median(times)


def bench_levenshtein(corpus: list[str], queries: list[str]) -> tuple[float, float]:
    """Returns (build_ms, mean_query_ms). Build is trivial (lowercase corpus)."""
    t0 = time.perf_counter()
    lowered = [n.lower() for n in corpus]
    build_ms = (time.perf_counter() - t0) * 1000

    # Warmup
    for q in queries[:5]:
        ql = q.lower()
        max((Levenshtein.normalized_similarity(ql, c), c) for c in lowered)

    times = []
    for q in queries:
        t0 = time.perf_counter()
        ql = q.lower()
        best_score, _ = max((Levenshtein.normalized_similarity(ql, c), c) for c in lowered)
        _ = best_score >= LEV_THRESHOLD
        times.append((time.perf_counter() - t0) * 1000)

    return build_ms, statistics.median(times)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Latency benchmark: engine vs. Levenshtein")
    parser.add_argument(
        "--queries",
        type=int,
        default=DEFAULT_N_QUERIES,
        help="Number of queries per benchmark run (default: %(default)s)",
    )
    args = parser.parse_args()

    seed = _load_seed()
    rng = random.Random(0)

    # Pre-warm: force espeak-ng library load before timing starts
    print("Warming up espeak-ng...", end=" ", flush=True)
    _warm_records = _make_records(seed[:5])
    build_employee_index(_warm_records, match_fields=[MatchField.LAST_NAME], language=LANGUAGE)
    print("done.\n")

    header = (
        f"{'Corpus':>8}  {'System':<22}  {'Build (ms)':>10}  {'Query p50 (ms)':>14}  {'QPS':>8}"
    )
    print()
    print(header)
    print("-" * len(header))

    for size in CORPUS_SIZES:
        corpus = _build_corpus(seed, size)
        queries = _sample_queries(corpus, args.queries, rng)

        eng_build, eng_query = bench_engine(corpus, queries)
        lev_build, lev_query = bench_levenshtein(corpus, queries)

        eng_qps = 1000 / eng_query if eng_query > 0 else float("inf")
        lev_qps = 1000 / lev_query if lev_query > 0 else float("inf")

        print(
            f"{size:>8}  {'Phonetics engine':<22}  {eng_build:>10.1f}  {eng_query:>14.3f}  {eng_qps:>8.0f}"
        )
        print(
            f"{'':>8}  {'Levenshtein':<22}  {lev_build:>10.3f}  {lev_query:>14.3f}  {lev_qps:>8.0f}"
        )
        print()


if __name__ == "__main__":
    main()
