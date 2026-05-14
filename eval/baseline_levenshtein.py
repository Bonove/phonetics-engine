#!/usr/bin/env python3
"""
Levenshtein baseline for surname matching.

Uses rapidfuzz (the same package as twyn) for fast normalized Levenshtein
similarity.  This is the "naive" baseline: no phoneme pipeline, no IPA,
no FAISS — just edit distance on the raw lowercased string.

Decision rule: match if similarity >= threshold, no-match otherwise.
Since there is no AMBIGUOUS concept here, every match above the threshold
is a direct hit (or miss if it returns the wrong name).

Usage:
    uv run python eval/baseline_levenshtein.py                    # default threshold sweep
    uv run python eval/baseline_levenshtein.py --threshold 0.80   # fixed threshold
    uv run python eval/baseline_levenshtein.py --output results.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

from rapidfuzz.distance import Levenshtein

# ---------------------------------------------------------------------------
# Paths (mirror evaluate.py layout)
# ---------------------------------------------------------------------------
_EVAL_DIR = Path(__file__).parent

GOLD_FILES: dict[str, Path] = {
    "Dutch": _EVAL_DIR / "gold_dutch.csv",
    "French": _EVAL_DIR / "gold_french.csv",
}
BACKGROUND_FILES: dict[str, Path] = {
    "Dutch": _EVAL_DIR / "surnames.txt",
    "French": _EVAL_DIR / "surnames_fr.txt",
}

DEFAULT_THRESHOLD = 0.80  # normalized Levenshtein similarity cut-off

# Same OOV names as evaluate.py — must not match anything in any language corpus.
_NEGATIVE_NAMES = [
    "Shakespeare",
    "Jefferson",
    "Roosevelt",
    "Churchill",
    "Wellington",
    "Hamilton",
    "Fitzgerald",
    "MacGregor",
    "Yamamoto",
    "Kowalski",
    "Fernandez",
    "Petrov",
    "Nakamura",
    "Okonkwo",
    "Papadopoulos",
    "Szczepanski",
    "Vanderbilt",
    "Worthington",
    "Nkrumah",
    "Brzezinski",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GoldPair = tuple[str, str, str, str]  # (original, query, category, note)


def _load_gold(path: Path) -> list[GoldPair]:
    with path.open(encoding="utf-8") as f:
        return [
            (row["original"], row["query"], row["category"], row["note"])
            for row in csv.DictReader(f)
        ]


def _load_corpus(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def _normalized_similarity(a: str, b: str) -> float:
    """Normalized Levenshtein similarity in [0, 1]. 1 = identical."""
    return Levenshtein.normalized_similarity(a.lower(), b.lower())


def _best_match(query: str, corpus: list[str]) -> tuple[str, float]:
    """Return (best_name, similarity) from corpus."""
    best_name, best_sim = "", 0.0
    for name in corpus:
        sim = _normalized_similarity(query, name)
        if sim > best_sim:
            best_sim = sim
            best_name = name
    return best_name, best_sim


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def evaluate(
    pairs: list[GoldPair],
    corpus: list[str],
    threshold: float,
) -> tuple[list[dict], dict]:
    """
    Evaluate Levenshtein baseline on *pairs* against *corpus*.

    A query is a TP if the best match is the expected original AND
    similarity >= threshold.  It is a FP if similarity >= threshold but the
    best match is a *different* name.  It is a FN if similarity < threshold
    (no match returned).
    """
    rows: list[dict] = []
    for original, query, category, note in pairs:
        best_name, best_sim = _best_match(query, corpus)
        matched = best_sim >= threshold

        if matched and best_name == original:
            outcome = "TP"
        elif matched:
            outcome = "FP"
        else:
            outcome = "FN"

        rows.append(
            {
                "original": original,
                "query": query,
                "category": category,
                "note": note,
                "best_match": best_name,
                "best_sim": best_sim,
                "matched": matched,
                "outcome": outcome,
            }
        )

    tp = sum(1 for r in rows if r["outcome"] == "TP")
    fp = sum(1 for r in rows if r["outcome"] == "FP")
    fn = sum(1 for r in rows if r["outcome"] == "FN")
    p, r, f1 = _prf(tp, fp, fn)

    by_cat: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for row in rows:
        by_cat[row["category"]][row["outcome"].lower()] += 1

    return rows, {
        "threshold": threshold,
        "precision": p,
        "recall": r,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total": len(pairs),
        "by_category": {
            cat: dict(zip(("precision", "recall", "f1"), _prf(d["tp"], d["fp"], d["fn"])))
            | {"tp": d["tp"], "fp": d["fp"], "fn": d["fn"]}
            for cat, d in sorted(by_cat.items())
        },
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def print_report(lang: str, summary: dict, rows: list[dict]) -> None:
    flag = {"Dutch": "NL", "French": "FR"}.get(lang, "")
    threshold = summary["threshold"]
    print(f"\n{'=' * 72}")
    print(f"  [{flag}]  {lang.upper()}  (threshold={threshold:.2f}, {summary['total']} pairs)")
    print(f"{'=' * 72}")
    print(
        f"  Overall ->  Precision: {summary['precision'] * 100:.1f}%"
        f"   Recall: {summary['recall'] * 100:.1f}%"
        f"   F1: {summary['f1'] * 100:.1f}%"
        f"   (TP={summary['tp']} FP={summary['fp']} FN={summary['fn']})"
    )

    print(f"\n  {'Category':<18} {'Prec':>7}  {'Recall':>7}  {'F1':>7}   TP  FP  FN")
    print(f"  {'-' * 60}")
    for cat, s in summary["by_category"].items():
        print(
            f"  {cat:<18} {s['precision'] * 100:>6.1f}%  {s['recall'] * 100:>6.1f}%"
            f"  {s['f1'] * 100:>6.1f}%   {s['tp']:>2}  {s['fp']:>2}  {s['fn']:>2}"
        )

    misses = [r for r in rows if r["outcome"] != "TP"]
    if misses:
        print(f"\n  Misses ({len(misses)}/{summary['total']}):")
        for r in misses:
            outcome_tag = f"[{r['outcome']}]"
            print(
                f"    '{r['query']}' (want '{r['original']}') "
                f"-> '{r['best_match']}' (sim={r['best_sim']:.3f}) {outcome_tag}"
                f"  # {r['note']}"
            )


def print_combined(summaries: dict[str, dict]) -> None:
    total_tp = sum(s["tp"] for s in summaries.values())
    total_fp = sum(s["fp"] for s in summaries.values())
    total_fn = sum(s["fn"] for s in summaries.values())
    combined_p, combined_r, combined_f1 = _prf(total_tp, total_fp, total_fn)
    macro_f1 = sum(s["f1"] for s in summaries.values()) / len(summaries)

    f1_values = [s["f1"] for s in summaries.values()]
    gen_ratio = min(f1_values) / max(f1_values) if max(f1_values) > 0 else 0.0
    if gen_ratio >= 0.90:
        gen_label = "LANGUAGE-NEUTRAL"
    elif gen_ratio >= 0.75:
        gen_label = "MILD BIAS"
    elif gen_ratio >= 0.55:
        gen_label = "SIGNIFICANT BIAS"
    else:
        gen_label = "STRONGLY BIASED"

    print(f"\n{'=' * 72}")
    print("  COMBINED (LEVENSHTEIN BASELINE)")
    print(f"{'=' * 72}")
    print(f"  {'Language':<12} {'Prec':>7}  {'Recall':>7}  {'F1':>7}  {'Pairs':>6}")
    print(f"  {'-' * 52}")
    for lang, s in summaries.items():
        print(
            f"  {lang:<12} {s['precision'] * 100:>6.1f}%  {s['recall'] * 100:>6.1f}%"
            f"  {s['f1'] * 100:>6.1f}%  {s['total']:>6}"
        )
    print(f"  {'-' * 52}")
    print(
        f"  {'Micro avg':<12} {combined_p * 100:>6.1f}%  {combined_r * 100:>6.1f}%"
        f"  {combined_f1 * 100:>6.1f}%  {sum(s['total'] for s in summaries.values()):>6}"
    )
    print(f"  {'Macro F1':<12} {'':>7}  {'':>7}  {macro_f1 * 100:>6.1f}%")
    print(f"\n  Generalizability (min F1 / max F1): {gen_ratio:.2f}  ->  {gen_label}")
    print(f"{'=' * 72}")


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------


def sweep(gold_files: dict[str, Path], background_files: dict[str, Path]) -> None:
    """
    Find the best threshold for each language and combined Macro F1.
    Builds corpora once, then replays the decision at every threshold in 0.01 steps.
    """
    THRESHOLDS = [round(t / 100, 2) for t in range(50, 100)]

    # Pre-load everything
    lang_data: dict[str, tuple[list[GoldPair], list[str]]] = {}
    for lang, gpath in gold_files.items():
        pairs = _load_gold(gpath)
        bg_path = background_files.get(lang)
        background = _load_corpus(bg_path) if bg_path and bg_path.exists() else []
        originals = list(dict.fromkeys(p[0] for p in pairs))
        corpus = list(dict.fromkeys(background + originals))
        lang_data[lang] = (pairs, corpus)

    # Pre-compute all similarities (expensive part — done once per pair)
    print("  Pre-computing Levenshtein similarities...", flush=True)
    lang_sims: dict[str, list[tuple[str, str, str, list[tuple[str, float]]]]] = {}
    lang_neg_sims: dict[str, dict[str, list[tuple[str, float]]]] = {}
    for lang, (pairs, corpus) in lang_data.items():
        cache = []
        for original, query, category, note in pairs:
            sims = [(name, _normalized_similarity(query, name)) for name in corpus]
            sims.sort(key=lambda x: x[1], reverse=True)
            cache.append((original, query, category, sims))
        lang_sims[lang] = cache

        neg_cache: dict[str, list[tuple[str, float]]] = {}
        for neg in _NEGATIVE_NAMES:
            sims = [(name, _normalized_similarity(neg, name)) for name in corpus]
            sims.sort(key=lambda x: x[1], reverse=True)
            neg_cache[neg] = sims
        lang_neg_sims[lang] = neg_cache

    langs = list(gold_files.keys())
    results: list[dict] = []

    for t in THRESHOLDS:
        lang_stats: dict[str, dict] = {}
        total_neg_fp = 0
        for lang, cache in lang_sims.items():
            tp = fp = fn = 0
            for original, _query, _cat, sims in cache:
                best_name, best_sim = sims[0]
                if best_sim >= t:
                    if best_name == original:
                        tp += 1
                    else:
                        fp += 1
                else:
                    fn += 1
            p, r, f1 = _prf(tp, fp, fn)

            # Negative control: how many OOV names incorrectly match this corpus?
            neg_fp = sum(1 for neg in _NEGATIVE_NAMES if lang_neg_sims[lang][neg][0][1] >= t)
            total_neg_fp += neg_fp

            lang_stats[lang] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": p,
                "recall": r,
                "f1": f1,
                "neg_fp": neg_fp,
            }

        f1_values = [s["f1"] for s in lang_stats.values()]
        macro_f1 = sum(f1_values) / len(f1_values)
        gen_ratio = min(f1_values) / max(f1_values) if max(f1_values) > 0 else 0.0
        total_neg = len(_NEGATIVE_NAMES) * len(langs)
        neg_fp_rate = total_neg_fp / total_neg
        results.append(
            {
                "threshold": t,
                "macro_f1": macro_f1,
                "gen_ratio": gen_ratio,
                "neg_fp_rate": neg_fp_rate,
                **{f"f1_{lang}": s["f1"] for lang, s in lang_stats.items()},
                **{f"prec_{lang}": s["precision"] for lang, s in lang_stats.items()},
                **{f"rec_{lang}": s["recall"] for lang, s in lang_stats.items()},
            }
        )

    results.sort(key=lambda x: x["macro_f1"], reverse=True)

    bar = "=" * 72
    print(f"\n{bar}")
    print("  LEVENSHTEIN BASELINE — THRESHOLD SWEEP")
    print(bar)

    col_heads = f"  {'thresh':>6}  "
    for lang in langs:
        col_heads += f"{'F1_' + lang[:2].upper():>7}  "
    col_heads += f"{'MacroF1':>8}  {'Gen':>5}  {'FP_neg':>6}"
    print(f"\n  All thresholds (0.50 – 0.99):\n{col_heads}")
    print(f"  {'-' * 65}")
    for r in results:
        row = f"  {r['threshold']:>6.2f}  "
        for lang in langs:
            row += f"{r[f'f1_{lang}'] * 100:>6.1f}%  "
        row += (
            f"{r['macro_f1'] * 100:>7.1f}%  {r['gen_ratio']:>5.2f}  {r['neg_fp_rate'] * 100:>5.1f}%"
        )
        print(row)

    best = results[0]
    print(
        f"\n  Best overall threshold: {best['threshold']:.2f}  ->  Macro F1 = {best['macro_f1'] * 100:.1f}%  (OOV FP={best['neg_fp_rate'] * 100:.1f}%)"
    )
    for lang in langs:
        print(
            f"    {lang}: F1={best[f'f1_{lang}'] * 100:.1f}%"
            f"  P={best[f'prec_{lang}'] * 100:.1f}%"
            f"  R={best[f'rec_{lang}'] * 100:.1f}%"
        )

    zero_fp = [r for r in results if r["neg_fp_rate"] == 0.0]
    if zero_fp:
        best_zero = zero_fp[0]
        print(
            f"\n  Best at 0% OOV FP:      {best_zero['threshold']:.2f}  ->  Macro F1 = {best_zero['macro_f1'] * 100:.1f}%"
        )
        for lang in langs:
            print(
                f"    {lang}: F1={best_zero[f'f1_{lang}'] * 100:.1f}%"
                f"  P={best_zero[f'prec_{lang}'] * 100:.1f}%"
                f"  R={best_zero[f'rec_{lang}'] * 100:.1f}%"
            )
    print(bar)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Levenshtein baseline for surname matching evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python eval/baseline_levenshtein.py                    # sweep, find best threshold
  uv run python eval/baseline_levenshtein.py --threshold 0.80   # fixed threshold
  uv run python eval/baseline_levenshtein.py --output results.csv
        """,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Fixed normalized Levenshtein similarity threshold (0–1). "
            "Omit to run the full sweep (default)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write raw per-query results to this CSV file (only with --threshold).",
    )
    args = parser.parse_args()

    for path in [*GOLD_FILES.values(), *BACKGROUND_FILES.values()]:
        if not path.exists():
            print(f"ERROR: file not found: {path}", file=sys.stderr)
            sys.exit(1)

    if args.threshold is None:
        sweep(GOLD_FILES, BACKGROUND_FILES)
        return

    threshold = args.threshold
    if not (0.0 < threshold < 1.0):
        print("ERROR: --threshold must be between 0 and 1 (exclusive)", file=sys.stderr)
        sys.exit(1)

    all_rows: list[dict] = []
    summaries: dict[str, dict] = {}

    for lang, gpath in GOLD_FILES.items():
        pairs = _load_gold(gpath)
        bg_path = BACKGROUND_FILES[lang]
        background = _load_corpus(bg_path)
        originals = list(dict.fromkeys(p[0] for p in pairs))
        corpus = list(dict.fromkeys(background + originals))

        print(f"[{lang}] {len(pairs)} pairs, corpus size={len(corpus)}")
        rows, summary = evaluate(pairs, corpus, threshold)
        summaries[lang] = summary
        print_report(lang, summary, rows)
        all_rows.extend({"lang": lang, **r} for r in rows)

    print_combined(summaries)

    if args.output:
        with args.output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n  Raw results saved -> {args.output}")


if __name__ == "__main__":
    main()
