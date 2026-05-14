#!/usr/bin/env python3
"""
Offline evaluation of phonetics-engine surname matching.

Imports engine modules directly — no HTTP server or database needed.

Usage:
    uv run python eval/evaluate.py                    # gold mode (default)
    uv run python eval/evaluate.py --mode synthetic   # rule-based distortion mode
    uv run python eval/evaluate.py --sweep            # threshold sweep (synthetic)
    uv run python eval/evaluate.py --output results.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

from phonetics_engine.decision import classify
from phonetics_engine.enums import Decision, MatchField
from phonetics_engine.loader import EmployeeRecord
from phonetics_engine.matcher import build_employee_index
from phonetics_engine.models import Thresholds

# ---------------------------------------------------------------------------
# Defaults (mirror config.py employee values)
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLDS = Thresholds(
    min_match=0.60,
    high_confidence=0.65,
    ambiguity_margin=0.08,
)

MATCH_FIELDS = [MatchField.LAST_NAME, MatchField.FULL_NAME]
TOP_K = 5

_GOLD_DIR = Path(__file__).parent

# espeak-ng language code per gold set
LANG_ESPEAK: dict[str, str] = {
    "Dutch": "nl",
    "French": "fr-fr",
}

# Per-language background corpus (realistic same-language names, no gold originals)
LANG_BACKGROUND: dict[str, Path] = {
    "Dutch": _GOLD_DIR / "surnames.txt",
    "French": _GOLD_DIR / "surnames_fr.txt",
}

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _make_records(names: list[str]) -> list[EmployeeRecord]:
    """Wrap each name as a minimal EmployeeRecord; id == name for lookup."""
    return [
        EmployeeRecord(id=n, first_name="", infix=None, last_name=n, full_name=n) for n in names
    ]


def _query(index, query: str, thresholds: Thresholds) -> tuple[Decision, list]:
    scored = index.search(query, TOP_K)
    return classify(query, scored, thresholds, TOP_K)


# ---------------------------------------------------------------------------
# GOLD EVALUATION
# ---------------------------------------------------------------------------

GoldPair = tuple[str, str, str, str]  # (original, query, category, note)


def load_gold(path: Path) -> list[GoldPair]:
    rows = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["original"], row["query"], row["category"], row["note"]))
    return rows


def evaluate_gold(
    pairs: list[GoldPair],
    thresholds: Thresholds,
    background_corpus: list[str] | None = None,
    language: str = "nl",
) -> tuple[list[dict], dict]:
    """
    Build an index from the unique originals in *pairs* (plus an optional
    background corpus for realistic corpus-size conditions), query each STT
    variant, and compute P/R/F1 overall and per error category.

    The *background_corpus* is important: with only 25 gold names the engine
    rarely fires AMBIGUOUS (no close runner-ups), so distorted queries hit the
    [min_match, high_confidence) dead-zone and return NO_MATCH even though the
    raw similarity is well above min_match.  A realistic background restores
    the competitive pressure that the engine was designed to operate under.
    """
    originals = list(dict.fromkeys(p[0] for p in pairs))  # unique, order-preserving
    index_names = list(dict.fromkeys((background_corpus or []) + originals))
    index = build_employee_index(
        _make_records(index_names), match_fields=MATCH_FIELDS, language=language
    )

    rows: list[dict] = []
    for original, query, category, note in pairs:
        # Raw search — used to surface the score for the expected match even
        # when classify() returns NO_MATCH (dead-zone visibility).
        raw_scored = index.search(query, max(TOP_K, len(index_names)))
        raw_score_for_original = next((c.score for c in raw_scored if c.id == original), 0.0)

        decision, matches = classify(query, raw_scored[:TOP_K], thresholds, TOP_K)
        hit = any(m.id == original for m in matches)
        rows.append(
            {
                "original": original,
                "query": query,
                "category": category,
                "note": note,
                "decision": str(decision),
                "top_match": matches[0].id if matches else "",
                "top_score": matches[0].score if matches else 0.0,
                "raw_score": raw_score_for_original,
                "hit": hit,
            }
        )

    tp = sum(1 for r in rows if r["hit"])
    fp_wrong = sum(1 for r in rows if not r["hit"] and r["decision"] != str(Decision.NO_MATCH))
    fn = sum(1 for r in rows if not r["hit"] and r["decision"] == str(Decision.NO_MATCH))
    p, r, f1 = _prf(tp, fp_wrong, fn)

    by_cat: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fp_wrong": 0, "fn": 0})
    for row in rows:
        cat = row["category"]
        if row["hit"]:
            by_cat[cat]["tp"] += 1
        elif row["decision"] != str(Decision.NO_MATCH):
            by_cat[cat]["fp_wrong"] += 1
        else:
            by_cat[cat]["fn"] += 1

    return rows, {
        "precision": p,
        "recall": r,
        "f1": f1,
        "tp": tp,
        "fp_wrong": fp_wrong,
        "fn": fn,
        "total": len(pairs),
        "by_category": {
            cat: dict(zip(("precision", "recall", "f1"), _prf(d["tp"], d["fp_wrong"], d["fn"])))
            | {"tp": d["tp"], "fp_wrong": d["fp_wrong"], "fn": d["fn"]}
            for cat, d in sorted(by_cat.items())
        },
    }


def print_gold_report(lang: str, summary: dict, rows: list[dict], thresholds: Thresholds) -> None:
    flag = {"Dutch": "NL", "French": "FR"}.get(lang, "")
    print(f"\n{'=' * 72}")
    print(f"  [{flag}]  {lang.upper()} GOLD SET  ({summary['total']} pairs)")
    print(f"{'=' * 72}")
    print(
        f"  Thresholds: min={thresholds.min_match}"
        f"  high={thresholds.high_confidence}"
        f"  margin={thresholds.ambiguity_margin}"
    )
    print(
        f"  Overall ->  Precision: {summary['precision'] * 100:.1f}%"
        f"   Recall: {summary['recall'] * 100:.1f}%"
        f"   F1: {summary['f1'] * 100:.1f}%"
        f"   (TP={summary['tp']} FP={summary['fp_wrong']} FN={summary['fn']})"
    )

    print(f"\n  {'Category':<18} {'Prec':>7}  {'Recall':>7}  {'F1':>7}   TP  FP  FN")
    print(f"  {'-' * 60}")
    for cat, s in summary["by_category"].items():
        print(
            f"  {cat:<18} {s['precision'] * 100:>6.1f}%  {s['recall'] * 100:>6.1f}%"
            f"  {s['f1'] * 100:>6.1f}%   {s['tp']:>2}  {s['fp_wrong']:>2}  {s['fn']:>2}"
        )

    misses = [r for r in rows if not r["hit"]]
    if misses:
        print(f"\n  Misses ({len(misses)}/{summary['total']}):")
        for r in misses:
            verdict = f"-> '{r['top_match']}'" if r["top_match"] else "-> NO_MATCH"
            raw = r.get("raw_score", 0.0)
            # Flag dead-zone cases: engine found it but thresholds blocked it
            zone = " [DEAD ZONE]" if raw >= thresholds.min_match else ""
            print(
                f"    '{r['query']}' (want '{r['original']}') {verdict}"
                f"  [{r['decision']}, top={r['top_score']:.3f}, raw={raw:.3f}]{zone}"
                f"  # {r['note']}"
            )


def print_combined_report(summaries: dict[str, dict]) -> None:
    total_tp = sum(s["tp"] for s in summaries.values())
    total_fp = sum(s["fp_wrong"] for s in summaries.values())
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
    print("  COMBINED RESULTS & GENERALIZABILITY")
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
    print("  (1.00 = identical performance across languages, <0.75 = language-specific)")
    print(f"{'=' * 72}")


# ---------------------------------------------------------------------------
# SYNTHETIC EVALUATION (kept for development / threshold sweeping)
# ---------------------------------------------------------------------------

_SYNTHETIC_RULES: list[list[tuple[str, str]]] = [
    [
        ("ij", "ei"),
        ("ei", "ij"),
        ("au", "ou"),
        ("ou", "au"),
        ("v", "f"),
        ("f", "v"),
        ("d", "t"),
        ("t", "d"),
        ("b", "p"),
        ("p", "b"),
        ("z", "s"),
        ("s", "z"),
        ("ng", "n"),
    ],
    [("aa", "a"), ("ee", "e"), ("oo", "o"), ("ie", "i"), ("ck", "k"), ("ui", "u"), ("oe", "u")],
    [("sch", "sk"), ("sch", "s"), ("ch", "k"), ("g", "ch"), ("th", "t"), ("eu", "u"), ("uw", "u")],
]
_LEVEL_LABELS = {
    0: "Exact (baseline)      ",
    1: "L1 -- homophones/voicing",
    2: "L2 -- vowel length      ",
    3: "L3 -- clusters/digraphs ",
}
# Names that must NOT match anything in the gold/background corpus.
# Rules: (1) no linguistic overlap with Dutch or French;
#        (2) must not appear in any gold set as an original;
#        (3) must not appear in surnames.txt.
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
    "Petrov",  # replaced Dubois (Dubois is a French gold original)
    "Nakamura",
    "Okonkwo",
    "Papadopoulos",
    "Szczepanski",
    "Vanderbilt",
    "Worthington",
    "Nkrumah",
    "Brzezinski",
]


def _load_surnames(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]


def _distort(name: str, level: int) -> list[str]:
    rules = _SYNTHETIC_RULES[level - 1]
    lower = name.lower()
    variants: set[str] = set()
    for pattern, replacement in rules:
        pos = lower.find(pattern)
        if pos != -1:
            variant = lower[:pos] + replacement + lower[pos + len(pattern) :]
            if variant != lower:
                variants.add(variant)
    return sorted(variants)


def evaluate_synthetic(corpus: list[str], thresholds: Thresholds) -> tuple[list[dict], dict]:
    index = build_employee_index(_make_records(corpus), match_fields=MATCH_FIELDS)
    rows: list[dict] = []

    for name in corpus:
        decision, matches = _query(index, name, thresholds)
        hit = any(m.id == name for m in matches)
        rows.append(
            {
                "original": name,
                "query": name,
                "level": 0,
                "decision": str(decision),
                "top_match": matches[0].id if matches else "",
                "top_score": matches[0].score if matches else 0.0,
                "hit": hit,
            }
        )

    for name in corpus:
        for level in range(1, 4):
            for variant in _distort(name, level):
                decision, matches = _query(index, variant, thresholds)
                hit = any(m.id == name for m in matches)
                rows.append(
                    {
                        "original": name,
                        "query": variant,
                        "level": level,
                        "decision": str(decision),
                        "top_match": matches[0].id if matches else "",
                        "top_score": matches[0].score if matches else 0.0,
                        "hit": hit,
                    }
                )

    fp_hits = 0
    fp_details: list[dict] = []
    for neg in _NEGATIVE_NAMES:
        decision, matches = _query(index, neg, thresholds)
        if decision != Decision.NO_MATCH:
            fp_hits += 1
        fp_details.append(
            {
                "query": neg,
                "decision": str(decision),
                "top_match": matches[0].id if matches else "",
                "top_score": matches[0].score if matches else 0.0,
            }
        )

    by_level: dict = defaultdict(
        lambda: {"tp": 0, "fp_wrong": 0, "fn": 0, "total": 0, "scores": []}
    )
    for r in rows:
        lv = r["level"]
        by_level[lv]["total"] += 1
        by_level[lv]["scores"].append(r["top_score"])
        if r["hit"]:
            by_level[lv]["tp"] += 1
        elif r["decision"] != str(Decision.NO_MATCH):
            by_level[lv]["fp_wrong"] += 1
        else:
            by_level[lv]["fn"] += 1

    by_level_stats = {}
    for lv, d in sorted(by_level.items()):
        p, r, f1 = _prf(d["tp"], d["fp_wrong"], d["fn"])
        by_level_stats[lv] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "tp": d["tp"],
            "fp_wrong": d["fp_wrong"],
            "fn": d["fn"],
            "total": d["total"],
            "mean_score": sum(d["scores"]) / len(d["scores"]) if d["scores"] else 0.0,
        }

    return rows, {
        "by_level": by_level_stats,
        "false_positive_rate": fp_hits / len(_NEGATIVE_NAMES),
        "false_positives": fp_hits,
        "negatives_tested": len(_NEGATIVE_NAMES),
        "fp_details": fp_details,
    }


def print_synthetic_report(summary: dict, thresholds: Thresholds) -> None:
    print("\n" + "=" * 80)
    print("  PHONETICS ENGINE -- SYNTHETIC DISTORTION EVALUATION")
    print("=" * 80)
    print(
        f"  Thresholds: min={thresholds.min_match}  high={thresholds.high_confidence}"
        f"  margin={thresholds.ambiguity_margin}"
    )
    print("-" * 80)
    print(
        f"  {'Level':<26} {'Prec':>7}  {'Recall':>7}  {'F1':>7}   {'TP':>4} {'FP':>4} {'FN':>4}  {'Total':>6}"
    )
    print("-" * 80)
    for lv, s in summary["by_level"].items():
        label = _LEVEL_LABELS.get(lv, f"Level {lv}")
        print(
            f"  {label:<26} {s['precision'] * 100:>6.1f}%  {s['recall'] * 100:>6.1f}%"
            f"  {s['f1'] * 100:>6.1f}%   {s['tp']:>4} {s['fp_wrong']:>4} {s['fn']:>4}  {s['total']:>6}"
        )
    print("-" * 80)
    print("  TP=correct match returned  FP=wrong match returned  FN=no match returned")
    print("-" * 80)
    fp_pct = summary["false_positive_rate"] * 100
    print(
        f"  Negative control -- false positives: {summary['false_positives']}/{summary['negatives_tested']} ({fp_pct:.1f}%)"
    )
    for r in [r for r in summary["fp_details"] if r["decision"] != str(Decision.NO_MATCH)]:
        print(
            f"    '{r['query']}' -> '{r['top_match']}' ({r['decision']}, score={r['top_score']:.4f})"
        )
    print("=" * 80)


def sweep_synthetic(corpus: list[str]) -> None:
    """Threshold sensitivity sweep over synthetic distortion data."""
    import itertools

    configs = [
        (mn, hc) for mn, hc in itertools.product([0.45, 0.55, 0.65], [0.75, 0.82, 0.86]) if mn < hc
    ]
    print("\n=== Threshold Sweep (L1 = single Dutch phoneme swap) ===")
    print(f"{'min':>6} {'high':>6}  {'L1 P':>7} {'L1 R':>7} {'L1 F1':>7}  {'neg FP%':>7}")
    print("-" * 52)
    for min_match, high_conf in configs:
        t = Thresholds(min_match=min_match, high_confidence=high_conf, ambiguity_margin=0.10)
        _, summary = evaluate_synthetic(corpus, t)
        l1 = summary["by_level"].get(1, {})
        p = l1.get("precision", 0.0) * 100
        r = l1.get("recall", 0.0) * 100
        f1 = l1.get("f1", 0.0) * 100
        fp = summary["false_positive_rate"] * 100
        print(
            f"{min_match:>6.2f} {high_conf:>6.2f}  {p:>6.1f}%  {r:>6.1f}%  {f1:>6.1f}%   {fp:>6.1f}%"
        )


def sweep_gold(gold_files: dict[str, Path]) -> None:
    """
    Full hyperparameter sweep over (min_match, high_confidence, ambiguity_margin)
    evaluated on the curated gold data.

    Key design: build each language index *once*, pre-fetch all scored
    candidates per query (the model), then replay classify() across the entire
    parameter grid (the decision layer).  This decouples model quality from
    threshold calibration and keeps the sweep fast.
    """
    import itertools

    MIN_MATCHES = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    HIGH_CONFS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.86]
    MARGINS = [0.08, 0.10, 0.12, 0.15]
    CURRENT_DEFAULT = (0.55, 0.86, 0.12)

    # ------------------------------------------------------------------
    # 1. Build indices once; pre-fetch scored candidates for every query.
    # ------------------------------------------------------------------
    print("  Building indices and pre-fetching candidates...", flush=True)
    lang_data: dict[str, dict] = {}
    for lang, path in gold_files.items():
        espeak_lang = LANG_ESPEAK.get(lang, "nl")
        bg_path = LANG_BACKGROUND.get(lang)
        background = _load_surnames(bg_path) if bg_path and bg_path.exists() else []
        pairs = load_gold(path)
        originals = list(dict.fromkeys(p[0] for p in pairs))
        index_names = list(dict.fromkeys(background + originals))
        index = build_employee_index(
            _make_records(index_names), match_fields=MATCH_FIELDS, language=espeak_lang
        )
        n = len(index_names)

        pair_cache: list[tuple[str, str, list]] = []
        for original, query, _cat, _note in pairs:
            # Fetch all candidates so every threshold can be replayed fairly.
            scored = index.search(query, n)
            pair_cache.append((original, query, scored))

        neg_cache: list[tuple[str, list]] = []
        for neg in _NEGATIVE_NAMES:
            scored = index.search(neg, TOP_K)
            neg_cache.append((neg, scored))

        lang_data[lang] = {"pairs": pair_cache, "negatives": neg_cache}

    langs = list(gold_files.keys())
    total_configs = sum(
        1 for mn, hc, mg in itertools.product(MIN_MATCHES, HIGH_CONFS, MARGINS) if mn < hc
    )

    # ------------------------------------------------------------------
    # 2. Replay classify() across the full parameter grid.
    # ------------------------------------------------------------------
    results: list[dict] = []
    for min_m, high_c, margin in itertools.product(MIN_MATCHES, HIGH_CONFS, MARGINS):
        if min_m >= high_c:
            continue
        t = Thresholds(min_match=min_m, high_confidence=high_c, ambiguity_margin=margin)

        lang_stats: dict[str, dict] = {}
        for lang, data in lang_data.items():
            tp = fp = fn = 0
            for original, query, scored in data["pairs"]:
                decision, matches = classify(query, scored[:TOP_K], t, TOP_K)
                hit = any(m.id == original for m in matches)
                if hit:
                    tp += 1
                elif decision != Decision.NO_MATCH:
                    fp += 1
                else:
                    fn += 1

            neg_fp = sum(
                1
                for _neg, scored in data["negatives"]
                if classify(_neg, scored, t, TOP_K)[0] != Decision.NO_MATCH
            )
            p, r, f1 = _prf(tp, fp, fn)
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
        neg_fp_rate = sum(s["neg_fp"] for s in lang_stats.values()) / total_neg

        results.append(
            {
                "min_m": min_m,
                "high_c": high_c,
                "margin": margin,
                "macro_f1": macro_f1,
                "gen_ratio": gen_ratio,
                "neg_fp_rate": neg_fp_rate,
                **{f"f1_{lang}": s["f1"] for lang, s in lang_stats.items()},
                **{f"prec_{lang}": s["precision"] for lang, s in lang_stats.items()},
                **{f"rec_{lang}": s["recall"] for lang, s in lang_stats.items()},
            }
        )

    results.sort(key=lambda x: x["macro_f1"], reverse=True)

    # ------------------------------------------------------------------
    # 3. Report.
    # ------------------------------------------------------------------
    bar = "=" * 80
    print(f"\n{bar}")
    print("  GOLD THRESHOLD SWEEP")
    print(bar)
    print(
        f"  Grid: {len(MIN_MATCHES)} min_match × {len(HIGH_CONFS)} high_conf × {len(MARGINS)} margin"
        f"  →  {total_configs} valid configs"
    )
    print(
        f"  Negative control: {len(_NEGATIVE_NAMES)} out-of-vocabulary names"
        f" × {len(langs)} language indices"
    )

    # Header
    col_heads = "  " + f"{'min':>5} {'high':>5} {'marg':>5}  "
    for lang in langs:
        col_heads += f"{'F1_' + lang[:2].upper():>7}  "
    col_heads += f"{'MacroF1':>8}  {'Gen':>5}  {'FP_neg':>6}"
    print(f"\n  Top 25 by Macro F1:\n{col_heads}")
    print(f"  {'-' * 72}")

    for r in results[:25]:
        is_default = (r["min_m"], r["high_c"], r["margin"]) == CURRENT_DEFAULT
        row = f"  {r['min_m']:>5.2f} {r['high_c']:>5.2f} {r['margin']:>5.2f}  "
        for lang in langs:
            row += f"{r[f'f1_{lang}'] * 100:>6.1f}%  "
        row += (
            f"{r['macro_f1'] * 100:>7.1f}%  {r['gen_ratio']:>5.2f}  {r['neg_fp_rate'] * 100:>5.1f}%"
        )
        if is_default:
            row += "  ← current default"
        print(row)

    # Current default rank
    current = next(
        (r for r in results if (r["min_m"], r["high_c"], r["margin"]) == CURRENT_DEFAULT), None
    )
    if current and results.index(current) >= 25:
        rank = results.index(current) + 1
        row = f"  {current['min_m']:>5.2f} {current['high_c']:>5.2f} {current['margin']:>5.2f}  "
        for lang in langs:
            row += f"{current[f'f1_{lang}'] * 100:>6.1f}%  "
        row += (
            f"{current['macro_f1'] * 100:>7.1f}%  {current['gen_ratio']:>5.2f}"
            f"  {current['neg_fp_rate'] * 100:>5.1f}%  ← current default (rank {rank}/{len(results)})"
        )
        print(f"  {'·' * 72}")
        print(row)

    # Pareto front: best Macro F1 at each precision floor (with 0% neg FP)
    print("\n  Pareto front — best Macro F1 at precision constraints (0 neg FPs required):")
    print(f"  {'Prec ≥':>8}  {'min':>5} {'high':>5} {'marg':>5}  {'MacroF1':>8}  per-language F1")
    print(f"  {'-' * 65}")
    for prec_floor in [0.70, 0.80, 0.90, 1.00]:
        candidates = [
            r
            for r in results
            if all(r[f"prec_{lang}"] >= prec_floor for lang in langs) and r["neg_fp_rate"] == 0.0
        ]
        if candidates:
            best = candidates[0]
            lang_detail = "  ".join(
                f"{lang[:2].upper()}={best[f'f1_{lang}'] * 100:.1f}%" for lang in langs
            )
            print(
                f"  {prec_floor * 100:>7.0f}%  "
                f"{best['min_m']:>5.2f} {best['high_c']:>5.2f} {best['margin']:>5.2f}  "
                f"{best['macro_f1'] * 100:>7.1f}%  {lang_detail}"
            )
        else:
            print(f"  {prec_floor * 100:>7.0f}%  (no config meets this constraint with 0 neg FPs)")

    # Diagnostic: which negatives hit at the current default threshold?
    if current:
        t_default = Thresholds(
            min_match=current["min_m"],
            high_confidence=current["high_c"],
            ambiguity_margin=current["margin"],
        )
        neg_hits: list[tuple[str, str, str, float]] = []  # (lang, name, matched, score)
        for lang, data in lang_data.items():
            for neg_name, scored in data["negatives"]:
                decision, matches = classify(neg_name, scored, t_default, TOP_K)
                if decision != Decision.NO_MATCH:
                    top = matches[0] if matches else None
                    neg_hits.append(
                        (
                            lang,
                            neg_name,
                            top.id if top else "?",
                            top.score if top else 0.0,
                        )
                    )
        if neg_hits:
            print(
                f"\n  Negative control hits at current default ({current['min_m']}/{current['high_c']}/{current['margin']}):"
            )
            for lang, neg_name, matched, score in neg_hits:
                print(f"    [{lang[:2].upper()}] '{neg_name}' -> '{matched}' ({score:.3f})")
        else:
            print(
                "\n  Negative control: 0 hits at current default (all OOV names correctly rejected)"
            )

    print(bar)


# ---------------------------------------------------------------------------
# Shared CSV output
# ---------------------------------------------------------------------------


def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Raw results saved -> {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate phonetics-engine surname matching.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python eval/evaluate.py                     # gold eval at default thresholds
  uv run python eval/evaluate.py --sweep             # hyperparameter sweep on gold data
  uv run python eval/evaluate.py --mode synthetic    # synthetic distortion eval
  uv run python eval/evaluate.py --mode synthetic --sweep  # synthetic threshold sweep
  uv run python eval/evaluate.py --output results.csv
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["gold", "synthetic"],
        default="gold",
        help="Evaluation data source. Default: gold",
    )
    parser.add_argument(
        "--names",
        type=Path,
        default=_GOLD_DIR / "surnames.txt",
        help="Surnames corpus (background for gold mode; primary data for synthetic). Default: eval/surnames.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write raw per-query results to this CSV file (non-sweep modes only).",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help=(
            "Run a full threshold hyperparameter sweep. "
            "In gold mode (default): sweeps (min_match, high_confidence, margin) on curated pairs. "
            "In synthetic mode: sweeps L1-distortion performance."
        ),
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Synthetic mode
    # ------------------------------------------------------------------
    if args.mode == "synthetic":
        if not args.names.exists():
            print(f"ERROR: corpus file not found: {args.names}", file=sys.stderr)
            sys.exit(1)
        corpus = _load_surnames(args.names)
        print(f"Loaded {len(corpus)} surnames from {args.names}")
        if args.sweep:
            sweep_synthetic(corpus)
        else:
            rows, summary = evaluate_synthetic(corpus, DEFAULT_THRESHOLDS)
            print_synthetic_report(summary, DEFAULT_THRESHOLDS)
            if args.output:
                save_csv(rows, args.output)
        return

    # ------------------------------------------------------------------
    # Gold mode (default)
    # ------------------------------------------------------------------
    gold_files = {
        "Dutch": _GOLD_DIR / "gold_dutch.csv",
        "French": _GOLD_DIR / "gold_french.csv",
    }
    for lang, path in gold_files.items():
        if not path.exists():
            print(f"ERROR: gold file not found: {path}", file=sys.stderr)
            sys.exit(1)

    if args.sweep:
        sweep_gold(gold_files)
        return

    all_rows: list[dict] = []
    summaries: dict[str, dict] = {}

    for lang, path in gold_files.items():
        espeak_lang = LANG_ESPEAK.get(lang, "nl")
        bg_path = LANG_BACKGROUND.get(lang)
        background: list[str] = []
        if bg_path and bg_path.exists():
            background = _load_surnames(bg_path)
            print(f"[{lang}] background corpus: {len(background)} names from {bg_path.name}")
        else:
            print(f"[{lang}] no background corpus — index contains gold originals only")
        pairs = load_gold(path)
        rows, summary = evaluate_gold(
            pairs, DEFAULT_THRESHOLDS, background_corpus=background, language=espeak_lang
        )
        summaries[lang] = summary
        print_gold_report(lang, summary, rows, DEFAULT_THRESHOLDS)
        all_rows.extend({"lang": lang, **r} for r in rows)

    print_combined_report(summaries)

    if args.output:
        save_csv(all_rows, args.output)


if __name__ == "__main__":
    main()
