from dataclasses import dataclass

from phonetics_engine.enums import Decision, MatchField
from phonetics_engine.models import Match, Thresholds
from phonetics_engine.normalize import canonicalize


@dataclass(slots=True)
class ScoredCandidate:
    id: str
    display_name: str
    canonical_name: str
    score: float
    matched_field: MatchField | None = None
    matched_value: str | None = None


def _is_exact(query_canon: str, c: ScoredCandidate) -> bool:
    return query_canon == c.canonical_name or query_canon == canonicalize(c.display_name)


def _to_match(c: ScoredCandidate, margin: float) -> Match:
    return Match(
        id=c.id,
        display_name=c.display_name,
        canonical_name=c.canonical_name,
        score=round(c.score, 4),
        margin_to_next=round(margin, 4),
        matched_field=c.matched_field,
        matched_value=c.matched_value,
    )


def classify(
    query: str,
    scored: list[ScoredCandidate],
    thresholds: Thresholds,
    top_k: int,
) -> tuple[Decision, list[Match]]:
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    if not scored:
        return Decision.NO_MATCH, []

    scored = sorted(scored, key=lambda c: c.score, reverse=True)
    query_canon = canonicalize(query)

    exact_matches = [c for c in scored if _is_exact(query_canon, c)]
    if len(exact_matches) == 1:
        # single exact match: margin_to_next = score (no runner-up to subtract)
        return Decision.EXACT, [_to_match(exact_matches[0], exact_matches[0].score)]
    if len(exact_matches) >= 2:
        kept = exact_matches[:top_k]
        out = []
        for i, c in enumerate(kept):
            # last element: margin_to_next = full score (no runner-up exists)
            margin = c.score - kept[i + 1].score if i + 1 < len(kept) else c.score
            out.append(_to_match(c, margin))
        return Decision.AMBIGUOUS, out

    best = scored[0]
    if best.score < thresholds.min_match:
        return Decision.NO_MATCH, []

    runner_up_score = scored[1].score if len(scored) > 1 else 0.0
    margin = round(best.score - runner_up_score, 4)

    if margin < thresholds.ambiguity_margin:
        kept = [c for c in scored[:top_k] if c.score >= thresholds.min_match
                and round(best.score - c.score, 4) < thresholds.ambiguity_margin]
        out = []
        for i, c in enumerate(kept):
            # last element: margin_to_next = full score (no runner-up exists)
            m = c.score - kept[i + 1].score if i + 1 < len(kept) else c.score
            out.append(_to_match(c, m))
        return Decision.AMBIGUOUS, out

    if best.score >= thresholds.high_confidence:
        return Decision.SINGLE_HIGH_CONFIDENCE, [_to_match(best, margin)]

    # best.score in [min_match, high_confidence) and margin OK -> still no_match per spec
    return Decision.NO_MATCH, []
