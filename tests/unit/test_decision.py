from phonetics_engine.decision import ScoredCandidate, classify
from phonetics_engine.enums import Decision, MatchField
from phonetics_engine.models import Thresholds

COMPANY_THRESHOLDS = Thresholds(min_match=0.55, high_confidence=0.82, ambiguity_margin=0.10)


def _sc(id_, display, canonical, score, mf=None, mv=None):
    return ScoredCandidate(
        id=id_,
        display_name=display,
        canonical_name=canonical,
        score=score,
        matched_field=mf,
        matched_value=mv,
    )


def test_no_match_when_top_below_min():
    decision, matches = classify(
        query="ssspmlx",
        scored=[_sc("c1", "Waysis", "waysis", 0.20)],
        thresholds=COMPANY_THRESHOLDS,
        top_k=5,
    )
    assert decision == Decision.NO_MATCH
    assert matches == []


def test_exact_single_match():
    decision, matches = classify(
        query="Waysis",
        scored=[
            _sc("c1", "Waysis", "waysis", 0.99),
            _sc("c2", "Waste", "waste", 0.40),
        ],
        thresholds=COMPANY_THRESHOLDS,
        top_k=5,
    )
    assert decision == Decision.EXACT
    assert len(matches) == 1
    assert matches[0].id == "c1"
    assert matches[0].margin_to_next == matches[0].score  # only one returned


def test_exact_two_matches_becomes_ambiguous():
    decision, matches = classify(
        query="steven",
        scored=[
            _sc("e1", "Steven", "steven", 0.99),
            _sc("e2", "Steven", "steven", 0.99),
            _sc("e3", "Stefan", "stefan", 0.60),
        ],
        thresholds=COMPANY_THRESHOLDS,
        top_k=5,
    )
    assert decision == Decision.AMBIGUOUS
    assert len(matches) == 2
    assert {m.id for m in matches} == {"e1", "e2"}


def test_single_high_confidence():
    decision, matches = classify(
        query="wasteless",
        scored=[
            _sc("c1", "Waysis", "waysis", 0.91),
            _sc("c2", "Waste", "waste", 0.51),
        ],
        thresholds=COMPANY_THRESHOLDS,
        top_k=5,
    )
    assert decision == Decision.SINGLE_HIGH_CONFIDENCE
    assert len(matches) == 1
    assert matches[0].id == "c1"
    assert matches[0].margin_to_next == 0.40


def test_ambiguous_close_scores():
    decision, matches = classify(
        query="vries",
        scored=[
            _sc("e1", "Sanne de Vries", "sanne de vries", 0.80, MatchField.LAST_NAME, "Vries"),
            _sc("e2", "Bert Vries",      "bert vries",     0.78, MatchField.LAST_NAME, "Vries"),
        ],
        thresholds=COMPANY_THRESHOLDS,
        top_k=5,
    )
    assert decision == Decision.AMBIGUOUS
    assert len(matches) == 2


def test_top_k_truncates_ambiguous_matches():
    """All 10 candidates are within ambiguity_margin of the best — top_k clips the returned list."""
    scored = [_sc(f"c{i}", f"Co{i}", f"co{i}", 0.90 - i * 0.01) for i in range(10)]
    decision, matches = classify(
        query="xxxxx",
        scored=scored,
        thresholds=COMPANY_THRESHOLDS,
        top_k=3,
    )
    assert decision == Decision.AMBIGUOUS
    assert len(matches) == 3


def test_margin_to_next_for_last_match():
    decision, matches = classify(
        query="xxxxx",
        scored=[
            _sc("c1", "A", "a", 0.95),
            _sc("c2", "B", "b", 0.50),
        ],
        thresholds=COMPANY_THRESHOLDS,
        top_k=5,
    )
    assert decision == Decision.SINGLE_HIGH_CONFIDENCE
    assert round(matches[0].margin_to_next, 4) == 0.45
