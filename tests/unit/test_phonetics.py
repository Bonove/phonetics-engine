import pytest

from phonetics_engine.phonetics import (
    PhoneticIndex,
    phonemize_batch,
    phonemize_name,
)

pytestmark = pytest.mark.skipif(
    pytest.importorskip("phonemizer", reason="espeak-ng not available") is None,
    reason="espeak-ng not installed",
)


def test_phonemize_name_returns_phonemes():
    p = phonemize_name("Steven")
    assert p
    assert " " in p  # spaces between phonemes


def test_phonemize_batch_preserves_length():
    out = phonemize_batch(["Steven", "", "Wasteless"])
    assert len(out) == 3
    assert out[1] == ""


def test_index_search_finds_close_match():
    idx = PhoneticIndex(["Steven", "Stefan", "Marie", "Wasteless"])
    results = idx.search("Steeve", top_k=3)
    assert results
    assert results[0]["name"] in {"Steven", "Stefan"}
    assert 0.0 <= results[0]["score"] <= 1.0


def test_index_empty_returns_empty_search():
    idx = PhoneticIndex([])
    assert idx.search("anything") == []
