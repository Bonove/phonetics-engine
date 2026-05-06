from phonetics_engine.normalize import canonicalize


def test_canonicalize_lowercases_and_strips():
    assert canonicalize("  Waysis  ") == "waysis"


def test_canonicalize_strips_diacritics():
    assert canonicalize("Café") == "cafe"
    assert canonicalize("Müller") == "muller"


def test_canonicalize_handles_empty():
    assert canonicalize("") == ""
    assert canonicalize("   ") == ""


def test_canonicalize_collapses_internal_whitespace():
    assert canonicalize("de  Vries") == "de vries"
    assert canonicalize("van\t der  Berg") == "van der berg"
