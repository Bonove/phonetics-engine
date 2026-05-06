from dataclasses import dataclass

import faiss
import numpy as np

from phonetics_engine.decision import ScoredCandidate
from phonetics_engine.enums import MatchField
from phonetics_engine.loader import CompanyRecord, EmployeeRecord
from phonetics_engine.phonetics import _phonemes_to_vector, phonemize_batch, phonemize_name


@dataclass
class _Entry:
    record_id: str
    display_name: str
    canonical_name: str
    matched_field: MatchField
    matched_value: str


class TenantIndex:
    def __init__(self, entries: list[_Entry], dim: int = 128):
        self._entries = entries
        self._dim = dim
        self._index: faiss.Index | None = None

        if not entries:
            return

        phonemes = phonemize_batch([e.matched_value for e in entries])
        vectors = np.array(
            [_phonemes_to_vector(p, dim) for p in phonemes],
            dtype=np.float32,
        )
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(vectors)

    def search(self, query: str, top_k: int) -> list[ScoredCandidate]:
        if not self._index or not self._entries:
            return []

        qp = phonemize_name(query)
        if not qp:
            return []
        qv = _phonemes_to_vector(qp, self._dim).reshape(1, -1)

        # Pull more raw matches than top_k since dedup may reduce the result count.
        k_raw = min(len(self._entries), max(top_k * 4, 16))
        scores, indices = self._index.search(qv, k_raw)

        best: dict[str, ScoredCandidate] = {}
        for s, i in zip(scores[0], indices[0], strict=False):
            if i < 0:
                continue
            entry = self._entries[i]
            clamped = float(max(0.0, min(1.0, s)))
            existing = best.get(entry.record_id)
            if existing is None or clamped > existing.score:
                best[entry.record_id] = ScoredCandidate(
                    id=entry.record_id,
                    display_name=entry.display_name,
                    canonical_name=entry.canonical_name,
                    score=clamped,
                    matched_field=entry.matched_field,
                    matched_value=entry.matched_value,
                )

        out = sorted(best.values(), key=lambda c: c.score, reverse=True)
        return out[:top_k]


def build_company_index(
    records: list[CompanyRecord], *, match_fields: list[MatchField]
) -> TenantIndex:
    entries: list[_Entry] = []
    for r in records:
        if MatchField.DISPLAY_NAME in match_fields:
            entries.append(
                _Entry(
                    r.id, r.display_name, r.canonical_name, MatchField.DISPLAY_NAME, r.display_name
                )
            )
        if MatchField.CANONICAL_NAME in match_fields:
            entries.append(
                _Entry(
                    r.id,
                    r.display_name,
                    r.canonical_name,
                    MatchField.CANONICAL_NAME,
                    r.canonical_name,
                )
            )
        if MatchField.ALIAS in match_fields:
            for alias in r.aliases:
                entries.append(
                    _Entry(r.id, r.display_name, r.canonical_name, MatchField.ALIAS, alias)
                )
    return TenantIndex(entries)


def build_employee_index(
    records: list[EmployeeRecord], *, match_fields: list[MatchField]
) -> TenantIndex:
    entries: list[_Entry] = []
    for r in records:
        display = r.full_name
        canonical = r.full_name.lower()
        if MatchField.LAST_NAME in match_fields:
            entries.append(_Entry(r.id, display, canonical, MatchField.LAST_NAME, r.last_name))
            if r.infix and r.infix.strip():
                value = f"{r.infix.strip()} {r.last_name}"
                entries.append(
                    _Entry(r.id, display, canonical, MatchField.LAST_NAME_WITH_INFIX, value)
                )
        if MatchField.FULL_NAME in match_fields:
            entries.append(_Entry(r.id, display, canonical, MatchField.FULL_NAME, r.full_name))
    return TenantIndex(entries)
