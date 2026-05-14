import espeakng_loader
import faiss
import numpy as np
from phonemizer import phonemize
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from phonemizer.separator import Separator

# Bundle espeak-ng via pip wheel so we don't depend on apt/brew system installs.
EspeakWrapper.set_library(espeakng_loader.get_library_path())
EspeakWrapper.set_data_path(espeakng_loader.get_data_path())

PHONEMIZER_LANGUAGE = "nl"
PHONEMIZER_BACKEND = "espeak"
SIMILARITY_THRESHOLD = 0.0  # we filter at decision-layer with thresholds; keep raw here

_SEPARATOR = Separator(phone=" ", word="", syllable="")


def phonemize_name(name: str, language: str = PHONEMIZER_LANGUAGE) -> str:
    if not name or not name.strip():
        return ""
    result = phonemize(
        name.strip(),
        backend=PHONEMIZER_BACKEND,
        language=language,
        separator=_SEPARATOR,
        strip=True,
    )
    return result.strip()


def phonemize_batch(names: list[str], language: str = PHONEMIZER_LANGUAGE) -> list[str]:
    cleaned = [n.strip() if n else "" for n in names]
    non_empty = [n for n in cleaned if n]

    if not non_empty:
        return [""] * len(names)

    results = phonemize(
        non_empty,
        backend=PHONEMIZER_BACKEND,
        language=language,
        separator=_SEPARATOR,
        strip=True,
    )

    out: list[str] = []
    rit = iter(results if isinstance(results, list) else [results])
    for n in cleaned:
        if n:
            out.append(next(rit).strip())
        else:
            out.append("")
    return out


def _phonemes_to_vector(phonemes: str, dim: int = 128) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    chars = phonemes.replace(" ", "")
    if not chars:
        return vec
    for n in (2, 3):
        for i in range(len(chars) - n + 1):
            ngram = chars[i : i + n]
            h = hash(ngram) % dim
            vec[h] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


class PhoneticIndex:
    def __init__(self, names: list[str]):
        self._names = list(names)
        self._phonemes: list[str] = []
        self._index: faiss.Index | None = None

        if not names:
            return

        self._phonemes = phonemize_batch(names)
        dim = 128
        vectors = np.array(
            [_phonemes_to_vector(p, dim) for p in self._phonemes],
            dtype=np.float32,
        )
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(vectors)

    @property
    def size(self) -> int:
        return len(self._names)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not self._index or self.size == 0:
            return []
        qp = phonemize_name(query)
        if not qp:
            return []
        qv = _phonemes_to_vector(qp).reshape(1, -1)
        k = min(top_k, self.size)
        scores, indices = self._index.search(qv, k)
        results: list[dict] = []
        for s, i in zip(scores[0], indices[0], strict=False):
            if i < 0:
                continue
            clamped = float(max(0.0, min(1.0, s)))
            if clamped >= SIMILARITY_THRESHOLD:
                results.append(
                    {
                        "name": self._names[i],
                        "score": round(clamped, 4),
                        "phonemes": self._phonemes[i],
                    }
                )
        return results
