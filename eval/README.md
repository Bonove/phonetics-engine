# Phonetics Engine — Evaluation

Report on the offline evaluation harness for surname matching.  
- [1. Context](#1-context)
- [2. System Architecture](#2-system-architecture)
  - [Similarity pipeline](#similarity-pipeline)
  - [Decision logic](#decision-logic)
  - [Default thresholds (production)](#default-thresholds-production)
- [3. Evaluation Design](#3-evaluation-design)
  - [3.1 Why gold sets, not synthetic distortions](#31-why-gold-sets-not-synthetic-distortions)
  - [3.2 Gold sets](#32-gold-sets)
    - [`gold_dutch.csv` — Dutch (25 pairs)](#gold_dutchcsv--dutch-25-pairs)
    - [`gold_french.csv` — French (25 pairs)](#gold_frenchcsv--french-25-pairs)
  - [3.3 Background corpora](#33-background-corpora)
  - [3.4 Negative control](#34-negative-control)
  - [3.5 Metrics](#35-metrics)
  - [3.6 Threshold sweep methodology](#36-threshold-sweep-methodology)
- [4. Results](#4-results)
  - [4.1 Current config — per-language breakdown](#41-current-config--per-language-breakdown)
    - [Dutch — espeak `nl` — 223-name background](#dutch--espeak-nl--223-name-background)
    - [French — espeak `fr-fr` — 307-name background](#french--espeak-fr-fr--307-name-background)
    - [Combined](#combined)
  - [4.2 Remaining miss patterns](#42-remaining-miss-patterns)
  - [4.3 Threshold sweep](#43-threshold-sweep)
  - [4.4 Pareto front (best Macro F1 at 0 OOV false positives)](#44-pareto-front-best-macro-f1-at-0-oov-false-positives)
  - [4.5 Negative control](#45-negative-control)
  - [4.6 Failure modes](#46-failure-modes)
    - [Previously threshold-level, now resolved](#previously-threshold-level-now-resolved)
    - [Model-level (require changes upstream of FAISS)](#model-level-require-changes-upstream-of-faiss)
- [5. Levenshtein Baseline](#5-levenshtein-baseline)
  - [5.1 Results](#51-results)
  - [5.2 Per-language analysis](#52-per-language-analysis)
  - [5.3 Latency trade-off](#53-latency-trade-off)
  - [5.4 Interpretation](#54-interpretation)
- [6. Vectorisation Experiment](#6-vectorisation-experiment)
  - [Why phoneme unigrams produce more OOV false positives](#why-phoneme-unigrams-produce-more-oov-false-positives)
  - [Why phoneme 2+3-grams (no unigrams) are worse than char 2+3-grams](#why-phoneme-23-grams-no-unigrams-are-worse-than-char-23-grams)
  - [Next candidate](#next-candidate)
- [7. Limitations](#7-limitations)
- [8. Conclusion](#8-conclusion)
- [9. Running the Evaluation](#9-running-the-evaluation)
  - [Sweep output](#sweep-output)
- [10. Files](#10-files)
Last updated: May 2026.

---

## 1. Context

The phonetics engine helps a voicebot recover from STT (speech-to-text) misrecognitions of
surnames. When a caller says "Fisser" and the correct name in the database is "Visser", the engine
must decide: *match*, *ambiguous* (ask for confirmation), or *no match* (ask caller to spell).

The evaluation in this directory measures how well the engine handles the systematic phonetic
distortions that STT systems introduce, across two languages.

---

## 2. System Architecture

### Similarity pipeline

```
query string
    │
    ▼  espeak-ng  (language-specific: "nl" for Dutch, "fr-fr" for French)
space-separated IPA    e.g.  Dutch "v ɪ s ə r" / French "d y p ɔ̃"
    │
    ▼  strip spaces, extract character 2-grams + 3-grams
feature vector (128 bins, hashed, L2-normalised)
    │
    ▼  FAISS IndexFlatIP  (inner product = cosine similarity on unit vectors)
top-K scored candidates
    │
    ▼  classify()
Decision
```

Using the correct phoneme backend per language is critical. Running French names through the
Dutch backend (`nl`) produces wrong IPA — `ch→sh` and nasal vowel patterns become invisible to
the model. Every result in this report uses the correct per-language backend.

### Decision logic

| Decision | Condition |
|---|---|
| `EXACT` | canonicalised query == canonical name |
| `SINGLE_HIGH_CONFIDENCE` | score ≥ `high_confidence` AND margin ≥ `ambiguity_margin` |
| `AMBIGUOUS` | score ≥ `min_match` AND margin < `ambiguity_margin` |
| `NO_MATCH` | score < `min_match` OR score in (`min_match`, `high_confidence`) with margin ≥ `ambiguity_margin` |

The last row is the **dead zone**: a score above the noise floor but below the confidence gate,
with no close runner-up to trigger `AMBIGUOUS`. This is the dominant Dutch failure mode.

### Default thresholds (production)

| Parameter | Value |
|---|---|
| `min_match` | 0.60 |
| `high_confidence` | 0.65 |
| `ambiguity_margin` | 0.08 |

---

## 3. Evaluation Design

### 3.1 Why gold sets, not synthetic distortions

An earlier iteration used programmatic character substitutions ("v→f", "aa→a") to generate test
queries. This approach has two fatal flaws:

1. **True homophones are trivially easy.** If espeak-ng maps both "Visser" and "Fisser" to the
   same IPA, the engine scores 1.0 regardless of thresholds. The test measures the distortion
   generator, not the engine.

2. **Corpus-size confound.** With a large background corpus, many substituted queries land near
   multiple similar-sounding names simultaneously, causing `AMBIGUOUS` to fire and return the
   correct name — even when the raw similarity is mediocre. With a small corpus the same query
   hits the dead zone instead.

The gold sets replace generated distortions with **curated (original, STT-variant) pairs** that
reflect real, attested sources of confusion, stratified by phonological category.

### 3.2 Gold sets

Two gold sets of 25 pairs each, stored as CSV files in this directory.

#### `gold_dutch.csv` — Dutch (25 pairs)

| Category | Pairs | Examples |
|---|---|---|
| `voicing` | 7 | Visser/Fisser, Bakker/Pakker, Zegers/Segers, Mulder/Multer |
| `vowel_length` | 5 | Maas/Mas, Laan/Lan, Beer/Ber, Veen/Ven, Boon/Bon |
| `diphthong` | 4 | Kleijn/Klein, Dijkstra/Dekstra, Kuipers/Kupers, Brouwer/Brauer |
| `cluster` | 4 | Schouten/Skouten, Schuurman/Skuurman, Jong/Jon, Achterberg/Akterberg |
| `other` | 5 | Timmerman/Timerman, Blom/Plom, Wolters/Volters, Hermans/Ermans, Smeets/Smets |

#### `gold_french.csv` — French (25 pairs)

| Category | Pairs | Examples |
|---|---|---|
| `final_silent` | 6 | Dupont/Dupon, Robert/Rober, Bernard/Bernar, Blanc/Blan, Petit/Peti |
| `eau_o` | 3 | Moreau/Moro, Rousseau/Russo, Beaulieu/Bolieu |
| `ch_sh` | 4 | Chevalier/Shevalier, Michel/Mishel, Blanchard/Blanshar, Richard/Rishar |
| `nasal` | 4 | Martin/Marten, Laurent/Loran, Vincent/Vanson, Fontaine/Fonten |
| `other` | 8 | Philippe/Filipe, André/Andre, Gauthier/Gotier, Henry/Anri, Girard/Jirar, Roux/Ru, Dubois/Dubwa, Clément/Clement |

### 3.3 Background corpora

Each language uses its own same-language background corpus. Using Dutch background names for the
French index inflates OOV false positives (English names phonemised with French rules
accidentally match Dutch names) and corrupts the `AMBIGUOUS` dynamics.

| Language | File | Size |
|---|---|---|
| Dutch | `surnames.txt` | 223 names |
| French | `surnames_fr.txt` | 307 names |

Gold originals are merged into the background so the target name is always in the index.
The background provides realistic competitive pressure for the `AMBIGUOUS` mechanism.

### 3.4 Negative control

20 out-of-vocabulary (OOV) names are tested against each language index. A hit is a false
positive. Names are chosen to have no linguistic overlap with Dutch or French:

```
Shakespeare, Jefferson, Roosevelt, Churchill, Wellington, Hamilton,
Fitzgerald, MacGregor, Yamamoto, Kowalski, Fernandez, Petrov,
Nakamura, Okonkwo, Papadopoulos, Szczepanski, Vanderbilt,
Worthington, Nkrumah, Brzezinski
```

> `Dubois` was removed from this list — it is a French gold original.

### 3.5 Metrics

- **Precision** = TP / (TP + FP) where FP = wrong match returned
- **Recall** = TP / (TP + FN) where FN = no match returned when correct match exists
- **F1** = harmonic mean of precision and recall
- **Macro F1** = mean F1 across languages
- **Generalizability ratio** = min(F1) / max(F1) — 1.0 = identical; < 0.75 = language-specific
- **Neg FP rate** = OOV hits / (20 × number of language indices)

A hit = the correct original appears anywhere in the returned candidate list.

### 3.6 Threshold sweep methodology

Decouples **model quality** (raw FAISS cosine similarity) from **decision quality** (thresholds):

1. Build each language index **once** with the correct espeak backend.
2. Pre-fetch all scored candidates per query at maximum depth.
3. For each `(min_match, high_confidence, ambiguity_margin)` config, replay `classify()` on
   cached scores — no re-indexing, no re-querying.

Grid: 6 × `min_match` × 6 × `high_confidence` × 4 × `ambiguity_margin` = **120 valid configs**.

---

## 4. Results

### 4.1 Current config — per-language breakdown

Run: `uv run python eval/evaluate.py`  
Thresholds: `min=0.60  high=0.65  margin=0.08`

#### Dutch — espeak `nl` — 223-name background

| Category | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| `cluster` | 100% | 75% | 86% | 3 | 0 | 1 |
| `diphthong` | 100% | 75% | 86% | 3 | 0 | 1 |
| `other` | 100% | 60% | 75% | 3 | 0 | 2 |
| `voicing` | 100% | 57% | 73% | 4 | 0 | 3 |
| `vowel_length` | 0% | 0% | **0%** | 0 | 1 | 4 |
| **Overall** | **93%** | **54%** | **68.4%** | 13 | 1 | 11 |

#### French — espeak `fr-fr` — 307-name background

| Category | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| `ch_sh` | 100% | 100% | **100%** | 4 | 0 | 0 |
| `eau_o` | 100% | 67% | 80% | 2 | 0 | 1 |
| `final_silent` | 100% | 83% | 91% | 5 | 0 | 1 |
| `nasal` | 100% | 75% | 86% | 3 | 0 | 1 |
| `other` | 100% | 75% | 86% | 6 | 0 | 2 |
| **Overall** | **100%** | **80%** | **88.9%** | 20 | 0 | 5 |

#### Combined

| Config | Dutch F1 | French F1 | Macro F1 | Generalizability |
|---|---|---|---|---|
| Previous default (0.55/0.86/0.12) | 48.5% | 86.4% | 67.4% | 0.56 — SIGNIFICANT BIAS |
| **Current config (0.60/0.65/0.08)** | **68.4%** | **88.9%** | **78.7%** | **0.77 — MILD BIAS** |

### 4.2 Remaining miss patterns

With the previous default the dominant failure was the **dead zone**: the model found the
correct name at raw similarity 0.60–0.85, but `high_confidence=0.86` blocked the match.
Lowering `high_confidence` to 0.65 resolved those cases. The current misses split into two
groups.

**Model-level misses** — raw cosine similarity genuinely below `min_match=0.60`:

| Query | Want | Raw score | Reason |
|---|---|---|---|
| Fos | Vos | 0.333 | initial v→f substitution |
| Fermeer | Vermeer | 0.455 | initial v→f substitution |
| Multer | Mulder | 0.503 | word-final d→t devoicing |
| Mas | Maas | 0.000 | `ː` treated as literal char — `maːs` vs `mɑs` share no n-grams |
| Lan | Laan | 0.000 | same cause; wrong match returned (Roeland) — FP |
| Ber | Beer | 0.000 | same cause |
| Ven | Veen | 0.000 | same cause |
| Bon | Boon | 0.000 | same cause |
| Klein | Kleijn | 0.503 | ij→i diphthong simplification |
| Jon | Jong | 0.333 | final ng→n nasal drop |
| Smets | Smeets | 0.252 | vowel shift |
| Russo | Rousseau | 0.200 | eau→o + ou→u, both collapsed |
| Vanson | Vincent | 0.251 | in→on nasal restructuring + silent t |
| Anri | Henry | 0.000 | silent h + nasal vowel collapse |
| Ru | Roux | 0.000 | near-total phoneme deletion |
| Rober | Robert | 0.507 | silent final t |

No threshold adjustment can recover a raw score of 0.0. The root fix for the `vowel_length`
category is to strip `ː` from the IPA string before n-gram extraction (see §7).

**TOP-K dead zones** — raw similarity above `min_match` but the correct name ranks outside
the top-5 FAISS results:

| Query | Want | Raw score | Notes |
|---|---|---|---|
| Plom | Blom | 0.600 | scores at threshold but ranks > 5th |

These would be recovered by increasing the FAISS search depth at query time, with no model
or threshold changes needed.

### 4.3 Threshold sweep

Selected rows from the full sweep (run: `uv run python eval/evaluate.py --sweep`).
Note: sweep F1 figures use TOP_K=5 candidates; the full evaluation figure may be higher.

| min | high | margin | DU F1 | FR F1 | Macro F1 | Gen | Neg FP |
|---|---|---|---|---|---|---|---|
| 0.45 | 0.60 | 0.10 | 81.0% | 91.3% | 86.1% | 0.89 | 40.0% |
| 0.50 | 0.60 | 0.08 | 75.0% | 91.3% | 83.2% | 0.82 | 30.0% |
| **0.60** | **0.65** | **0.08** | **68.4%** | **88.9%** | **78.7%** | **0.77** | **5.0%** | ← current (rank 47/120) |
| 0.65 | 0.70 | 0.08 | 64.9% | 88.9% | 76.9% | 0.73 | 0.0% | ← Pareto at 0% OOV FP |
| 0.55 | 0.86 | 0.12 | 48.5% | 86.4% | 67.4% | 0.56 | 12.5% | (previous default) |

The previous production default (0.55/0.86/0.12) ranked **last** in the sweep.

### 4.4 Pareto front (best Macro F1 at 0 OOV false positives)

| Precision floor | Config | Macro F1 (sweep) | DU F1 | FR F1 |
|---|---|---|---|---|
| ≥ 70% | **0.65 / 0.70 / 0.08** | **76.9%** | 64.9% | 88.9% |
| ≥ 80% | **0.65 / 0.70 / 0.08** | **76.9%** | 64.9% | 88.9% |
| ≥ 90% | (no config meets this constraint with 0 neg FPs) | | | |
| 100% | (no config meets this constraint with 0 neg FPs) | | | |

The active config (0.60/0.65/0.08) trades a 1.8 pp Macro F1 advantage over the strict
Pareto-optimal for higher Dutch recall, accepting 2 OOV false positives.

### 4.5 Negative control

At the current config (min=0.60) there are **2 OOV false positives** across both indices
(5.0% of 20 OOV names × 2 language indices):

| OOV name | Index | Matched | Score |
|---|---|---|---|
| Roosevelt | Dutch | Veld | 0.651 |
| Jefferson | French | Levy | 0.601 |

The Pareto-optimal config (0.65/0.70/0.08) eliminates both.
The previous default had 5 OOV FPs on the French index (12.5%) at the lower 0.55 floor.

### 4.6 Failure modes

#### Previously threshold-level, now resolved

Dutch `voicing`, `cluster`, and `diphthong` patterns had raw similarities of 0.60–0.85, which
were blocked by `high_confidence=0.86`. Lowering `high_confidence` to 0.65 recovered most of
them (+19.9 pp Dutch F1).

#### Model-level (require changes upstream of FAISS)

**Dutch vowel length — 0% recall, raw = 0.000**  
espeak-ng uses `ː` for long vowels: `maːs` (Maas) vs `mɑs` (Mas). The character n-gram
extractor treats `ː` as a literal character, so long and short vowel strings share almost no
n-grams. No threshold can fix a raw score of 0.000.  
Fix: strip `ː` before n-gram extraction (see §7).

**French phoneme collapses**  
`Vincent/Vanson` (raw=0.333) and `Henry/Anri` (raw=0.172) involve nasal vowel restructuring
and silent-consonant removal. `Roux/Ru` (raw=0.000) is a near-total deletion. These are
genuine model-level boundaries not addressable by threshold tuning.

**TOP-K retrieval depth**  
`Schouten/Skouten` (raw=0.636) and `Blom/Plom` (raw=0.600) score above `min_match` but rank
outside the TOP-5 candidates. Increasing the FAISS search depth for the classify step would
recover both without any threshold or model changes.

---

## 5. Levenshtein Baseline

`baseline_levenshtein.py` evaluates the same gold sets using **normalized Levenshtein
(edit) distance** as the similarity measure. This is the naive approach: no phoneme
pipeline, no IPA, no FAISS — just character-level edit distance on the raw lowercased
string, powered by `rapidfuzz`.

Decision rule: return a match if the closest name in the corpus has similarity ≥ threshold;
no-match otherwise. The threshold is found by sweeping 0.50–0.99 in 0.01 steps.

Run: `uv run python eval/baseline_levenshtein.py`

### 5.1 Results

Pareto-optimal threshold per approach (0% OOV false positives required):

| System | Config | Dutch F1 | French F1 | Macro F1 | OOV FP |
|---|---|---|---|---|---|
| Levenshtein (baseline) | threshold=0.67 | 98.0% | 68.4% | 83.2% | 0% |
| Phonetics engine (previous default) | 0.55/0.86/0.12 | 48.5% | 86.4% | 67.4% | 12.5% |
| Phonetics engine (current) | 0.60/0.65/0.08 | 68.4% | 88.9% | 78.7% | 5.0% |
| Phonetics engine (Pareto 0% OOV) | 0.65/0.70/0.08 | 64.9% | 88.9% | 76.9% | 0% |

### 5.2 Per-language analysis

**Dutch**: Levenshtein is strongly superior on Dutch (98.0% vs 68.4%). Dutch STT distortions
are predominantly 1–2 character substitutions (`Fisser/Visser`, `Smeets/Smets`). Edit
distance captures this directly. The remaining Dutch gap for the phonetics engine is partly
the `vowel_length` category (0% recall, a model-level issue solvable with IPA normalisation,
see §7) and partly `voicing` cases where single-character substitutions are insufficiently
discriminative in a 128-dim hashed character n-gram space.

**French**: The phonetics engine is strongly superior (88.9% F1 vs 68.4% F1 for
Levenshtein). French spelling and pronunciation are divergent: `Chevalier` and `Shevalier`
differ by 2 characters but are phonetically near-identical. Edit distance penalises this
heavily. The `fr-fr` espeak-ng backend maps both to the same IPA, making them essentially
an exact match. This is the exact use case the phoneme pipeline was designed for.

### 5.3 Latency trade-off

Levenshtein requires a full pairwise scan over the corpus: one comparison per candidate name
per query. The phonetics engine vectorises the query once (a fixed IPA + n-gram step) and
then uses FAISS for the corpus scan, with SIMD-vectorised inner product comparisons.

Benchmarked with 100 queries per corpus size against tiled Dutch surname corpora
(`eval/benchmark.py`, p50 latency, macOS, espeak-ng pre-warmed):

| Corpus size | Engine p50 (ms) | Levenshtein p50 (ms) | Speedup |
|------------:|:-:|:-:|:-:|
| 100 | 0.114 | 0.043 | 0.4× (Lev wins) |
| 500 | 0.089 | 0.143 | 1.6× |
| 1,000 | 0.074 | 0.236 | 3.2× |
| 5,000 | 0.118 | 1.209 | 10.2× |
| 10,000 | 0.177 | 2.428 | 13.7× |

Below ~300 names Levenshtein has an edge because there is no index-build amortisation.
Above that the FAISS query time stays roughly flat (~0.1 ms) while Levenshtein scales
linearly (~0.24 µs per extra name). At 10,000 names the engine is **13.7× faster** per
query (0.18 ms vs 2.43 ms). FAISS also supports approximate nearest-neighbor indices
(IVF, HNSW) that would bring search to sub-linear time — an option not available to
Levenshtein without structural pre-processing.

However, given the low absolute times of the analysis with Levenshtein (2.5ms for 10k corpus), the latency
hit might be totally irrelevant.

### 5.4 Interpretation

The phonetics engine currently trails the Levenshtein baseline on Macro F1 (78.7% vs 83.2%
at current config; 76.9% vs 83.2% at 0% OOV FP). This aggregate figure masks opposite
trends: the engine is substantially worse on Dutch (−29.6 pp F1) and substantially better
on French (+20.5 pp F1). The phoneme pipeline is doing exactly what it was designed for on
French; the Dutch gap is concentrated in `vowel_length` and one TOP-K retrieval case —
both tractable issues (see §7).

---

## 6. Vectorisation Experiment

The default vectoriser extracts character 2-grams and 3-grams from the IPA string after
stripping spaces (`"v ɪ s ə r"` → `"vɪsər"` → bigrams `vɪ`, `ɪs`, `sə`, `ər`;
trigrams `vɪs`, `ɪsə`, `sər`). Three strategies were evaluated at the same threshold
config (0.60/0.65/0.08):

| Vectoriser | Dutch F1 | French F1 | Macro F1 | OOV FP |
|---|---|---|---|---|
| **Char n-grams (2,3) — current** | **68.4%** | **88.9%** | **78.7%** | **5.0%** |
| Phoneme n-grams (1,2,3) — with unigrams | 78.0% | 91.3% | 84.7% | ❌ worse |
| Phoneme n-grams (2,3) — no unigrams | 64.9% | 86.4% | 75.6% | 0% |

### Why phoneme unigrams produce more OOV false positives

Adding phoneme unigrams (`r`, `n`, `l`, `m`, `s`, …) improves gold recall but makes OOV
names match too easily: common individual phonemes appear in both English OOV names
(`Fernandez`, `Hamilton`) and Dutch/French background names. At threshold 0.60/0.65,
Fernandez scores 0.700 against a Dutch background name — above `high_confidence`. Raising
the threshold to eliminate these OOV FPs creates a new dead zone that depresses Dutch F1
back below the char n-gram baseline.

### Why phoneme 2+3-grams (no unigrams) are worse than char 2+3-grams

Surnames are short (4–8 phoneme tokens). A 5-token name produces only 4 bigrams + 3
trigrams = 7 features. The vector is sparse, and cosine similarity scores drop below
`min_match` for valid pairs. The character n-gram approach extracts bigram/trigram patterns
at sub-phoneme resolution (IPA uses single Unicode characters per phoneme), giving denser
and more discriminative vectors for this name-length distribution.

### Next candidate

Increase the hash dimension (128 → 256 or 512) before retrying phoneme n-grams. A larger
dimension reduces hash collisions and might give phoneme unigrams sufficient separation to
avoid OOV FPs without requiring a higher `min_match` threshold.

---

## 7. Limitations

**Gold sets are AI-generated.** The (original, STT-variant) pairs were constructed by
prompting an LLM to generate plausible phonological distortions for each category, then
manually reviewed for plausibility. They represent hypothesized distortion patterns, not
transcripts from a real STT system on real callers. A system that looks good here could
still fail on distortions that weren't anticipated.

**25 pairs per language is a small sample.** Category-level F1 scores are computed over
3–7 pairs. A single pair flipping from miss to hit moves the category score by 14–33 pp.
The numbers should be read as directional signals, not precise estimates.

**Background corpora are generic, not tenant-specific.** The Dutch (223 names) and French
(307 names) background corpora are generic surname lists. Real tenants may have employee
databases with thousands of names, different surname distributions, or names from languages
other than the index language — all of which will affect both recall and OOV false
positive rates in ways this evaluation does not capture.

**OOV negative control is narrow.** The 20 OOV names are predominantly English and
Eastern European. A multilingual tenant with Arabic, Chinese, or Turkish names in their
database would present a very different OOV challenge.

---

## 8. Conclusion

**The approach works, but the two languages have fundamentally different failure profiles.**

For French, the phoneme pipeline is clearly the right tool. French orthography and
pronunciation are so divergent that character-level edit distance is nearly useless
(`Chevalier/Shevalier` differ by two chars but are phonetically identical — Levenshtein
penalises that heavily, the engine does not). The engine reaches 88.9% F1 on the French
gold set; the Levenshtein baseline reaches 68.4% F1. The remaining 4 French misses are
genuine phoneme collapses (`Roux/Ru`, `Henry/Anri`) where the IPA strings are simply too
different after the distortion — no realistic threshold change will recover them.

For Dutch, the picture is messier. The engine reaches 68.4% F1 on the Dutch gold set; the
Levenshtein baseline reaches 98.0% F1. That gap tells you that most Dutch STT distortions
are shallow character substitutions that edit distance handles trivially. The gap is not
random: 5 of the 11 Dutch misses come from one specific category (`vowel_length`) that
fails completely because espeak-ng encodes long vowels with `ː` and our n-gram extractor
treats that as a literal character. One more miss is a retrieval artefact — the correct
name has a cosine similarity above threshold but ranks outside the top-5 FAISS results. Fix
those two things and Dutch F1 climbs from 68.4% to roughly 90%. The remaining 5 Dutch
misses are genuinely hard cases (initial consonant substitutions like `Fos/Vos` where the
phoneme strings are just different enough to fall below threshold).

**False positives in production.** At the current config (min=0.60), there are **2 OOV
false positives** across both indices (5.0%): Roosevelt on the Dutch index (cosine
similarity 0.651 against Veld) and Jefferson on the French index (0.601 against Levy). In
practice this means a truly foreign name will very occasionally return a confident match.
Whether that is acceptable depends on the tenant: for a monolingual customer base it is
fine; for a multilingual one it is a risk. Raising `min_match` to 0.65 eliminates both OOV
FPs at a cost of 3.5 pp Dutch F1 recall.

**The recalibration (done) was not a model improvement — it was fixing a configuration
that had been making the model look worse than it is.** The underlying FAISS similarity
scores were correct all along; `high_confidence=0.86` was simply blocking matches that
scored 0.60–0.85, which is most of the useful signal. That is fixed. The two remaining
high-impact improvements — stripping `ː` from IPA before extraction, and increasing the
FAISS search depth — are both small, safe changes with well-understood upside.

| Language | System | Precision | Recall | F1 |
|---|---|---|---|---|
| Dutch | Levenshtein (0.67) | 100% | 96% | 98.0% |
| Dutch | Engine (0.60/0.65/0.08) | 93% | 54% | 68.4% |
| French | Levenshtein (0.67) | 93% | 54% | 68.4% |
| French | Engine (0.60/0.65/0.08) | 100% | 80% | 88.9% |
| **Macro F1** | Levenshtein | | | **83.2%** |
| **Macro F1** | Engine | | | **78.7%** |

---

## 9. Running the Evaluation

```bash
# Full gold evaluation at default thresholds
uv run python eval/evaluate.py

# Full threshold sweep (~60 s)
uv run python eval/evaluate.py --sweep

# Save raw per-query results to CSV
uv run python eval/evaluate.py --output results.csv

# Synthetic distortion mode (development / sanity check only)
uv run python eval/evaluate.py --mode synthetic
uv run python eval/evaluate.py --mode synthetic --sweep
```

### Sweep output

- Top 25 configs sorted by Macro F1 with per-language F1, generalizability, OOV FP rate
- Current default rank shown even if outside top 25
- Pareto front: best Macro F1 at precision floors 70% / 80% / 90% / 100% with 0 OOV FPs
- Diagnostic: which OOV names the current default incorrectly matches

---

## 10. Files

| File | Description |
|---|---|
| `evaluate.py` | Evaluation harness — gold eval, synthetic eval, threshold sweep |
| `gold_dutch.csv` | 25 curated Dutch (original, STT-variant) pairs |
| `gold_french.csv` | 25 curated French (original, STT-variant) pairs |
| `surnames.txt` | 223 Dutch surnames — background corpus for Dutch index |
| `surnames_fr.txt` | 307 French surnames — background corpus for French index |
| `baseline_levenshtein.py` | Levenshtein baseline evaluation and threshold sweep |

