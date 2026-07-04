# Algorithms

`polars-stringsim` ships 24 similarity metrics, all returning `Float64` in
`[0, 1]` (1.0 = identical). Distance-only variants (`levenshtein`,
`damerau_levenshtein`, `osa`, `hamming`) return a raw `Float64` count instead.

This document explains what each metric does, how it's normalized, and how it
behaves on edge cases. For the Python API, see the
[README](README.md#api-reference).

> **Convention:** every `[0,1]` similarity returns **1.0 for two empty
> strings** and **0.0 when only one side is empty** (token/Jaro/phonetic/LCS).
> Null in either Polars input column → `null` output.

---

## Contents

- [Edit distances](#edit-distances) — Levenshtein, Damerau-Levenshtein, OSA, Hamming
- [Jaro family](#jaro-family) — Jaro, Jaro-Winkler
- [Token & n-gram](#token--n-gram) — Jaccard, Sørensen-Dice, q-grams, trigrams
- [Subsequence](#subsequence) — LCS
- [Phonetic encoders](#phonetic-encoders) — Soundex, Metaphone, DoubleMetaphone, NYSIIS
- [Choosing a metric](#choosing-a-metric)

---

## Edit distances

All delegate to the [`strsim`](https://crates.io/crates/strsim) crate.

### `levenshtein` / `levenshtein_norm`

Minimum number of single-character **insertions, deletions, or substitutions**
to turn `a` into `b`.

| Function | Returns | Formula |
|---|---|---|
| `pf.levenshtein(a, b)` | raw distance (count) | — |
| `pf.levenshtein_norm(a, b)` | similarity in `[0,1]` | `1 − d / max(|a|, |b|)` |

> Reference: `levenshtein("kitten", "sitting") = 3`,
> `levenshtein_norm("kitten", "sitting") = 1 − 3/7 ≈ 0.571`.
> `levenshtein_norm("", "") = 1.0`.

### `damerau_levenshtein` / `damerau_levenshtein_norm`

Like Levenshtein, but also counts **adjacent transpositions** (swapping two
neighbouring characters) as a single operation. Lower than or equal to
Levenshtein when transpositions help.

| Function | Returns | Formula |
|---|---|---|
| `pf.damerau_levenshtein(a, b)` | raw distance | — |
| `pf.damerau_levenshtein_norm(a, b)` | similarity in `[0,1]` | `1 − d / max(|a|, |b|)` |

> Reference: `damerau_levenshtein("ca", "ac") = 1` (one transposition),
> whereas Levenshtein would cost 2 (two substitutions).
> `damerau_levenshtein_norm("", "") = 1.0`.

### `osa` (Optimal String Alignment)

Damerau-Levenshtein with a restriction: no substring may be edited more than
once. Exposed as a **raw distance only** (no normalized variant).

| Function | Returns |
|---|---|
| `pf.osa(a, b)` | raw distance (count) |

> Reference: `osa("ca", "ac") = 1`.

### `hamming` / `hamming_norm`

Counts positions at which two equal-length strings differ. **Unlike standard
Hamming, mismatched lengths do not error** — the distance degrades gracefully
to `max(|a|, |b|)`.

| Function | Returns | Formula |
|---|---|---|
| `pf.hamming(a, b)` | raw distance (count) | — |
| `pf.hamming_norm(a, b)` | similarity in `[0,1]` | `1 − d / max(|a|, |b|)` |

> Reference: `hamming("karolin", "kathrin") = 3`.
> `hamming("abc", "a") = 3` (graceful degradation, not an error).

---

## Jaro family

Delegate to `strsim`. Designed for short strings like personal names; rewards
matching characters within a matching window and accounts for transpositions.

### `jaro`

| Function | Returns | Formula |
|---|---|---|
| `pf.jaro(a, b)` | similarity in `[0,1]` | Jaro similarity (see below) |

The Jaro similarity is:

```
m == 0  →  0
else    →  (1/3) · ( m/|a| + m/|b| + (m − t)/m )
```

where `m` = matched characters (within a window of `⌊max(|a|,|b|)/2⌋ − 1⌋`)
and `t` = half the number of transposed matches.

> `jaro("", "") = 1.0`; `jaro("abc", "") = 0.0`; `jaro("abc", "abc") = 1.0`.

### `jaro_winkler`

Jaro similarity with a **prefix bonus**: equal characters in the first 4
positions boost the score by up to 0.1 × prefix_length × scaling (scaling =
0.1, capped at 4 chars). This makes it especially strong for names that share
a prefix (e.g. typos in surnames).

```
jw = jaro + (p · scaling · (1 − jaro))      where p ≤ 4
```

> Reference: `jaro_winkler("MARTHA", "MARHTA") ≈ 0.9611`
> (Jaro alone is 0.9444; the shared `MAR` prefix adds the bonus).

---

## Token & n-gram

Split strings into sets and compare set overlap. **Tokenization is
whitespace-only and case-sensitive** — upper-case and lower-case variants of
the same word count as different tokens. Pre-`.lower()` your columns if you
want case-insensitive matching.

### `token_jaccard` / `token_sorensen_dice`

Whitespace-split token sets `A`, `B`.

| Function | Formula |
|---|---|
| `pf.token_jaccard(a, b)` | `|A ∩ B| / |A ∪ B|` |
| `pf.token_sorensen_dice(a, b)` | `2·|A ∩ B| / (|A| + |B|)` |

> Reference: `token_jaccard("foo bar", "foo baz") = 1/3` (`{foo,bar}` vs
> `{foo,baz}`: ∩=1, ∪=3). `token_sorensen_dice("foo bar", "foo baz") = 0.5`.
> `("","") = 1.0`; `("foo","bar") = 0.0`.

### `trigram_jaccard` / `trigram_sorensen_dice`

Same as the token variants, but over the set of **character trigrams**
(3-character substrings) instead of whitespace tokens. Trigrams capture
character-level shape and tolerate word reordering, unlike token sets.

| Function | Formula |
|---|---|
| `pf.trigram_jaccard(a, b)` | `|A₃ ∩ B₃| / |A₃ ∪ B₃|` |
| `pf.trigram_sorensen_dice(a, b)` | `2·|A₃ ∩ B₃| / (|A₃| + |B₃|)` |

where `A₃` = trigram set of `a`. Strings shorter than 3 characters are padded
to a single gram equal to the whole string.

> Reference: `trigram_jaccard("abcd", "abcd") = 1.0`.

### `qgram_jaccard(a, b, q=3)`

Generalized n-gram Jaccard. The only metric that takes a parameter: `q`
(default 3). Trigram Jaccard is `qgram_jaccard` with `q=3`.

```python
df.with_columns(pf.qgram_jaccard("a", "b", q=2).alias("bigram_jaccard"))
```

---

## Subsequence

### `lcs_sim`

Length of the **longest common subsequence** (not necessarily contiguous),
normalized by the longer string's length.

```
lcs_sim = |LCS(a, b)| / max(|a|, |b|)
```

> Reference: `lcs_len("AGCAT", "GAC") = 2` (the subsequence `GA` or `AC`),
> so `lcs_sim("AGCAT", "GAC") = 2/5 = 0.4`.
> `lcs_sim("abc", "abc") = 1.0`; `lcs_sim("abc", "xyz") = 0.0`.

---

## Phonetic encoders

Phonetic algorithms encode strings by how they **sound**, so names with
different spellings but the same pronunciation get the same code
(`Robert` ≡ `Rupert` under Soundex). All encoders delegate to the
[`rphonetic`](https://crates.io/crates/rphonetic) crate.

Each encoder is exposed in **two flavors**:

| Suffix | Behaviour | Output range |
|---|---|---|
| `*_sim` (e.g. `soundex_sim`) | **exact match** of the encoded codes: `1.0` if equal, else `0.0` | `{0, 1}` |
| `*_jw_sim` (e.g. `soundex_jw_sim`) | **Jaro-Winkler** on the encoded codes — gives partial credit when codes are close but not identical | `[0, 1]` |

The `_jw_sim` variants are useful for fuzzy phonetic matching where a hard
match/miss is too coarse.

| Encoder | Algorithm | Notes |
|---|---|---|
| `soundex` | [Soundex](https://en.wikipedia.org/wiki/Soundex) | Classic letter→digit code; first letter preserved. `Robert` and `Rupert` both encode to `R163`. |
| `metaphone` | [Metaphone](https://en.wikipedia.org/wiki/Metaphone) | Improvement on Soundex; handles more English pronunciation rules. |
| `double_metaphone` | [Double Metaphone](https://en.wikipedia.org/wiki/Metaphone#Double_Metaphone) | Returns a primary code (secondary code is ignored here); more accurate for non-English names. |
| `nysiis` | [NYSIIS](https://en.wikipedia.org/wiki/NYSIIS) | NY State Identification and Intelligence System; better than Soundex for surname matching. |

> Edge cases (all phonetic metrics): `("","") = 1.0`;
> `(non-empty, "") = 0.0`.

**Registry aliases.** In `hybrid_score`, `fuzzy_join`, `deduplicate`, and
`pairwise_compare`, the short names `soundex`, `metaphone`, `double_metaphone`,
`nysiis` map to the `*_sim` variants, and `soundex_jw`, `metaphone_jw`, etc.
map to the `*_jw_sim` variants. The full name list is in the
[Algorithm name registry](README.md#algorithm-name-registry).

---

## Choosing a metric

| Use case | Recommended metrics |
|---|---|
| Typos in short strings / names | `jaro_winkler`, `levenshtein_norm` |
| Transposition-heavy typos (`teh` vs `the`) | `damerau_levenshtein_norm` |
| Word reordering (`John Smith` vs `Smith, John`) | `token_jaccard`, `trigram_jaccard` |
| Equal-length codes / identifiers | `hamming_norm` |
| Names that sound alike (`Robert` / `Rupert`) | `soundex_jw`, `double_metaphone_jw`, `nysiis_jw` |
| Robust name matching (default) | `pf.name_default` (pre-built: JW + double_metaphone + trigram) |
| Record linkage at scale | `hybrid_score` with `block=...` to avoid O(n²) |

When in doubt, start with [`hybrid_score`](README.md#hybrid-scoring) combining
`jaro_winkler` + `levenshtein_norm` + a phonetic metric, or use one of the
[pre-built scorers](README.md#hybrid-scoring) (`name_default`, `phonetic_edit`,
`token_char`, `prefix_ngram`).
