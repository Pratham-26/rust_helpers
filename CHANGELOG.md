# Changelog

All notable changes to **polars-stringsim** are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-07-04

### Changed
- **`hybrid_score` is now parallel and significantly faster.** The row scan now
  runs on the rayon thread pool (one contiguous index range per thread) instead
  of a single serial loop, and per-row metric scores are buffered in a
  stack-allocated `SmallVec<[f64; 8]>` instead of a heap `Vec`.

  Measured on the FEBRL4 record-linkage benchmark (3-metric ensemble:
  `jaro_winkler` + `trigram_jaccard` + `token_jaccard`):

  | workload | 0.1.0 | 0.2.0 | speedup |
  |---|---|---|---|
  | 50k pairs, default threads | 133 ms | **21 ms** | **6.3×** |
  | 200k pairs, default threads | — | **83 ms** | **5.4×** vs `combine()` |
  | 1M pairs, 1 thread  | 2738 ms | — | — |
  | 1M pairs, 16 threads | ~no scaling | **407 ms** | **6.7×** vs 1-thread |

  Thread scaling (1M pairs, `set_num_threads(n)`):

  | n threads | time | speedup |
  |---|---|---|
  | 1 | 3348 ms | 1.0× |
  | 4 | 1316 ms | 2.5× |
  | 8 | 684 ms | 4.9× |
  | 16 | 434 ms | 7.7× |

  `hybrid_score` was previously *slower* than the equivalent `combine()` over
  separate expressions (0.79× at 4 threads); it is now ~5× faster. Most of the
  win comes from eliminating the intermediate struct column plus the per-row
  heap allocation; the rest is rayon parallelism.

### Added
- **Tunable worker pool** for `hybrid_score`. Parallel metrics now run on a
  dedicated, swappable rayon pool that is independent of the Polars engine pool
  (`POLARS_MAX_THREADS`). Configure it from Python:

  ```python
  import polars_stringsim as pf
  pf.set_num_threads(8)        # use 8 threads for hybrid_score
  pf.get_num_threads()         # -> 8
  pf.set_num_threads(0)        # restore default (env var or all cores)
  ```

  Or at process start via env var:

  ```
  POLARS_STRINGSIM_THREADS=8 python your_script.py
  ```

  Resolution order: `set_num_threads()` > `POLARS_STRINGSIM_THREADS` > all
  logical cores. Values are clamped to 1024.

- `rayon` and `smallvec` as explicit dependencies (both were already in the
  transitive dependency tree via Polars).

## [0.1.0] — 2026-07-04

### Added
- 24 metric expressions: edit distances (Levenshtein, Damerau-Levenshtein, OSA,
  Hamming, normalized variants), Jaro family (Jaro, Jaro-Winkler), token/n-gram
  (token Jaccard, token Sørensen-Dice, q-gram/trigram Jaccard & Sørensen-Dice),
  LCS similarity, and phonetic encoders (Soundex, Metaphone, DoubleMetaphone,
  NYSIIS) with exact-match and Jaro-Winkler-on-codes similarity variants.
- `combine()` ensemble with `weighted_avg` / `mean` / `max` / `min` / `median`
  / `vote` combine methods.
- `hybrid_score()` multi-metric scoring (single Rust call) plus 4 pre-built
  scorers: `phonetic_edit`, `token_char`, `prefix_ngram`, `name_default`.
- DataFrame helpers: `fuzzy_join` (cross / blocked, `top_k`, threshold),
  `deduplicate` (union-find clustering), `pairwise_compare`, and blocking
  indexes (`block_first_chars`, `block_char_bag`).
- Explainability via `return_breakdown=True` (per-metric score struct).
- Prebuilt wheels on PyPI for CPython 3.9–3.13 on Linux x86_64 and Windows
  x86_64; sdist for all other platforms.

[0.2.0]: https://github.com/Pratham-26/rust_helpers/releases/tag/v0.2.0
[0.1.0]: https://github.com/Pratham-26/rust_helpers/releases/tag/v0.1.0
