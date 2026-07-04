# PRD: Polars Fuzzy Name Matching Extension (`polars-fuzzy`)

## 1. Overview / Problem Statement

Users performing record linkage, entity resolution, and name matching on
Polars DataFrames need fast, composable fuzzy string matching. Single
algorithms are insufficient for real-world name data; combining multiple
metrics (e.g., Jaro-Winkler for spelling + Metaphone for phonetics +
Jaccard for tokens) yields better precision/recall.

`polars-fuzzy` is a Polars plugin (Rust core + Python bindings via PyO3
+ `pyo3-polars`) that exposes a comprehensive suite of fuzzy matching
algorithms as Polars expressions, with first-class support for combining
multiple metrics into a single hybrid score.

## 2. Objectives

- Expose a comprehensive set of fuzzy matching algorithms as native
  Polars expressions (`pl.col`-based, vectorized in Rust).
- Provide strong support for **multi-algorithm combination** (hybrid
  scoring, ensembles, weighted fusion).
- Keep all hot paths in Rust — no Python loops over rows.
- Support both eager and lazy Polars frames.
- Be ergonomic: simple cases are one-liners; advanced cases are
  composable.

## 3. Target Users

- Data engineers doing record linkage / deduplication at scale.
- Data scientists cleaning name fields in Polars pipelines.
- Anyone who has outgrown single-metric libraries like `polars-strsim`
  or `fuzzyrust` and needs composability.

## 4. Core Features

### 4.1. Algorithms (Rust Core)

Comprehensive list:

- **Edit distances**: Levenshtein, Damerau-Levenshtein, normalized
  Levenshtein, OSA.
- **Jaro family**: Jaro, Jaro-Winkler.
- **Phonetic**: Soundex, Double Metaphone, Metaphone, NYSIIS (exposed as
  similarity via set comparison of codes).
- **Token / n-gram**: Jaccard and Sørensen-Dice over tokens, character
  bigrams (trigram), q-grams.
- **Other**: Hamming, LCS-based similarity, Sørensen-Dice.

All algorithms return a similarity score in `[0, 1]` (where applicable),
or the raw code for phonetic encoders when used standalone.

### 4.2. Combination / Hybrid Features (Key Emphasis)

Rust-native combiners for performance:

- **Weighted Average / Sum**: User-defined weights for multiple metrics.
  - Example: `0.6 * jaro_winkler + 0.3 * metaphone_jaccard + 0.1 * levenshtein_norm`

- **Hybrid Scorers (pre-built)**:
  - Phonetic + Edit (e.g., Soundex code match + Jaro-Winkler on original).
  - Token + Character level.
  - Prefix-boosted + n-gram.

- **Ensemble Methods**:
  - Max / Min / Mean / Median across metrics.
  - Voting (count how many metrics exceed threshold).
  - Custom combiner functions (Rust closures or Python-callable).

- **Multi-Metric Expressions**:
  ```python
  df.with_columns(
      combined_score=pf.combine(
          [
              pf.jaro_winkler("name_a", "name_b"),
              pf.double_metaphone_jaccard("name_a", "name_b"),
              pf.levenshtein_norm("name_a", "name_b")
          ],
          weights=[0.5, 0.3, 0.2],
          method="weighted_avg"  # or "mean", "max", etc.
      )
  )
  ```

- **Thresholding & Ranking**:
  - Filter by combined score.
  - Return top-k candidates per record with per-metric breakdowns (for
    debugging/explainability).

### 4.3. DataFrame-level Helpers (Enhanced)

- `fuzzy_join(..., metrics=["jaro_winkler", "metaphone"], weights=..., combiner="weighted")`
- `deduplicate(..., composite_threshold=0.85)` — uses combined score internally.
- Batch pairwise comparison between two columns/DataFrames with multi-metric
  output (Struct column with scores per algorithm + combined).

### 4.4. Configuration & Extensibility

- Config objects/structs for reusable combiners.
- Allow users to register custom Rust combiners via plugins.
- Normalization options before combining (e.g., all metrics to `[0,1]`
  similarity).

## 5. Technical Implementation Notes

- **Rust Side**: Use enums for metrics, trait-based scorers, and a
  `Combiner` struct that takes a vec of scores + weights. Leverage
  `polars` `Series` ops for vectorized combination where possible.
- **Python API**: High-level `pf.combine()` + namespace methods that
  call into Rust.
- **Performance**: All combinations stay in Rust (no Python loops).
  Support for lazy evaluation.
- **Existing Packages**: Build on lessons from `polars-strsim` /
  `fuzzyrust` but expand heavily on composability.

## 6. Usage Examples

```python
import polars as pl
import polars_fuzzy as pf

# Simple hybrid expression
df = df.with_columns(
    hybrid=pf.hybrid_score(
        pl.col("name_a"), pl.col("name_b"),
        algorithms=["jaro_winkler", "soundex_jaccard"],
        weights=[0.7, 0.3]
    )
)

# Advanced join with combination
result = pf.fuzzy_join(
    left, right,
    left_on="customer_name",
    right_on="db_name",
    algorithms=["jaro_winkler", "double_metaphone", "trigram_jaccard"],
    weights=[0.5, 0.3, 0.2],
    threshold=0.75,
    top_k=5
)
```

## 7. Phasing

- **MVP**: Core algorithms + basic expressions + simple weighted combiner.
- **Phase 2**: Full hybrids, DataFrame helpers, phonetic integration.
- **Phase 3**: Advanced ensembles, custom combiners, explainability.

## 8. Other

- **Non-functional**: Sub-linear scaling where possible; multi-threaded
  via Polars' thread pool; deterministic outputs; no global state.
- **Architecture**: Cargo workspace with Rust crate (`src/`) + Python
  package (`python/`) built via `maturin develop` / `maturin build`.
- **Risks**: Polars plugin API churn (pin versions); phonetic algorithm
  correctness vs. reference implementations; performance of pairwise
  joins on large frames (mitigate with blocking/indexing in Phase 2+).
