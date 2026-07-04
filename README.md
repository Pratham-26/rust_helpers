# polars-stringsim

Composable fuzzy string matching for [Polars](https://pola.rs/), implemented as
a native Polars plugin (Rust core + Python bindings via PyO3). All scoring runs
in Rust — no Python loops over rows — and works in both eager and lazy frames.

## Features

- **24 metric expressions** — edit distances (Levenshtein, Damerau-Levenshtein, OSA, Hamming), the Jaro family, token/n-gram (Jaccard, Sørensen-Dice, q-grams), LCS, and four phonetic encoders (Soundex, Metaphone, DoubleMetaphone, NYSIIS). See [`ALGORITHMS.md`](ALGORITHMS.md) for the math and normalization behind each one.
- **Hybrid scoring** — combine multiple algorithms in a single Rust call (`hybrid_score`), plus 4 pre-built scorers (`name_default`, `phonetic_edit`, `token_char`, `prefix_ngram`).
- **DataFrame helpers** — `fuzzy_join`, `deduplicate`, `pairwise_compare` with blocking indexes (first-chars, char-bag) to avoid O(n²) cross joins.
- **Explainability** — `return_breakdown=True` returns per-metric scores alongside the combined score, so you can see *why* two strings matched.
- **Ensembles** — `weighted_avg`, `mean`, `max`, `min`, `median`, `vote`.

All similarity functions return `Float64` in `[0, 1]`. Null in either input → null output.

## Install

### Users (no Rust toolchain required)

Prebuilt wheels are published on PyPI for **CPython 3.9–3.13** on
**Linux x86_64** and **Windows x86_64**. macOS and aarch64-linux wheels are
planned (see [Platform support](#platform-support)); on those platforms pip
falls back to a source build.

```bash
pip install polars-stringsim
```

That's it — `import polars_stringsim as pf` works out of the box.

> If a wheel for your platform is missing, pip will fall back to a source
> build, which **does** require Rust (see below).

### From source / development

For hacking on the plugin itself, or for a platform without a prebuilt wheel:

```bash
git clone https://github.com/Pratham-26/rust_helpers.git
cd rust_helpers

# Iterative dev install (rebuild on every change):
pip install maturin
maturin develop --release

# Or: install directly from the repo's source (compiles Rust, needs a toolchain):
pip install git+https://github.com/Pratham-26/rust_helpers.git
```

Requires Rust stable. Pinned to `polars = 0.54.4` / `pyo3-polars = 0.27`.
See [`RELEASE.md`](RELEASE.md) for how wheels are built and published.

## Usage

```python
import polars as pl
import polars_stringsim as pf

customers = pl.DataFrame({"name": ["Robert Smith", "Catherine Jones", "Jon Smyth"]})
db = pl.DataFrame({"name": ["Robert Smyth", "Katherine Jones", "William Brown"]})

# 1. Single metric
customers.join(db, how="cross").with_columns(
    s=pf.jaro_winkler("name", "name_right")
)

# 2. Hybrid: spelling + phonetic + token, fused in one Rust call
df.with_columns(
    hybrid=pf.hybrid_score("a", "b",
        algorithms=["jaro_winkler", "double_metaphone", "trigram_jaccard"],
        weights=[0.5, 0.3, 0.2])
)

# 3. Pre-built scorer
df.with_columns(s=pf.name_default("a", "b"))   # JW + Double Metaphone + trigram

# 4. Per-metric breakdown (explainability)
df.with_columns(
    bd=pf.combine(
        [pf.jaro_winkler("a","b"), pf.double_metaphone_sim("a","b")],
        weights=[0.6, 0.4], return_breakdown=True,
    )
)

# 5. fuzzy_join with blocking (avoids O(n*m) cross product)
pf.fuzzy_join(customers, db, left_on="name", right_on="name",
    algorithms=["jaro_winkler", "double_metaphone"], weights=[0.6, 0.4],
    threshold=0.75, top_k=1, block="first_chars", block_n=1)

# 6. deduplicate near-duplicate name variants
pf.deduplicate(messy_names, on="name",
    algorithms=["jaro_winkler"], weights=[1.0],
    composite_threshold=0.8, block="first_chars", block_n=1)

# 7. pairwise_compare for threshold tuning (returns combined score per pair)
pf.pairwise_compare(customers, db, left_on="name", right_on="name",
    algorithms=["jaro_winkler", "trigram_jaccard"], weights=[0.6, 0.4])
```

## API reference

### Per-metric expressions (`pf.<name>(left, right) → pl.Expr`)

`jaro`, `jaro_winkler`, `levenshtein`, `levenshtein_norm`, `damerau_levenshtein`,
`damerau_levenshtein_norm`, `osa`, `hamming`, `hamming_norm`, `token_jaccard`,
`token_sorensen_dice`, `trigram_jaccard`, `trigram_sorensen_dice`,
`qgram_jaccard(left, right, q=3)`, `lcs_sim`, `soundex_sim`, `soundex_jw_sim`,
`metaphone_sim`, `metaphone_jw_sim`, `double_metaphone_sim`,
`double_metaphone_jw_sim`, `nysiis_sim`, `nysiis_jw_sim`.

### Combiners / hybrid

- `pf.combine(metrics, *, weights=None, method="weighted_avg", threshold=None, return_breakdown=False)` — fuse pre-built metric expressions.
- `pf.hybrid_score(left, right, *, algorithms, weights=None, method="weighted_avg", threshold=None)` — same, but builds metrics in Rust (no intermediate struct column).
- **Pre-built scorers**: `pf.phonetic_edit`, `pf.token_char`, `pf.prefix_ngram`, `pf.name_default`.

### DataFrame helpers

- `pf.fuzzy_join(left, right, *, left_on, right_on, algorithms, weights, method, threshold, top_k, block, block_n, how, add_breakdown)`
- `pf.deduplicate(frame, *, on, algorithms, weights, method, composite_threshold, block, block_n)`
- `pf.pairwise_compare(left, right, *, left_on, right_on, algorithms, weights, method, block, block_n)`
- **Blocking**: `pf.block_first_chars(col, n=2)`, `pf.block_char_bag(col)`

### Combine methods

`weighted_avg` (default; weights normalized to sum to 1), `mean`, `max`, `min`, `median`, `vote` (count of metrics ≥ `threshold`, normalized by N).

## Algorithm name registry

`hybrid_score`, `fuzzy_join`, `deduplicate`, and `pairwise_compare` accept algorithm names:

`jaro`, `jaro_winkler`, `levenshtein`, `levenshtein_norm`, `damerau_levenshtein`, `damerau_levenshtein_norm`, `osa`, `hamming`, `hamming_norm`, `token_jaccard`, `token_sorensen_dice`, `trigram_jaccard`, `trigram_sorensen_dice`, `lcs_sim`, `soundex`/`soundex_jw`, `metaphone`/`metaphone_jw`, `double_metaphone`/`double_metaphone_jw`, `nysiis`/`nysiis_jw`.

For what each metric computes and how it's normalized, see [`ALGORITHMS.md`](ALGORITHMS.md).

## Platform support

Prebuilt wheels on PyPI (no Rust toolchain needed):

| Platform | Wheel | Status |
|---|---|---|
| Linux x86_64 | `manylinux_2_28_x86_64` | ✅ v0.1.0 |
| Windows x86_64 | `win_amd64` | ✅ v0.1.0 |
| macOS x86_64 / arm64 | — | ⏳ planned (sdist fallback works; needs Rust) |
| Linux aarch64 | — | ⏳ planned (sdist fallback works; needs Rust) |

CPython 3.9–3.13 supported wherever a wheel exists. `pip install polars-stringsim` picks the right wheel automatically; on uncovered platforms it builds from the sdist (requires Rust stable).

## Tests

```bash
cargo test --lib          # 30 Rust unit tests
pytest tests/python       # 38 Python end-to-end tests
```

Run the example end-to-end with `uv`:

```bash
maturin build --release
WHL=target/wheels/polars_stringsim-*.whl
uv run --with "$WHL" --with polars --with pytest python -m pytest tests/python/
uv run --with "$WHL" --with polars python examples/record_linkage.py
```

## Architecture

```
src/
├── algorithms/   # pure Rust: edit, jaro, token, lcs, phonetic
├── combiner.rs   # CombineMethod enum + combine_row (weighted_avg/mean/max/min/median/vote)
├── expr/mod.rs   # #[polars_expr] wrappers, one per metric
├── expr_combine.rs   # combine_expr + combine_breakdown_expr (Struct output)
├── expr_hybrid.rs    # hybrid_score_expr (multi-algo in one Rust call)
├── series_util.rs    # str-column readers, Float64 builder, null handling
└── lib.rs        # #[pymodule]

python/polars_stringsim/
├── _expression.py  # per-metric expr builders + combine()
├── _registry.py    # algorithm name → builder map
├── hybrid.py       # hybrid_score + pre-built scorers
├── frame.py        # fuzzy_join, deduplicate, pairwise_compare, blocking
└── __init__.py     # public API
```

## Features & status

Everything in the original PRD is implemented and shipped:

- ✅ **24 metric expressions** — edit distances, Jaro family, token/n-gram, LCS, and phonetic encoders. Full reference: [`ALGORITHMS.md`](ALGORITHMS.md).
- ✅ **Composable ensembles** — `combine()` with `weighted_avg` / `mean` / `max` / `min` / `median` / `vote`.
- ✅ **Hybrid scoring** — `hybrid_score()` (builds metrics in Rust, no intermediate struct column) + 4 pre-built scorers.
- ✅ **DataFrame helpers** — `fuzzy_join` (blocked/cross, threshold, `top_k`, `how`), `deduplicate` (union-find clustering), `pairwise_compare`, and blocking indexes.
- ✅ **Explainability** — `return_breakdown=True` returns a per-metric score breakdown.
- ✅ **Native Polars plugin** — all scoring in Rust via PyO3; works in eager and lazy frames.
- ✅ **Prebuilt wheels on PyPI** — `pip install polars-stringsim`, no Rust toolchain required (see [Platform support](#platform-support)).

### Not yet implemented

- ⏳ **Platform wheels**: macOS (x86_64 + arm64) and aarch64-linux — currently sdist-only on those platforms.
- 🔲 Custom combiner registration (user-supplied Rust closures).
- 🔲 GPU acceleration.
- 🔲 More phonetic encoders (Caverphone, Beider-Morse).
