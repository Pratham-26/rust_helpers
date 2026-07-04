"""DataFrame-level fuzzy matching helpers.

``fuzzy_join`` and ``deduplicate`` compose the expression-level API to do
record-linkage work. Large joins use a blocking index (first-N-chars or
character-block key) to avoid a full cross product.

All scoring stays in Rust via the plugin expressions; Python only orchestrates
the join shape.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import polars as pl

from polars_stringsim._registry import build_metrics
from polars_stringsim._expression import _validate_weights, combine
from polars_stringsim.hybrid import hybrid_score

ColLike = Union[str, "pl.Expr"]

_DEFAULT_ALGOS = ["jaro_winkler", "trigram_jaccard"]
_DEFAULT_WEIGHTS = [0.6, 0.4]


# ---------------------------------------------------------------------------
# Blocking helpers
# ---------------------------------------------------------------------------

def block_first_chars(col: ColLike, n: int = 2) -> pl.Expr:
    """Block key = first ``n`` characters, uppercased.

    Two records share a block iff their first ``n`` chars match. Cheap and
    effective for names where the start is usually reliable.
    """
    e = col if isinstance(col, pl.Expr) else pl.col(col)
    return e.str.to_uppercase().str.slice(0, n)


def block_char_bag(col: ColLike) -> pl.Expr:
    """Block key = sorted set of characters in the string.

    Order-independent: "Smith" and "tmihS" share a block. Coarser than
    first-chars; use when initials may be reordered. Empty/whitespace-only
    strings map to the empty-string block; nulls stay null.
    """
    e = col if isinstance(col, pl.Expr) else pl.col(col)
    # Strip whitespace and lowercase, then split into chars, unique, sort, join.
    # str.split("") on an empty string is version-fragile, so short-circuit
    # the empty case explicitly via when/then.
    cleaned = e.str.to_lowercase().str.replace_all(r"\s+", "", literal=False)
    return pl.when(cleaned.str.len_chars() == 0).then(pl.lit("")).otherwise(
        cleaned.str.split("")
        .list.unique()
        .list.sort()
        .list.join("")
    )


# ---------------------------------------------------------------------------
# fuzzy_join
# ---------------------------------------------------------------------------

def fuzzy_join(
    left: pl.DataFrame | pl.LazyFrame,
    right: pl.DataFrame | pl.LazyFrame,
    *,
    left_on: str,
    right_on: str,
    algorithms: Optional[Sequence[str]] = None,
    weights: Optional[Sequence[float]] = None,
    method: str = "weighted_avg",
    threshold: Optional[float] = None,
    top_k: Optional[int] = None,
    block: Optional[str] = None,
    block_n: int = 2,
    how: str = "inner",
    add_breakdown: bool = False,
) -> pl.DataFrame | pl.LazyFrame:
    """Fuzzy-join two frames on a string column using combined metrics.

    Parameters
    ----------
    left, right:
        Frames to join.
    left_on, right_on:
        String columns to match.
    algorithms:
        Algorithm names from the registry. Defaults to
        ``["jaro_winkler", "trigram_jaccard"]``.
    weights:
        Per-algorithm weights (default ``[0.6, 0.4]``).
    method:
        Combine method (see :func:`polars_stringsim.combine`).
    threshold:
        Minimum combined score to keep a pair. ``None`` keeps all.
    top_k:
        Keep only the top-k best right matches per left row (by score).
        ``None`` keeps all matches above ``threshold``.
    block:
        Blocking strategy to avoid a full cross join:

        - ``"first_chars"`` (default when set): block on first ``block_n``
          uppercased chars.
        - ``"char_bag"``: block on the sorted unique character set.
    block_n:
        Number of leading chars for ``block="first_chars"``.
    how:
        ``"inner"`` (default) or ``"left"``. ``"left"`` keeps unmatched left
        rows with null right columns.
    add_breakdown:
        If True, the ``score`` column is replaced by a ``breakdown`` struct
        ``{ score: Float64, scores: Struct<metric_0: Float64, ...> }`` giving
        each algorithm's per-pair score alongside the combined score. Scoring
        then routes through :func:`polars_stringsim.combine` (which materializes
        the per-metric columns) rather than the single-call ``hybrid_score``,
        so it is slightly slower but provides explainability.

    Returns
    -------
    Frame with left columns, right columns (suffixed ``_right`` where names
    collide), a ``score`` Float64 column, and optionally a ``scores`` struct.

    Notes
    -----
    Without ``block``, this performs a full cross join before filtering —
    O(n*m). For large frames always pass ``block``.
    """
    algos = list(algorithms) if algorithms is not None else list(_DEFAULT_ALGOS)
    if weights is not None:
        w = _validate_weights(weights, len(algos))
    else:
        w = list(_DEFAULT_WEIGHTS)

    # Suffix right columns to avoid collisions; rename the join key for clarity.
    right_renamed = right.rename({right_on: "__right_val__"})

    # Tag left rows with a stable row index so top_k can rank per *row* rather
    # than per distinct key value (the latter collapses duplicate keys into one
    # window and mis-splits their top-k slots). Dropped before returning.
    left_tagged = left.with_row_index("__left_row__")

    if block is not None:
        if block == "first_chars":
            l_block = block_first_chars(left_on, block_n)
            r_block = block_first_chars("__right_val__", block_n)
        elif block == "char_bag":
            l_block = block_char_bag(left_on)
            r_block = block_char_bag("__right_val__")
        else:
            raise ValueError(f"unknown block strategy {block!r}")
        joined = (
            left_tagged.with_columns(__block__=l_block)
            .join(
                right_renamed.with_columns(__block__=r_block),
                on="__block__",
                how="inner",
                suffix="_right",
            )
            .drop("__block__")
        )
    else:
        # Full cross product, then filter.
        joined = left_tagged.join(
            right_renamed, how="cross", suffix="_right"
        )

    if add_breakdown:
        # Explainability path: build per-algorithm metric columns and combine
        # them with return_breakdown=True, yielding a struct column
        # { score: Float64, scores: Struct<metric_i: Float64, ...> }.
        # Uses combine() rather than hybrid_score() so the per-metric scores are
        # materialized; slightly slower but observable.
        scored = joined.with_columns(
            breakdown=combine(
                build_metrics(algos, left_on, "__right_val__"),
                weights=w, method=method, return_breakdown=True,
            )
        ).with_columns(score=pl.col("breakdown").struct.field("score"))
    else:
        scored = joined.with_columns(
            score=hybrid_score(
                left_on, "__right_val__",
                algorithms=algos, weights=w, method=method, threshold=threshold,
            )
        )

    if threshold is not None:
        scored = scored.filter(pl.col("score") >= threshold)

    if top_k is not None:
        # Rank right matches per left row by score descending, keep top_k.
        # Partition by the row id, not the key value, so duplicate left keys
        # each get their own full top-k allowance.
        scored = (
            scored.with_columns(
                __rank=pl.col("score")
                .rank(method="ordinal", descending=True)
                .over("__left_row__")
            )
            .filter(pl.col("__rank") <= top_k)
            .drop("__rank")
        )

    scored = scored.drop("__left_row__")

    if how == "left":
        # Re-attach unmatched left rows with null right columns.
        matched_keys = scored.select(left_on).unique()
        unmatched = left.join(matched_keys, on=left_on, how="anti")
        # Align schema: add the right/score columns as null.
        for c in scored.columns:
            if c not in unmatched.columns:
                unmatched = unmatched.with_columns(**{c: None})
        scored = pl.concat([scored, unmatched.select(scored.columns)], how="vertical_relaxed")

    scored = scored.rename({"__right_val__": f"{right_on}_right"})
    return scored


# ---------------------------------------------------------------------------
# deduplicate
# ---------------------------------------------------------------------------

def deduplicate(
    frame: pl.DataFrame | pl.LazyFrame,
    *,
    on: str,
    algorithms: Optional[Sequence[str]] = None,
    weights: Optional[Sequence[float]] = None,
    method: str = "weighted_avg",
    composite_threshold: float = 0.85,
    block: Optional[str] = None,
    block_n: int = 2,
) -> pl.DataFrame | pl.LazyFrame:
    """Collapse near-duplicate rows on column ``on`` using combined scoring.

    Two records are considered duplicates when their combined score >=
    ``composite_threshold``. Within each duplicate cluster, the first-occurring
    row is kept as the canonical record.

    Parameters
    ----------
    on:
        String column to deduplicate.
    algorithms, weights, method:
        Combined-score configuration (see :func:`polars_stringsim.combine`).
    composite_threshold:
        Combined score at/above which two rows are duplicates.
    block:
        Optional blocking to avoid O(n^2) all-pairs. Same options as
        :func:`fuzzy_join`.

    Returns
    -------
    Frame with duplicates removed, plus a ``cluster_id`` Int64 column grouping
    duplicates together (rows sharing a cluster_id were collapsed into the
    canonical row of that cluster).
    """
    algos = list(algorithms) if algorithms is not None else list(_DEFAULT_ALGOS)
    if weights is not None:
        w = _validate_weights(weights, len(algos))
    else:
        w = list(_DEFAULT_WEIGHTS)

    # Self-join pairs within blocks (or all pairs), keep i<j to avoid duplicates.
    # union-find needs the input row count up front; LazyFrame has no .height,
    # so resolve it via a cheap count query (single scalar, no full collect).
    if isinstance(frame, pl.DataFrame):
        n_rows = frame.height
    else:
        n_rows = frame.select(pl.len()).collect().item()
    renamed = frame.rename({on: "__val__"}).with_row_index("__row__")

    if block is not None:
        if block == "first_chars":
            block_expr = block_first_chars("__val__", block_n)
        elif block == "char_bag":
            block_expr = block_char_bag("__val__")
        else:
            raise ValueError(f"unknown block strategy {block!r}")
        renamed = renamed.with_columns(__block__=block_expr)
        pairs = renamed.join(renamed, on="__block__", suffix="__r").filter(
            pl.col("__row__") < pl.col("__row____r")
        ).drop("__block__")
    else:
        pairs = renamed.join(renamed, how="cross", suffix="__r").filter(
            pl.col("__row__") < pl.col("__row____r")
        )

    dup_pairs = pairs.with_columns(
        score=hybrid_score(
            "__val__", "__val____r",
            algorithms=algos, weights=w, method=method,
        )
    ).filter(pl.col("score") >= composite_threshold).select(
        "__row__", "__row____r"
    )

    # Union-Find to collapse transitive duplicate pairs into clusters.
    # Materialize for the clustering step (small relative to all-pairs).
    if isinstance(dup_pairs, pl.LazyFrame):
        dup_pairs = dup_pairs.collect()

    clusters = _union_find(dup_pairs, n_rows)

    out = frame.with_columns(cluster_id=pl.Series(clusters, dtype=pl.Int64))
    # Keep first row of each cluster as canonical.
    return out.group_by("cluster_id", maintain_order=True).first().drop("cluster_id")


def _union_find(pairs: pl.DataFrame, n_rows: int) -> list[int]:
    """Collapse (a, b) duplicate pairs into cluster ids via union-find."""
    parent = list(range(n_rows))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    if pairs.height > 0:
        a_col = pairs["__row__"].to_list()
        b_col = pairs["__row____r"].to_list()
        for a, b in zip(a_col, b_col):
            union(a, b)

    # Assign contiguous cluster ids in order of first appearance.
    seen: dict[int, int] = {}
    out: list[int] = []
    next_id = 0
    for i in range(n_rows):
        root = find(i)
        if root not in seen:
            seen[root] = next_id
            next_id += 1
        out.append(seen[root])
    return out


# ---------------------------------------------------------------------------
# pairwise_compare
# ---------------------------------------------------------------------------

def pairwise_compare(
    left: pl.DataFrame | pl.LazyFrame,
    right: pl.DataFrame | pl.LazyFrame,
    *,
    left_on: str,
    right_on: str,
    algorithms: Optional[Sequence[str]] = None,
    weights: Optional[Sequence[float]] = None,
    method: str = "weighted_avg",
    block: Optional[str] = None,
    block_n: int = 2,
) -> pl.DataFrame | pl.LazyFrame:
    """Compute multi-metric + combined scores for all (or blocked) pairs.

    Returns a frame with the left/right values and a ``scores`` struct:
    ``{ <algo>: Float64, ..., combined: Float64 }`` — useful for
    explainability and threshold tuning.

    Like :func:`fuzzy_join` but does not filter/rank; it returns every pair's
    full breakdown.
    """
    algos = list(algorithms) if algorithms is not None else list(_DEFAULT_ALGOS)
    if weights is not None:
        w = _validate_weights(weights, len(algos))
    else:
        w = list(_DEFAULT_WEIGHTS)

    right_renamed = right.rename({right_on: "__right_val__"})
    if block is not None:
        if block == "first_chars":
            l_block = block_first_chars(left_on, block_n)
            r_block = block_first_chars("__right_val__", block_n)
        elif block == "char_bag":
            l_block = block_char_bag(left_on)
            r_block = block_char_bag("__right_val__")
        else:
            raise ValueError(f"unknown block strategy {block!r}")
        joined = (
            left.with_columns(__block__=l_block)
            .join(
                right_renamed.with_columns(__block__=r_block),
                on="__block__",
                how="inner",
                suffix="_right",
            )
            .drop("__block__")
        )
    else:
        joined = left.join(right_renamed, how="cross", suffix="_right")

    # Per-algorithm metric columns, each named after its algorithm so the
    # output struct is self-describing: { <algo>: Float64, ..., combined: Float64 }.
    # Deduplicate names so two invocations of the same algorithm get distinct
    # columns (e.g. "jaro_winkler", "jaro_winkler__2").
    names: list[str] = []
    seen: dict[str, int] = {}
    aliased_metrics: list[pl.Expr] = []
    builders = build_metrics(algos, left_on, "__right_val__")
    for name, expr in zip(algos, builders):
        if name in seen:
            seen[name] += 1
            unique = f"{name}__{seen[name]}"
        else:
            seen[name] = 1
            unique = name
        names.append(unique)
        aliased_metrics.append(expr.alias(unique))

    out = joined.with_columns(*aliased_metrics).with_columns(
        scores=pl.struct([pl.col(n) for n in names]),
        combined=combine(aliased_metrics, weights=w, method=method),
    ).rename({"__right_val__": f"{right_on}_right"})
    return out
