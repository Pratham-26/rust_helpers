"""Expression builders for polars-fuzzy.

Each function returns a Polars expression that calls into the Rust plugin.
The plugin function names match the `#[polars_expr] fn <name>_expr` in Rust.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import polars as pl

# Import the compiled extension module that hosts the plugin symbols.
from polars_fuzzy import _polars_fuzzy as _pf

__all__ = [
    "jaro",
    "jaro_winkler",
    "levenshtein",
    "levenshtein_norm",
    "damerau_levenshtein",
    "damerau_levenshtein_norm",
    "osa",
    "hamming",
    "hamming_norm",
    "token_jaccard",
    "token_sorensen_dice",
    "trigram_jaccard",
    "trigram_sorensen_dice",
    "qgram_jaccard",
    "lcs_sim",
    "soundex_sim",
    "soundex_jw_sim",
    "metaphone_sim",
    "metaphone_jw_sim",
    "double_metaphone_sim",
    "double_metaphone_jw_sim",
    "nysiis_sim",
    "nysiis_jw_sim",
    "combine",
]

ColLike = Union[str, "pl.Expr"]


def _cols(left: ColLike, right: ColLike) -> list[pl.Expr]:
    """Normalize the two inputs to a list of two expressions."""
    return [
        left if isinstance(left, pl.Expr) else pl.col(left),
        right if isinstance(right, pl.Expr) else pl.col(right),
    ]


def _binary_expr(left: ColLike, right: ColLike, fn_name: str) -> pl.Expr:
    """Wrap a two-column plugin call."""
    return pl.plugins.register_plugin_function(
        plugin_path=_pf.__file__,
        function_name=fn_name,
        args=_cols(left, right),
        is_elementwise=True,
    )


def jaro(left: ColLike, right: ColLike) -> pl.Expr:
    """Jaro similarity in `[0,1]`. Null-in → null-out."""
    return _binary_expr(left, right, "jaro_expr")


def jaro_winkler(left: ColLike, right: ColLike) -> pl.Expr:
    """Jaro-Winkler similarity in `[0,1]`."""
    return _binary_expr(left, right, "jaro_winkler_expr")


def levenshtein(left: ColLike, right: ColLike) -> pl.Expr:
    """Levenshtein edit distance (raw count, Float64)."""
    return _binary_expr(left, right, "levenshtein_expr")


def levenshtein_norm(left: ColLike, right: ColLike) -> pl.Expr:
    """Normalized Levenshtein similarity in `[0,1]`."""
    return _binary_expr(left, right, "levenshtein_norm_expr")


def damerau_levenshtein(left: ColLike, right: ColLike) -> pl.Expr:
    """Damerau-Levenshtein distance (raw count)."""
    return _binary_expr(left, right, "damerau_levenshtein_expr")


def damerau_levenshtein_norm(left: ColLike, right: ColLike) -> pl.Expr:
    """Normalized Damerau-Levenshtein similarity in `[0,1]`."""
    return _binary_expr(left, right, "damerau_levenshtein_norm_expr")


def osa(left: ColLike, right: ColLike) -> pl.Expr:
    """Optimal String Alignment distance (raw count)."""
    return _binary_expr(left, right, "osa_expr")


def hamming(left: ColLike, right: ColLike) -> pl.Expr:
    """Hamming distance (raw count). Degrades gracefully on length mismatch."""
    return _binary_expr(left, right, "hamming_expr")


def hamming_norm(left: ColLike, right: ColLike) -> pl.Expr:
    """Normalized Hamming similarity in `[0,1]`."""
    return _binary_expr(left, right, "hamming_norm_expr")


def token_jaccard(left: ColLike, right: ColLike) -> pl.Expr:
    """Jaccard similarity over whitespace tokens."""
    return _binary_expr(left, right, "token_jaccard_expr")


def token_sorensen_dice(left: ColLike, right: ColLike) -> pl.Expr:
    """Sørensen-Dice similarity over whitespace tokens."""
    return _binary_expr(left, right, "token_sorensen_dice_expr")


def trigram_jaccard(left: ColLike, right: ColLike) -> pl.Expr:
    """Jaccard similarity over character trigrams."""
    return _binary_expr(left, right, "trigram_jaccard_expr")


def trigram_sorensen_dice(left: ColLike, right: ColLike) -> pl.Expr:
    """Sørensen-Dice over character trigrams."""
    return _binary_expr(left, right, "trigram_sorensen_dice_expr")


def qgram_jaccard(left: ColLike, right: ColLike, q: int = 3) -> pl.Expr:
    """Jaccard similarity over character q-grams. `q` defaults to 3."""
    return pl.plugins.register_plugin_function(
        plugin_path=_pf.__file__,
        function_name="qgram_jaccard_expr",
        args=_cols(left, right),
        kwargs={"q": q},
        is_elementwise=True,
    )


def lcs_sim(left: ColLike, right: ColLike) -> pl.Expr:
    """LCS-based similarity in `[0,1]`: `|lcs| / max(|a|,|b|)`."""
    return _binary_expr(left, right, "lcs_sim_expr")


def soundex_sim(left: ColLike, right: ColLike) -> pl.Expr:
    """1.0 if Soundex codes match, else 0.0."""
    return _binary_expr(left, right, "soundex_sim_expr")


def soundex_jw_sim(left: ColLike, right: ColLike) -> pl.Expr:
    """Jaro-Winkler similarity between Soundex codes."""
    return _binary_expr(left, right, "soundex_jw_sim_expr")


def metaphone_sim(left: ColLike, right: ColLike) -> pl.Expr:
    """1.0 if Metaphone codes match, else 0.0."""
    return _binary_expr(left, right, "metaphone_sim_expr")


def metaphone_jw_sim(left: ColLike, right: ColLike) -> pl.Expr:
    """Jaro-Winkler similarity between Metaphone codes."""
    return _binary_expr(left, right, "metaphone_jw_sim_expr")


def double_metaphone_sim(left: ColLike, right: ColLike) -> pl.Expr:
    """1.0 if Double Metaphone primary codes match, else 0.0."""
    return _binary_expr(left, right, "double_metaphone_sim_expr")


def double_metaphone_jw_sim(left: ColLike, right: ColLike) -> pl.Expr:
    """Jaro-Winkler similarity between Double Metaphone primary codes."""
    return _binary_expr(left, right, "double_metaphone_jw_sim_expr")


def nysiis_sim(left: ColLike, right: ColLike) -> pl.Expr:
    """1.0 if NYSIIS codes match, else 0.0."""
    return _binary_expr(left, right, "nysiis_sim_expr")


def nysiis_jw_sim(left: ColLike, right: ColLike) -> pl.Expr:
    """Jaro-Winkler similarity between NYSIIS codes."""
    return _binary_expr(left, right, "nysiis_jw_sim_expr")


def combine(
    metrics: Sequence[pl.Expr],
    *,
    weights: Optional[Sequence[float]] = None,
    method: str = "weighted_avg",
    threshold: Optional[float] = None,
    return_breakdown: bool = False,
) -> pl.Expr:
    """Combine multiple per-row metric scores into one Float64 column.

    Parameters
    ----------
    metrics:
        Sequence of metric expressions (e.g. ``pf.jaro_winkler("a","b")``).
        All must be elementwise and aligned to the same rows.
    weights:
        Per-metric weights for ``method="weighted_avg"``. If omitted or not
        summing to 1, they are normalized. If arity mismatches the metrics,
        falls back to unweighted mean.
    method:
        One of ``"weighted_avg"``, ``"mean"``, ``"max"``, ``"min"``,
        ``"median"``, ``"vote"``.
    threshold:
        Required for ``method="vote"``; defaults to 0.5.

    Returns
    -------
    pl.Expr
        Float64 column of combined scores. Null in any metric → null output.

    Examples
    --------
    >>> df.with_columns(
    ...     score=pf.combine(
    ...         [pf.jaro_winkler("a","b"), pf.levenshtein_norm("a","b")],
    ...         weights=[0.7, 0.3],
    ...     )
    ... )
    """
    if len(metrics) == 0:
        raise ValueError("combine() requires at least one metric expression")

    kwargs: dict = {"method": method}
    if weights is not None:
        if len(weights) != len(metrics):
            raise ValueError(
                f"weights arity ({len(weights)}) != metrics arity ({len(metrics)})"
            )
        kwargs["weights"] = list(weights)
    if threshold is not None:
        kwargs["threshold"] = float(threshold)

    # Pack the metric expressions into a single struct column so the plugin
    # receives them as one Series with N Float64 fields. Each metric gets a
    # unique alias so the struct fields don't collide (polars derives field
    # names from expression output names, which default to the source column).
    aliased = [m.alias(f"metric_{i}") for i, m in enumerate(metrics)]
    struct_expr = pl.struct(aliased)
    fn_name = "combine_breakdown_expr" if return_breakdown else "combine_expr"
    return pl.plugins.register_plugin_function(
        plugin_path=_pf.__file__,
        function_name=fn_name,
        args=[struct_expr],
        kwargs=kwargs,
        is_elementwise=True,
    )
