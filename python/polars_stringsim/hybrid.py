"""Hybrid scoring: combine multiple algorithms over two columns in one Rust call.

Wraps the ``hybrid_score_expr`` plugin, plus pre-built hybrid scorers for
common record-linkage recipes.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import polars as pl

from polars_stringsim import _polars_stringsim as _pf
from polars_stringsim._expression import _validate_weights

ColLike = Union[str, "pl.Expr"]


def hybrid_score(
    left: ColLike,
    right: ColLike,
    *,
    algorithms: Sequence[str],
    weights: Optional[Sequence[float]] = None,
    method: str = "weighted_avg",
    threshold: Optional[float] = None,
) -> pl.Expr:
    """Score two columns with multiple algorithms, fused in one Rust call.

    Equivalent to ``pf.combine([pf.<algo>(left, right) for algo in algorithms])``
    but avoids materializing the intermediate per-metric columns.

    Parameters
    ----------
    algorithms:
        Names from the registry, e.g. ``["jaro_winkler", "soundex"]``.
    weights, method, threshold:
        See :func:`polars_stringsim.combine`.

    Examples
    --------
    >>> df.with_columns(
    ...     s=pf.hybrid_score("a", "b",
    ...                       algorithms=["jaro_winkler", "double_metaphone"],
    ...                       weights=[0.7, 0.3])
    ... )
    """
    if len(algorithms) == 0:
        raise ValueError("hybrid_score requires at least one algorithm")

    kwargs: dict = {"algorithms": list(algorithms), "method": method}
    if weights is not None:
        kwargs["weights"] = _validate_weights(weights, len(algorithms))
    if threshold is not None:
        kwargs["threshold"] = float(threshold)

    args = [
        left if isinstance(left, pl.Expr) else pl.col(left),
        right if isinstance(right, pl.Expr) else pl.col(right),
    ]
    return pl.plugins.register_plugin_function(
        plugin_path=_pf.__file__,
        function_name="hybrid_score_expr",
        args=args,
        kwargs=kwargs,
        is_elementwise=True,
    )


# -----------------------------------------------------------------------
# Pre-built hybrid scorers
# -----------------------------------------------------------------------
# Each returns a pl.Expr combining a curated set of algorithms with sensible
# default weights. Override weights by calling hybrid_score directly.

def phonetic_edit(
    left: ColLike,
    right: ColLike,
    *,
    weights: Sequence[float] = (0.5, 0.3, 0.2),
    phonetic: str = "double_metaphone",
) -> pl.Expr:
    """Phonetic + edit hybrid: Jaro-Winkler + normalized Levenshtein + phonetic.

    Good for names where spelling and pronunciation both matter.
    """
    default_algos = ["jaro_winkler", "levenshtein_norm", phonetic]
    return hybrid_score(left, right, algorithms=default_algos, weights=weights)


def token_char(
    left: ColLike,
    right: ColLike,
    *,
    weights: Sequence[float] = (0.5, 0.5),
) -> pl.Expr:
    """Token-level + character-level hybrid: trigram Jaccard + Jaro-Winkler.

    Good for multi-word strings where word order varies.
    """
    return hybrid_score(
        left, right, algorithms=["trigram_jaccard", "jaro_winkler"], weights=weights
    )


def prefix_ngram(
    left: ColLike,
    right: ColLike,
    *,
    weights: Sequence[float] = (0.6, 0.4),
) -> pl.Expr:
    """Prefix-boosted + n-gram hybrid: Jaro-Winkler + trigram Jaccard.

    Jaro-Winkler boosts common prefixes; trigram catches interior edits.
    """
    return hybrid_score(
        left, right, algorithms=["jaro_winkler", "trigram_jaccard"], weights=weights
    )


def name_default(left: ColLike, right: ColLike) -> pl.Expr:
    """Opinionated all-rounder for person/org names.

    Jaro-Winkler (spelling) + Double Metaphone (phonetic) + trigram Jaccard
    (token/shape), weighted 0.5 / 0.3 / 0.2.
    """
    return hybrid_score(
        left,
        right,
        algorithms=["jaro_winkler", "double_metaphone", "trigram_jaccard"],
        weights=[0.5, 0.3, 0.2],
    )
