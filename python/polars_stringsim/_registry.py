"""Algorithm name → expression builder registry.

Used by ``hybrid_score``, ``fuzzy_join``, and ``pairwise_compare`` to resolve
string algorithm names (e.g. ``"jaro_winkler"``) to the underlying expression
builders without callers having to know the function names.
"""
from __future__ import annotations

from typing import Callable, Sequence, Union

import polars as pl

from polars_stringsim import _expression as _e

ColLike = Union[str, "pl.Expr"]

# Each entry maps a public algorithm name to a builder taking (left, right).
# Names are stable and match the user-facing API in ``polars_stringsim``.
REGISTRY: dict[str, Callable[[ColLike, ColLike], pl.Expr]] = {
    # Jaro family
    "jaro": _e.jaro,
    "jaro_winkler": _e.jaro_winkler,
    # Edit distances
    "levenshtein": _e.levenshtein,
    "levenshtein_norm": _e.levenshtein_norm,
    "damerau_levenshtein": _e.damerau_levenshtein,
    "damerau_levenshtein_norm": _e.damerau_levenshtein_norm,
    "osa": _e.osa,
    "hamming": _e.hamming,
    "hamming_norm": _e.hamming_norm,
    # Token / n-gram
    "token_jaccard": _e.token_jaccard,
    "token_sorensen_dice": _e.token_sorensen_dice,
    "trigram_jaccard": _e.trigram_jaccard,
    "trigram_sorensen_dice": _e.trigram_sorensen_dice,
    # LCS
    "lcs_sim": _e.lcs_sim,
    # Phonetic (exact-match variants used for hybrid scoring)
    "soundex": _e.soundex_sim,
    "metaphone": _e.metaphone_sim,
    "double_metaphone": _e.double_metaphone_sim,
    "nysiis": _e.nysiis_sim,
    "soundex_jw": _e.soundex_jw_sim,
    "metaphone_jw": _e.metaphone_jw_sim,
    "double_metaphone_jw": _e.double_metaphone_jw_sim,
    "nysiis_jw": _e.nysiis_jw_sim,
}


def resolve(name: str) -> Callable[[ColLike, ColLike], pl.Expr]:
    """Resolve an algorithm name to its expression builder."""
    try:
        return REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(REGISTRY))
        raise ValueError(
            f"unknown algorithm {name!r}. Known: {known}"
        ) from None


def build_metrics(
    algorithms: Sequence[str],
    left: ColLike,
    right: ColLike,
) -> list[pl.Expr]:
    """Build a list of metric expressions from algorithm names."""
    return [resolve(name)(left, right) for name in algorithms]
