"""polars-stringsim — composable fuzzy string matching for Polars.

All metrics are implemented in Rust and exposed as Polars expressions.
Combine multiple metrics with :func:`combine` or :func:`hybrid_score` for
hybrid scoring. Use :func:`fuzzy_join` / :func:`deduplicate` /
:func:`pairwise_compare` for record-linkage workflows.
"""
from polars_stringsim._expression import (
    combine,
    damerau_levenshtein,
    damerau_levenshtein_norm,
    double_metaphone_jw_sim,
    double_metaphone_sim,
    hamming,
    hamming_norm,
    jaro,
    jaro_winkler,
    lcs_sim,
    levenshtein,
    levenshtein_norm,
    metaphone_jw_sim,
    metaphone_sim,
    nysiis_jw_sim,
    nysiis_sim,
    osa,
    qgram_jaccard,
    soundex_jw_sim,
    soundex_sim,
    token_jaccard,
    token_sorensen_dice,
    trigram_jaccard,
    trigram_sorensen_dice,
)
from polars_stringsim.hybrid import (
    hybrid_score,
    name_default,
    phonetic_edit,
    prefix_ngram,
    token_char,
)
from polars_stringsim.frame import (
    block_char_bag,
    block_first_chars,
    deduplicate,
    fuzzy_join,
    pairwise_compare,
)

try:
    from polars_stringsim._polars_stringsim import __version__ as __version__
except Exception:  # pragma: no cover
    __version__ = "0.1.0"

__all__ = [
    # Jaro family
    "jaro",
    "jaro_winkler",
    # Edit distances
    "levenshtein",
    "levenshtein_norm",
    "damerau_levenshtein",
    "damerau_levenshtein_norm",
    "osa",
    "hamming",
    "hamming_norm",
    # Token / n-gram
    "token_jaccard",
    "token_sorensen_dice",
    "trigram_jaccard",
    "trigram_sorensen_dice",
    "qgram_jaccard",
    # LCS
    "lcs_sim",
    # Phonetic
    "soundex_sim",
    "soundex_jw_sim",
    "metaphone_sim",
    "metaphone_jw_sim",
    "double_metaphone_sim",
    "double_metaphone_jw_sim",
    "nysiis_sim",
    "nysiis_jw_sim",
    # Combiner / hybrid
    "combine",
    "hybrid_score",
    "phonetic_edit",
    "token_char",
    "prefix_ngram",
    "name_default",
    # DataFrame helpers
    "fuzzy_join",
    "deduplicate",
    "pairwise_compare",
    "block_first_chars",
    "block_char_bag",
]
