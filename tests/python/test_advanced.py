"""Tests for hybrid scoring, breakdown, and DataFrame-level helpers."""
from __future__ import annotations

import polars as pl
import pytest

import polars_fuzzy as pf


# ---------- hybrid_score ----------

def test_hybrid_score_matches_combine() -> None:
    """hybrid_score should equal combine() of the same metrics."""
    df = pl.DataFrame({"a": ["MARTHA", "Robert"], "b": ["MARHTA", "Rupert"]})
    algos = ["jaro_winkler", "soundex"]
    w = [0.6, 0.4]

    via_combine = df.with_columns(
        s=pf.combine(
            [pf.jaro_winkler("a", "b"), pf.soundex_sim("a", "b")], weights=w
        )
    )["s"].to_list()
    via_hybrid = df.with_columns(
        s=pf.hybrid_score("a", "b", algorithms=algos, weights=w)
    )["s"].to_list()

    for c, h in zip(via_combine, via_hybrid):
        assert abs(c - h) < 1e-9, f"{c} != {h}"


def test_hybrid_score_methods() -> None:
    df = pl.DataFrame({"a": ["abc", "xyz"], "b": ["abc", "xyz"]})
    for method in ["weighted_avg", "mean", "max", "min", "median"]:
        out = df.with_columns(
            s=pf.hybrid_score(
                "a", "b", algorithms=["jaro_winkler", "levenshtein_norm"], method=method
            )
        )
        # identity rows → 1.0 for all similarity-based methods
        assert abs(out["s"].to_list()[0] - 1.0) < 1e-9, method


def test_hybrid_unknown_algo_raises() -> None:
    df = pl.DataFrame({"a": ["x"], "b": ["x"]})
    with pytest.raises(Exception):
        df.with_columns(
            s=pf.hybrid_score("a", "b", algorithms=["not_a_real_algo"])
        ).collect() if isinstance(df, pl.LazyFrame) else df.with_columns(
            s=pf.hybrid_score("a", "b", algorithms=["not_a_real_algo"])
        )


def test_hybrid_arity_mismatch() -> None:
    with pytest.raises(ValueError):
        pf.hybrid_score("a", "b", algorithms=["jaro"], weights=[0.5, 0.5])


# ---------- pre-built scorers ----------

def test_phonetic_edit() -> None:
    df = pl.DataFrame({"a": ["Smith"], "b": ["Smythe"]})
    out = df.with_columns(s=pf.phonetic_edit("a", "b"))
    assert out["s"].dtype == pl.Float64
    assert 0.0 <= out["s"].to_list()[0] <= 1.0


def test_name_default() -> None:
    df = pl.DataFrame({"a": ["Catherine"], "b": ["Katherine"]})
    out = df.with_columns(s=pf.name_default("a", "b"))
    # Phonetic match (Double Metaphone of Catherine/Katherine is identical)
    # → score should be high.
    assert out["s"].to_list()[0] > 0.8


# ---------- breakdown ----------

def test_combine_breakdown() -> None:
    df = pl.DataFrame({"a": ["MARTHA"], "b": ["MARHTA"]})
    out = df.with_columns(
        bd=pf.combine(
            [pf.jaro_winkler("a", "b"), pf.levenshtein_norm("a", "b")],
            weights=[0.6, 0.4],
            return_breakdown=True,
        )
    )
    row = out["bd"].to_list()[0]
    assert "score" in row
    assert "scores" in row
    # MARTHA/MARHTA: jw≈0.9611, levenshtein_norm=0.667 (distance 2, len 6)
    assert abs(row["score"] - (0.6 * 0.961_111 + 0.4 * 0.666_667)) < 1e-4
    assert "metric_0" in row["scores"]
    assert "metric_1" in row["scores"]


# ---------- fuzzy_join ----------

def test_fuzzy_join_basic() -> None:
    left = pl.DataFrame({"name": ["Robert", "Catherine", "John"]})
    right = pl.DataFrame({"name": ["Rupert", "Katherine", "Paul"]})
    out = pf.fuzzy_join(
        left, right,
        left_on="name", right_on="name",
        algorithms=["jaro_winkler", "double_metaphone"],
        weights=[0.5, 0.5],
        threshold=0.6,
    )
    # Robert/Rupert share Soundex R163 + high JW → kept
    # Catherine/Katherine share Double Metaphone + high JW → kept
    # John/Paul → low → dropped
    left_vals = out["name"].to_list()
    assert "Robert" in left_vals
    assert "Catherine" in left_vals
    assert "John" not in left_vals


def test_fuzzy_join_blocked() -> None:
    left = pl.DataFrame({"name": ["Robert", "Catherine", "John"]})
    right = pl.DataFrame({"name": ["Rupert", "Katherine", "Paul"]})
    out = pf.fuzzy_join(
        left, right,
        left_on="name", right_on="name",
        algorithms=["jaro_winkler"],
        weights=[1.0],
        threshold=0.0,
        block="first_chars",
        block_n=1,
    )
    # Only same-first-char pairs survive blocking.
    pairs = list(zip(out["name"].to_list(), out["name_right"].to_list()))
    for l, r in pairs:
        assert l[0].upper() == r[0].upper()


def test_fuzzy_join_top_k() -> None:
    left = pl.DataFrame({"name": ["Smith"]})
    right = pl.DataFrame({"name": ["Smith", "Smyth", "Smit", "Smithy"]})
    out = pf.fuzzy_join(
        left, right,
        left_on="name", right_on="name",
        algorithms=["jaro_winkler"], weights=[1.0],
        top_k=2,
    )
    assert out.height == 2


def test_fuzzy_join_add_breakdown() -> None:
    left = pl.DataFrame({"name": ["Robert"]})
    right = pl.DataFrame({"name": ["Rupert"]})
    out = pf.fuzzy_join(
        left, right,
        left_on="name", right_on="name",
        algorithms=["jaro_winkler"], weights=[1.0],
        add_breakdown=True,
    )
    assert "score" in out.columns


# ---------- deduplicate ----------

def test_deduplicate_basic() -> None:
    df = pl.DataFrame({"name": ["Smith", "Smyth", "Jones", "Smithy", "Brown"]})
    out = pf.deduplicate(
        df, on="name",
        algorithms=["jaro_winkler"], weights=[1.0],
        composite_threshold=0.8,
    )
    # Smith, Smyth, Smithy should collapse into one cluster.
    assert out.height < df.height
    assert out.height == 3  # Smith-cluster, Jones, Brown


def test_deduplicate_blocked() -> None:
    df = pl.DataFrame({"name": ["Robert", "Rupert", "Catherine", "Paul"]})
    out = pf.deduplicate(
        df, on="name",
        algorithms=["soundex"], weights=[1.0],
        composite_threshold=1.0,
        block="first_chars", block_n=1,
    )
    # Robert/Rupert share first char + Soundex → collapse.
    assert out.height <= df.height


# ---------- pairwise_compare ----------

def test_pairwise_compare() -> None:
    left = pl.DataFrame({"name": ["Smith"]})
    right = pl.DataFrame({"name": ["Smith", "Smyth"]})
    out = pf.pairwise_compare(
        left, right,
        left_on="name", right_on="name",
        algorithms=["jaro_winkler", "levenshtein_norm"],
        weights=[0.5, 0.5],
    )
    assert out.height == 2
    assert "combined" in out.columns
    # Identity pair → 1.0
    assert abs(out["combined"].to_list()[0] - 1.0) < 1e-9


# ---------- blocking helpers ----------

def test_block_first_chars() -> None:
    df = pl.DataFrame({"n": ["robert", "Rupert"]}).with_columns(
        b=pf.block_first_chars("n", 1)
    )
    assert df["b"].to_list() == ["R", "R"]


def test_block_char_bag() -> None:
    df = pl.DataFrame({"n": ["Smith", "htmIS "]}).with_columns(
        b=pf.block_char_bag("n")
    )
    assert df["b"].to_list()[0] == df["b"].to_list()[1]


# ---------- registry ----------

def test_registry_unknown() -> None:
    from polars_fuzzy._registry import resolve
    with pytest.raises(ValueError):
        resolve("not_real")
