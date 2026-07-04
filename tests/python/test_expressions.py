"""End-to-end tests for polars-fuzzy on Polars DataFrames."""
from __future__ import annotations

import math

import polars as pl
import pytest

import polars_fuzzy as pf


@pytest.fixture
def names_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "a": ["MARTHA", "Robert", "kitten", "abc", "", None, "José"],
            "b": ["MARHTA", "Rupert", "sitting", "abc", "", "x", "José"],
        }
    )


# ---------- Jaro family ----------
def test_jaro_winkler_martha(names_df: pl.DataFrame) -> None:
    out = names_df.with_columns(s=pf.jaro_winkler("a", "b"))
    s = out["s"].to_list()
    assert abs(s[0] - 0.961_111) < 1e-5
    # identity
    assert abs(s[3] - 1.0) < 1e-12
    # empty/empty → 1.0
    assert abs(s[4] - 1.0) < 1e-12
    # null propagation
    assert s[5] is None


def test_jaro_returns_float64(names_df: pl.DataFrame) -> None:
    out = names_df.with_columns(s=pf.jaro("a", "b"))
    assert out["s"].dtype == pl.Float64


# ---------- Edit distances ----------
def test_levenshtein_kitten(names_df: pl.DataFrame) -> None:
    out = names_df.with_columns(s=pf.levenshtein("a", "b"))
    assert out["s"].to_list()[2] == 3.0


def test_levenshtein_norm(names_df: pl.DataFrame) -> None:
    out = names_df.with_columns(s=pf.levenshtein_norm("a", "b"))
    val = out["s"].to_list()[2]
    assert abs(val - (1.0 - 3.0 / 7.0)) < 1e-12


def test_damerau_transposition() -> None:
    df = pl.DataFrame({"a": ["ca"], "b": ["ac"]})
    out = df.with_columns(s=pf.damerau_levenshtein("a", "b"))
    assert out["s"].to_list()[0] == 1.0


def test_hamming_basic() -> None:
    df = pl.DataFrame({"a": ["karolin"], "b": ["kathrin"]})
    out = df.with_columns(s=pf.hamming("a", "b"))
    assert out["s"].to_list()[0] == 3.0


# ---------- Token / n-gram ----------
def test_token_jaccard_half() -> None:
    df = pl.DataFrame({"a": ["foo bar"], "b": ["foo baz"]})
    out = df.with_columns(s=pf.token_jaccard("a", "b"))
    assert abs(out["s"].to_list()[0] - 1.0 / 3.0) < 1e-12


def test_token_jaccard_identical() -> None:
    df = pl.DataFrame({"a": ["foo bar"], "b": ["foo bar"]})
    out = df.with_columns(s=pf.token_jaccard("a", "b"))
    assert abs(out["s"].to_list()[0] - 1.0) < 1e-12


def test_trigram_identical() -> None:
    df = pl.DataFrame({"a": ["abcd"], "b": ["abcd"]})
    out = df.with_columns(s=pf.trigram_jaccard("a", "b"))
    assert abs(out["s"].to_list()[0] - 1.0) < 1e-12


def test_qgram_custom_q() -> None:
    df = pl.DataFrame({"a": ["abcd"], "b": ["abcd"]})
    out = df.with_columns(s=pf.qgram_jaccard("a", "b", q=2))
    assert abs(out["s"].to_list()[0] - 1.0) < 1e-12


# ---------- LCS ----------
def test_lcs_sim_disjoint() -> None:
    df = pl.DataFrame({"a": ["abc"], "b": ["xyz"]})
    out = df.with_columns(s=pf.lcs_sim("a", "b"))
    assert abs(out["s"].to_list()[0] - 0.0) < 1e-12


# ---------- Phonetic ----------
def test_soundex_match() -> None:
    df = pl.DataFrame({"a": ["Robert"], "b": ["Rupert"]})
    out = df.with_columns(s=pf.soundex_sim("a", "b"))
    assert out["s"].to_list()[0] == 1.0


def test_double_metaphone_identity() -> None:
    df = pl.DataFrame({"a": ["Smith"], "b": ["Smith"]})
    out = df.with_columns(s=pf.double_metaphone_sim("a", "b"))
    assert out["s"].to_list()[0] == 1.0


# ---------- Unicode ----------
def test_unicode_names() -> None:
    df = pl.DataFrame({"a": ["José"], "b": ["José"]})
    out = df.with_columns(s=pf.jaro_winkler("a", "b"))
    assert abs(out["s"].to_list()[0] - 1.0) < 1e-12


# ---------- Combiner ----------
def test_combine_weighted_avg() -> None:
    df = pl.DataFrame({"a": ["MARTHA", "Robert"], "b": ["MARHTA", "Rupert"]})
    out = df.with_columns(
        s=pf.combine(
            [pf.jaro_winkler("a", "b"), pf.soundex_sim("a", "b")],
            weights=[0.6, 0.4],
        )
    )
    vals = out["s"].to_list()
    # row 0: MARTHA/MARHTA share Soundex M630 → soundex=1.0;
    #        jaro_winkler=0.9611 → 0.6*0.9611 + 0.4*1.0
    assert abs(vals[0] - (0.6 * 0.961_111 + 0.4 * 1.0)) < 1e-5
    # row 1: soundex Robert/Rupert = 1.0, jw > 0
    assert vals[1] > 0.4


def test_combine_methods() -> None:
    df = pl.DataFrame({"a": ["x", "y"], "b": ["x", "z"]})
    metrics = [pf.jaro_winkler("a", "b"), pf.levenshtein_norm("a", "b")]

    mean = df.with_columns(s=pf.combine(metrics, method="mean"))["s"].to_list()
    mx = df.with_columns(s=pf.combine(metrics, method="max"))["s"].to_list()
    mn = df.with_columns(s=pf.combine(metrics, method="min"))["s"].to_list()
    med = df.with_columns(s=pf.combine(metrics, method="median"))["s"].to_list()
    vote = df.with_columns(
        s=pf.combine(metrics, method="vote", threshold=0.5)
    )["s"].to_list()

    # identity row: all metrics 1.0
    assert abs(mean[0] - 1.0) < 1e-12
    assert abs(mx[0] - 1.0) < 1e-12
    assert abs(mn[0] - 1.0) < 1e-12
    assert abs(med[0] - 1.0) < 1e-12
    assert abs(vote[0] - 1.0) < 1e-12


def test_combine_null_propagation() -> None:
    # Explicit String dtype so the column isn't inferred as Null.
    df = pl.DataFrame(
        {"a": [None], "b": ["x"]}, schema={"a": pl.String, "b": pl.String}
    )
    out = df.with_columns(
        s=pf.combine([pf.jaro_winkler("a", "b")], weights=[1.0])
    )
    assert out["s"].to_list()[0] is None


def test_combine_weights_normalized() -> None:
    df = pl.DataFrame({"a": ["x", "x"], "b": ["x", "x"]})
    # weights [2,2] → normalized → equal to mean of two 1.0s → 1.0
    out = df.with_columns(
        s=pf.combine(
            [pf.jaro_winkler("a", "b"), pf.levenshtein_norm("a", "b")],
            weights=[2.0, 2.0],
        )
    )
    assert abs(out["s"].to_list()[0] - 1.0) < 1e-12


def test_combine_arity_mismatch_raises() -> None:
    df = pl.DataFrame({"a": ["x"], "b": ["x"]})
    with pytest.raises(ValueError):
        df.with_columns(
            s=pf.combine(
                [pf.jaro_winkler("a", "b")], weights=[0.5, 0.5]
            )
        )


# ---------- Lazy parity ----------
def test_lazy_parity(names_df: pl.DataFrame) -> None:
    eager = names_df.with_columns(s=pf.jaro_winkler("a", "b"))
    lazy = names_df.lazy().with_columns(s=pf.jaro_winkler("a", "b")).collect()
    assert eager["s"].to_list() == lazy["s"].to_list()


# ---------- Expr input ----------
def test_expr_input(names_df: pl.DataFrame) -> None:
    out = names_df.with_columns(s=pf.jaro_winkler(pl.col("a"), pl.col("b")))
    assert abs(out["s"].to_list()[0] - 0.961_111) < 1e-5
