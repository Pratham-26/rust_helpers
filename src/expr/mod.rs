//! Polars expression wrappers around the algorithm functions.
//!
//! Each `#[polars_expr]` function takes two string columns (or a single
//! 2-field struct) and returns a Float64 column. Null-in → null-out.

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

use crate::algorithms::{edit, jaro, lcs, phonetic, token};
use crate::series_util::{assert_same_len, extract_two_str_columns, score_pairs};

// ---- Jaro family ----
#[polars_expr(output_type=Float64)]
fn jaro_expr(inputs: &[Series]) -> PolarsResult<Series> {
    let (l, r) = extract_two_str_columns(inputs)?;
    assert_same_len(&l, &r)?;
    score_pairs(&l, &r, jaro::jaro)
}

#[polars_expr(output_type=Float64)]
fn jaro_winkler_expr(inputs: &[Series]) -> PolarsResult<Series> {
    let (l, r) = extract_two_str_columns(inputs)?;
    assert_same_len(&l, &r)?;
    score_pairs(&l, &r, jaro::jaro_winkler)
}

// ---- Edit distances ----
#[polars_expr(output_type=Float64)]
fn levenshtein_expr(inputs: &[Series]) -> PolarsResult<Series> {
    let (l, r) = extract_two_str_columns(inputs)?;
    assert_same_len(&l, &r)?;
    score_pairs(&l, &r, |a, b| edit::levenshtein(a, b) as f64)
}

#[polars_expr(output_type=Float64)]
fn levenshtein_norm_expr(inputs: &[Series]) -> PolarsResult<Series> {
    let (l, r) = extract_two_str_columns(inputs)?;
    assert_same_len(&l, &r)?;
    score_pairs(&l, &r, edit::levenshtein_norm)
}

#[polars_expr(output_type=Float64)]
fn damerau_levenshtein_expr(inputs: &[Series]) -> PolarsResult<Series> {
    let (l, r) = extract_two_str_columns(inputs)?;
    assert_same_len(&l, &r)?;
    score_pairs(&l, &r, |a, b| edit::damerau_levenshtein(a, b) as f64)
}

#[polars_expr(output_type=Float64)]
fn damerau_levenshtein_norm_expr(inputs: &[Series]) -> PolarsResult<Series> {
    let (l, r) = extract_two_str_columns(inputs)?;
    assert_same_len(&l, &r)?;
    score_pairs(&l, &r, edit::damerau_levenshtein_norm)
}

#[polars_expr(output_type=Float64)]
fn osa_expr(inputs: &[Series]) -> PolarsResult<Series> {
    let (l, r) = extract_two_str_columns(inputs)?;
    assert_same_len(&l, &r)?;
    score_pairs(&l, &r, |a, b| edit::osa(a, b) as f64)
}

#[polars_expr(output_type=Float64)]
fn hamming_expr(inputs: &[Series]) -> PolarsResult<Series> {
    let (l, r) = extract_two_str_columns(inputs)?;
    assert_same_len(&l, &r)?;
    score_pairs(&l, &r, |a, b| edit::hamming(a, b) as f64)
}

#[polars_expr(output_type=Float64)]
fn hamming_norm_expr(inputs: &[Series]) -> PolarsResult<Series> {
    let (l, r) = extract_two_str_columns(inputs)?;
    assert_same_len(&l, &r)?;
    score_pairs(&l, &r, edit::hamming_norm)
}

// ---- Token / n-gram ----
#[polars_expr(output_type=Float64)]
fn token_jaccard_expr(inputs: &[Series]) -> PolarsResult<Series> {
    let (l, r) = extract_two_str_columns(inputs)?;
    assert_same_len(&l, &r)?;
    score_pairs(&l, &r, token::token_jaccard)
}

#[polars_expr(output_type=Float64)]
fn token_sorensen_dice_expr(inputs: &[Series]) -> PolarsResult<Series> {
    let (l, r) = extract_two_str_columns(inputs)?;
    assert_same_len(&l, &r)?;
    score_pairs(&l, &r, token::token_sorensen_dice)
}

#[polars_expr(output_type=Float64)]
fn trigram_jaccard_expr(inputs: &[Series]) -> PolarsResult<Series> {
    let (l, r) = extract_two_str_columns(inputs)?;
    assert_same_len(&l, &r)?;
    score_pairs(&l, &r, token::trigram_jaccard)
}

#[polars_expr(output_type=Float64)]
fn trigram_sorensen_dice_expr(inputs: &[Series]) -> PolarsResult<Series> {
    let (l, r) = extract_two_str_columns(inputs)?;
    assert_same_len(&l, &r)?;
    score_pairs(&l, &r, token::trigram_sorensen_dice)
}

#[polars_expr(output_type=Float64)]
fn qgram_jaccard_expr(inputs: &[Series], kwargs: QGramKwargs) -> PolarsResult<Series> {
    let (l, r) = extract_two_str_columns(inputs)?;
    assert_same_len(&l, &r)?;
    let q = kwargs.q.unwrap_or(3).max(1);
    score_pairs(&l, &r, |a, b| token::qgram_jaccard(a, b, q))
}

#[derive(serde::Deserialize)]
struct QGramKwargs {
    q: Option<usize>,
}

// ---- LCS ----
#[polars_expr(output_type=Float64)]
fn lcs_sim_expr(inputs: &[Series]) -> PolarsResult<Series> {
    let (l, r) = extract_two_str_columns(inputs)?;
    assert_same_len(&l, &r)?;
    score_pairs(&l, &r, lcs::lcs_sim)
}

// ---- Phonetic ----
#[polars_expr(output_type=Float64)]
fn soundex_sim_expr(inputs: &[Series]) -> PolarsResult<Series> {
    let (l, r) = extract_two_str_columns(inputs)?;
    assert_same_len(&l, &r)?;
    score_pairs(&l, &r, phonetic::soundex_sim)
}

#[polars_expr(output_type=Float64)]
fn soundex_jw_sim_expr(inputs: &[Series]) -> PolarsResult<Series> {
    let (l, r) = extract_two_str_columns(inputs)?;
    assert_same_len(&l, &r)?;
    score_pairs(&l, &r, phonetic::soundex_jw_sim)
}

#[polars_expr(output_type=Float64)]
fn metaphone_sim_expr(inputs: &[Series]) -> PolarsResult<Series> {
    let (l, r) = extract_two_str_columns(inputs)?;
    assert_same_len(&l, &r)?;
    score_pairs(&l, &r, phonetic::metaphone_sim)
}

#[polars_expr(output_type=Float64)]
fn metaphone_jw_sim_expr(inputs: &[Series]) -> PolarsResult<Series> {
    let (l, r) = extract_two_str_columns(inputs)?;
    assert_same_len(&l, &r)?;
    score_pairs(&l, &r, phonetic::metaphone_jw_sim)
}

#[polars_expr(output_type=Float64)]
fn double_metaphone_sim_expr(inputs: &[Series]) -> PolarsResult<Series> {
    let (l, r) = extract_two_str_columns(inputs)?;
    assert_same_len(&l, &r)?;
    score_pairs(&l, &r, phonetic::double_metaphone_sim)
}

#[polars_expr(output_type=Float64)]
fn double_metaphone_jw_sim_expr(inputs: &[Series]) -> PolarsResult<Series> {
    let (l, r) = extract_two_str_columns(inputs)?;
    assert_same_len(&l, &r)?;
    score_pairs(&l, &r, phonetic::double_metaphone_jw_sim)
}

#[polars_expr(output_type=Float64)]
fn nysiis_sim_expr(inputs: &[Series]) -> PolarsResult<Series> {
    let (l, r) = extract_two_str_columns(inputs)?;
    assert_same_len(&l, &r)?;
    score_pairs(&l, &r, phonetic::nysiis_sim)
}

#[polars_expr(output_type=Float64)]
fn nysiis_jw_sim_expr(inputs: &[Series]) -> PolarsResult<Series> {
    let (l, r) = extract_two_str_columns(inputs)?;
    assert_same_len(&l, &r)?;
    score_pairs(&l, &r, phonetic::nysiis_jw_sim)
}
