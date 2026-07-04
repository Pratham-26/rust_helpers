//! Hybrid-score expression plugin.
//!
//! Takes two string columns and a list of algorithm names (+ optional weights
//! and method) and produces a single combined Float64 score, all in one Rust
//! call. Equivalent to building N metric expressions in Python and passing
//! them through ``combine_expr``, but avoids the intermediate struct column.

use polars_arrow::array::{MutablePrimitiveArray, PrimitiveArray};
use polars_arrow::pushable::Pushable;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

use crate::algorithms::{edit, jaro, lcs, phonetic, token};
use crate::combiner::{combine_row, CombineKwargs};
use crate::series_util::{assert_same_len, extract_two_str_columns};

/// Algorithm name → scoring function. All return `[0,1]` similarity.
type ScoreFn = fn(&str, &str) -> f64;

fn resolve(name: &str) -> PolarsResult<ScoreFn> {
    Ok(match name {
        "jaro" => jaro::jaro,
        "jaro_winkler" => jaro::jaro_winkler,
        "levenshtein_norm" => edit::levenshtein_norm,
        "damerau_levenshtein_norm" => edit::damerau_levenshtein_norm,
        "hamming_norm" => edit::hamming_norm,
        "token_jaccard" => token::token_jaccard,
        "token_sorensen_dice" => token::token_sorensen_dice,
        "trigram_jaccard" => token::trigram_jaccard,
        "trigram_sorensen_dice" => token::trigram_sorensen_dice,
        "lcs_sim" => lcs::lcs_sim,
        "soundex" => phonetic::soundex_sim,
        "metaphone" => phonetic::metaphone_sim,
        "double_metaphone" => phonetic::double_metaphone_sim,
        "nysiis" => phonetic::nysiis_sim,
        "soundex_jw" => phonetic::soundex_jw_sim,
        "metaphone_jw" => phonetic::metaphone_jw_sim,
        "double_metaphone_jw" => phonetic::double_metaphone_jw_sim,
        "nysiis_jw" => phonetic::nysiis_jw_sim,
        other => {
            polars_bail!(ComputeError: "unknown algorithm {other:?} in hybrid_score");
        }
    })
}

#[derive(serde::Deserialize)]
pub(crate) struct HybridKwargs {
    algorithms: Vec<String>,
    #[serde(default)]
    method: crate::combiner::CombineMethod,
    weights: Option<Vec<f64>>,
    threshold: Option<f64>,
}

impl HybridKwargs {
    fn to_combine(&self) -> CombineKwargs {
        CombineKwargs {
            weights: self.weights.clone(),
            method: self.method,
            threshold: self.threshold,
        }
    }
}

#[polars_expr(output_type=Float64)]
pub(crate) fn hybrid_score_expr(inputs: &[Series], kwargs: HybridKwargs) -> PolarsResult<Series> {
    let (l, r) = extract_two_str_columns(inputs)?;
    assert_same_len(&l, &r)?;

    let fns: Vec<ScoreFn> = kwargs
        .algorithms
        .iter()
        .map(|n| resolve(n))
        .collect::<PolarsResult<_>>()?;
    if fns.is_empty() {
        polars_bail!(ComputeError: "hybrid_score requires at least one algorithm");
    }

    let combine_kw = kwargs.to_combine();
    let n = l.len();
    let mut out = MutablePrimitiveArray::<f64>::with_capacity(n);
    let mut row: Vec<f64> = Vec::with_capacity(fns.len());

    for i in 0..n {
        match (l.get(i), r.get(i)) {
            (Some(a), Some(b)) => {
                row.clear();
                for f in &fns {
                    row.push(f(a, b));
                }
                out.push(Some(combine_row(&row, &combine_kw)));
            }
            _ => out.push_null(),
        }
    }

    let arr: PrimitiveArray<f64> = out.into();
    Ok(Float64Chunked::from_chunk_iter(PlSmallStr::EMPTY, [arr]).into_series())
}
