//! Helpers for reading UTF-8 string columns and building Float64 result columns.

use polars_arrow::array::{MutablePrimitiveArray, PrimitiveArray};
use polars_arrow::pushable::Pushable;
use polars::prelude::*;

/// Extract two aligned string columns from the plugin inputs.
/// Supports both two separate columns and a single `struct` column with two
/// string fields.
pub fn extract_two_str_columns(inputs: &[Series]) -> PolarsResult<(StringChunked, StringChunked)> {
    if inputs.len() == 2 {
        let l = inputs[0].str()?.clone();
        let r = inputs[1].str()?.clone();
        Ok((l, r))
    } else if inputs.len() == 1 {
        let s = inputs[0].struct_()?;
        let fields = s.fields_as_series();
        if fields.len() != 2 {
            polars_bail!(
                ComputeError: "expected a struct with exactly 2 string fields, got {}",
                fields.len()
            );
        }
        let l = fields[0].str()?.clone();
        let r = fields[1].str()?.clone();
        Ok((l, r))
    } else {
        polars_bail!(
            ComputeError: "expected 2 string columns or 1 struct column, got {} inputs",
            inputs.len()
        );
    }
}

/// Build a Float64 column from a per-row scoring closure. Null when either
/// input is null.
pub fn score_pairs(
    left: &StringChunked,
    right: &StringChunked,
    score: impl Fn(&str, &str) -> f64,
) -> PolarsResult<Series> {
    let n = left.len();
    let mut out = MutablePrimitiveArray::<f64>::with_capacity(n);
    for i in 0..n {
        match (left.get(i), right.get(i)) {
            (Some(l), Some(r)) => out.push(Some(score(l, r))),
            _ => out.push_null(),
        }
    }
    let arr: PrimitiveArray<f64> = out.into();
    Ok(Float64Chunked::from_chunk_iter(PlSmallStr::EMPTY, [arr]).into_series())
}

/// Validate that two columns have equal length.
pub fn assert_same_len(a: &StringChunked, b: &StringChunked) -> PolarsResult<()> {
    if a.len() != b.len() {
        polars_bail!(ComputeError: "length mismatch: {} vs {}", a.len(), b.len());
    }
    Ok(())
}
