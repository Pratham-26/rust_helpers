//! Combine-expression plugin.
//!
//! Input: a single struct column whose fields are Float64 metric scores.
//! Output (default): a single Float64 column with the combined score per row.
//! Output (return_breakdown=true): a Struct column `{ score: Float64,
//! scores: Struct<metric_0: Float64, ...> }` for explainability.
//! Null in any field → null output.

use polars_arrow::array::{MutablePrimitiveArray, PrimitiveArray};
use polars_arrow::pushable::Pushable;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

use crate::combiner::{combine_row, CombineKwargs};

/// Per-metric scores extracted from the input struct, row-aligned.
struct MetricCols {
    /// Owned Float64 columns (one per metric).
    cols: Vec<ChunkedArray<Float64Type>>,
    /// Field names from the input struct, in order.
    names: Vec<PlSmallStr>,
    n_rows: usize,
}

fn extract_metrics(inputs: &[Series]) -> PolarsResult<MetricCols> {
    if inputs.is_empty() {
        polars_bail!(ComputeError: "combine requires a struct input");
    }
    let s = inputs[0].struct_()?;
    let fields = s.fields_as_series();
    if fields.is_empty() {
        polars_bail!(ComputeError: "combine struct must have at least one metric field");
    }
    let names = s.struct_fields().iter().map(|f| f.name().clone()).collect();
    let cols: Vec<ChunkedArray<Float64Type>> = fields
        .iter()
        .map(|f| f.cast(&DataType::Float64).map(|c| c.f64().unwrap().clone()))
        .collect::<PolarsResult<_>>()?;
    let n_rows = cols[0].len();
    Ok(MetricCols { cols, names, n_rows })
}

/// Build the combined Float64 column. Null in any field → null output.
fn build_combined(m: &MetricCols, kwargs: &CombineKwargs) -> PolarsResult<Series> {
    let n = m.n_rows;
    let mut out = MutablePrimitiveArray::<f64>::with_capacity(n);
    let mut row: Vec<f64> = Vec::with_capacity(m.cols.len());

    for i in 0..n {
        let mut any_null = false;
        row.clear();
        for c in &m.cols {
            match c.get(i) {
                Some(v) => row.push(v),
                None => {
                    any_null = true;
                    break;
                }
            }
        }
        if any_null {
            out.push_null();
        } else {
            out.push(Some(combine_row(&row, kwargs)));
        }
    }

    let arr: PrimitiveArray<f64> = out.into();
    Ok(Float64Chunked::from_chunk_iter(PlSmallStr::EMPTY, [arr]).into_series())
}

#[polars_expr(output_type=Float64)]
pub(crate) fn combine_expr(inputs: &[Series], kwargs: CombineKwargs) -> PolarsResult<Series> {
    let m = extract_metrics(inputs)?;
    build_combined(&m, &kwargs)
}

#[polars_expr(output_type_func=output_type_breakdown)]
pub(crate) fn combine_breakdown_expr(
    inputs: &[Series],
    kwargs: CombineBreakdownKwargs,
) -> PolarsResult<Series> {
    let m = extract_metrics(inputs)?;
    let combined = build_combined(&m, &kwargs.inner)?.f64()?.clone();
    let n = m.n_rows;

    // Build the inner "scores" struct from the metric columns.
    let mut score_fields: Vec<Series> = Vec::with_capacity(m.cols.len());
    for (c, name) in m.cols.iter().zip(m.names.iter()) {
        score_fields.push(c.clone().with_name(name.clone()).into_series());
    }
    let scores_struct =
        StructChunked::from_series(PlSmallStr::from_static("scores"), n, score_fields.iter())?
            .into_series();

    // Outer struct: { score: Float64, scores: Struct<...> }
    let score_field = combined
        .clone()
        .with_name(PlSmallStr::from_static("score"))
        .into_series();
    let outer_fields = [score_field, scores_struct];
    let outer = StructChunked::from_series(PlSmallStr::EMPTY, n, outer_fields.iter())?;
    Ok(outer.into_series())
}

#[derive(serde::Deserialize)]
pub(crate) struct CombineBreakdownKwargs {
    #[serde(flatten)]
    inner: CombineKwargs,
}

fn output_type_breakdown(input: &[Field]) -> PolarsResult<Field> {
    // Mirror input struct fields as Float64 inside the nested "scores" struct.
    let inner_fields: Vec<Field> = match input.first().map(|f| f.dtype()) {
        Some(DataType::Struct(fs)) => fs
            .iter()
            .map(|f| Field::new(f.name().clone(), DataType::Float64))
            .collect(),
        _ => vec![],
    };
    let scores = Field::new(
        PlSmallStr::from_static("scores"),
        DataType::Struct(inner_fields),
    );
    Ok(Field::new(
        PlSmallStr::EMPTY,
        DataType::Struct(vec![
            Field::new(PlSmallStr::from_static("score"), DataType::Float64),
            scores,
        ]),
    ))
}
