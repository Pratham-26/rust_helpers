//! `polars-stringsim` — composable fuzzy string matching for Polars.
//!
//! Native Polars plugin (Rust core + Python bindings via PyO3).
//! All hot paths stay in Rust; no Python loops over rows.
#![allow(clippy::needless_pass_by_value)]

pub mod algorithms;
pub mod combiner;
pub mod expr;
pub mod expr_combine;
pub mod expr_hybrid;
pub mod series_util;

#[pymodule]
fn _polars_stringsim(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Expression plugins are discovered dynamically via the polars_expr macro;
    // no manual registration needed here.
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

use pyo3::prelude::*;
