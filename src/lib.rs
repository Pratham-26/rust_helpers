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
pub mod thread_pool;

#[pymodule]
fn _polars_stringsim(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Expression plugins are discovered dynamically via the polars_expr macro;
    // no manual registration needed here.
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    m.add_function(wrap_pyfunction!(py_set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(py_get_num_threads, m)?)?;
    Ok(())
}

/// Set the number of threads used by parallel metrics (`hybrid_score`).
///
/// `n == 0` restores the default (the `POLARS_STRINGSIM_THREADS` env var, or
/// all logical cores if unset). Values above 1024 are clamped. Returns the
/// resolved thread count. The change applies to all subsequent calls.
///
/// This pool is independent of the Polars engine pool (`POLARS_MAX_THREADS`),
/// so you can tune them separately — e.g. leave Polars at 4 threads while
/// giving `hybrid_score` the whole machine.
#[pyfunction]
#[pyo3(signature = (n))]
fn py_set_num_threads(n: usize) -> usize {
    thread_pool::set_num_threads(n)
}

/// Get the current thread count of the stringsim worker pool.
#[pyfunction]
fn py_get_num_threads() -> usize {
    thread_pool::num_threads()
}

use pyo3::prelude::*;
