//! Swappable dedicated rayon thread pool for `polars-stringsim`.
//!
//! `hybrid_score` (and any future parallel metric) runs on this pool rather
//! than the global rayon pool, so users can tune its width independently of
//! both the Polars engine (`POLARS_MAX_THREADS`) and any other rayon user in
//! the process.
//!
//! The pool is held behind an `RwLock<Arc<ThreadPool>>`. Reads on the hot
//! path (every `hybrid_score` call) take only a read lock, clone the `Arc`
//! (cheap, atomic inc), and release the lock before running — so concurrent
//! `hybrid_score` calls never contend on the lock and never block a reconfig.
//!
//! Configuration order (first wins):
//!   1. `set_num_threads(n)` from Python,
//!   2. `POLARS_STRINGSIM_THREADS` env var,
//!   3. rayon default (number of logical cores).

use std::env;
use std::sync::{Arc, OnceLock, RwLock};

use rayon::{ThreadPool, ThreadPoolBuilder};

/// Upper bound on what we'll accept, to reject obviously-wrong input early.
const MAX_THREADS: usize = 1024;

static POOL: OnceLock<RwLock<Arc<ThreadPool>>> = OnceLock::new();

/// Initialize the lazily-created pool with the configured thread count.
fn init() -> &'static RwLock<Arc<ThreadPool>> {
    POOL.get_or_init(|| RwLock::new(Arc::new(build(default_num_threads()).expect("build thread pool"))))
}

/// Build a rayon pool with `n` threads. `n == 0` falls back to the default.
pub(crate) fn build(n: usize) -> Result<ThreadPool, Box<dyn std::error::Error + Send + Sync>> {
    let n = if n == 0 { default_num_threads() } else { n.min(MAX_THREADS) };
    let pool = ThreadPoolBuilder::new()
        .num_threads(n)
        .thread_name(move |i| format!("polars-stringsim-{i}"))
        .build()?;
    Ok(pool)
}

/// Resolve the default thread count: env override, else rayon's default.
fn default_num_threads() -> usize {
    if let Ok(v) = env::var("POLARS_STRINGSIM_THREADS") {
        if let Ok(n) = v.parse::<usize>() {
            if n > 0 {
                return n.min(MAX_THREADS);
            }
        }
    }
    // Rayon picks num_cpus at pool-build time.
    num_cpus_fallback()
}

/// `std::thread::available_parallelism`, falling back to 1 if unavailable.
fn num_cpus_fallback() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

/// Run `op` on the current stringsim pool. Clones the `Arc` first so the
/// lock is never held while user work runs.
pub(crate) fn install<OP, R>(op: OP) -> R
where
    OP: FnOnce() -> R + Send,
    R: Send,
{
    let pool = { init().read().expect("pool lock poisoned").clone() };
    pool.install(op)
}

/// Replace the pool with one of `n` threads. Affects all subsequent calls.
/// Returns the new thread count. `n == 0` restores the default.
pub fn set_num_threads(n: usize) -> usize {
    let resolved = if n == 0 { default_num_threads() } else { n.min(MAX_THREADS) };
    let new = Arc::new(build(resolved).expect("build thread pool"));
    // If somehow the pool is already initialized, swap; otherwise just store.
    if let Some(lock) = POOL.get() {
        *lock.write().expect("pool lock poisoned") = new;
    } else {
        // Race-resilient: another thread may have initialized concurrently.
        let _ = init();
        *POOL.get().expect("pool initialized").write().expect("pool lock poisoned") = new;
    }
    resolved
}

/// Current pool width. Builds the pool on first call.
pub fn num_threads() -> usize {
    init().read().expect("pool lock poisoned").current_num_threads()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_has_at_least_one_thread() {
        assert!(num_threads() >= 1);
    }

    #[test]
    fn set_and_get_thread_count() {
        let prev = num_threads();
        // Use 2 if available_parallelism allows, else 1.
        let target = prev.min(2).max(1);
        let got = set_num_threads(target);
        assert_eq!(got, target);
        assert_eq!(num_threads(), target);
        // Restore.
        set_num_threads(prev);
    }

    #[test]
    fn install_runs_closure() {
        let sum = install(|| (1..=100).sum::<i64>());
        assert_eq!(sum, 5050);
    }

    #[test]
    fn zero_falls_back_to_default() {
        let prev = num_threads();
        let got = set_num_threads(0);
        assert!(got >= 1);
        set_num_threads(prev);
    }

    #[test]
    fn huge_value_is_clamped() {
        let prev = num_threads();
        let got = set_num_threads(usize::MAX);
        assert_eq!(got, MAX_THREADS);
        set_num_threads(prev);
    }
}
