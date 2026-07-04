//! Multi-metric score combiner.
//!
//! Fuses multiple per-row metric scores into a single score. All hot paths
//! here operate on `&[f64]` slices — no allocations beyond the result.

use serde::Deserialize;

/// Combination strategy.
#[derive(Clone, Copy, Deserialize, PartialEq, Debug)]
#[serde(rename_all = "snake_case")]
pub enum CombineMethod {
    /// Sum_i w_i * s_i. Weights normalized to sum to 1.0 if they don't.
    WeightedAvg,
    /// Unweighted mean: (1/n) * Sum s_i.
    Mean,
    /// Max score across metrics.
    Max,
    /// Min score across metrics.
    Min,
    /// Median score across metrics.
    Median,
    /// Count of metrics whose score >= threshold, normalized to [0,1] by /n.
    Vote,
}

impl Default for CombineMethod {
    fn default() -> Self {
        Self::WeightedAvg
    }
}

/// Per-row scores for the combine expression. Passed as kwargs.
#[derive(Deserialize, Default)]
pub struct CombineKwargs {
    pub weights: Option<Vec<f64>>,
    #[serde(default)]
    pub method: CombineMethod,
    /// Only used by `Vote`. Defaults to 0.5 when `None`.
    pub threshold: Option<f64>,
}

/// Combine one row's metric scores into a single score. `scores.len()` must
/// equal `weights.len()` when `method == WeightedAvg`.
pub fn combine_row(scores: &[f64], kwargs: &CombineKwargs) -> f64 {
    if scores.is_empty() {
        return f64::NAN;
    }
    match kwargs.method {
        CombineMethod::WeightedAvg => {
            let weights = kwargs.weights.as_deref().unwrap_or(&[]);
            let w = if weights.len() == scores.len() {
                let sum: f64 = weights.iter().sum();
                if sum > 0.0 {
                    weights
                } else {
                    // Fall back to equal weights.
                    return mean(scores);
                }
            } else {
                // Mismatched arity: fall back to mean rather than panicking.
                return mean(scores);
            };
            let wsum: f64 = w.iter().sum();
            if wsum == 0.0 {
                return mean(scores);
            }
            let mut acc = 0.0;
            for (s, wi) in scores.iter().zip(w.iter()) {
                acc += s * wi;
            }
            acc / wsum
        }
        CombineMethod::Mean => mean(scores),
        CombineMethod::Max => {
            let mut m = f64::NEG_INFINITY;
            for &s in scores {
                if s > m {
                    m = s;
                }
            }
            m
        }
        CombineMethod::Min => {
            let mut m = f64::INFINITY;
            for &s in scores {
                if s < m {
                    m = s;
                }
            }
            m
        }
        CombineMethod::Median => {
            let mut v: Vec<f64> = scores.to_vec();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = v.len();
            if n % 2 == 1 {
                v[n / 2]
            } else {
                (v[n / 2 - 1] + v[n / 2]) / 2.0
            }
        }
        CombineMethod::Vote => {
            let t = kwargs.threshold.unwrap_or(0.5);
            let c = scores.iter().filter(|&&s| s >= t).count() as f64;
            c / scores.len() as f64
        }
    }
}

fn mean(s: &[f64]) -> f64 {
    s.iter().sum::<f64>() / s.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kw(method: CombineMethod, weights: Option<Vec<f64>>, threshold: Option<f64>) -> CombineKwargs {
        CombineKwargs {
            weights,
            method,
            threshold,
        }
    }

    #[test]
    fn weighted_avg() {
        let k = kw(
            CombineMethod::WeightedAvg,
            Some(vec![0.6, 0.4]),
            None,
        );
        let r = combine_row(&[0.9, 0.8], &k);
        assert!((r - 0.86).abs() < 1e-12, "got {r}");
    }

    #[test]
    fn weighted_avg_normalizes() {
        // weights sum to 2.0 → normalize so result is still weighted mean.
        let k = kw(CombineMethod::WeightedAvg, Some(vec![1.0, 1.0]), None);
        let r = combine_row(&[1.0, 0.0], &k);
        assert!((r - 0.5).abs() < 1e-12, "got {r}");
    }

    #[test]
    fn mean_max_min() {
        let s = [0.1, 0.4, 0.9];
        assert!((combine_row(&s, &kw(CombineMethod::Mean, None, None)) - 0.466_666_6).abs() < 1e-6);
        assert!((combine_row(&s, &kw(CombineMethod::Max, None, None)) - 0.9).abs() < 1e-12);
        assert!((combine_row(&s, &kw(CombineMethod::Min, None, None)) - 0.1).abs() < 1e-12);
    }

    #[test]
    fn median_even_odd() {
        assert!((combine_row(&[0.1, 0.4, 0.9], &kw(CombineMethod::Median, None, None)) - 0.4).abs() < 1e-12);
        assert!(
            (combine_row(&[0.1, 0.4, 0.5, 0.9], &kw(CombineMethod::Median, None, None)) - 0.45).abs()
                < 1e-12
        );
    }

    #[test]
    fn vote() {
        let k = kw(CombineMethod::Vote, None, Some(0.5));
        // 2 of 3 >= 0.5 → 2/3
        let r = combine_row(&[0.9, 0.6, 0.1], &k);
        assert!((r - 2.0 / 3.0).abs() < 1e-12, "got {r}");
    }

    #[test]
    fn empty_returns_nan() {
        assert!(combine_row(&[], &CombineKwargs::default()).is_nan());
    }
}
