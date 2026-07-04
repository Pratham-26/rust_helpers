//! Edit-distance based metrics.
//!
//! Distances are returned as raw counts; `_norm` variants convert to a
//! `[0,1]` similarity where 1.0 means identical.

/// Levenshtein edit distance (raw count).
pub fn levenshtein(a: &str, b: &str) -> usize {
    strsim::levenshtein(a, b)
}

/// Normalized Levenshtein similarity in `[0,1]`.
/// 1.0 = identical. Empty/empty → 1.0.
pub fn levenshtein_norm(a: &str, b: &str) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    strsim::normalized_levenshtein(a, b)
}

/// Damerau-Levenshtein distance (with adjacent transpositions).
pub fn damerau_levenshtein(a: &str, b: &str) -> usize {
    strsim::damerau_levenshtein(a, b)
}

/// Normalized Damerau-Levenshtein similarity in `[0,1]`.
pub fn damerau_levenshtein_norm(a: &str, b: &str) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    strsim::normalized_damerau_levenshtein(a, b)
}

/// Optimal String Alignment (OSA) distance.
pub fn osa(a: &str, b: &str) -> usize {
    strsim::osa_distance(a, b)
}

/// Hamming distance. Strings must be equal length; otherwise returns the
/// length difference as a lower bound (no error) — see `hamming_strict` if
/// you need error-on-mismatch semantics.
pub fn hamming(a: &str, b: &str) -> usize {
    // strsim::hamming errors on length mismatch; we degrade gracefully.
    match strsim::hamming(a, b) {
        Ok(d) => d,
        Err(_) => a.chars().count().max(b.chars().count()),
    }
}

/// Hamming distance, normalized to `[0,1]` similarity against the longer string.
pub fn hamming_norm(a: &str, b: &str) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let d = hamming(a, b) as f64;
    let n = a.chars().count().max(b.chars().count()) as f64;
    if n == 0.0 {
        1.0
    } else {
        1.0 - d / n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kitten_to_sitting() {
        assert_eq!(levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn levenshtein_norm_basic() {
        let s = levenshtein_norm("kitten", "sitting");
        assert!((s - (1.0 - 3.0 / 7.0)).abs() < 1e-12, "got {s}");
    }

    #[test]
    fn damerau_transposition() {
        assert_eq!(damerau_levenshtein("ca", "ac"), 1);
        assert_eq!(osa("ca", "ac"), 1);
    }

    #[test]
    fn hamming_basic() {
        assert_eq!(hamming("karolin", "kathrin"), 3);
    }

    #[test]
    fn hamming_mismatched_len() {
        // Degrades gracefully: distance is the longer length.
        assert_eq!(hamming("abc", "a"), 3);
    }

    #[test]
    fn empty_strings() {
        assert!((levenshtein_norm("", "") - 1.0).abs() < 1e-12);
        assert!((damerau_levenshtein_norm("", "") - 1.0).abs() < 1e-12);
        assert!((hamming_norm("", "") - 1.0).abs() < 1e-12);
    }
}
