//! Longest Common Subsequence based similarity.

/// Length of the longest common subsequence.
pub fn lcs_len(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (m, n) = (a.len(), b.len());
    if m == 0 || n == 0 {
        return 0;
    }
    let mut prev = vec![0usize; n + 1];
    let mut curr = vec![0usize; n + 1];
    for i in 1..=m {
        for j in 1..=n {
            curr[j] = if a[i - 1] == b[j - 1] {
                prev[j - 1] + 1
            } else {
                prev[j].max(curr[j - 1])
            };
        }
        std::mem::swap(&mut prev, &mut curr);
        curr.iter_mut().for_each(|x| *x = 0);
    }
    prev[n]
}

/// LCS-based similarity in `[0,1]`, defined as `|lcs| / max(|a|,|b|)`.
/// Empty/empty → 1.0.
pub fn lcs_sim(a: &str, b: &str) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let l = lcs_len(a, b) as f64;
    let m = a.chars().count().max(b.chars().count()) as f64;
    if m == 0.0 {
        1.0
    } else {
        l / m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lcs_basic() {
        assert_eq!(lcs_len("AGCAT", "GAC"), 2);
    }

    #[test]
    fn lcs_sim_identical() {
        assert!((lcs_sim("abc", "abc") - 1.0).abs() < 1e-12);
    }

    #[test]
    fn lcs_sim_disjoint() {
        assert!(lcs_sim("abc", "xyz").abs() < 1e-12);
    }

    #[test]
    fn lcs_empty() {
        assert!((lcs_sim("", "") - 1.0).abs() < 1e-12);
        assert!(lcs_sim("abc", "").abs() < 1e-12);
    }
}
