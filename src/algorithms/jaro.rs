//! Jaro and Jaro-Winkler similarity.

/// Jaro similarity in `[0,1]`. Empty/empty → 1.0.
pub fn jaro(a: &str, b: &str) -> f64 {
    strsim::jaro(a, b)
}

/// Jaro-Winkler similarity in `[0,1]`. Empty/empty → 1.0.
pub fn jaro_winkler(a: &str, b: &str) -> f64 {
    strsim::jaro_winkler(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jaro_winkler_martha() {
        let s = jaro_winkler("MARTHA", "MARHTA");
        assert!((s - 0.961_111_111_111_111).abs() < 1e-9, "got {s}");
    }

    #[test]
    fn jaro_identity() {
        assert!((jaro("abc", "abc") - 1.0).abs() < 1e-12);
    }

    #[test]
    fn jaro_empty() {
        assert!((jaro("", "") - 1.0).abs() < 1e-12);
        assert!(jaro("abc", "").abs() < 1e-12);
    }
}
