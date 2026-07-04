//! Phonetic-encoder-based similarity metrics.
//!
//! Each metric encodes both inputs with a phonetic encoder, then compares the
//! resulting codes. We expose two comparison strategies:
//!   - exact-code equality (1.0 if codes match, else 0.0)
//!   - Jaro-Winkler on the codes (smoother gradient, helps ranking)
//!
//! All functions return `[0,1]`. Empty/empty → 1.0.

use crate::algorithms::jaro::jaro_winkler;
use rphonetic::{DoubleMetaphone, Encoder, Metaphone, Nysiis, Soundex};

/// 1.0 if Soundex codes match, else 0.0.
pub fn soundex_sim(a: &str, b: &str) -> f64 {
    code_eq_sim(a, b, |s| Soundex::default().encode(s))
}

/// Jaro-Winkler similarity between Soundex codes.
pub fn soundex_jw_sim(a: &str, b: &str) -> f64 {
    let enc = |s: &str| Soundex::default().encode(s);
    code_jw_sim(a, b, &enc)
}

/// 1.0 if Metaphone codes match, else 0.0.
pub fn metaphone_sim(a: &str, b: &str) -> f64 {
    code_eq_sim(a, b, |s| Metaphone::default().encode(s))
}

/// Jaro-Winkler similarity between Metaphone codes.
pub fn metaphone_jw_sim(a: &str, b: &str) -> f64 {
    let enc = |s: &str| Metaphone::default().encode(s);
    code_jw_sim(a, b, &enc)
}

/// 1.0 if Double Metaphone *primary* codes match, else 0.0.
pub fn double_metaphone_sim(a: &str, b: &str) -> f64 {
    code_eq_sim(a, b, |s| {
        DoubleMetaphone::default().encode(s)
    })
}

/// Jaro-Winkler similarity between Double Metaphone primary codes.
pub fn double_metaphone_jw_sim(a: &str, b: &str) -> f64 {
    let enc = |s: &str| DoubleMetaphone::default().encode(s);
    code_jw_sim(a, b, &enc)
}

/// 1.0 if NYSIIS codes match, else 0.0.
pub fn nysiis_sim(a: &str, b: &str) -> f64 {
    code_eq_sim(a, b, |s| Nysiis::default().encode(s))
}

/// Jaro-Winkler similarity between NYSIIS codes.
pub fn nysiis_jw_sim(a: &str, b: &str) -> f64 {
    let enc = |s: &str| Nysiis::default().encode(s);
    code_jw_sim(a, b, &enc)
}

fn code_eq_sim(a: &str, b: &str, enc: impl Fn(&str) -> String) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    if enc(a) == enc(b) {
        1.0
    } else {
        0.0
    }
}

fn code_jw_sim(a: &str, b: &str, enc: &impl Fn(&str) -> String) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    jaro_winkler(&enc(a), &enc(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soundex_robert_rupert() {
        // Classic: Robert and Rupert share Soundex R163.
        assert!((soundex_sim("Robert", "Rupert") - 1.0).abs() < 1e-12);
        assert!((soundex_sim("Robert", "Ashcraft") - 0.0).abs() < 1e-12);
    }

    #[test]
    fn double_metaphone_basic() {
        assert!((double_metaphone_sim("Smith", "Smith") - 1.0).abs() < 1e-12);
    }

    #[test]
    fn nysiis_identity() {
        assert!((nysiis_sim("Washington", "Washington") - 1.0).abs() < 1e-12);
    }

    #[test]
    fn empty_strings() {
        assert!((soundex_sim("", "") - 1.0).abs() < 1e-12);
        assert!((metaphone_sim("", "") - 1.0).abs() < 1e-12);
        assert!((soundex_sim("abc", "") - 0.0).abs() < 1e-12);
    }

    #[test]
    fn jw_variants_finite() {
        let s = soundex_jw_sim("Robert", "Rupert");
        assert!(s.is_finite() && (0.0..=1.0).contains(&s));
    }
}
