//! Token- and n-gram-based similarity metrics. All return `[0,1]`.

use std::collections::HashSet;
use std::hash::Hash;

/// Tokenize on whitespace, lowercase.
fn tokens(s: &str) -> HashSet<&str> {
    s.split_whitespace().collect()
}

/// Jaccard similarity over whitespace tokens.
pub fn token_jaccard(a: &str, b: &str) -> f64 {
    let ta = tokens(a);
    let tb = tokens(b);
    jaccard_sets(&ta, &tb)
}

/// Sørensen-Dice similarity over whitespace tokens.
pub fn token_sorensen_dice(a: &str, b: &str) -> f64 {
    let ta = tokens(a);
    let tb = tokens(b);
    sorensen_dice_sets(&ta, &tb)
}

/// Jaccard similarity over character q-grams (default q=3 → trigrams).
pub fn qgram_jaccard(a: &str, b: &str, q: usize) -> f64 {
    let ga = qgrams(a, q);
    let gb = qgrams(b, q);
    jaccard_sets(&ga, &gb)
}

/// Trigram (q=3) Jaccard similarity.
pub fn trigram_jaccard(a: &str, b: &str) -> f64 {
    qgram_jaccard(a, b, 3)
}

/// Sørensen-Dice over trigrams.
pub fn trigram_sorensen_dice(a: &str, b: &str) -> f64 {
    let ga = qgrams(a, 3);
    let gb = qgrams(b, 3);
    sorensen_dice_sets(&ga, &gb)
}

fn qgrams(s: &str, q: usize) -> HashSet<String> {
    if q == 0 {
        return HashSet::new();
    }
    let chars: Vec<char> = s.chars().collect();
    if chars.len() < q {
        // Pad: a single short string yields one gram equal to itself.
        return [s.to_string()].into_iter().collect();
    }
    (0..=chars.len() - q)
        .map(|i| chars[i..i + q].iter().collect())
        .collect()
}

fn jaccard_sets<T: Hash + Eq>(a: &HashSet<T>, b: &HashSet<T>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let inter = a.intersection(b).count() as f64;
    let union = a.len() + b.len() - a.intersection(b).count();
    if union == 0 {
        1.0
    } else {
        inter / union as f64
    }
}

fn sorensen_dice_sets<T: Hash + Eq>(a: &HashSet<T>, b: &HashSet<T>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let inter = a.intersection(b).count() as f64;
    let total = a.len() + b.len();
    if total == 0 {
        1.0
    } else {
        2.0 * inter / total as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_jaccard_identical() {
        assert!((token_jaccard("foo bar", "foo bar") - 1.0).abs() < 1e-12);
    }

    #[test]
    fn token_jaccard_half() {
        // {foo, bar} vs {foo, baz}: 1/3 shared.
        let s = token_jaccard("foo bar", "foo baz");
        assert!((s - 1.0 / 3.0).abs() < 1e-12, "got {s}");
    }

    #[test]
    fn token_jaccard_disjoint() {
        assert!(token_jaccard("foo", "bar").abs() < 1e-12);
    }

    #[test]
    fn trigram_identical() {
        assert!((trigram_jaccard("abcd", "abcd") - 1.0).abs() < 1e-12);
    }

    #[test]
    fn sorensen_dice_half() {
        let s = token_sorensen_dice("foo bar", "foo baz");
        assert!((s - 2.0 / 4.0).abs() < 1e-12, "got {s}");
    }

    #[test]
    fn empty_strings() {
        assert!((token_jaccard("", "") - 1.0).abs() < 1e-12);
        assert!((trigram_jaccard("", "") - 1.0).abs() < 1e-12);
    }
}
