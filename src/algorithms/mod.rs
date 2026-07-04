//! Fuzzy matching algorithm implementations.
//!
//! Each module exposes pure functions returning similarity scores in `[0,1]`
//! (or raw integer distances where noted), operating on `&str` slices so they
//! are UTF-8 safe and trivially testable.

pub mod edit;
pub mod jaro;
pub mod lcs;
pub mod phonetic;
pub mod token;
