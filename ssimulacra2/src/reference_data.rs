//! C++ ssimulacra2 reference data (to be auto-generated).
//!
//! To generate this file:
//!   SSIMULACRA2_BIN=/path/to/ssimulacra2 cargo run --example capture_cpp_reference
//!
//! This stub exists so the code compiles before reference data is captured.

#![allow(clippy::excessive_precision)]

/// A reference test case with expected C++ ssimulacra2 score.
#[derive(Debug, Clone)]
pub struct ReferenceCase {
    pub name: &'static str,
    pub width: usize,
    pub height: usize,
    pub expected_score: f64,
}

/// All reference test cases.
///
/// NOTE: This is a stub! Run capture_cpp_reference to populate with real data.
pub const REFERENCE_CASES: &[ReferenceCase] = &[];
