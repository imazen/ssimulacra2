//! Tests that verify ssimulacra2 scores against C++ reference values.
//!
//! These tests use pre-captured C++ ssimulacra2 scores for synthetic test images,
//! allowing parity verification without requiring the C++ binary at runtime.
//!
//! To regenerate reference data:
//!   SSIMULACRA2_BIN=/path/to/ssimulacra2 cargo run --example capture_cpp_reference
//!
//! Run tests with: cargo test --test reference_parity

use sha2::{Digest, Sha256};
use ssimulacra2::compute_frame_ssimulacra2;
use ssimulacra2::reference_data::{ReferenceCase, REFERENCE_CASES};
use yuvxyb::{ColorPrimaries, Rgb, TransferCharacteristic};

// ============================================================================
// Image Generation Functions (must match capture_cpp_reference.rs exactly)
// ============================================================================

/// LCG pseudo-random number generator (deterministic)
struct Lcg {
    state: u64,
}

impl Lcg {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u8(&mut self) -> u8 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.state >> 33) & 0xFF) as u8
    }
}

/// Generate uniform color image
fn gen_uniform(width: usize, height: usize, r: u8, g: u8, b: u8) -> Vec<u8> {
    vec![r, g, b]
        .into_iter()
        .cycle()
        .take(width * height * 3)
        .collect()
}

/// Generate horizontal gradient
fn gen_gradient_h(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    for _y in 0..height {
        for x in 0..width {
            let val = if width > 1 {
                (x * 255 / (width - 1)) as u8
            } else {
                128
            };
            data.extend_from_slice(&[val, val, val]);
        }
    }
    data
}

/// Generate vertical gradient
fn gen_gradient_v(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        let val = if height > 1 {
            (y * 255 / (height - 1)) as u8
        } else {
            128
        };
        for _x in 0..width {
            data.extend_from_slice(&[val, val, val]);
        }
    }
    data
}

/// Generate diagonal gradient
fn gen_gradient_diag(width: usize, height: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    let max_dist = width + height - 2;
    for y in 0..height {
        for x in 0..width {
            let val = if max_dist > 0 {
                ((x + y) * 255 / max_dist) as u8
            } else {
                128
            };
            data.extend_from_slice(&[val, val, val]);
        }
    }
    data
}

/// Generate checkerboard pattern
fn gen_checkerboard(width: usize, height: usize, cell_size: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let val = if ((x / cell_size) + (y / cell_size)) % 2 == 0 {
                255
            } else {
                0
            };
            data.extend_from_slice(&[val, val, val]);
        }
    }
    data
}

/// Generate random noise
fn gen_noise(width: usize, height: usize, seed: u64) -> Vec<u8> {
    let mut lcg = Lcg::new(seed);
    let mut data = Vec::with_capacity(width * height * 3);
    for _ in 0..width * height {
        data.push(lcg.next_u8());
        data.push(lcg.next_u8());
        data.push(lcg.next_u8());
    }
    data
}

/// Generate edge pattern
fn gen_edge(width: usize, height: usize, vertical: bool) -> Vec<u8> {
    let mut data = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let val = if vertical {
                if x < width / 2 { 0 } else { 255 }
            } else {
                if y < height / 2 { 0 } else { 255 }
            };
            data.extend_from_slice(&[val, val, val]);
        }
    }
    data
}

// ============================================================================
// Test Case Generator
// ============================================================================

fn generate_test_image(case: &ReferenceCase) -> (Vec<u8>, Vec<u8>) {
    let name = case.name;
    let width = case.width;
    let height = case.height;

    // Parse test case name to generate correct images
    if name.starts_with("perfect_match") {
        let data = gen_uniform(width, height, 128, 128, 128);
        (data.clone(), data)
    } else if let Some(shift_str) = name.strip_prefix("uniform_shift_") {
        if let Some(shift) = shift_str.split('_').next().and_then(|s| s.parse::<u8>().ok()) {
            let source = gen_uniform(width, height, 128, 128, 128);
            let distorted = gen_uniform(width, height, 128 + shift, 128 + shift, 128 + shift);
            (source, distorted)
        } else {
            panic!("Invalid uniform_shift test case: {}", name);
        }
    } else if name.starts_with("gradient_h_") {
        let grad = gen_gradient_h(width, height);
        (grad.clone(), grad)
    } else if name.starts_with("gradient_v_") {
        let grad = gen_gradient_v(width, height);
        (grad.clone(), grad)
    } else if name.starts_with("gradient_diag_") {
        let grad = gen_gradient_diag(width, height);
        (grad.clone(), grad)
    } else if let Some(rest) = name.strip_prefix("checkerboard_") {
        if let Some(cell_size) = rest.split('_').next().and_then(|s| s.parse::<usize>().ok()) {
            let checker = gen_checkerboard(width, height, cell_size);
            (checker.clone(), checker)
        } else {
            panic!("Invalid checkerboard test case: {}", name);
        }
    } else if let Some(rest) = name.strip_prefix("noise_seed_") {
        if let Some(seed) = rest.split('_').next().and_then(|s| s.parse::<u64>().ok()) {
            let noise = gen_noise(width, height, seed);
            (noise.clone(), noise)
        } else {
            panic!("Invalid noise test case: {}", name);
        }
    } else if name.starts_with("edge_vertical") {
        let edge = gen_edge(width, height, true);
        (edge.clone(), edge)
    } else if name.starts_with("edge_horizontal") {
        let edge = gen_edge(width, height, false);
        (edge.clone(), edge)
    } else if name.contains("gradient_vs_uniform") {
        let grad = gen_gradient_h(width, height);
        let uniform = gen_uniform(width, height, 128, 128, 128);
        (grad, uniform)
    } else if name.contains("noise_vs_uniform") {
        let noise = gen_noise(width, height, 42);
        let uniform = gen_uniform(width, height, 128, 128, 128);
        (noise, uniform)
    } else {
        panic!("Unknown test case pattern: {}", name);
    }
}

// ============================================================================
// Tests
// ============================================================================

#[test]
fn test_reference_parity() {
    if REFERENCE_CASES.is_empty() {
        eprintln!("WARNING: No reference cases loaded!");
        eprintln!("Run: SSIMULACRA2_BIN=/path/to/ssimulacra2 cargo run --example capture_cpp_reference");
        return;
    }

    let mut failures = Vec::new();
    let mut max_error = 0.0f64;

    for (i, case) in REFERENCE_CASES.iter().enumerate() {
        let (source_data, distorted_data) = generate_test_image(case);

        // Verify hashes match (detects changes in image generation)
        let source_hash = format!("{:x}", Sha256::digest(&source_data));
        let distorted_hash = format!("{:x}", Sha256::digest(&distorted_data));

        if source_hash != case.source_hash {
            eprintln!(
                "\nERROR: Source image hash mismatch for {}!\nExpected: {}\nGot:      {}\nThis indicates the image generation algorithm changed.",
                case.name, case.source_hash, source_hash
            );
            panic!("Image generation changed for {}", case.name);
        }

        if distorted_hash != case.distorted_hash {
            eprintln!(
                "\nERROR: Distorted image hash mismatch for {}!\nExpected: {}\nGot:      {}\nThis indicates the image generation algorithm changed.",
                case.name, case.distorted_hash, distorted_hash
            );
            panic!("Image generation changed for {}", case.name);
        }

        // Convert to RGB format
        let source_rgb: Vec<[f32; 3]> = source_data
            .chunks_exact(3)
            .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
            .collect();

        let distorted_rgb: Vec<[f32; 3]> = distorted_data
            .chunks_exact(3)
            .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
            .collect();

        let source = Rgb::new(
            source_rgb,
            case.width,
            case.height,
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .unwrap();

        let distorted = Rgb::new(
            distorted_rgb,
            case.width,
            case.height,
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .unwrap();

        let score = compute_frame_ssimulacra2(source, distorted).unwrap();
        let error = (score - case.expected_score).abs();
        max_error = max_error.max(error);

        if error > 1.5 {
            // Tolerance: 1.5 (allows for floating-point and edge case differences)
            failures.push((i, case.name, case.expected_score, score, error));
        }
    }

    if !failures.is_empty() {
        eprintln!("\n{} / {} tests FAILED:", failures.len(), REFERENCE_CASES.len());
        eprintln!("{:<5} {:<50} {:>15} {:>15} {:>10}", "Index", "Name", "Expected", "Actual", "Error");
        eprintln!("{:-<100}", "");
        for (i, name, expected, actual, error) in &failures {
            eprintln!(
                "{:<5} {:<50} {:>15.6} {:>15.6} {:>10.6}",
                i, name, expected, actual, error
            );
        }
        eprintln!("\nMax error: {:.6}", max_error);
        panic!("{} tests failed", failures.len());
    }

    println!("All {} reference tests passed! Max error: {:.2e}", REFERENCE_CASES.len(), max_error);
}
