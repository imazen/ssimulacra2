//! Comprehensive benchmark comparing all feature combinations on 512x512 images
//!
//! Run with:
//! ```bash
//! # Single-threaded baselines
//! cargo run --release --no-default-features --features blur-accurate --example feature_benchmark
//! cargo run --release --no-default-features --features blur-transpose --example feature_benchmark
//! cargo run --release --no-default-features --features blur-simd --example feature_benchmark
//!
//! # With SIMD ops
//! cargo run --release --no-default-features --features blur-transpose,simd-ops --example feature_benchmark
//! cargo run --release --no-default-features --features blur-simd,simd-ops --example feature_benchmark
//!
//! # With rayon (multi-threaded)
//! cargo run --release --features blur-transpose,rayon --example feature_benchmark
//! cargo run --release --features blur-simd,simd-ops,rayon --example feature_benchmark
//! ```

use ssimulacra2::compute_frame_ssimulacra2;
use std::time::Instant;
use yuvxyb::{ColorPrimaries, Rgb, TransferCharacteristic};

fn create_test_image_512x512() -> (Rgb, Rgb) {
    let width = 512;
    let height = 512;
    let size = width * height;

    // Create a gradient pattern (common test pattern)
    let source_data: Vec<[f32; 3]> = (0..size)
        .map(|i| {
            let x = (i % width) as f32 / width as f32;
            let y = (i / width) as f32 / height as f32;
            [x, y, (x + y) / 2.0]
        })
        .collect();

    // Create distorted version with slight variations
    let distorted_data: Vec<[f32; 3]> = source_data
        .iter()
        .map(|&[r, g, b]| {
            [
                (r * 0.95).min(1.0),
                (g * 1.02).min(1.0),
                (b * 0.98).min(1.0),
            ]
        })
        .collect();

    let source = Rgb::new(
        source_data,
        width,
        height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .unwrap();

    let distorted = Rgb::new(
        distorted_data,
        width,
        height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .unwrap();

    (source, distorted)
}

fn benchmark_with_warmup(source: &Rgb, distorted: &Rgb, iterations: usize) -> (f64, f64, f64) {
    // Warmup
    for _ in 0..3 {
        let _ = compute_frame_ssimulacra2(source.clone(), distorted.clone()).unwrap();
    }

    // Actual benchmark
    let mut times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        let _score = compute_frame_ssimulacra2(source.clone(), distorted.clone()).unwrap();
        times.push(start.elapsed().as_secs_f64() * 1000.0); // Convert to ms
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let median = times[times.len() / 2];
    let p95 = times[(times.len() as f32 * 0.95) as usize];

    (mean, median, p95)
}

fn main() {
    println!("SSIMULACRA2 Feature Benchmark - 512x512 images");
    println!("==============================================\n");

    // Print active features
    println!("Active features:");
    #[cfg(feature = "rayon")]
    println!("  - rayon (multi-threading)");
    #[cfg(feature = "blur-accurate")]
    println!("  - blur-accurate (f64 IIR)");
    #[cfg(feature = "blur-transpose")]
    println!("  - blur-transpose (f32 IIR transpose)");
    #[cfg(feature = "blur-simd")]
    println!("  - blur-simd (SIMD vertical pass)");
    #[cfg(feature = "simd-ops")]
    println!("  - simd-ops (SIMD compute pipeline)");
    #[cfg(feature = "inaccurate-libblur")]
    println!("  - inaccurate-libblur");

    #[cfg(not(any(
        feature = "rayon",
        feature = "blur-accurate",
        feature = "blur-transpose",
        feature = "blur-simd",
        feature = "simd-ops",
        feature = "inaccurate-libblur"
    )))]
    println!("  - (no features - using blur-accurate fallback)");

    println!();

    println!("Creating test images...");
    let (source, distorted) = create_test_image_512x512();

    println!("Running benchmark with 100 iterations...\n");
    let (mean, median, p95) = benchmark_with_warmup(&source, &distorted, 100);

    println!("Results:");
    println!("  Mean:   {:.3} ms", mean);
    println!("  Median: {:.3} ms", median);
    println!("  P95:    {:.3} ms", p95);
    println!();

    // Calculate score for verification
    let score = compute_frame_ssimulacra2(source, distorted).unwrap();
    println!("SSIMULACRA2 score: {:.6}", score);
}
