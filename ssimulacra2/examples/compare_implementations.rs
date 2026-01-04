//! Side-by-side comparison of different SSIMULACRA2 implementations
//!
//! Run with: cargo run --release --example compare_implementations

use ssimulacra2::{
    compute_frame_ssimulacra2_with_config, BlurImpl, ComputeImpl, Ssimulacra2Config, XybImpl,
};
use std::time::Instant;
use yuvxyb::{ColorPrimaries, Rgb, TransferCharacteristic};

fn main() {
    println!("SSIMULACRA2 Implementation Comparison");
    println!("======================================\n");

    let sizes = [(256, 256), (512, 512), (1024, 1024)];
    let iterations = 10;

    for (width, height) in sizes {
        println!("Image size: {}x{}", width, height);
        println!("{:-<60}", "");

        // Create test images
        let source_data: Vec<[f32; 3]> = (0..width * height)
            .map(|i| {
                let x = (i % width) as f32 / width as f32;
                let y = (i / width) as f32 / height as f32;
                [x, y, (x + y) / 2.0]
            })
            .collect();

        let distorted_data: Vec<[f32; 3]> = source_data
            .iter()
            .map(|&[r, g, b]| [r * 0.95, g * 1.02, b * 0.98])
            .collect();

        // Test configurations
        let configs = [
            (
                "Scalar (baseline)",
                Ssimulacra2Config::scalar(),
            ),
            (
                "SIMD (safe wide crate)",
                Ssimulacra2Config::simd(),
            ),
            #[cfg(feature = "unsafe-simd")]
            (
                "Unsafe SIMD (raw intrinsics)",
                Ssimulacra2Config::unsafe_simd(),
            ),
        ];

        let mut results = Vec::new();

        for (name, config) in &configs {
            let source = Rgb::new(
                source_data.clone(),
                width,
                height,
                TransferCharacteristic::SRGB,
                ColorPrimaries::BT709,
            )
            .unwrap();

            let distorted = Rgb::new(
                distorted_data.clone(),
                width,
                height,
                TransferCharacteristic::SRGB,
                ColorPrimaries::BT709,
            )
            .unwrap();

            // Warmup
            let _ = compute_frame_ssimulacra2_with_config(source.clone(), distorted.clone(), *config);

            // Time the computation
            let start = Instant::now();
            let mut score = 0.0;
            for _ in 0..iterations {
                let src = Rgb::new(
                    source_data.clone(),
                    width,
                    height,
                    TransferCharacteristic::SRGB,
                    ColorPrimaries::BT709,
                )
                .unwrap();
                let dst = Rgb::new(
                    distorted_data.clone(),
                    width,
                    height,
                    TransferCharacteristic::SRGB,
                    ColorPrimaries::BT709,
                )
                .unwrap();
                score = compute_frame_ssimulacra2_with_config(src, dst, *config).unwrap();
            }
            let elapsed = start.elapsed();
            let ms_per_iter = elapsed.as_secs_f64() * 1000.0 / iterations as f64;

            results.push((name, score, ms_per_iter));
        }

        // Print results
        let baseline_ms = results[0].2;
        for (name, score, ms) in &results {
            let speedup = baseline_ms / ms;
            println!(
                "  {:<30} score={:.6}  time={:.2}ms  speedup={:.2}x",
                name, score, ms, speedup
            );
        }

        // Verify scores match
        let scores: Vec<f64> = results.iter().map(|(_, s, _)| *s).collect();
        let max_diff = scores
            .windows(2)
            .map(|w| (w[0] - w[1]).abs())
            .fold(0.0f64, f64::max);

        if max_diff < 0.001 {
            println!("  Scores match within tolerance (max diff: {:.6})", max_diff);
        } else {
            println!("  WARNING: Score difference: {:.6}", max_diff);
        }
        println!();
    }

    // Print implementation details
    println!("Implementation details:");
    println!("  Blur implementations:");
    println!("    - Scalar: f64 IIR recursive Gaussian");
    println!("    - SIMD: wide crate f32x4");
    #[cfg(feature = "unsafe-simd")]
    println!("    - Unsafe SIMD: raw AVX2/FMA intrinsics");
    println!();
    println!("  XYB conversion:");
    println!("    - Scalar: yuvxyb library");
    println!("    - SIMD: wide crate f32x16");
    #[cfg(feature = "unsafe-simd")]
    println!("    - Unsafe SIMD: raw AVX2/FMA intrinsics");
    println!();
    println!("  SSIM/EdgeDiff:");
    println!("    - Scalar/SIMD: f64 accumulation");
    #[cfg(feature = "unsafe-simd")]
    println!("    - Unsafe SIMD: AVX2 f32 with f64 accumulation");
}
