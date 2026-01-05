//! Benchmark comparing all blur implementations and full SSIMULACRA2 pipeline
//!
//! Run with:
//!   cargo run --release --example benchmark_unsafe_simd
//!   cargo run --release --example benchmark_unsafe_simd --features unsafe-simd

use std::time::Instant;

use ssimulacra2::{compute_frame_ssimulacra2_with_config, Blur, BlurImpl, Ssimulacra2Config};
use yuvxyb::{ColorPrimaries, Rgb, TransferCharacteristic};

fn create_test_image(width: usize, height: usize, seed: u64) -> Rgb {
    let mut state = seed;
    let data: Vec<[f32; 3]> = (0..width * height)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = ((state >> 33) & 0xFF) as f32 / 255.0;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let g = ((state >> 33) & 0xFF) as f32 / 255.0;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let b = ((state >> 33) & 0xFF) as f32 / 255.0;
            [r, g, b]
        })
        .collect();

    Rgb::new(
        data,
        width,
        height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .unwrap()
}

fn benchmark_blur(width: usize, height: usize, impl_type: BlurImpl, iterations: usize) -> f64 {
    // Create test plane
    let plane: Vec<f32> = (0..width * height)
        .map(|i| (i as f32 / (width * height) as f32))
        .collect();

    let mut blur = Blur::with_impl(width, height, impl_type);
    let img = [plane.clone(), plane.clone(), plane.clone()];

    // Warmup
    for _ in 0..5 {
        let _ = blur.blur(&img);
    }

    // Timed runs
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = blur.blur(&img);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

fn benchmark_full_ssimulacra2(
    width: usize,
    height: usize,
    config: Ssimulacra2Config,
    iterations: usize,
) -> f64 {
    let source = create_test_image(width, height, 12345);
    let distorted = create_test_image(width, height, 67890);

    // Warmup
    for _ in 0..3 {
        let _ = compute_frame_ssimulacra2_with_config(source.clone(), distorted.clone(), config);
    }

    // Timed runs
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = compute_frame_ssimulacra2_with_config(source.clone(), distorted.clone(), config);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

fn main() {
    println!("SSIMULACRA2 Implementation Benchmark");
    println!("=====================================\n");

    // Test different image sizes
    let sizes = [
        (256, 256, "256x256", 200),
        (512, 512, "512x512", 100),
        (1024, 1024, "1024x1024", 50),
        (1920, 1080, "1920x1080 (FHD)", 30),
    ];

    // Blur-only benchmarks
    println!("Blur-only benchmark (3 planes):");
    println!(
        "{:20} {:>12} {:>12} {:>12} {:>12}",
        "Size", "Scalar", "SIMD", "Transpose", "Unsafe"
    );
    println!("{:-<75}", "");

    for (width, height, name, iters) in sizes.iter() {
        let scalar_ms = benchmark_blur(*width, *height, BlurImpl::Scalar, *iters);
        let simd_ms = benchmark_blur(*width, *height, BlurImpl::Simd, *iters);
        let transpose_ms = benchmark_blur(*width, *height, BlurImpl::SimdTranspose, *iters);

        #[cfg(feature = "unsafe-simd")]
        let unsafe_ms = benchmark_blur(*width, *height, BlurImpl::UnsafeSimd, *iters);
        #[cfg(not(feature = "unsafe-simd"))]
        let unsafe_ms = f64::NAN;

        println!(
            "{:20} {:>12.3} {:>12.3} {:>12.3} {:>12.3}",
            name, scalar_ms, simd_ms, transpose_ms, unsafe_ms
        );
    }

    println!("\nFull SSIMULACRA2 benchmark:");
    println!(
        "{:20} {:>12} {:>12} {:>12} {:>12}",
        "Size", "Scalar", "SIMD", "Transpose", "Unsafe"
    );
    println!("{:-<75}", "");

    for (width, height, name, iters) in sizes.iter() {
        let iters = iters / 2;

        let scalar_ms =
            benchmark_full_ssimulacra2(*width, *height, Ssimulacra2Config::scalar(), iters);
        let simd_ms = benchmark_full_ssimulacra2(*width, *height, Ssimulacra2Config::simd(), iters);
        let transpose_ms =
            benchmark_full_ssimulacra2(*width, *height, Ssimulacra2Config::simd_transpose(), iters);

        #[cfg(feature = "unsafe-simd")]
        let unsafe_ms =
            benchmark_full_ssimulacra2(*width, *height, Ssimulacra2Config::unsafe_simd(), iters);
        #[cfg(not(feature = "unsafe-simd"))]
        let unsafe_ms = f64::NAN;

        println!(
            "{:20} {:>12.3} {:>12.3} {:>12.3} {:>12.3}",
            name, scalar_ms, simd_ms, transpose_ms, unsafe_ms
        );
    }

    println!("\nDone.");

    #[cfg(not(feature = "unsafe-simd"))]
    println!("\nNote: Unsafe column shows NaN - run with --features unsafe-simd to enable");
}
