//! Benchmark comparing unsafe SIMD blur vs safe SIMD blur
//!
//! Run with:
//!   cargo run --release --example benchmark_unsafe_simd --no-default-features --features blur-unsafe-simd
//!   cargo run --release --example benchmark_unsafe_simd --features blur-simd

use std::time::Instant;

use ssimulacra2::{compute_frame_ssimulacra2, Blur};
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

fn benchmark_blur(width: usize, height: usize, iterations: usize) -> f64 {
    // Create test plane
    let plane: Vec<f32> = (0..width * height)
        .map(|i| (i as f32 / (width * height) as f32))
        .collect();

    let mut blur = Blur::new(width, height);
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

fn benchmark_full_ssimulacra2(width: usize, height: usize, iterations: usize) -> f64 {
    let source = create_test_image(width, height, 12345);
    let distorted = create_test_image(width, height, 67890);

    // Warmup
    for _ in 0..3 {
        let _ = compute_frame_ssimulacra2(source.clone(), distorted.clone());
    }

    // Timed runs
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = compute_frame_ssimulacra2(source.clone(), distorted.clone());
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

fn main() {
    println!("SSIMULACRA2 Blur Backend Benchmark");
    println!("===================================\n");

    #[cfg(feature = "blur-unsafe-simd")]
    println!("Backend: blur-unsafe-simd (raw x86 intrinsics)");
    #[cfg(feature = "blur-simd")]
    println!("Backend: blur-simd (wide crate)");
    #[cfg(all(not(feature = "blur-simd"), not(feature = "blur-unsafe-simd")))]
    println!("Backend: baseline (f64 IIR)");

    println!();

    // Test different image sizes
    let sizes = [
        (256, 256, "256x256", 200),
        (512, 512, "512x512", 100),
        (1024, 1024, "1024x1024", 50),
        (1920, 1080, "1920x1080 (FHD)", 30),
        (3840, 2160, "3840x2160 (4K)", 10),
    ];

    println!("Blur-only benchmark (3 planes):");
    println!("{:20} {:>12} {:>12}", "Size", "Time (ms)", "MP/s");
    println!("{:-<50}", "");

    for (width, height, name, iters) in sizes.iter() {
        let ms = benchmark_blur(*width, *height, *iters);
        let mpixels = (*width * *height) as f64 / 1_000_000.0;
        let mp_per_sec = mpixels * 3.0 / (ms / 1000.0); // 3 planes
        println!("{:20} {:>12.3} {:>12.1}", name, ms, mp_per_sec);
    }

    println!("\nFull SSIMULACRA2 benchmark:");
    println!("{:20} {:>12} {:>12}", "Size", "Time (ms)", "MP/s");
    println!("{:-<50}", "");

    for (width, height, name, iters) in sizes.iter() {
        let ms = benchmark_full_ssimulacra2(*width, *height, *iters / 2);
        let mpixels = (*width * *height) as f64 / 1_000_000.0;
        let mp_per_sec = mpixels / (ms / 1000.0);
        println!("{:20} {:>12.3} {:>12.1}", name, ms, mp_per_sec);
    }

    println!("\nDone.");
}
