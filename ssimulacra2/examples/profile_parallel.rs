//! Profile parallel vs sequential to identify bottlenecks
//!
//! Run with: cargo run --release --features rayon --example profile_parallel

#[cfg(feature = "rayon")]
use ssimulacra2::compute_frame_ssimulacra2;
#[cfg(feature = "rayon")]
use ssimulacra2::compute_frame_ssimulacra2_parallel;
#[cfg(feature = "rayon")]
use std::time::Instant;
#[cfg(feature = "rayon")]
use yuvxyb::{ColorPrimaries, Rgb, TransferCharacteristic};

#[cfg(feature = "rayon")]
fn create_test_images(width: usize, height: usize) -> (Rgb, Rgb) {
    let size = width * height;

    let source_data: Vec<[f32; 3]> = (0..size)
        .map(|i| {
            let x = (i % width) as f32 / width as f32;
            let y = (i / width) as f32 / height as f32;
            [x, y, (x + y) / 2.0]
        })
        .collect();

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

#[cfg(feature = "rayon")]
fn main() {
    use rayon::prelude::*;

    println!("Parallel vs Sequential Profiling");
    println!("=================================\n");

    let num_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    println!("Available CPU cores: {}\n", num_cpus);

    // Test 2048x2048 specifically to understand the bottleneck
    let (width, height) = (2048, 2048);
    let (source, distorted) = create_test_images(width, height);

    println!("Profiling {}x{} images...\n", width, height);

    // Measure just allocation time
    println!("1. Measuring allocation overhead:");
    let size = width * height;
    let alloc_start = Instant::now();
    for _ in 0..6 {
        // Simulate what each parallel task allocates
        let _mul: [Vec<f32>; 3] = [vec![0.0f32; size], vec![0.0f32; size], vec![0.0f32; size]];
        let _img1_planar: [Vec<f32>; 3] =
            [vec![0.0f32; size], vec![0.0f32; size], vec![0.0f32; size]];
        let _img2_planar: [Vec<f32>; 3] =
            [vec![0.0f32; size], vec![0.0f32; size], vec![0.0f32; size]];
        let _sigma1_sq: [Vec<f32>; 3] =
            [vec![0.0f32; size], vec![0.0f32; size], vec![0.0f32; size]];
        let _sigma2_sq: [Vec<f32>; 3] =
            [vec![0.0f32; size], vec![0.0f32; size], vec![0.0f32; size]];
        let _sigma12: [Vec<f32>; 3] = [vec![0.0f32; size], vec![0.0f32; size], vec![0.0f32; size]];
        let _mu1: [Vec<f32>; 3] = [vec![0.0f32; size], vec![0.0f32; size], vec![0.0f32; size]];
        let _mu2: [Vec<f32>; 3] = [vec![0.0f32; size], vec![0.0f32; size], vec![0.0f32; size]];
    }
    let sequential_alloc = alloc_start.elapsed();
    println!(
        "   Sequential (6 scales): {:.2}ms",
        sequential_alloc.as_secs_f64() * 1000.0
    );

    // Parallel allocation
    let alloc_start = Instant::now();
    let _results: Vec<_> = (0..6)
        .into_par_iter()
        .map(|_| {
            let _mul: [Vec<f32>; 3] = [vec![0.0f32; size], vec![0.0f32; size], vec![0.0f32; size]];
            let _img1_planar: [Vec<f32>; 3] =
                [vec![0.0f32; size], vec![0.0f32; size], vec![0.0f32; size]];
            let _img2_planar: [Vec<f32>; 3] =
                [vec![0.0f32; size], vec![0.0f32; size], vec![0.0f32; size]];
            let _sigma1_sq: [Vec<f32>; 3] =
                [vec![0.0f32; size], vec![0.0f32; size], vec![0.0f32; size]];
            let _sigma2_sq: [Vec<f32>; 3] =
                [vec![0.0f32; size], vec![0.0f32; size], vec![0.0f32; size]];
            let _sigma12: [Vec<f32>; 3] =
                [vec![0.0f32; size], vec![0.0f32; size], vec![0.0f32; size]];
            let _mu1: [Vec<f32>; 3] = [vec![0.0f32; size], vec![0.0f32; size], vec![0.0f32; size]];
            let _mu2: [Vec<f32>; 3] = [vec![0.0f32; size], vec![0.0f32; size], vec![0.0f32; size]];
            0
        })
        .collect();
    let parallel_alloc = alloc_start.elapsed();
    println!(
        "   Parallel (6 scales):   {:.2}ms",
        parallel_alloc.as_secs_f64() * 1000.0
    );
    println!(
        "   Allocation slowdown:   {:.2}x\n",
        parallel_alloc.as_secs_f64() / sequential_alloc.as_secs_f64()
    );

    // Measure buffer sizes
    let total_per_scale = size * 4 * 8 * 3; // 8 arrays of 3 planes, f32 each
    println!("2. Memory usage per scale:");
    println!(
        "   Scale 0 (2048x2048): {} MB per task",
        total_per_scale / 1024 / 1024
    );
    println!(
        "   Total if all 6 run parallel: {} MB\n",
        total_per_scale * 6 / 1024 / 1024
    );

    // Full sequential run
    println!("3. Full computation comparison:");
    let warmup_src = Rgb::new(
        source.data().to_vec(),
        width,
        height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .unwrap();
    let warmup_dst = Rgb::new(
        distorted.data().to_vec(),
        width,
        height,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .unwrap();
    let _ = compute_frame_ssimulacra2(warmup_src, warmup_dst);

    let iterations = 5;

    let seq_start = Instant::now();
    for _ in 0..iterations {
        let src = Rgb::new(
            source.data().to_vec(),
            width,
            height,
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .unwrap();
        let dst = Rgb::new(
            distorted.data().to_vec(),
            width,
            height,
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .unwrap();
        let _ = compute_frame_ssimulacra2(src, dst);
    }
    let seq_time = seq_start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

    let par_start = Instant::now();
    for _ in 0..iterations {
        let src = Rgb::new(
            source.data().to_vec(),
            width,
            height,
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .unwrap();
        let dst = Rgb::new(
            distorted.data().to_vec(),
            width,
            height,
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .unwrap();
        let _ = compute_frame_ssimulacra2_parallel(src, dst);
    }
    let par_time = par_start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

    println!("   Sequential: {:.2}ms", seq_time);
    println!("   Parallel:   {:.2}ms", par_time);
    println!("   Speedup:    {:.2}x", seq_time / par_time);

    // Calculate overhead
    let ideal_parallel = seq_time / num_cpus as f64;
    let overhead = par_time - ideal_parallel;
    println!("\n4. Overhead analysis:");
    println!(
        "   Ideal parallel time ({}x):  {:.2}ms",
        num_cpus, ideal_parallel
    );
    println!("   Actual parallel time:        {:.2}ms", par_time);
    println!(
        "   Overhead:                    {:.2}ms ({:.0}%)",
        overhead,
        overhead / seq_time * 100.0
    );
}

#[cfg(not(feature = "rayon"))]
fn main() {
    println!("ERROR: This example requires the 'rayon' feature.");
    println!("Run with: cargo run --release --features rayon --example profile_parallel");
}
