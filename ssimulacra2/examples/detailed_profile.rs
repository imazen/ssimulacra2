//! Detailed profiling of SSIMULACRA2 operations
//!
//! Instruments each operation to measure actual runtime percentage
//!
//! Run with:
//! ```bash
//! cargo run --release --example detailed_profile
//! cargo run --release --no-default-features --features blur-simd,simd-ops --example detailed_profile
//! ```

use ssimulacra2::*;
use std::time::Instant;
use yuvxyb::{ColorPrimaries, LinearRgb, Rgb, TransferCharacteristic, Xyb};

#[derive(Default)]
struct MsssimScale {
    avg_ssim: [f64; 3 * 2],
    avg_edgediff: [f64; 3 * 4],
}

#[derive(Default)]
struct Msssim {
    scales: Vec<MsssimScale>,
}

impl Msssim {
    fn score(&self) -> f64 {
        let mut ssim = 1.0;
        let mut ssim_max = 1.0;
        for scale in &self.scales {
            for c in 0..3 {
                ssim *= scale.avg_ssim[c * 2 + 1];
                ssim_max *= scale.avg_edgediff[c * 4 + 1].min(scale.avg_edgediff[c * 4 + 3]);
            }
        }
        ssim = ssim.powf(1.0 / (self.scales.len() as f64 * 3.0));
        ssim_max = ssim_max.powf(1.0 / (self.scales.len() as f64 * 3.0));
        if ssim < ssim_max {
            ssim = ssim_max;
        }
        (ssim * 100.0_f64 - 100.0).max(-500.0)
    }
}

fn make_positive_xyb(xyb: &mut Xyb) {
    for pix in xyb.data_mut().iter_mut() {
        pix[2] = (pix[2] - pix[1]) + 0.55;
        pix[0] = (pix[0]).mul_add(14.0, 0.42);
        pix[1] += 0.01;
    }
}

fn xyb_to_planar(xyb: &Xyb) -> [Vec<f32>; 3] {
    #[cfg(feature = "simd-ops")]
    {
        use ssimulacra2::simd_ops;
        return simd_ops::xyb_to_planar_simd(xyb.data(), xyb.width(), xyb.height());
    }

    #[cfg(not(feature = "simd-ops"))]
    {
        let mut out1 = vec![0.0f32; xyb.width() * xyb.height()];
        let mut out2 = vec![0.0f32; xyb.width() * xyb.height()];
        let mut out3 = vec![0.0f32; xyb.width() * xyb.height()];
        for (((i, o1), o2), o3) in xyb
            .data()
            .iter()
            .copied()
            .zip(out1.iter_mut())
            .zip(out2.iter_mut())
            .zip(out3.iter_mut())
        {
            *o1 = i[0];
            *o2 = i[1];
            *o3 = i[2];
        }
        [out1, out2, out3]
    }
}

fn image_multiply(img1: &[Vec<f32>; 3], img2: &[Vec<f32>; 3], out: &mut [Vec<f32>; 3]) {
    #[cfg(feature = "simd-ops")]
    {
        use ssimulacra2::simd_ops;
        simd_ops::image_multiply_simd(img1, img2, out);
    }

    #[cfg(not(feature = "simd-ops"))]
    {
        for ((plane1, plane2), out_plane) in img1.iter().zip(img2.iter()).zip(out.iter_mut()) {
            for ((&p1, &p2), o) in plane1.iter().zip(plane2.iter()).zip(out_plane.iter_mut()) {
                *o = p1 * p2;
            }
        }
    }
}

fn downscale_by_2(in_data: &LinearRgb) -> LinearRgb {
    const SCALE: usize = 2;
    let in_w = in_data.width();
    let in_h = in_data.height();
    let out_w = (in_w + SCALE - 1) / SCALE;
    let out_h = (in_h + SCALE - 1) / SCALE;
    let mut out_data = vec![[0.0f32; 3]; out_w * out_h];

    let in_data = &in_data.data();
    for oy in 0..out_h {
        for ox in 0..out_w {
            let mut sum = [0.0f32; 3];
            for iy in 0..SCALE {
                let y = oy * SCALE + iy;
                if y >= in_h {
                    continue;
                }
                for ix in 0..SCALE {
                    let x = ox * SCALE + ix;
                    if x >= in_w {
                        continue;
                    }
                    let pix = in_data[y * in_w + x];
                    sum[0] += pix[0];
                    sum[1] += pix[1];
                    sum[2] += pix[2];
                }
            }
            out_data[oy * out_w + ox] = [sum[0] / 4.0, sum[1] / 4.0, sum[2] / 4.0];
        }
    }
    LinearRgb::new(out_data, out_w, out_h).expect("Resolution and data size match")
}

fn ssim_map(
    width: usize,
    height: usize,
    m1: &[Vec<f32>; 3],
    m2: &[Vec<f32>; 3],
    s11: &[Vec<f32>; 3],
    s22: &[Vec<f32>; 3],
    s12: &[Vec<f32>; 3],
) -> [f64; 3 * 2] {
    #[cfg(feature = "simd-ops")]
    {
        use ssimulacra2::simd_ops;
        return simd_ops::ssim_map_simd(width, height, m1, m2, s11, s22, s12);
    }

    #[cfg(not(feature = "simd-ops"))]
    {
        const C2: f32 = 0.0009f32;
        let one_per_pixels = 1.0f64 / (width * height) as f64;
        let mut plane_averages = [0f64; 3 * 2];

        for c in 0..3 {
            let mut sum1 = [0.0f64; 2];
            for (row_m1, (row_m2, (row_s11, (row_s22, row_s12)))) in m1[c].chunks_exact(width).zip(
                m2[c].chunks_exact(width).zip(
                    s11[c]
                        .chunks_exact(width)
                        .zip(s22[c].chunks_exact(width).zip(s12[c].chunks_exact(width))),
                ),
            ) {
                for x in 0..width {
                    let mu1 = row_m1[x];
                    let mu2 = row_m2[x];
                    let mu11 = mu1 * mu1;
                    let mu22 = mu2 * mu2;
                    let mu12 = mu1 * mu2;
                    let mu_diff = mu1 - mu2;
                    let num_m = f64::from(mu_diff).mul_add(-f64::from(mu_diff), 1.0f64);
                    let num_s = 2f64.mul_add(f64::from(row_s12[x] - mu12), f64::from(C2));
                    let denom_s =
                        f64::from(row_s11[x] - mu11) + f64::from(row_s22[x] - mu22) + f64::from(C2);
                    let mut d = 1.0f64 - (num_m * num_s) / denom_s;
                    d = d.max(0.0);
                    sum1[0] += d;
                    sum1[1] += d.powi(4);
                }
            }
            plane_averages[c * 2] = one_per_pixels * sum1[0];
            plane_averages[c * 2 + 1] = (one_per_pixels * sum1[1]).sqrt().sqrt();
        }
        plane_averages
    }
}

fn edge_diff_map(
    width: usize,
    height: usize,
    img1: &[Vec<f32>; 3],
    mu1: &[Vec<f32>; 3],
    img2: &[Vec<f32>; 3],
    mu2: &[Vec<f32>; 3],
) -> [f64; 3 * 4] {
    #[cfg(feature = "simd-ops")]
    {
        use ssimulacra2::simd_ops;
        return simd_ops::edge_diff_map_simd(width, height, img1, mu1, img2, mu2);
    }

    #[cfg(not(feature = "simd-ops"))]
    {
        let one_per_pixels = 1.0f64 / (width * height) as f64;
        let mut plane_averages = [0f64; 3 * 4];

        for c in 0..3 {
            let mut sum1 = [0.0f64; 4];
            for (row1, (row2, (rowm1, rowm2))) in img1[c].chunks_exact(width).zip(
                img2[c]
                    .chunks_exact(width)
                    .zip(mu1[c].chunks_exact(width).zip(mu2[c].chunks_exact(width))),
            ) {
                for x in 0..width {
                    let d1: f64 = (1.0 + f64::from((row2[x] - rowm2[x]).abs()))
                        / (1.0 + f64::from((row1[x] - rowm1[x]).abs()))
                        - 1.0;
                    let artifact = d1.max(0.0);
                    sum1[0] += artifact;
                    sum1[1] += artifact.powi(4);
                    let detail_lost = (-d1).max(0.0);
                    sum1[2] += detail_lost;
                    sum1[3] += detail_lost.powi(4);
                }
            }
            plane_averages[c * 4] = one_per_pixels * sum1[0];
            plane_averages[c * 4 + 1] = (one_per_pixels * sum1[1]).sqrt().sqrt();
            plane_averages[c * 4 + 2] = one_per_pixels * sum1[2];
            plane_averages[c * 4 + 3] = (one_per_pixels * sum1[3]).sqrt().sqrt();
        }
        plane_averages
    }
}

#[derive(Default, Debug)]
struct Timings {
    xyb_conversion: f64,
    xyb_to_planar: f64,
    image_multiply: f64,
    blur: f64,
    ssim_map: f64,
    edge_diff_map: f64,
    downscale: f64,
    other: f64,
}

fn compute_with_timing(source: Rgb, distorted: Rgb) -> (f64, Timings) {
    let mut timings = Timings::default();

    let t0 = Instant::now();
    let Ok(mut img1) = LinearRgb::try_from(source) else {
        panic!("conversion failed");
    };
    let Ok(mut img2) = LinearRgb::try_from(distorted) else {
        panic!("conversion failed");
    };
    timings.other += t0.elapsed().as_secs_f64() * 1000.0;

    let mut width = img1.width();
    let mut height = img1.height();

    let mut mul = [
        vec![0.0f32; width * height],
        vec![0.0f32; width * height],
        vec![0.0f32; width * height],
    ];
    let mut blur = Blur::new(width, height);
    let mut msssim = Msssim::default();

    for scale in 0..6 {
        if width < 8 || height < 8 {
            break;
        }

        if scale > 0 {
            let t0 = Instant::now();
            img1 = downscale_by_2(&img1);
            img2 = downscale_by_2(&img2);
            timings.downscale += t0.elapsed().as_secs_f64() * 1000.0;

            width = img1.width();
            height = img2.height();
        }
        for c in &mut mul {
            c.truncate(width * height);
        }
        blur.shrink_to(width, height);

        let t0 = Instant::now();
        let mut img1_xyb = Xyb::from(img1.clone());
        let mut img2_xyb = Xyb::from(img2.clone());
        make_positive_xyb(&mut img1_xyb);
        make_positive_xyb(&mut img2_xyb);
        timings.xyb_conversion += t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        let img1_planar = xyb_to_planar(&img1_xyb);
        let img2_planar = xyb_to_planar(&img2_xyb);
        timings.xyb_to_planar += t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        image_multiply(&img1_planar, &img1_planar, &mut mul);
        timings.image_multiply += t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        let sigma1_sq = blur.blur(&mul);
        timings.blur += t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        image_multiply(&img2_planar, &img2_planar, &mut mul);
        timings.image_multiply += t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        let sigma2_sq = blur.blur(&mul);
        timings.blur += t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        image_multiply(&img1_planar, &img2_planar, &mut mul);
        timings.image_multiply += t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        let sigma12 = blur.blur(&mul);
        timings.blur += t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        let mu1 = blur.blur(&img1_planar);
        timings.blur += t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        let mu2 = blur.blur(&img2_planar);
        timings.blur += t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        let avg_ssim = ssim_map(width, height, &mu1, &mu2, &sigma1_sq, &sigma2_sq, &sigma12);
        timings.ssim_map += t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        let avg_edgediff = edge_diff_map(width, height, &img1_planar, &mu1, &img2_planar, &mu2);
        timings.edge_diff_map += t0.elapsed().as_secs_f64() * 1000.0;

        msssim.scales.push(MsssimScale {
            avg_ssim,
            avg_edgediff,
        });
    }

    (msssim.score(), timings)
}

fn create_test_image_512x512() -> (Rgb, Rgb) {
    let width = 512;
    let height = 512;
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

fn main() {
    println!("SSIMULACRA2 Detailed Runtime Profile - 512x512");
    println!("================================================\n");

    println!("Active features:");
    #[cfg(feature = "blur-simd")]
    println!("  - blur-simd");
    #[cfg(not(feature = "blur-simd"))]
    println!("  - blur-transpose (or blur-accurate)");
    #[cfg(feature = "simd-ops")]
    println!("  - simd-ops");
    #[cfg(not(feature = "simd-ops"))]
    println!("  - scalar compute ops");
    println!();

    let (source, distorted) = create_test_image_512x512();

    // Warmup
    for _ in 0..3 {
        let _ = compute_with_timing(source.clone(), distorted.clone());
    }

    // Profile 100 iterations
    println!("Running 100 iterations with detailed timing...\n");
    let mut total_timings = Timings::default();

    for _ in 0..100 {
        let (_score, timings) = compute_with_timing(source.clone(), distorted.clone());
        total_timings.xyb_conversion += timings.xyb_conversion;
        total_timings.xyb_to_planar += timings.xyb_to_planar;
        total_timings.image_multiply += timings.image_multiply;
        total_timings.blur += timings.blur;
        total_timings.ssim_map += timings.ssim_map;
        total_timings.edge_diff_map += timings.edge_diff_map;
        total_timings.downscale += timings.downscale;
        total_timings.other += timings.other;
    }

    let total = total_timings.xyb_conversion
        + total_timings.xyb_to_planar
        + total_timings.image_multiply
        + total_timings.blur
        + total_timings.ssim_map
        + total_timings.edge_diff_map
        + total_timings.downscale
        + total_timings.other;

    println!("Total runtime (100 iterations): {:.3} ms", total);
    println!("Average per iteration: {:.3} ms\n", total / 100.0);

    println!("Runtime breakdown:");
    println!(
        "  blur:             {:.3} ms ({:.1}%)",
        total_timings.blur / 100.0,
        total_timings.blur / total * 100.0
    );
    println!(
        "  xyb_to_planar:    {:.3} ms ({:.1}%)",
        total_timings.xyb_to_planar / 100.0,
        total_timings.xyb_to_planar / total * 100.0
    );
    println!(
        "  image_multiply:   {:.3} ms ({:.1}%)",
        total_timings.image_multiply / 100.0,
        total_timings.image_multiply / total * 100.0
    );
    println!(
        "  ssim_map:         {:.3} ms ({:.1}%)",
        total_timings.ssim_map / 100.0,
        total_timings.ssim_map / total * 100.0
    );
    println!(
        "  edge_diff_map:    {:.3} ms ({:.1}%)",
        total_timings.edge_diff_map / 100.0,
        total_timings.edge_diff_map / total * 100.0
    );
    println!(
        "  downscale:        {:.3} ms ({:.1}%)",
        total_timings.downscale / 100.0,
        total_timings.downscale / total * 100.0
    );
    println!(
        "  xyb_conversion:   {:.3} ms ({:.1}%)",
        total_timings.xyb_conversion / 100.0,
        total_timings.xyb_conversion / total * 100.0
    );
    println!(
        "  other:            {:.3} ms ({:.1}%)",
        total_timings.other / 100.0,
        total_timings.other / total * 100.0
    );
}
