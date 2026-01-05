/// SIMD-optimized operations for SSIMULACRA2 computation
///
/// Uses the `wide` crate for portable SIMD across x86 (SSE/AVX) and ARM (NEON)

use multiversion::multiversion;
use wide::f32x16;

/// SIMD-optimized SSIM map computation
///
/// Processes 16 pixels at once using f32x16, then accumulates in f64 for precision
#[inline(always)]
#[multiversion(targets("x86_64+avx2+fma", "x86_64+sse2", "aarch64+neon"))]
pub fn ssim_map_simd(
    width: usize,
    height: usize,
    m1: &[Vec<f32>; 3],
    m2: &[Vec<f32>; 3],
    s11: &[Vec<f32>; 3],
    s22: &[Vec<f32>; 3],
    s12: &[Vec<f32>; 3],
) -> [f64; 3 * 2] {
    const C2: f32 = 0.0009f32;
    let c2_simd = f32x16::splat(C2);
    let one_simd = f32x16::splat(1.0);
    let two_simd = f32x16::splat(2.0);
    let zero_simd = f32x16::splat(0.0);

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
            let mut x = 0;

            // Process 16 pixels at a time with SIMD
            while x + 16 <= width {
                // Load 16 pixels
                let mu1 = f32x16::new([
                    row_m1[x],
                    row_m1[x + 1],
                    row_m1[x + 2],
                    row_m1[x + 3],
                    row_m1[x + 4],
                    row_m1[x + 5],
                    row_m1[x + 6],
                    row_m1[x + 7],
                    row_m1[x + 8],
                    row_m1[x + 9],
                    row_m1[x + 10],
                    row_m1[x + 11],
                    row_m1[x + 12],
                    row_m1[x + 13],
                    row_m1[x + 14],
                    row_m1[x + 15],
                ]);
                let mu2 = f32x16::new([
                    row_m2[x],
                    row_m2[x + 1],
                    row_m2[x + 2],
                    row_m2[x + 3],
                    row_m2[x + 4],
                    row_m2[x + 5],
                    row_m2[x + 6],
                    row_m2[x + 7],
                    row_m2[x + 8],
                    row_m2[x + 9],
                    row_m2[x + 10],
                    row_m2[x + 11],
                    row_m2[x + 12],
                    row_m2[x + 13],
                    row_m2[x + 14],
                    row_m2[x + 15],
                ]);
                let s11_vals = f32x16::new([
                    row_s11[x],
                    row_s11[x + 1],
                    row_s11[x + 2],
                    row_s11[x + 3],
                    row_s11[x + 4],
                    row_s11[x + 5],
                    row_s11[x + 6],
                    row_s11[x + 7],
                    row_s11[x + 8],
                    row_s11[x + 9],
                    row_s11[x + 10],
                    row_s11[x + 11],
                    row_s11[x + 12],
                    row_s11[x + 13],
                    row_s11[x + 14],
                    row_s11[x + 15],
                ]);
                let s22_vals = f32x16::new([
                    row_s22[x],
                    row_s22[x + 1],
                    row_s22[x + 2],
                    row_s22[x + 3],
                    row_s22[x + 4],
                    row_s22[x + 5],
                    row_s22[x + 6],
                    row_s22[x + 7],
                    row_s22[x + 8],
                    row_s22[x + 9],
                    row_s22[x + 10],
                    row_s22[x + 11],
                    row_s22[x + 12],
                    row_s22[x + 13],
                    row_s22[x + 14],
                    row_s22[x + 15],
                ]);
                let s12_vals = f32x16::new([
                    row_s12[x],
                    row_s12[x + 1],
                    row_s12[x + 2],
                    row_s12[x + 3],
                    row_s12[x + 4],
                    row_s12[x + 5],
                    row_s12[x + 6],
                    row_s12[x + 7],
                    row_s12[x + 8],
                    row_s12[x + 9],
                    row_s12[x + 10],
                    row_s12[x + 11],
                    row_s12[x + 12],
                    row_s12[x + 13],
                    row_s12[x + 14],
                    row_s12[x + 15],
                ]);

                // Compute intermediate values
                let mu11 = mu1 * mu1;
                let mu22 = mu2 * mu2;
                let mu12 = mu1 * mu2;
                let mu_diff = mu1 - mu2;

                // num_m = 1.0 - mu_diff^2
                let num_m = mu_diff.mul_add(-mu_diff, one_simd);

                // num_s = 2 * (s12 - mu12) + C2
                let num_s = two_simd.mul_add(s12_vals - mu12, c2_simd);

                // denom_s = (s11 - mu11) + (s22 - mu22) + C2
                let denom_s = (s11_vals - mu11) + (s22_vals - mu22) + c2_simd;

                // d = 1.0 - (num_m * num_s) / denom_s
                let d = one_simd - (num_m * num_s) / denom_s;

                // Clamp to 0.0 (max with zero)
                let d = d.max(zero_simd);

                // Extract values and accumulate in f64 for precision
                let d_arr = d.to_array();
                for i in 0..16 {
                    let d_f64 = f64::from(d_arr[i]);
                    sum1[0] += d_f64;
                    sum1[1] += d_f64.powi(4);
                }

                x += 16;
            }

            // Handle remaining pixels with scalar code
            for x in x..width {
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

/// SIMD-optimized edge difference map computation
#[inline(always)]
#[multiversion(targets("x86_64+avx2+fma", "x86_64+sse2", "aarch64+neon"))]
pub fn edge_diff_map_simd(
    width: usize,
    height: usize,
    img1: &[Vec<f32>; 3],
    mu1: &[Vec<f32>; 3],
    img2: &[Vec<f32>; 3],
    mu2: &[Vec<f32>; 3],
) -> [f64; 3 * 4] {
    let one_per_pixels = 1.0f64 / (width * height) as f64;
    let mut plane_averages = [0f64; 3 * 4];

    let one_simd = f32x16::splat(1.0);
    let zero_simd = f32x16::splat(0.0);

    for c in 0..3 {
        let mut sum1 = [0.0f64; 4];

        for (row1, (row2, (rowm1, rowm2))) in img1[c].chunks_exact(width).zip(
            img2[c]
                .chunks_exact(width)
                .zip(mu1[c].chunks_exact(width).zip(mu2[c].chunks_exact(width))),
        ) {
            let mut x = 0;

            // Process 16 pixels at once with SIMD
            while x + 16 <= width {
                // Load values
                let r1 = f32x16::new([
                    row1[x],
                    row1[x + 1],
                    row1[x + 2],
                    row1[x + 3],
                    row1[x + 4],
                    row1[x + 5],
                    row1[x + 6],
                    row1[x + 7],
                    row1[x + 8],
                    row1[x + 9],
                    row1[x + 10],
                    row1[x + 11],
                    row1[x + 12],
                    row1[x + 13],
                    row1[x + 14],
                    row1[x + 15],
                ]);
                let r2 = f32x16::new([
                    row2[x],
                    row2[x + 1],
                    row2[x + 2],
                    row2[x + 3],
                    row2[x + 4],
                    row2[x + 5],
                    row2[x + 6],
                    row2[x + 7],
                    row2[x + 8],
                    row2[x + 9],
                    row2[x + 10],
                    row2[x + 11],
                    row2[x + 12],
                    row2[x + 13],
                    row2[x + 14],
                    row2[x + 15],
                ]);
                let rm1 = f32x16::new([
                    rowm1[x],
                    rowm1[x + 1],
                    rowm1[x + 2],
                    rowm1[x + 3],
                    rowm1[x + 4],
                    rowm1[x + 5],
                    rowm1[x + 6],
                    rowm1[x + 7],
                    rowm1[x + 8],
                    rowm1[x + 9],
                    rowm1[x + 10],
                    rowm1[x + 11],
                    rowm1[x + 12],
                    rowm1[x + 13],
                    rowm1[x + 14],
                    rowm1[x + 15],
                ]);
                let rm2 = f32x16::new([
                    rowm2[x],
                    rowm2[x + 1],
                    rowm2[x + 2],
                    rowm2[x + 3],
                    rowm2[x + 4],
                    rowm2[x + 5],
                    rowm2[x + 6],
                    rowm2[x + 7],
                    rowm2[x + 8],
                    rowm2[x + 9],
                    rowm2[x + 10],
                    rowm2[x + 11],
                    rowm2[x + 12],
                    rowm2[x + 13],
                    rowm2[x + 14],
                    rowm2[x + 15],
                ]);

                // d1 = (1 + |row2 - rowm2|) / (1 + |row1 - rowm1|) - 1
                let d1_temp = r1 - rm1;
                let diff1 = d1_temp.max(-d1_temp); // abs() = max(x, -x)
                let d2_temp = r2 - rm2;
                let diff2 = d2_temp.max(-d2_temp); // abs() = max(x, -x)
                let d1 = (one_simd + diff2) / (one_simd + diff1) - one_simd;

                // artifact = max(d1, 0)
                let artifact = d1.max(zero_simd);

                // detail_lost = max(-d1, 0)
                let detail_lost = (-d1).max(zero_simd);

                // Accumulate
                let artifact_arr = artifact.to_array();
                let detail_arr = detail_lost.to_array();

                for i in 0..16 {
                    let a = f64::from(artifact_arr[i]);
                    let d = f64::from(detail_arr[i]);
                    sum1[0] += a;
                    sum1[1] += a.powi(4);
                    sum1[2] += d;
                    sum1[3] += d.powi(4);
                }

                x += 16;
            }

            // Handle remaining pixels with scalar code
            for x in x..width {
                let d1: f64 = (1.0 + f64::from((row2[x] - rowm2[x]).abs()))
                    / (1.0 + f64::from((row1[x] - rowm1[x]).abs()))
                    - 1.0;
                let artifact = d1.max(0.0);
                let detail_lost = (-d1).max(0.0);
                sum1[0] += artifact;
                sum1[1] += artifact.powi(4);
                sum1[2] += detail_lost;
                sum1[3] += detail_lost.powi(4);
            }
        }

        for i in 0..4 {
            plane_averages[c * 4 + i] = one_per_pixels * sum1[i];
        }
        plane_averages[c * 4 + 1] = plane_averages[c * 4 + 1].sqrt().sqrt();
        plane_averages[c * 4 + 3] = plane_averages[c * 4 + 3].sqrt().sqrt();
    }

    plane_averages
}

/// SIMD-optimized image multiplication
#[inline(always)]
#[multiversion(targets("x86_64+avx2+fma", "x86_64+sse2", "aarch64+neon"))]
pub fn image_multiply_simd(img1: &[Vec<f32>; 3], img2: &[Vec<f32>; 3], out: &mut [Vec<f32>; 3]) {
    for c in 0..3 {
        let plane1 = &img1[c];
        let plane2 = &img2[c];
        let out_plane = &mut out[c];

        let mut i = 0;

        // Process 16 elements at a time
        while i + 16 <= plane1.len() {
            let p1 = f32x16::new([
                plane1[i],
                plane1[i + 1],
                plane1[i + 2],
                plane1[i + 3],
                plane1[i + 4],
                plane1[i + 5],
                plane1[i + 6],
                plane1[i + 7],
                plane1[i + 8],
                plane1[i + 9],
                plane1[i + 10],
                plane1[i + 11],
                plane1[i + 12],
                plane1[i + 13],
                plane1[i + 14],
                plane1[i + 15],
            ]);
            let p2 = f32x16::new([
                plane2[i],
                plane2[i + 1],
                plane2[i + 2],
                plane2[i + 3],
                plane2[i + 4],
                plane2[i + 5],
                plane2[i + 6],
                plane2[i + 7],
                plane2[i + 8],
                plane2[i + 9],
                plane2[i + 10],
                plane2[i + 11],
                plane2[i + 12],
                plane2[i + 13],
                plane2[i + 14],
                plane2[i + 15],
            ]);
            let result = p1 * p2;
            let result_arr = result.to_array();

            for j in 0..16 {
                out_plane[i + j] = result_arr[j];
            }

            i += 16;
        }

        // Handle remaining elements
        for i in i..plane1.len() {
            out_plane[i] = plane1[i] * plane2[i];
        }
    }
}
