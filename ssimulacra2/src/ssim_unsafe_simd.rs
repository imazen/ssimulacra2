//! Unsafe SIMD implementation of SSIM map and edge diff map
//!
//! Uses raw AVX2/SSE intrinsics for maximum performance.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const C2: f32 = 0.0009f32;

/// Fast horizontal sum of 8 f32s in an AVX register
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx")]
unsafe fn hsum_ps_avx(v: __m256) -> f32 {
    // Add high 128 to low 128
    let low = _mm256_castps256_ps128(v);
    let high = _mm256_extractf128_ps(v, 1);
    let sum128 = _mm_add_ps(low, high);
    // Horizontal add within 128-bit
    let shuf = _mm_movehdup_ps(sum128); // [1,1,3,3]
    let sums = _mm_add_ps(sum128, shuf);
    let shuf = _mm_movehl_ps(sums, sums);
    let sums = _mm_add_ss(sums, shuf);
    _mm_cvtss_f32(sums)
}

/// Computes SSIM map using unsafe SIMD
pub fn ssim_map_unsafe(
    width: usize,
    height: usize,
    m1: &[Vec<f32>; 3],
    m2: &[Vec<f32>; 3],
    s11: &[Vec<f32>; 3],
    s22: &[Vec<f32>; 3],
    s12: &[Vec<f32>; 3],
) -> [f64; 3 * 2] {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { ssim_map_avx2(width, height, m1, m2, s11, s22, s12) };
        }
    }
    ssim_map_scalar(width, height, m1, m2, s11, s22, s12)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn ssim_map_avx2(
    width: usize,
    height: usize,
    m1: &[Vec<f32>; 3],
    m2: &[Vec<f32>; 3],
    s11: &[Vec<f32>; 3],
    s22: &[Vec<f32>; 3],
    s12: &[Vec<f32>; 3],
) -> [f64; 3 * 2] {
    let one_per_pixels = 1.0f64 / (width * height) as f64;
    let mut plane_averages = [0f64; 3 * 2];

    let c2_vec = _mm256_set1_ps(C2);
    let one_vec = _mm256_set1_ps(1.0);
    let zero_vec = _mm256_setzero_ps();

    for c in 0..3 {
        let mut sum_d = 0.0f64;
        let mut sum_d4 = 0.0f64;

        let m1_plane = &m1[c];
        let m2_plane = &m2[c];
        let s11_plane = &s11[c];
        let s22_plane = &s22[c];
        let s12_plane = &s12[c];

        let chunks_8 = m1_plane.len() / 8;

        for chunk in 0..chunks_8 {
            let base = chunk * 8;

            let mu1 = _mm256_loadu_ps(m1_plane.as_ptr().add(base));
            let mu2 = _mm256_loadu_ps(m2_plane.as_ptr().add(base));
            let sigma11 = _mm256_loadu_ps(s11_plane.as_ptr().add(base));
            let sigma22 = _mm256_loadu_ps(s22_plane.as_ptr().add(base));
            let sigma12 = _mm256_loadu_ps(s12_plane.as_ptr().add(base));

            // mu11 = mu1 * mu1
            let mu11 = _mm256_mul_ps(mu1, mu1);
            // mu22 = mu2 * mu2
            let mu22 = _mm256_mul_ps(mu2, mu2);
            // mu12 = mu1 * mu2
            let mu12 = _mm256_mul_ps(mu1, mu2);
            // mu_diff = mu1 - mu2
            let mu_diff = _mm256_sub_ps(mu1, mu2);

            // num_m = 1 - mu_diff * mu_diff
            let mu_diff_sq = _mm256_mul_ps(mu_diff, mu_diff);
            let num_m = _mm256_sub_ps(one_vec, mu_diff_sq);

            // num_s = 2 * (sigma12 - mu12) + C2
            let s12_minus_mu12 = _mm256_sub_ps(sigma12, mu12);
            let two_s12 = _mm256_add_ps(s12_minus_mu12, s12_minus_mu12);
            let num_s = _mm256_add_ps(two_s12, c2_vec);

            // denom_s = (sigma11 - mu11) + (sigma22 - mu22) + C2
            let s11_minus_mu11 = _mm256_sub_ps(sigma11, mu11);
            let s22_minus_mu22 = _mm256_sub_ps(sigma22, mu22);
            let denom_s = _mm256_add_ps(_mm256_add_ps(s11_minus_mu11, s22_minus_mu22), c2_vec);

            // d = 1 - (num_m * num_s) / denom_s
            let num = _mm256_mul_ps(num_m, num_s);
            let ratio = _mm256_div_ps(num, denom_s);
            let d = _mm256_sub_ps(one_vec, ratio);

            // d = max(d, 0)
            let d = _mm256_max_ps(d, zero_vec);

            // d^4 = d * d * d * d
            let d2 = _mm256_mul_ps(d, d);
            let d4 = _mm256_mul_ps(d2, d2);

            // Efficient horizontal sum using hadd
            sum_d += hsum_ps_avx(d) as f64;
            sum_d4 += hsum_ps_avx(d4) as f64;
        }

        // Handle remainder with scalar
        let remaining_start = chunks_8 * 8;
        for x in remaining_start..m1_plane.len() {
            let mu1 = m1_plane[x];
            let mu2 = m2_plane[x];
            let mu11 = mu1 * mu1;
            let mu22 = mu2 * mu2;
            let mu12 = mu1 * mu2;
            let mu_diff = mu1 - mu2;

            let num_m = f64::from(mu_diff).mul_add(-f64::from(mu_diff), 1.0f64);
            let num_s = 2f64.mul_add(f64::from(s12_plane[x] - mu12), f64::from(C2));
            let denom_s =
                f64::from(s11_plane[x] - mu11) + f64::from(s22_plane[x] - mu22) + f64::from(C2);
            let mut d = 1.0f64 - (num_m * num_s) / denom_s;
            d = d.max(0.0);
            sum_d += d;
            sum_d4 += d.powi(4);
        }

        plane_averages[c * 2] = one_per_pixels * sum_d;
        plane_averages[c * 2 + 1] = (one_per_pixels * sum_d4).sqrt().sqrt();
    }

    plane_averages
}

fn ssim_map_scalar(
    width: usize,
    height: usize,
    m1: &[Vec<f32>; 3],
    m2: &[Vec<f32>; 3],
    s11: &[Vec<f32>; 3],
    s22: &[Vec<f32>; 3],
    s12: &[Vec<f32>; 3],
) -> [f64; 3 * 2] {
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

/// Computes edge diff map using unsafe SIMD
pub fn edge_diff_map_unsafe(
    width: usize,
    height: usize,
    img1: &[Vec<f32>; 3],
    mu1: &[Vec<f32>; 3],
    img2: &[Vec<f32>; 3],
    mu2: &[Vec<f32>; 3],
) -> [f64; 3 * 4] {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { edge_diff_map_avx2(width, height, img1, mu1, img2, mu2) };
        }
    }
    edge_diff_map_scalar(width, height, img1, mu1, img2, mu2)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn edge_diff_map_avx2(
    width: usize,
    height: usize,
    img1: &[Vec<f32>; 3],
    mu1: &[Vec<f32>; 3],
    img2: &[Vec<f32>; 3],
    mu2: &[Vec<f32>; 3],
) -> [f64; 3 * 4] {
    let one_per_pixels = 1.0f64 / (width * height) as f64;
    let mut plane_averages = [0f64; 3 * 4];

    let one_vec = _mm256_set1_ps(1.0);
    let zero_vec = _mm256_setzero_ps();
    let sign_mask = _mm256_set1_ps(-0.0); // For absolute value

    for c in 0..3 {
        let mut sum_artifact = 0.0f64;
        let mut sum_artifact4 = 0.0f64;
        let mut sum_detail_lost = 0.0f64;
        let mut sum_detail_lost4 = 0.0f64;

        let img1_plane = &img1[c];
        let mu1_plane = &mu1[c];
        let img2_plane = &img2[c];
        let mu2_plane = &mu2[c];

        let chunks_8 = img1_plane.len() / 8;

        for chunk in 0..chunks_8 {
            let base = chunk * 8;

            let row1 = _mm256_loadu_ps(img1_plane.as_ptr().add(base));
            let rowm1 = _mm256_loadu_ps(mu1_plane.as_ptr().add(base));
            let row2 = _mm256_loadu_ps(img2_plane.as_ptr().add(base));
            let rowm2 = _mm256_loadu_ps(mu2_plane.as_ptr().add(base));

            // edge1 = |row1 - rowm1|
            let diff1 = _mm256_sub_ps(row1, rowm1);
            let edge1 = _mm256_andnot_ps(sign_mask, diff1); // abs

            // edge2 = |row2 - rowm2|
            let diff2 = _mm256_sub_ps(row2, rowm2);
            let edge2 = _mm256_andnot_ps(sign_mask, diff2); // abs

            // d1 = (1 + edge2) / (1 + edge1) - 1
            let num = _mm256_add_ps(one_vec, edge2);
            let denom = _mm256_add_ps(one_vec, edge1);
            let ratio = _mm256_div_ps(num, denom);
            let d1 = _mm256_sub_ps(ratio, one_vec);

            // artifact = max(d1, 0)
            let artifact = _mm256_max_ps(d1, zero_vec);

            // detail_lost = max(-d1, 0)
            let neg_d1 = _mm256_sub_ps(zero_vec, d1);
            let detail_lost = _mm256_max_ps(neg_d1, zero_vec);

            // Compute 4th powers
            let artifact2 = _mm256_mul_ps(artifact, artifact);
            let artifact4 = _mm256_mul_ps(artifact2, artifact2);
            let detail_lost2 = _mm256_mul_ps(detail_lost, detail_lost);
            let detail_lost4 = _mm256_mul_ps(detail_lost2, detail_lost2);

            // Efficient horizontal sum
            sum_artifact += hsum_ps_avx(artifact) as f64;
            sum_artifact4 += hsum_ps_avx(artifact4) as f64;
            sum_detail_lost += hsum_ps_avx(detail_lost) as f64;
            sum_detail_lost4 += hsum_ps_avx(detail_lost4) as f64;
        }

        // Handle remainder with scalar
        let remaining_start = chunks_8 * 8;
        for x in remaining_start..img1_plane.len() {
            let d1: f64 = (1.0 + f64::from((img2_plane[x] - mu2_plane[x]).abs()))
                / (1.0 + f64::from((img1_plane[x] - mu1_plane[x]).abs()))
                - 1.0;

            let artifact = d1.max(0.0);
            sum_artifact += artifact;
            sum_artifact4 += artifact.powi(4);

            let detail_lost = (-d1).max(0.0);
            sum_detail_lost += detail_lost;
            sum_detail_lost4 += detail_lost.powi(4);
        }

        plane_averages[c * 4] = one_per_pixels * sum_artifact;
        plane_averages[c * 4 + 1] = (one_per_pixels * sum_artifact4).sqrt().sqrt();
        plane_averages[c * 4 + 2] = one_per_pixels * sum_detail_lost;
        plane_averages[c * 4 + 3] = (one_per_pixels * sum_detail_lost4).sqrt().sqrt();
    }

    plane_averages
}

fn edge_diff_map_scalar(
    width: usize,
    height: usize,
    img1: &[Vec<f32>; 3],
    mu1: &[Vec<f32>; 3],
    img2: &[Vec<f32>; 3],
    mu2: &[Vec<f32>; 3],
) -> [f64; 3 * 4] {
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
