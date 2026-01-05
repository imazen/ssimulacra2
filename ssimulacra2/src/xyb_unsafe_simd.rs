//! SIMD XYB conversion using x86 intrinsics
//!
//! This module provides fast XYB conversion using:
//! - `safe_unaligned_simd` for safe memory load/store operations
//! - Safe SIMD arithmetic (Rust 1.87+)

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
use safe_unaligned_simd::x86_64 as safe_simd;

// XYB color space constants
const K_M02: f32 = 0.078f32;
const K_M00: f32 = 0.30f32;
const K_M01: f32 = 1.0f32 - K_M02 - K_M00;
const K_M12: f32 = 0.078f32;
const K_M10: f32 = 0.23f32;
const K_M11: f32 = 1.0f32 - K_M12 - K_M10;
const K_M20: f32 = 0.243_422_69_f32;
const K_M21: f32 = 0.204_767_45_f32;
const K_M22: f32 = 1.0f32 - K_M20 - K_M21;
const K_B0: f32 = 0.003_793_073_4_f32;

/// Fast scalar cube root using bit manipulation + Newton-Raphson
#[inline(always)]
fn cbrtf_fast(x: f32) -> f32 {
    const B1: u32 = 709_958_130;
    let mut ui: u32 = x.to_bits();
    let mut hx: u32 = ui & 0x7FFF_FFFF;
    hx = hx / 3 + B1;
    ui &= 0x8000_0000;
    ui |= hx;
    let mut t: f64 = f64::from(f32::from_bits(ui));
    let xf64 = f64::from(x);
    // Two Newton-Raphson iterations
    let mut r = t * t * t;
    t = t * (xf64 + xf64 + r) / (xf64 + r + r);
    r = t * t * t;
    t = t * (xf64 + xf64 + r) / (xf64 + r + r);
    t as f32
}

/// Converts linear RGB to XYB using unsafe SIMD intrinsics
pub fn linear_rgb_to_xyb_unsafe(input: &mut [[f32; 3]]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                linear_rgb_to_xyb_avx2(input);
            }
            return;
        }
    }
    // Fallback to scalar
    linear_rgb_to_xyb_scalar(input);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn linear_rgb_to_xyb_avx2(input: &mut [[f32; 3]]) {
    let absorbance_bias = -cbrtf_fast(K_B0);

    // Process 8 pixels at a time with AVX2
    let chunks_8 = input.len() / 8;

    // Preload matrix constants
    let m00 = _mm256_set1_ps(K_M00);
    let m01 = _mm256_set1_ps(K_M01);
    let m02 = _mm256_set1_ps(K_M02);
    let m10 = _mm256_set1_ps(K_M10);
    let m11 = _mm256_set1_ps(K_M11);
    let m12 = _mm256_set1_ps(K_M12);
    let m20 = _mm256_set1_ps(K_M20);
    let m21 = _mm256_set1_ps(K_M21);
    let m22 = _mm256_set1_ps(K_M22);
    let bias = _mm256_set1_ps(K_B0);
    let absorb_bias = _mm256_set1_ps(absorbance_bias);
    let zero = _mm256_setzero_ps();
    let half = _mm256_set1_ps(0.5);

    for chunk_idx in 0..chunks_8 {
        let base = chunk_idx * 8;

        // Load 8 pixels and transpose to SoA (gather R, G, B separately)
        let mut r_arr = [0.0f32; 8];
        let mut g_arr = [0.0f32; 8];
        let mut b_arr = [0.0f32; 8];

        for i in 0..8 {
            let p = input[base + i];
            r_arr[i] = p[0];
            g_arr[i] = p[1];
            b_arr[i] = p[2];
        }

        // Safe loads via safe_unaligned_simd (array refs, not raw pointers)
        let r = safe_simd::_mm256_loadu_ps(&r_arr);
        let g = safe_simd::_mm256_loadu_ps(&g_arr);
        let b = safe_simd::_mm256_loadu_ps(&b_arr);

        // Matrix multiply with FMA: mixed = M * rgb + bias
        let mixed0 = _mm256_fmadd_ps(
            m00,
            r,
            _mm256_fmadd_ps(m01, g, _mm256_fmadd_ps(m02, b, bias)),
        );
        let mixed1 = _mm256_fmadd_ps(
            m10,
            r,
            _mm256_fmadd_ps(m11, g, _mm256_fmadd_ps(m12, b, bias)),
        );
        let mixed2 = _mm256_fmadd_ps(
            m20,
            r,
            _mm256_fmadd_ps(m21, g, _mm256_fmadd_ps(m22, b, bias)),
        );

        // Clamp to zero
        let mixed0 = _mm256_max_ps(mixed0, zero);
        let mixed1 = _mm256_max_ps(mixed1, zero);
        let mixed2 = _mm256_max_ps(mixed2, zero);

        // Extract, compute cbrt, and reload (cbrt is hard to vectorize efficiently)
        let mut m0_arr = [0.0f32; 8];
        let mut m1_arr = [0.0f32; 8];
        let mut m2_arr = [0.0f32; 8];
        // Safe stores via safe_unaligned_simd
        safe_simd::_mm256_storeu_ps(&mut m0_arr, mixed0);
        safe_simd::_mm256_storeu_ps(&mut m1_arr, mixed1);
        safe_simd::_mm256_storeu_ps(&mut m2_arr, mixed2);

        for i in 0..8 {
            m0_arr[i] = cbrtf_fast(m0_arr[i]);
            m1_arr[i] = cbrtf_fast(m1_arr[i]);
            m2_arr[i] = cbrtf_fast(m2_arr[i]);
        }

        // Safe loads
        let mixed0 = _mm256_add_ps(safe_simd::_mm256_loadu_ps(&m0_arr), absorb_bias);
        let mixed1 = _mm256_add_ps(safe_simd::_mm256_loadu_ps(&m1_arr), absorb_bias);
        let mixed2 = _mm256_add_ps(safe_simd::_mm256_loadu_ps(&m2_arr), absorb_bias);

        // Convert to XYB
        let x = _mm256_mul_ps(half, _mm256_sub_ps(mixed0, mixed1));
        let y = _mm256_mul_ps(half, _mm256_add_ps(mixed0, mixed1));
        let b_out = mixed2;

        // Safe stores
        let mut x_arr = [0.0f32; 8];
        let mut y_arr = [0.0f32; 8];
        let mut b_arr = [0.0f32; 8];
        safe_simd::_mm256_storeu_ps(&mut x_arr, x);
        safe_simd::_mm256_storeu_ps(&mut y_arr, y);
        safe_simd::_mm256_storeu_ps(&mut b_arr, b_out);

        for i in 0..8 {
            input[base + i] = [x_arr[i], y_arr[i], b_arr[i]];
        }
    }

    // Handle remaining pixels with scalar
    let remaining_start = chunks_8 * 8;
    linear_rgb_to_xyb_scalar(&mut input[remaining_start..]);
}

fn linear_rgb_to_xyb_scalar(input: &mut [[f32; 3]]) {
    let absorbance_bias = -cbrtf_fast(K_B0);

    for pix in input.iter_mut() {
        let r = pix[0];
        let g = pix[1];
        let b = pix[2];

        let mut mixed0 = K_M00.mul_add(r, K_M01.mul_add(g, K_M02 * b)) + K_B0;
        let mut mixed1 = K_M10.mul_add(r, K_M11.mul_add(g, K_M12 * b)) + K_B0;
        let mut mixed2 = K_M20.mul_add(r, K_M21.mul_add(g, K_M22 * b)) + K_B0;

        mixed0 = mixed0.max(0.0);
        mixed1 = mixed1.max(0.0);
        mixed2 = mixed2.max(0.0);

        mixed0 = cbrtf_fast(mixed0) + absorbance_bias;
        mixed1 = cbrtf_fast(mixed1) + absorbance_bias;
        mixed2 = cbrtf_fast(mixed2) + absorbance_bias;

        pix[0] = 0.5 * (mixed0 - mixed1);
        pix[1] = 0.5 * (mixed0 + mixed1);
        pix[2] = mixed2;
    }
}
