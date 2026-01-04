//! SIMD-optimized RGB ↔ XYB conversions extracted from yuvxyb.
//!
//! This module contains only the SIMD variants of color space conversion
//! functions needed for SSIMULACRA2, extracted from the yuvxyb crate to avoid
//! the full dependency while getting the performance benefits.
//!
//! Original code from: https://github.com/rust-av/yuvxyb
//! License: BSD-2-Clause

use wide::{f32x8, f64x2};

// XYB color space constants from jpegli
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
const K_B1: f32 = K_B0;
const K_B2: f32 = K_B0;

const OPSIN_ABSORBANCE_MATRIX: [f32; 9] = [
    K_M00, K_M01, K_M02, K_M10, K_M11, K_M12, K_M20, K_M21, K_M22,
];

const OPSIN_ABSORBANCE_BIAS: [f32; 3] = [K_B0, K_B1, K_B2];

const INVERSE_OPSIN_ABSORBANCE_MATRIX: [f32; 9] = [
    11.031_566_9,
    -9.866_943_8,
    -0.164_623_05,
    -3.254_147_4,
    4.418_770_3,
    -0.164_623_05,
    -3.658_851_4,
    2.712_923,
    1.945_928_3,
];

const NEG_OPSIN_ABSORBANCE_BIAS: [f32; 3] = [-K_B0, -K_B1, -K_B2];

// SIMD cube root implementation
#[inline]
fn initial_approx(x: f32) -> f32 {
    const B1: u32 = 709_958_130;
    let ui: u32 = x.to_bits();
    let sign = ui & 0x8000_0000;
    let hx = ui & 0x7FFF_FFFF;
    let approx = hx / 3 + B1;
    f32::from_bits(sign | approx)
}

/// SIMD cube root for 8 f32 values (AVX2 optimal)
#[inline]
fn cbrtf_x8(x: f32x8) -> f32x8 {
    let x_arr: [f32; 8] = x.into();

    let t_arr: [f32; 8] = [
        initial_approx(x_arr[0]),
        initial_approx(x_arr[1]),
        initial_approx(x_arr[2]),
        initial_approx(x_arr[3]),
        initial_approx(x_arr[4]),
        initial_approx(x_arr[5]),
        initial_approx(x_arr[6]),
        initial_approx(x_arr[7]),
    ];

    // Process in four f64x2 chunks for precision
    let x0 = f64x2::new([x_arr[0] as f64, x_arr[1] as f64]);
    let x1 = f64x2::new([x_arr[2] as f64, x_arr[3] as f64]);
    let x2 = f64x2::new([x_arr[4] as f64, x_arr[5] as f64]);
    let x3 = f64x2::new([x_arr[6] as f64, x_arr[7] as f64]);

    let mut t0 = f64x2::new([t_arr[0] as f64, t_arr[1] as f64]);
    let mut t1 = f64x2::new([t_arr[2] as f64, t_arr[3] as f64]);
    let mut t2 = f64x2::new([t_arr[4] as f64, t_arr[5] as f64]);
    let mut t3 = f64x2::new([t_arr[6] as f64, t_arr[7] as f64]);

    let x2_0 = x0 + x0;
    let x2_1 = x1 + x1;
    let x2_2 = x2 + x2;
    let x2_3 = x3 + x3;

    // First Newton iteration
    let r0 = t0 * t0 * t0;
    let r1 = t1 * t1 * t1;
    let r2 = t2 * t2 * t2;
    let r3 = t3 * t3 * t3;
    t0 = t0 * (x2_0 + r0) / (x0 + r0 + r0);
    t1 = t1 * (x2_1 + r1) / (x1 + r1 + r1);
    t2 = t2 * (x2_2 + r2) / (x2 + r2 + r2);
    t3 = t3 * (x2_3 + r3) / (x3 + r3 + r3);

    // Second Newton iteration
    let r0 = t0 * t0 * t0;
    let r1 = t1 * t1 * t1;
    let r2 = t2 * t2 * t2;
    let r3 = t3 * t3 * t3;
    t0 = t0 * (x2_0 + r0) / (x0 + r0 + r0);
    t1 = t1 * (x2_1 + r1) / (x1 + r1 + r1);
    t2 = t2 * (x2_2 + r2) / (x2 + r2 + r2);
    t3 = t3 * (x2_3 + r3) / (x3 + r3 + r3);

    // Convert back to f32
    let t0_arr: [f64; 2] = t0.into();
    let t1_arr: [f64; 2] = t1.into();
    let t2_arr: [f64; 2] = t2.into();
    let t3_arr: [f64; 2] = t3.into();
    f32x8::new([
        t0_arr[0] as f32,
        t0_arr[1] as f32,
        t1_arr[0] as f32,
        t1_arr[1] as f32,
        t2_arr[0] as f32,
        t2_arr[1] as f32,
        t3_arr[0] as f32,
        t3_arr[1] as f32,
    ])
}

/// Converts linear RGB to XYB using f32x8 SIMD, in place.
///
/// Processes 8 pixels at a time - optimal for AVX2 hardware.
/// Input/output: [[R, G, B]] → [[X, Y, B]]
#[inline]
pub fn linear_rgb_to_xyb_simd(input: &mut [[f32; 3]]) {
    let absorbance_bias: [f32; 3] = [
        -OPSIN_ABSORBANCE_BIAS[0].cbrt(),
        -OPSIN_ABSORBANCE_BIAS[1].cbrt(),
        -OPSIN_ABSORBANCE_BIAS[2].cbrt(),
    ];

    let chunks_8 = input.len() / 8;

    for chunk_idx in 0..chunks_8 {
        let base = chunk_idx * 8;

        // Load 8 pixels and transpose to SoA
        let mut r_arr = [0.0f32; 8];
        let mut g_arr = [0.0f32; 8];
        let mut b_arr = [0.0f32; 8];

        for i in 0..8 {
            let p = input[base + i];
            r_arr[i] = p[0];
            g_arr[i] = p[1];
            b_arr[i] = p[2];
        }

        let r = f32x8::new(r_arr);
        let g = f32x8::new(g_arr);
        let b = f32x8::new(b_arr);

        // Matrix multiply: mixed = M * rgb + bias
        let m00 = f32x8::splat(OPSIN_ABSORBANCE_MATRIX[0]);
        let m01 = f32x8::splat(OPSIN_ABSORBANCE_MATRIX[1]);
        let m02 = f32x8::splat(OPSIN_ABSORBANCE_MATRIX[2]);
        let m10 = f32x8::splat(OPSIN_ABSORBANCE_MATRIX[3]);
        let m11 = f32x8::splat(OPSIN_ABSORBANCE_MATRIX[4]);
        let m12 = f32x8::splat(OPSIN_ABSORBANCE_MATRIX[5]);
        let m20 = f32x8::splat(OPSIN_ABSORBANCE_MATRIX[6]);
        let m21 = f32x8::splat(OPSIN_ABSORBANCE_MATRIX[7]);
        let m22 = f32x8::splat(OPSIN_ABSORBANCE_MATRIX[8]);

        let bias0 = f32x8::splat(OPSIN_ABSORBANCE_BIAS[0]);
        let bias1 = f32x8::splat(OPSIN_ABSORBANCE_BIAS[1]);
        let bias2 = f32x8::splat(OPSIN_ABSORBANCE_BIAS[2]);

        let mut mixed0 = m00 * r + m01 * g + m02 * b + bias0;
        let mut mixed1 = m10 * r + m11 * g + m12 * b + bias1;
        let mut mixed2 = m20 * r + m21 * g + m22 * b + bias2;

        // Clamp negative values to zero
        let zero = f32x8::splat(0.0);
        mixed0 = mixed0.max(zero);
        mixed1 = mixed1.max(zero);
        mixed2 = mixed2.max(zero);

        // Apply cube root + bias offset
        let absorb0 = f32x8::splat(absorbance_bias[0]);
        let absorb1 = f32x8::splat(absorbance_bias[1]);
        let absorb2 = f32x8::splat(absorbance_bias[2]);

        mixed0 = cbrtf_x8(mixed0) + absorb0;
        mixed1 = cbrtf_x8(mixed1) + absorb1;
        mixed2 = cbrtf_x8(mixed2) + absorb2;

        // Convert mixed to XYB
        let half = f32x8::splat(0.5);
        let x = half * (mixed0 - mixed1);
        let y = half * (mixed0 + mixed1);
        let b_out = mixed2;

        // Transpose back to AoS and store
        let x_arr: [f32; 8] = x.into();
        let y_arr: [f32; 8] = y.into();
        let b_arr: [f32; 8] = b_out.into();

        for i in 0..8 {
            input[base + i] = [x_arr[i], y_arr[i], b_arr[i]];
        }
    }

    // Process remaining pixels with scalar code
    let scalar_start = chunks_8 * 8;
    for pix in &mut input[scalar_start..] {
        let r = pix[0];
        let g = pix[1];
        let b = pix[2];

        let mut mixed0 = OPSIN_ABSORBANCE_MATRIX[0] * r
            + OPSIN_ABSORBANCE_MATRIX[1] * g
            + OPSIN_ABSORBANCE_MATRIX[2] * b
            + OPSIN_ABSORBANCE_BIAS[0];
        let mut mixed1 = OPSIN_ABSORBANCE_MATRIX[3] * r
            + OPSIN_ABSORBANCE_MATRIX[4] * g
            + OPSIN_ABSORBANCE_MATRIX[5] * b
            + OPSIN_ABSORBANCE_BIAS[1];
        let mut mixed2 = OPSIN_ABSORBANCE_MATRIX[6] * r
            + OPSIN_ABSORBANCE_MATRIX[7] * g
            + OPSIN_ABSORBANCE_MATRIX[8] * b
            + OPSIN_ABSORBANCE_BIAS[2];

        mixed0 = mixed0.max(0.0).cbrt() + absorbance_bias[0];
        mixed1 = mixed1.max(0.0).cbrt() + absorbance_bias[1];
        mixed2 = mixed2.max(0.0).cbrt() + absorbance_bias[2];

        pix[0] = 0.5 * (mixed0 - mixed1);
        pix[1] = 0.5 * (mixed0 + mixed1);
        pix[2] = mixed2;
    }
}
