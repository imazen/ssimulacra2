//! SIMD-optimized RGB ↔ XYB conversions extracted from yuvxyb.
//!
//! This module contains only the SIMD variants of color space conversion
//! functions needed for SSIMULACRA2, extracted from the yuvxyb crate to avoid
//! the full dependency while getting the performance benefits.
//!
//! Original code from: https://github.com/rust-av/yuvxyb
//! License: BSD-2-Clause

use wide::{f32x16, f32x8, f64x2};

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

// SIMD cube root implementation - initial approximation via bit manipulation
#[inline]
fn initial_approx(x: f32) -> f32 {
    // B1 = (127-127.0/3-0.03306235651)*2**23
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

/// SIMD cube root for 16 f32 values (AVX-512 optimal)
#[inline]
fn cbrtf_x16(x: f32x16) -> f32x16 {
    let x_arr: [f32; 16] = x.into();

    // Get initial approximations for all 16 elements
    let t_arr: [f32; 16] = [
        initial_approx(x_arr[0]),
        initial_approx(x_arr[1]),
        initial_approx(x_arr[2]),
        initial_approx(x_arr[3]),
        initial_approx(x_arr[4]),
        initial_approx(x_arr[5]),
        initial_approx(x_arr[6]),
        initial_approx(x_arr[7]),
        initial_approx(x_arr[8]),
        initial_approx(x_arr[9]),
        initial_approx(x_arr[10]),
        initial_approx(x_arr[11]),
        initial_approx(x_arr[12]),
        initial_approx(x_arr[13]),
        initial_approx(x_arr[14]),
        initial_approx(x_arr[15]),
    ];

    // Process in eight f64x2 chunks for f64 precision
    let x0 = f64x2::new([x_arr[0] as f64, x_arr[1] as f64]);
    let x1 = f64x2::new([x_arr[2] as f64, x_arr[3] as f64]);
    let x2 = f64x2::new([x_arr[4] as f64, x_arr[5] as f64]);
    let x3 = f64x2::new([x_arr[6] as f64, x_arr[7] as f64]);
    let x4 = f64x2::new([x_arr[8] as f64, x_arr[9] as f64]);
    let x5 = f64x2::new([x_arr[10] as f64, x_arr[11] as f64]);
    let x6 = f64x2::new([x_arr[12] as f64, x_arr[13] as f64]);
    let x7 = f64x2::new([x_arr[14] as f64, x_arr[15] as f64]);

    let mut t0 = f64x2::new([t_arr[0] as f64, t_arr[1] as f64]);
    let mut t1 = f64x2::new([t_arr[2] as f64, t_arr[3] as f64]);
    let mut t2 = f64x2::new([t_arr[4] as f64, t_arr[5] as f64]);
    let mut t3 = f64x2::new([t_arr[6] as f64, t_arr[7] as f64]);
    let mut t4 = f64x2::new([t_arr[8] as f64, t_arr[9] as f64]);
    let mut t5 = f64x2::new([t_arr[10] as f64, t_arr[11] as f64]);
    let mut t6 = f64x2::new([t_arr[12] as f64, t_arr[13] as f64]);
    let mut t7 = f64x2::new([t_arr[14] as f64, t_arr[15] as f64]);

    let x2_0 = x0 + x0;
    let x2_1 = x1 + x1;
    let x2_2 = x2 + x2;
    let x2_3 = x3 + x3;
    let x2_4 = x4 + x4;
    let x2_5 = x5 + x5;
    let x2_6 = x6 + x6;
    let x2_7 = x7 + x7;

    // First Newton iteration
    let r0 = t0 * t0 * t0;
    let r1 = t1 * t1 * t1;
    let r2 = t2 * t2 * t2;
    let r3 = t3 * t3 * t3;
    let r4 = t4 * t4 * t4;
    let r5 = t5 * t5 * t5;
    let r6 = t6 * t6 * t6;
    let r7 = t7 * t7 * t7;
    t0 = t0 * (x2_0 + r0) / (x0 + r0 + r0);
    t1 = t1 * (x2_1 + r1) / (x1 + r1 + r1);
    t2 = t2 * (x2_2 + r2) / (x2 + r2 + r2);
    t3 = t3 * (x2_3 + r3) / (x3 + r3 + r3);
    t4 = t4 * (x2_4 + r4) / (x4 + r4 + r4);
    t5 = t5 * (x2_5 + r5) / (x5 + r5 + r5);
    t6 = t6 * (x2_6 + r6) / (x6 + r6 + r6);
    t7 = t7 * (x2_7 + r7) / (x7 + r7 + r7);

    // Second Newton iteration
    let r0 = t0 * t0 * t0;
    let r1 = t1 * t1 * t1;
    let r2 = t2 * t2 * t2;
    let r3 = t3 * t3 * t3;
    let r4 = t4 * t4 * t4;
    let r5 = t5 * t5 * t5;
    let r6 = t6 * t6 * t6;
    let r7 = t7 * t7 * t7;
    t0 = t0 * (x2_0 + r0) / (x0 + r0 + r0);
    t1 = t1 * (x2_1 + r1) / (x1 + r1 + r1);
    t2 = t2 * (x2_2 + r2) / (x2 + r2 + r2);
    t3 = t3 * (x2_3 + r3) / (x3 + r3 + r3);
    t4 = t4 * (x2_4 + r4) / (x4 + r4 + r4);
    t5 = t5 * (x2_5 + r5) / (x5 + r5 + r5);
    t6 = t6 * (x2_6 + r6) / (x6 + r6 + r6);
    t7 = t7 * (x2_7 + r7) / (x7 + r7 + r7);

    // Convert back to f32
    let t0_arr: [f64; 2] = t0.into();
    let t1_arr: [f64; 2] = t1.into();
    let t2_arr: [f64; 2] = t2.into();
    let t3_arr: [f64; 2] = t3.into();
    let t4_arr: [f64; 2] = t4.into();
    let t5_arr: [f64; 2] = t5.into();
    let t6_arr: [f64; 2] = t6.into();
    let t7_arr: [f64; 2] = t7.into();
    f32x16::new([
        t0_arr[0] as f32,
        t0_arr[1] as f32,
        t1_arr[0] as f32,
        t1_arr[1] as f32,
        t2_arr[0] as f32,
        t2_arr[1] as f32,
        t3_arr[0] as f32,
        t3_arr[1] as f32,
        t4_arr[0] as f32,
        t4_arr[1] as f32,
        t5_arr[0] as f32,
        t5_arr[1] as f32,
        t6_arr[0] as f32,
        t6_arr[1] as f32,
        t7_arr[0] as f32,
        t7_arr[1] as f32,
    ])
}

/// Fast scalar cbrt matching the SIMD algorithm (FreeBSD/Newton-Raphson)
#[inline]
fn cbrtf_fast(x: f32) -> f32 {
    const B1: u32 = 709_958_130;
    let mut ui: u32 = x.to_bits();
    let mut hx: u32 = ui & 0x7FFF_FFFF;
    hx = hx / 3 + B1;
    ui &= 0x8000_0000;
    ui |= hx;
    let mut t: f64 = f64::from(f32::from_bits(ui));
    let xf64 = f64::from(x);
    let mut r = t * t * t;
    t = t * (xf64 + xf64 + r) / (xf64 + r + r);
    r = t * t * t;
    t = t * (xf64 + xf64 + r) / (xf64 + r + r);
    t as f32
}

/// Converts linear RGB to XYB using f32x16 SIMD, in place.
///
/// This processes the input in batches of 16 pixels for maximum performance,
/// falling back to f32x8 then scalar processing for remainders.
///
/// Input/output: [[R, G, B]] → [[X, Y, B]]
#[inline]
pub fn linear_rgb_to_xyb_simd(input: &mut [[f32; 3]]) {
    // Precompute the absorbance bias (negated cube root) - use cbrtf_fast to match SIMD
    let absorbance_bias: [f32; 3] = [
        -cbrtf_fast(OPSIN_ABSORBANCE_BIAS[0]),
        -cbrtf_fast(OPSIN_ABSORBANCE_BIAS[1]),
        -cbrtf_fast(OPSIN_ABSORBANCE_BIAS[2]),
    ];

    // Process 16 pixels at a time
    let chunks_16 = input.len() / 16;

    for chunk_idx in 0..chunks_16 {
        let base = chunk_idx * 16;

        // Load 16 pixels and transpose to SoA
        let mut r_arr = [0.0f32; 16];
        let mut g_arr = [0.0f32; 16];
        let mut b_arr = [0.0f32; 16];

        for i in 0..16 {
            let p = input[base + i];
            r_arr[i] = p[0];
            g_arr[i] = p[1];
            b_arr[i] = p[2];
        }

        let r = f32x16::new(r_arr);
        let g = f32x16::new(g_arr);
        let b = f32x16::new(b_arr);

        // Matrix multiply: mixed = M * rgb + bias
        let m00 = f32x16::splat(OPSIN_ABSORBANCE_MATRIX[0]);
        let m01 = f32x16::splat(OPSIN_ABSORBANCE_MATRIX[1]);
        let m02 = f32x16::splat(OPSIN_ABSORBANCE_MATRIX[2]);
        let m10 = f32x16::splat(OPSIN_ABSORBANCE_MATRIX[3]);
        let m11 = f32x16::splat(OPSIN_ABSORBANCE_MATRIX[4]);
        let m12 = f32x16::splat(OPSIN_ABSORBANCE_MATRIX[5]);
        let m20 = f32x16::splat(OPSIN_ABSORBANCE_MATRIX[6]);
        let m21 = f32x16::splat(OPSIN_ABSORBANCE_MATRIX[7]);
        let m22 = f32x16::splat(OPSIN_ABSORBANCE_MATRIX[8]);

        let bias0 = f32x16::splat(OPSIN_ABSORBANCE_BIAS[0]);
        let bias1 = f32x16::splat(OPSIN_ABSORBANCE_BIAS[1]);
        let bias2 = f32x16::splat(OPSIN_ABSORBANCE_BIAS[2]);

        // Use mul_add chain (matches scalar FMA precision)
        let mut mixed0 = m00.mul_add(r, m01.mul_add(g, m02.mul_add(b, bias0)));
        let mut mixed1 = m10.mul_add(r, m11.mul_add(g, m12.mul_add(b, bias1)));
        let mut mixed2 = m20.mul_add(r, m21.mul_add(g, m22.mul_add(b, bias2)));

        // Clamp negative values to zero
        let zero = f32x16::splat(0.0);
        mixed0 = mixed0.max(zero);
        mixed1 = mixed1.max(zero);
        mixed2 = mixed2.max(zero);

        // Apply cube root + bias offset
        let absorb0 = f32x16::splat(absorbance_bias[0]);
        let absorb1 = f32x16::splat(absorbance_bias[1]);
        let absorb2 = f32x16::splat(absorbance_bias[2]);

        mixed0 = cbrtf_x16(mixed0) + absorb0;
        mixed1 = cbrtf_x16(mixed1) + absorb1;
        mixed2 = cbrtf_x16(mixed2) + absorb2;

        // Convert mixed to XYB
        let half = f32x16::splat(0.5);
        let x = half * (mixed0 - mixed1);
        let y = half * (mixed0 + mixed1);
        let b_out = mixed2;

        // Transpose back to AoS and store
        let x_arr: [f32; 16] = x.into();
        let y_arr: [f32; 16] = y.into();
        let b_arr: [f32; 16] = b_out.into();

        for i in 0..16 {
            input[base + i] = [x_arr[i], y_arr[i], b_arr[i]];
        }
    }

    // Process remaining pixels with f32x8
    let remaining_start = chunks_16 * 16;
    let remaining = &mut input[remaining_start..];
    let chunks_8 = remaining.len() / 8;

    for chunk_idx in 0..chunks_8 {
        let base = chunk_idx * 8;

        // Load 8 pixels and transpose to SoA
        let mut r_arr = [0.0f32; 8];
        let mut g_arr = [0.0f32; 8];
        let mut b_arr = [0.0f32; 8];

        for i in 0..8 {
            let p = remaining[base + i];
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

        // Use mul_add chain (matches scalar FMA precision)
        let mut mixed0 = m00.mul_add(r, m01.mul_add(g, m02.mul_add(b, bias0)));
        let mut mixed1 = m10.mul_add(r, m11.mul_add(g, m12.mul_add(b, bias1)));
        let mut mixed2 = m20.mul_add(r, m21.mul_add(g, m22.mul_add(b, bias2)));

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
            remaining[base + i] = [x_arr[i], y_arr[i], b_arr[i]];
        }
    }

    // Process remaining pixels with scalar code (using cbrtf_fast to match SIMD)
    let scalar_start = chunks_8 * 8;
    for pix in &mut remaining[scalar_start..] {
        let mut mixed = opsin_absorbance_scalar(pix);
        for (m, absorb) in mixed.iter_mut().zip(absorbance_bias.iter()) {
            if *m < 0.0 {
                *m = 0.0;
            }
            *m = cbrtf_fast(*m) + *absorb;
        }
        *pix = mixed_to_xyb_scalar(&mixed);
    }
}

// Scalar helper functions for remainder processing
#[inline]
fn opsin_absorbance_scalar(rgb: &[f32; 3]) -> [f32; 3] {
    // Use mul_add chain to match the SIMD implementation
    [
        OPSIN_ABSORBANCE_MATRIX[0].mul_add(
            rgb[0],
            OPSIN_ABSORBANCE_MATRIX[1].mul_add(
                rgb[1],
                OPSIN_ABSORBANCE_MATRIX[2].mul_add(rgb[2], OPSIN_ABSORBANCE_BIAS[0]),
            ),
        ),
        OPSIN_ABSORBANCE_MATRIX[3].mul_add(
            rgb[0],
            OPSIN_ABSORBANCE_MATRIX[4].mul_add(
                rgb[1],
                OPSIN_ABSORBANCE_MATRIX[5].mul_add(rgb[2], OPSIN_ABSORBANCE_BIAS[1]),
            ),
        ),
        OPSIN_ABSORBANCE_MATRIX[6].mul_add(
            rgb[0],
            OPSIN_ABSORBANCE_MATRIX[7].mul_add(
                rgb[1],
                OPSIN_ABSORBANCE_MATRIX[8].mul_add(rgb[2], OPSIN_ABSORBANCE_BIAS[2]),
            ),
        ),
    ]
}

#[inline]
fn mixed_to_xyb_scalar(mixed: &[f32; 3]) -> [f32; 3] {
    [
        0.5 * (mixed[0] - mixed[1]),
        0.5 * (mixed[0] + mixed[1]),
        mixed[2],
    ]
}
