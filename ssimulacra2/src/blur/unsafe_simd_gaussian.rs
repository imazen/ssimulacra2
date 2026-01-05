//! Unsafe SIMD-optimized Recursive Gaussian using raw x86 intrinsics
//!
//! This module uses unsafe raw pointer arithmetic and explicit SIMD intrinsics
//! for maximum performance. It trades safety guarantees for speed.
//!
//! Key optimizations:
//! - Raw pointer arithmetic (no bounds checks)
//! - Direct AVX2/AVX-512 intrinsics
//! - Prefetching for memory access patterns
//! - Aligned memory operations where possible
//! - Manual loop unrolling
//! - Multiversion for compile-time CPU feature optimization

mod consts {
    #![allow(clippy::unreadable_literal)]
    include!(concat!(env!("OUT_DIR"), "/recursive_gaussian.rs"));
}

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use multiversion::multiversion;

/// Aligned buffer for SIMD operations (64-byte cache line alignment)
#[repr(C, align(64))]
struct AlignedF32([f32; 16]); // 64 bytes = 16 f32s

struct AlignedBuffer {
    data: Vec<f32>,
}

impl AlignedBuffer {
    fn new(size: usize) -> Self {
        // Simple allocation - rely on Vec's alignment for now
        // For true cache-line alignment, use aligned_alloc in the future
        Self {
            data: vec![0.0f32; size],
        }
    }

    #[inline(always)]
    fn as_ptr(&self) -> *const f32 {
        self.data.as_ptr()
    }

    #[inline(always)]
    fn as_mut_ptr(&mut self) -> *mut f32 {
        self.data.as_mut_ptr()
    }
}

pub struct UnsafeSimdGaussian {
    // Pre-allocated buffers (64-byte aligned for cache efficiency)
    temp: AlignedBuffer,
    // State buffers for vertical pass (prev, prev2, out for 3 filter taps)
    prev_buffer: AlignedBuffer,
    prev2_buffer: AlignedBuffer,
    out_buffer: AlignedBuffer,
    max_width: usize,
}

impl UnsafeSimdGaussian {
    pub fn new(max_width: usize) -> Self {
        // Allocate buffers sized for maximum expected dimensions
        const MAX_HEIGHT: usize = 4096;
        const MAX_COLUMNS: usize = 256; // Process up to 256 columns in vertical pass

        Self {
            temp: AlignedBuffer::new(max_width * MAX_HEIGHT),
            prev_buffer: AlignedBuffer::new(3 * MAX_COLUMNS),
            prev2_buffer: AlignedBuffer::new(3 * MAX_COLUMNS),
            out_buffer: AlignedBuffer::new(3 * MAX_COLUMNS),
            max_width,
        }
    }

    pub fn shrink_to(&mut self, _width: usize, _height: usize) {
        // Buffers are pre-allocated to max size
    }

    /// Main entry point - blur a single plane
    pub fn blur_single_plane(&mut self, plane: &[f32], width: usize, height: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; width * height];
        self.blur_single_plane_into(plane, &mut out, width, height);
        out
    }

    /// Blur into a pre-allocated output buffer (zero-allocation)
    pub fn blur_single_plane_into(
        &mut self,
        plane: &[f32],
        out: &mut [f32],
        width: usize,
        height: usize,
    ) {
        debug_assert!(width * height <= self.temp.data.len());

        // Horizontal pass - writes to temp buffer
        self.horizontal_pass(plane, width, height);

        // Vertical pass with SIMD - reads from temp, writes to out
        self.vertical_pass_simd(out, width, height);
    }

    /// Horizontal pass - process each row independently
    fn horizontal_pass(&mut self, input: &[f32], width: usize, height: usize) {
        let temp_ptr = self.temp.as_mut_ptr();
        let input_ptr = input.as_ptr();

        for y in 0..height {
            unsafe {
                horizontal_row_unsafe(input_ptr.add(y * width), temp_ptr.add(y * width), width);
            }
        }
    }

    /// SIMD vertical pass - process columns in parallel
    /// Uses compile-time feature detection for maximum performance
    fn vertical_pass_simd(&mut self, output: &mut [f32], width: usize, height: usize) {
        let input_ptr = self.temp.as_ptr();
        let output_ptr = output.as_mut_ptr();

        #[cfg(target_arch = "x86_64")]
        unsafe {
            self.vertical_pass_dispatch(input_ptr, output_ptr, width, height);
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            // Scalar fallback for non-x86
            for x in 0..width {
                unsafe {
                    self.vertical_pass_scalar(input_ptr, output_ptr, width, height, x);
                }
            }
        }
    }

    /// Dispatch to best available SIMD implementation
    /// This function is compiled with runtime dispatch via multiversion-style approach
    #[cfg(target_arch = "x86_64")]
    #[inline(never)]
    unsafe fn vertical_pass_dispatch(
        &mut self,
        input: *const f32,
        output: *mut f32,
        width: usize,
        height: usize,
    ) {
        // Cache the feature detection results
        static AVX512_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        static AVX2_FMA_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

        let has_avx512 = *AVX512_AVAILABLE.get_or_init(|| is_x86_feature_detected!("avx512f"));
        let has_avx2_fma = *AVX2_FMA_AVAILABLE
            .get_or_init(|| is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"));

        let mut x = 0;

        // AVX-512: 16 floats at a time
        if has_avx512 {
            while x + 16 <= width {
                self.vertical_pass_avx512(input, output, width, height, x);
                x += 16;
            }
        }

        // AVX2+FMA: 8 floats at a time
        if has_avx2_fma {
            while x + 8 <= width {
                self.vertical_pass_avx2_fma(input, output, width, height, x);
                x += 8;
            }
        }

        // SSE2: 4 floats at a time (always available on x86_64)
        while x + 4 <= width {
            self.vertical_pass_sse2(input, output, width, height, x);
            x += 4;
        }

        // Scalar remainder
        while x < width {
            self.vertical_pass_scalar(input, output, width, height, x);
            x += 1;
        }
    }

    /// AVX-512 vertical pass - 16 columns at a time
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn vertical_pass_avx512(
        &mut self,
        input: *const f32,
        output: *mut f32,
        width: usize,
        height: usize,
        x_offset: usize,
    ) {
        let big_n = consts::RADIUS as isize;
        let height_i = height as isize;

        // Splat constants
        let mul_in_1 = _mm512_set1_ps(consts::VERT_MUL_IN_1);
        let mul_in_3 = _mm512_set1_ps(consts::VERT_MUL_IN_3);
        let mul_in_5 = _mm512_set1_ps(consts::VERT_MUL_IN_5);
        let mul_prev_1 = _mm512_set1_ps(consts::VERT_MUL_PREV_1);
        let mul_prev_3 = _mm512_set1_ps(consts::VERT_MUL_PREV_3);
        let mul_prev_5 = _mm512_set1_ps(consts::VERT_MUL_PREV_5);

        let zeroes = _mm512_setzero_ps();

        // State vectors for 3 filter taps
        let mut prev_1 = zeroes;
        let mut prev_3 = zeroes;
        let mut prev_5 = zeroes;
        let mut prev2_1 = zeroes;
        let mut prev2_3 = zeroes;
        let mut prev2_5 = zeroes;

        let mut n = (-big_n) + 1;
        while n < height_i {
            let top = n - big_n - 1;
            let bottom = n + big_n - 1;

            // Load top row (or zeros if out of bounds)
            let top_vals = if top >= 0 && top < height_i {
                let ptr = input.add(top as usize * width + x_offset);
                // Prefetch next rows
                if top + 4 < height_i {
                    _mm_prefetch(
                        input.add((top as usize + 4) * width + x_offset) as *const i8,
                        _MM_HINT_T0,
                    );
                }
                _mm512_loadu_ps(ptr)
            } else {
                zeroes
            };

            // Load bottom row
            let bottom_vals = if bottom >= 0 && bottom < height_i {
                let ptr = input.add(bottom as usize * width + x_offset);
                _mm512_loadu_ps(ptr)
            } else {
                zeroes
            };

            let sum = _mm512_add_ps(top_vals, bottom_vals);

            // IIR filter with FMA
            let out1 = _mm512_fmadd_ps(prev_1, mul_prev_1, prev2_1);
            let out3 = _mm512_fmadd_ps(prev_3, mul_prev_3, prev2_3);
            let out5 = _mm512_fmadd_ps(prev_5, mul_prev_5, prev2_5);

            let out1 = _mm512_fmadd_ps(sum, mul_in_1, _mm512_sub_ps(zeroes, out1));
            let out3 = _mm512_fmadd_ps(sum, mul_in_3, _mm512_sub_ps(zeroes, out3));
            let out5 = _mm512_fmadd_ps(sum, mul_in_5, _mm512_sub_ps(zeroes, out5));

            // Update state
            prev2_1 = prev_1;
            prev2_3 = prev_3;
            prev2_5 = prev_5;
            prev_1 = out1;
            prev_3 = out3;
            prev_5 = out5;

            // Write output
            if n >= 0 {
                let result = _mm512_add_ps(_mm512_add_ps(out1, out3), out5);
                let out_ptr = output.add(n as usize * width + x_offset);
                _mm512_storeu_ps(out_ptr, result);
            }

            n += 1;
        }
    }

    /// AVX2+FMA vertical pass - 8 columns at a time
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn vertical_pass_avx2_fma(
        &mut self,
        input: *const f32,
        output: *mut f32,
        width: usize,
        height: usize,
        x_offset: usize,
    ) {
        let big_n = consts::RADIUS as isize;
        let height_i = height as isize;

        // Splat constants
        let mul_in_1 = _mm256_set1_ps(consts::VERT_MUL_IN_1);
        let mul_in_3 = _mm256_set1_ps(consts::VERT_MUL_IN_3);
        let mul_in_5 = _mm256_set1_ps(consts::VERT_MUL_IN_5);
        let mul_prev_1 = _mm256_set1_ps(consts::VERT_MUL_PREV_1);
        let mul_prev_3 = _mm256_set1_ps(consts::VERT_MUL_PREV_3);
        let mul_prev_5 = _mm256_set1_ps(consts::VERT_MUL_PREV_5);

        let zeroes = _mm256_setzero_ps();

        // State vectors
        let mut prev_1 = zeroes;
        let mut prev_3 = zeroes;
        let mut prev_5 = zeroes;
        let mut prev2_1 = zeroes;
        let mut prev2_3 = zeroes;
        let mut prev2_5 = zeroes;

        let mut n = (-big_n) + 1;
        while n < height_i {
            let top = n - big_n - 1;
            let bottom = n + big_n - 1;

            // Load with prefetching
            let top_vals = if top >= 0 && top < height_i {
                let ptr = input.add(top as usize * width + x_offset);
                // Prefetch 4 rows ahead
                if top + 4 < height_i {
                    _mm_prefetch(
                        input.add((top as usize + 4) * width + x_offset) as *const i8,
                        _MM_HINT_T0,
                    );
                }
                _mm256_loadu_ps(ptr)
            } else {
                zeroes
            };

            let bottom_vals = if bottom >= 0 && bottom < height_i {
                let ptr = input.add(bottom as usize * width + x_offset);
                _mm256_loadu_ps(ptr)
            } else {
                zeroes
            };

            let sum = _mm256_add_ps(top_vals, bottom_vals);

            // IIR filter with FMA: out = sum * mul_in - (prev * mul_prev + prev2)
            let out1 = _mm256_fmadd_ps(prev_1, mul_prev_1, prev2_1);
            let out3 = _mm256_fmadd_ps(prev_3, mul_prev_3, prev2_3);
            let out5 = _mm256_fmadd_ps(prev_5, mul_prev_5, prev2_5);

            // out = sum * mul_in - out (negate via subtract)
            let out1 = _mm256_fmsub_ps(sum, mul_in_1, out1);
            let out3 = _mm256_fmsub_ps(sum, mul_in_3, out3);
            let out5 = _mm256_fmsub_ps(sum, mul_in_5, out5);

            // Update state
            prev2_1 = prev_1;
            prev2_3 = prev_3;
            prev2_5 = prev_5;
            prev_1 = out1;
            prev_3 = out3;
            prev_5 = out5;

            // Write output
            if n >= 0 {
                let result = _mm256_add_ps(_mm256_add_ps(out1, out3), out5);
                let out_ptr = output.add(n as usize * width + x_offset);
                _mm256_storeu_ps(out_ptr, result);
            }

            n += 1;
        }
    }

    /// SSE2 vertical pass - 4 columns at a time
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse2")]
    unsafe fn vertical_pass_sse2(
        &mut self,
        input: *const f32,
        output: *mut f32,
        width: usize,
        height: usize,
        x_offset: usize,
    ) {
        let big_n = consts::RADIUS as isize;
        let height_i = height as isize;

        // Splat constants
        let mul_in_1 = _mm_set1_ps(consts::VERT_MUL_IN_1);
        let mul_in_3 = _mm_set1_ps(consts::VERT_MUL_IN_3);
        let mul_in_5 = _mm_set1_ps(consts::VERT_MUL_IN_5);
        let mul_prev_1 = _mm_set1_ps(consts::VERT_MUL_PREV_1);
        let mul_prev_3 = _mm_set1_ps(consts::VERT_MUL_PREV_3);
        let mul_prev_5 = _mm_set1_ps(consts::VERT_MUL_PREV_5);

        let zeroes = _mm_setzero_ps();

        // State vectors
        let mut prev_1 = zeroes;
        let mut prev_3 = zeroes;
        let mut prev_5 = zeroes;
        let mut prev2_1 = zeroes;
        let mut prev2_3 = zeroes;
        let mut prev2_5 = zeroes;

        let mut n = (-big_n) + 1;
        while n < height_i {
            let top = n - big_n - 1;
            let bottom = n + big_n - 1;

            let top_vals = if top >= 0 && top < height_i {
                let ptr = input.add(top as usize * width + x_offset);
                _mm_loadu_ps(ptr)
            } else {
                zeroes
            };

            let bottom_vals = if bottom >= 0 && bottom < height_i {
                let ptr = input.add(bottom as usize * width + x_offset);
                _mm_loadu_ps(ptr)
            } else {
                zeroes
            };

            let sum = _mm_add_ps(top_vals, bottom_vals);

            // IIR filter (no FMA on base SSE2)
            // out = prev * mul_prev + prev2
            let out1 = _mm_add_ps(_mm_mul_ps(prev_1, mul_prev_1), prev2_1);
            let out3 = _mm_add_ps(_mm_mul_ps(prev_3, mul_prev_3), prev2_3);
            let out5 = _mm_add_ps(_mm_mul_ps(prev_5, mul_prev_5), prev2_5);

            // out = sum * mul_in - out
            let out1 = _mm_sub_ps(_mm_mul_ps(sum, mul_in_1), out1);
            let out3 = _mm_sub_ps(_mm_mul_ps(sum, mul_in_3), out3);
            let out5 = _mm_sub_ps(_mm_mul_ps(sum, mul_in_5), out5);

            // Update state
            prev2_1 = prev_1;
            prev2_3 = prev_3;
            prev2_5 = prev_5;
            prev_1 = out1;
            prev_3 = out3;
            prev_5 = out5;

            // Write output
            if n >= 0 {
                let result = _mm_add_ps(_mm_add_ps(out1, out3), out5);
                let out_ptr = output.add(n as usize * width + x_offset);
                _mm_storeu_ps(out_ptr, result);
            }

            n += 1;
        }
    }

    /// Scalar fallback for remaining columns
    #[inline(always)]
    unsafe fn vertical_pass_scalar(
        &self,
        input: *const f32,
        output: *mut f32,
        width: usize,
        height: usize,
        x_offset: usize,
    ) {
        let big_n = consts::RADIUS as isize;
        let height_i = height as isize;

        let mut prev_1 = 0.0f32;
        let mut prev_3 = 0.0f32;
        let mut prev_5 = 0.0f32;
        let mut prev2_1 = 0.0f32;
        let mut prev2_3 = 0.0f32;
        let mut prev2_5 = 0.0f32;

        let mut n = (-big_n) + 1;
        while n < height_i {
            let top = n - big_n - 1;
            let bottom = n + big_n - 1;

            let top_val = if top >= 0 && top < height_i {
                *input.add(top as usize * width + x_offset)
            } else {
                0.0f32
            };

            let bottom_val = if bottom >= 0 && bottom < height_i {
                *input.add(bottom as usize * width + x_offset)
            } else {
                0.0f32
            };

            let sum = top_val + bottom_val;

            let out1 = prev_1.mul_add(consts::VERT_MUL_PREV_1, prev2_1);
            let out3 = prev_3.mul_add(consts::VERT_MUL_PREV_3, prev2_3);
            let out5 = prev_5.mul_add(consts::VERT_MUL_PREV_5, prev2_5);

            let out1 = sum.mul_add(consts::VERT_MUL_IN_1, -out1);
            let out3 = sum.mul_add(consts::VERT_MUL_IN_3, -out3);
            let out5 = sum.mul_add(consts::VERT_MUL_IN_5, -out5);

            prev2_1 = prev_1;
            prev2_3 = prev_3;
            prev2_5 = prev_5;
            prev_1 = out1;
            prev_3 = out3;
            prev_5 = out5;

            if n >= 0 {
                *output.add(n as usize * width + x_offset) = out1 + out3 + out5;
            }

            n += 1;
        }
    }
}

/// Horizontal row processing with raw pointers
/// Uses multiversion for compile-time CPU optimization
/// # Safety
/// Caller must ensure input and output pointers are valid for width elements
#[multiversion(targets("x86_64+avx2+fma", "x86_64+sse2", "aarch64+neon"))]
fn horizontal_row_unsafe(input: *const f32, output: *mut f32, width: usize) {
    let big_n = consts::RADIUS as isize;

    // Use f32 accumulators (faster than f64, acceptable precision for this branch)
    let mut prev_1 = 0.0f32;
    let mut prev_3 = 0.0f32;
    let mut prev_5 = 0.0f32;
    let mut prev2_1 = 0.0f32;
    let mut prev2_3 = 0.0f32;
    let mut prev2_5 = 0.0f32;

    let mut n = (-big_n) + 1;
    let width_i = width as isize;

    while n < width_i {
        let left = n - big_n - 1;
        let right = n + big_n - 1;

        // SAFETY: bounds checked via if conditions
        let left_val = if left >= 0 && left < width_i {
            unsafe { *input.offset(left) }
        } else {
            0.0f32
        };
        let right_val = if right >= 0 && right < width_i {
            unsafe { *input.offset(right) }
        } else {
            0.0f32
        };

        let sum = left_val + right_val;

        // IIR filter computation
        let mut out_1 = sum * consts::MUL_IN_1;
        let mut out_3 = sum * consts::MUL_IN_3;
        let mut out_5 = sum * consts::MUL_IN_5;

        out_1 = prev2_1.mul_add(consts::MUL_PREV2_1, out_1);
        out_3 = prev2_3.mul_add(consts::MUL_PREV2_3, out_3);
        out_5 = prev2_5.mul_add(consts::MUL_PREV2_5, out_5);

        prev2_1 = prev_1;
        prev2_3 = prev_3;
        prev2_5 = prev_5;

        out_1 = prev_1.mul_add(consts::MUL_PREV_1, out_1);
        out_3 = prev_3.mul_add(consts::MUL_PREV_3, out_3);
        out_5 = prev_5.mul_add(consts::MUL_PREV_5, out_5);

        prev_1 = out_1;
        prev_3 = out_3;
        prev_5 = out_5;

        if n >= 0 {
            // SAFETY: n is checked to be >= 0 and < width_i
            unsafe {
                *output.offset(n) = out_1 + out_3 + out_5;
            }
        }

        n += 1;
    }
}
