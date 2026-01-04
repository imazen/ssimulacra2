/// SIMD-optimized Recursive Gaussian using `wide` crate
///
/// Uses f32x4 (SSE2, 128-bit SIMD) to process 4 columns simultaneously
/// in the vertical pass. This is the fastest configuration on most CPUs.
use wide::f32x4;

mod consts {
    #![allow(clippy::unreadable_literal)]
    include!(concat!(env!("OUT_DIR"), "/recursive_gaussian.rs"));
}

use multiversion::multiversion;

pub struct SimdGaussian {
    // Pre-allocated buffers for vertical pass (avoids allocations)
    prev_buffer: Vec<f32>,
    prev2_buffer: Vec<f32>,
    out_buffer: Vec<f32>,
}

impl SimdGaussian {
    pub fn new(_max_width: usize) -> Self {
        // Allocate for max columns we'll process (128 columns = 32 SIMD lanes of 4)
        const MAX_COLUMNS: usize = 128;
        Self {
            prev_buffer: vec![0.0; 3 * MAX_COLUMNS],
            prev2_buffer: vec![0.0; 3 * MAX_COLUMNS],
            out_buffer: vec![0.0; 3 * MAX_COLUMNS],
        }
    }

    pub fn shrink_to(&mut self, _width: usize, _height: usize) {
        // Buffers are pre-allocated to max size, just reuse them
    }

    /// Public API matching other blur implementations
    pub fn blur_single_plane(&mut self, plane: &[f32], width: usize, height: usize) -> Vec<f32> {
        let mut temp = vec![0.0; width * height];
        let mut out = vec![0.0; width * height];

        // Horizontal pass
        Self::horizontal_pass(plane, &mut temp, width);

        // Vertical pass with SIMD
        self.vertical_pass_simd_chunked(&temp, &mut out, width, height);

        out
    }

    /// Horizontal pass - same as baseline (IIR is inherently sequential)
    fn horizontal_pass(input: &[f32], output: &mut [f32], width: usize) {
        assert_eq!(input.len(), output.len());

        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            input
                .par_chunks_exact(width)
                .zip(output.par_chunks_exact_mut(width))
                .for_each(|(input, output)| Self::horizontal_row(input, output, width));
        }

        #[cfg(not(feature = "rayon"))]
        {
            input
                .chunks_exact(width)
                .zip(output.chunks_exact_mut(width))
                .for_each(|(input, output)| Self::horizontal_row(input, output, width));
        }
    }

    #[inline(always)]
    #[multiversion(targets("x86_64+avx2+fma", "x86_64+sse2", "aarch64+neon"))]
    fn horizontal_row(input: &[f32], output: &mut [f32], width: usize) {
        let big_n = consts::RADIUS as isize;

        // Use f32 accumulators (matching transpose implementation)
        let mut prev_1 = 0f32;
        let mut prev_3 = 0f32;
        let mut prev_5 = 0f32;
        let mut prev2_1 = 0f32;
        let mut prev2_3 = 0f32;
        let mut prev2_5 = 0f32;

        let mut n = (-big_n) + 1;
        while n < width as isize {
            let left = n - big_n - 1;
            let right = n + big_n - 1;
            let left_val = if left >= 0 && (left as usize) < input.len() {
                input[left as usize]
            } else {
                0f32
            };
            let right_val = if right >= 0 && (right as usize) < input.len() {
                input[right as usize]
            } else {
                0f32
            };
            let sum = left_val + right_val;

            let mut out_1 = sum * consts::MUL_IN_1;
            let mut out_3 = sum * consts::MUL_IN_3;
            let mut out_5 = sum * consts::MUL_IN_5;

            out_1 = consts::MUL_PREV2_1.mul_add(prev2_1, out_1);
            out_3 = consts::MUL_PREV2_3.mul_add(prev2_3, out_3);
            out_5 = consts::MUL_PREV2_5.mul_add(prev2_5, out_5);
            prev2_1 = prev_1;
            prev2_3 = prev_3;
            prev2_5 = prev_5;

            out_1 = consts::MUL_PREV_1.mul_add(prev_1, out_1);
            out_3 = consts::MUL_PREV_3.mul_add(prev_3, out_3);
            out_5 = consts::MUL_PREV_5.mul_add(prev_5, out_5);
            prev_1 = out_1;
            prev_3 = out_3;
            prev_5 = out_5;

            if n >= 0 && (n as usize) < output.len() {
                output[n as usize] = out_1 + out_3 + out_5;
            }

            n += 1;
        }
    }

    /// SIMD-optimized vertical pass
    /// Processes 4 columns at a time using f32x4
    pub fn vertical_pass_simd_chunked(
        &mut self,
        input: &[f32],
        output: &mut [f32],
        width: usize,
        height: usize,
    ) {
        assert_eq!(input.len(), output.len());

        let mut x = 0;

        // Process 128 columns at a time (32 SIMD lanes of 4)
        while x + 128 <= width {
            Self::vertical_pass_simd::<128>(
                &input[x..],
                &mut output[x..],
                width,
                height,
                &mut self.prev_buffer[..3 * 128],
                &mut self.prev2_buffer[..3 * 128],
                &mut self.out_buffer[..3 * 128],
            );
            x += 128;
        }

        // Process 32 columns at a time (8 SIMD lanes of 4)
        while x + 32 <= width {
            Self::vertical_pass_simd::<32>(
                &input[x..],
                &mut output[x..],
                width,
                height,
                &mut self.prev_buffer[..3 * 32],
                &mut self.prev2_buffer[..3 * 32],
                &mut self.out_buffer[..3 * 32],
            );
            x += 32;
        }

        // Process 4 columns at a time (1 SIMD lane of 4)
        while x + 4 <= width {
            Self::vertical_pass_simd::<4>(
                &input[x..],
                &mut output[x..],
                width,
                height,
                &mut self.prev_buffer[..3 * 4],
                &mut self.prev2_buffer[..3 * 4],
                &mut self.out_buffer[..3 * 4],
            );
            x += 4;
        }

        // Handle remaining columns with scalar version
        while x < width {
            self.vertical_pass_scalar::<1>(&input[x..], &mut output[x..], width, height);
            x += 1;
        }
    }

    /// SIMD vertical pass - processes COLUMNS columns (must be multiple of 4)
    #[inline(always)]
    #[multiversion(targets("x86_64+avx2+fma", "x86_64+sse2", "aarch64+neon"))]
    fn vertical_pass_simd<const COLUMNS: usize>(
        input: &[f32],
        output: &mut [f32],
        width: usize,
        height: usize,
        prev: &mut [f32],
        prev2: &mut [f32],
        out: &mut [f32],
    ) {
        assert!(COLUMNS % 4 == 0, "COLUMNS must be multiple of 4 for SIMD");
        assert_eq!(input.len(), output.len());
        assert_eq!(prev.len(), 3 * COLUMNS);
        assert_eq!(prev2.len(), 3 * COLUMNS);
        assert_eq!(out.len(), 3 * COLUMNS);

        let big_n = consts::RADIUS as isize;
        let simd_lanes = COLUMNS / 4;

        // Clear buffers
        prev.fill(0.0);
        prev2.fill(0.0);
        out.fill(0.0);

        let zeroes = f32x4::splat(0.0);

        // Splat constants for SIMD operations
        let mul_in_1 = f32x4::splat(consts::VERT_MUL_IN_1);
        let mul_in_3 = f32x4::splat(consts::VERT_MUL_IN_3);
        let mul_in_5 = f32x4::splat(consts::VERT_MUL_IN_5);
        let mul_prev_1 = f32x4::splat(consts::VERT_MUL_PREV_1);
        let mul_prev_3 = f32x4::splat(consts::VERT_MUL_PREV_3);
        let mul_prev_5 = f32x4::splat(consts::VERT_MUL_PREV_5);

        let mut n = (-big_n) + 1;
        while n < height as isize {
            let top = n - big_n - 1;
            let bottom = n + big_n - 1;

            // Process 4 columns at a time using SIMD
            for lane in 0..simd_lanes {
                let i = lane * 4;

                // Load 4 values from top and bottom rows
                let top_vals = if top >= 0 && (top as usize * width + i + 3) < input.len() {
                    let idx = top as usize * width + i;
                    f32x4::new([input[idx], input[idx + 1], input[idx + 2], input[idx + 3]])
                } else {
                    zeroes
                };

                let bottom_vals = if bottom >= 0 && (bottom as usize * width + i + 3) < input.len()
                {
                    let idx = bottom as usize * width + i;
                    f32x4::new([input[idx], input[idx + 1], input[idx + 2], input[idx + 3]])
                } else {
                    zeroes
                };

                let sum = top_vals + bottom_vals;

                // Load previous values
                let i1 = i;
                let i3 = i1 + COLUMNS;
                let i5 = i3 + COLUMNS;

                let prev_1_vec = f32x4::new([prev[i1], prev[i1 + 1], prev[i1 + 2], prev[i1 + 3]]);
                let prev_3_vec = f32x4::new([prev[i3], prev[i3 + 1], prev[i3 + 2], prev[i3 + 3]]);
                let prev_5_vec = f32x4::new([prev[i5], prev[i5 + 1], prev[i5 + 2], prev[i5 + 3]]);

                let prev2_1_vec =
                    f32x4::new([prev2[i1], prev2[i1 + 1], prev2[i1 + 2], prev2[i1 + 3]]);
                let prev2_3_vec =
                    f32x4::new([prev2[i3], prev2[i3 + 1], prev2[i3 + 2], prev2[i3 + 3]]);
                let prev2_5_vec =
                    f32x4::new([prev2[i5], prev2[i5 + 1], prev2[i5 + 2], prev2[i5 + 3]]);

                // SIMD computation of IIR filter
                let out1 = prev_1_vec.mul_add(mul_prev_1, prev2_1_vec);
                let out3 = prev_3_vec.mul_add(mul_prev_3, prev2_3_vec);
                let out5 = prev_5_vec.mul_add(mul_prev_5, prev2_5_vec);

                let out1 = sum.mul_add(mul_in_1, -out1);
                let out3 = sum.mul_add(mul_in_3, -out3);
                let out5 = sum.mul_add(mul_in_5, -out5);

                // Store outputs (use array indexing)
                let out1_arr = out1.to_array();
                let out3_arr = out3.to_array();
                let out5_arr = out5.to_array();

                for j in 0..4 {
                    out[i1 + j] = out1_arr[j];
                    out[i3 + j] = out3_arr[j];
                    out[i5 + j] = out5_arr[j];
                }

                // Write final output if we're past the padding
                if n >= 0 {
                    let result = out1 + out3 + out5;
                    let result_arr = result.to_array();
                    for j in 0..4 {
                        output[n as usize * width + i + j] = result_arr[j];
                    }
                }
            }

            // Swap buffers (prev2 = prev, prev = out)
            prev2.copy_from_slice(prev);
            prev.copy_from_slice(out);

            n += 1;
        }
    }

    /// Scalar fallback for remaining columns
    fn vertical_pass_scalar<const COLUMNS: usize>(
        &mut self,
        input: &[f32],
        output: &mut [f32],
        width: usize,
        height: usize,
    ) {
        // Same as baseline implementation
        assert_eq!(input.len(), output.len());

        let big_n = consts::RADIUS as isize;

        let zeroes = vec![0f32; COLUMNS];
        let mut prev = vec![0f32; 3 * COLUMNS];
        let mut prev2 = vec![0f32; 3 * COLUMNS];
        let mut out = vec![0f32; 3 * COLUMNS];

        let mut n = (-big_n) + 1;
        while n < height as isize {
            let top = n - big_n - 1;
            let bottom = n + big_n - 1;
            let top_row = if top >= 0 {
                &input[top as usize * width..][..COLUMNS]
            } else {
                &zeroes
            };

            let bottom_row = if bottom < height as isize {
                &input[bottom as usize * width..][..COLUMNS]
            } else {
                &zeroes
            };

            for i in 0..COLUMNS {
                let sum = top_row[i] + bottom_row[i];

                let i1 = i;
                let i3 = i1 + COLUMNS;
                let i5 = i3 + COLUMNS;

                let out1 = prev[i1].mul_add(consts::VERT_MUL_PREV_1, prev2[i1]);
                let out3 = prev[i3].mul_add(consts::VERT_MUL_PREV_3, prev2[i3]);
                let out5 = prev[i5].mul_add(consts::VERT_MUL_PREV_5, prev2[i5]);

                let out1 = sum.mul_add(consts::VERT_MUL_IN_1, -out1);
                let out3 = sum.mul_add(consts::VERT_MUL_IN_3, -out3);
                let out5 = sum.mul_add(consts::VERT_MUL_IN_5, -out5);

                out[i1] = out1;
                out[i3] = out3;
                out[i5] = out5;

                if n >= 0 {
                    output[n as usize * width + i] = out1 + out3 + out5;
                }
            }

            prev2.copy_from_slice(&prev);
            prev.copy_from_slice(&out);

            n += 1;
        }
    }
}
