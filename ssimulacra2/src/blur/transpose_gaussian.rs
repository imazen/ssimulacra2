/// Transpose-optimized Recursive Gaussian blur from PR #28
///
/// Uses the same Charalampidis 2016 algorithm as the baseline, but:
/// - Transposes data for better cache locality in vertical pass
/// - Uses f32 accumulators instead of f64 (slightly less precise)
/// - Better parallelization with rayon

mod consts {
    #![allow(clippy::unreadable_literal)]
    include!(concat!(env!("OUT_DIR"), "/recursive_gaussian.rs"));
}

pub struct TransposeGaussian {
    temp: Vec<f32>,
    transposed_input: Vec<f32>,
    transposed_output: Vec<f32>,
}

impl TransposeGaussian {
    pub fn new(width: usize, height: usize) -> Self {
        let size = width * height;
        Self {
            temp: vec![0.0; size],
            transposed_input: vec![0.0; size],
            transposed_output: vec![0.0; size],
        }
    }

    pub fn shrink_to(&mut self, width: usize, height: usize) {
        let size = width * height;
        self.temp.truncate(size);
        self.transposed_input.truncate(size);
        self.transposed_output.truncate(size);
    }

    pub fn blur(&mut self, img: &[Vec<f32>; 3], width: usize, height: usize) -> [Vec<f32>; 3] {
        [
            self.blur_plane(&img[0], width, height),
            self.blur_plane(&img[1], width, height),
            self.blur_plane(&img[2], width, height),
        ]
    }

    pub fn blur_single_plane(&mut self, plane: &[f32], width: usize, height: usize) -> Vec<f32> {
        let mut out = vec![0.0; width * height];
        Self::horizontal_pass_static(plane, &mut self.temp, width);
        self.vertical_pass(&mut out, width, height);
        out
    }

    fn blur_plane(&mut self, plane: &[f32], width: usize, height: usize) -> Vec<f32> {
        self.blur_single_plane(plane, width, height)
    }

    #[inline(always)]
    fn horizontal_pass_static(input: &[f32], output: &mut [f32], width: usize) {
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
    fn horizontal_row(input: &[f32], output: &mut [f32], width: usize) {
        let big_n = consts::RADIUS as isize;
        let [mul_in_1, mul_in_3, mul_in_5] = [consts::MUL_IN_1, consts::MUL_IN_3, consts::MUL_IN_5];
        let [mul_prev_1, mul_prev_3, mul_prev_5] =
            [consts::MUL_PREV_1, consts::MUL_PREV_3, consts::MUL_PREV_5];
        let [mul_prev2_1, mul_prev2_3, mul_prev2_5] = [
            consts::MUL_PREV2_1,
            consts::MUL_PREV2_3,
            consts::MUL_PREV2_5,
        ];

        // Use f32 accumulators (faster but slightly less precise than f64)
        let mut prev = [0f32; 6]; // [prev_1, prev_3, prev_5, prev2_1, prev2_3, prev2_5]

        let mut n = (-big_n) + 1;
        while n < width as isize {
            let left = n - big_n - 1;
            let right = n + big_n - 1;

            let left_val = if left >= 0 {
                unsafe { *input.get_unchecked(left as usize) }
            } else {
                0f32
            };
            let right_val = if right < width as isize {
                unsafe { *input.get_unchecked(right as usize) }
            } else {
                0f32
            };
            let sum = left_val + right_val;

            let mut out = [sum * mul_in_1, sum * mul_in_3, sum * mul_in_5];

            out[0] = mul_prev2_1.mul_add(prev[3], out[0]);
            out[1] = mul_prev2_3.mul_add(prev[4], out[1]);
            out[2] = mul_prev2_5.mul_add(prev[5], out[2]);

            prev[3] = prev[0];
            prev[4] = prev[1];
            prev[5] = prev[2];

            out[0] = mul_prev_1.mul_add(prev[0], out[0]);
            out[1] = mul_prev_3.mul_add(prev[1], out[1]);
            out[2] = mul_prev_5.mul_add(prev[2], out[2]);

            prev[0] = out[0];
            prev[1] = out[1];
            prev[2] = out[2];

            if n >= 0 {
                unsafe {
                    *output.get_unchecked_mut(n as usize) = out[0] + out[1] + out[2];
                }
            }

            n += 1;
        }
    }

    #[inline(always)]
    fn transpose(input: &[f32], output: &mut [f32], width: usize, height: usize) {
        assert_eq!(input.len(), width * height);
        assert_eq!(output.len(), width * height);

        for y in 0..height {
            for x in 0..width {
                output[x * height + y] = input[y * width + x];
            }
        }
    }

    #[inline(always)]
    fn vertical_pass(&mut self, output: &mut [f32], width: usize, height: usize) {
        let size = width * height;
        assert_eq!(self.temp.len(), size);
        assert_eq!(output.len(), size);

        // Transpose the input data for better cache locality
        Self::transpose(&self.temp, &mut self.transposed_input, width, height);

        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;

            let chunk_size = height;
            let transposed_input = &self.transposed_input;
            self.transposed_output
                .par_chunks_exact_mut(chunk_size)
                .enumerate()
                .for_each(|(index, output_chunk)| {
                    let start = index * chunk_size;
                    let input_chunk = &transposed_input[start..start + chunk_size];
                    Self::horizontal_row(input_chunk, output_chunk, height);
                });
        }

        #[cfg(not(feature = "rayon"))]
        {
            for y in 0..width {
                let start = y * height;
                Self::horizontal_row(
                    &self.transposed_input[start..start + height],
                    &mut self.transposed_output[start..start + height],
                    height,
                );
            }
        }

        // Transpose back to original format
        Self::transpose(&self.transposed_output, output, height, width);
    }
}
