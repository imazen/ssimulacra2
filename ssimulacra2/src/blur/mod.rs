mod gaussian;
mod simd_gaussian;
mod transpose_gaussian;

#[cfg(feature = "unsafe-simd")]
mod unsafe_simd_gaussian;

use gaussian::RecursiveGaussian;
use simd_gaussian::SimdGaussian;
use transpose_gaussian::TransposeGaussian;

#[cfg(feature = "unsafe-simd")]
use unsafe_simd_gaussian::UnsafeSimdGaussian;

/// Implementation backend for blur operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlurImpl {
    /// Scalar implementation (baseline, most accurate)
    Scalar,
    /// Safe SIMD via wide crate
    #[default]
    Simd,
    /// Transpose-optimized blur (better cache locality, uses f32)
    SimdTranspose,
    /// Raw x86 intrinsics (fastest, experimental)
    #[cfg(feature = "unsafe-simd")]
    UnsafeSimd,
}

impl BlurImpl {
    /// Returns the name of this implementation
    pub fn name(&self) -> &'static str {
        match self {
            BlurImpl::Scalar => "scalar",
            BlurImpl::Simd => "simd (wide crate)",
            BlurImpl::SimdTranspose => "simd-transpose (cache-optimized)",
            #[cfg(feature = "unsafe-simd")]
            BlurImpl::UnsafeSimd => "unsafe-simd (raw intrinsics)",
        }
    }
}

/// Structure handling image blur with selectable implementation.
///
/// Supports runtime switching between:
/// - Scalar: f64 IIR baseline (most accurate)
/// - SIMD: Safe SIMD via wide crate
/// - SimdTranspose: Transpose-optimized for cache locality
/// - UnsafeSimd: Raw x86 intrinsics (fastest)
pub struct Blur {
    width: usize,
    height: usize,
    impl_type: BlurImpl,
    // Scalar backend
    scalar_kernel: RecursiveGaussian,
    scalar_temp: Vec<f32>,
    // Safe SIMD backend
    simd: SimdGaussian,
    // Transpose-optimized backend
    transpose: TransposeGaussian,
    // Unsafe SIMD backend
    #[cfg(feature = "unsafe-simd")]
    unsafe_simd: UnsafeSimdGaussian,
}

impl Blur {
    /// Create a new [Blur] with the default implementation (SIMD).
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self::with_impl(width, height, BlurImpl::default())
    }

    /// Create a new [Blur] with a specific implementation.
    #[must_use]
    pub fn with_impl(width: usize, height: usize, impl_type: BlurImpl) -> Self {
        Blur {
            width,
            height,
            impl_type,
            scalar_kernel: RecursiveGaussian,
            scalar_temp: vec![0.0f32; width * height],
            simd: SimdGaussian::new(width),
            transpose: TransposeGaussian::new(width, height),
            #[cfg(feature = "unsafe-simd")]
            unsafe_simd: UnsafeSimdGaussian::new(width),
        }
    }

    /// Get the current implementation type.
    pub fn impl_type(&self) -> BlurImpl {
        self.impl_type
    }

    /// Set the implementation type.
    pub fn set_impl(&mut self, impl_type: BlurImpl) {
        self.impl_type = impl_type;
    }

    /// Truncates the internal buffers to fit images of the given width and height.
    pub fn shrink_to(&mut self, width: usize, height: usize) {
        self.scalar_temp.truncate(width * height);
        self.simd.shrink_to(width, height);
        self.transpose.shrink_to(width, height);
        #[cfg(feature = "unsafe-simd")]
        self.unsafe_simd.shrink_to(width, height);
        self.width = width;
        self.height = height;
    }

    /// Blur the given image using the selected implementation.
    pub fn blur(&mut self, img: &[Vec<f32>; 3]) -> [Vec<f32>; 3] {
        [
            self.blur_plane(&img[0]),
            self.blur_plane(&img[1]),
            self.blur_plane(&img[2]),
        ]
    }

    fn blur_plane(&mut self, plane: &[f32]) -> Vec<f32> {
        match self.impl_type {
            BlurImpl::Scalar => self.blur_plane_scalar(plane),
            BlurImpl::Simd => self.blur_plane_simd(plane),
            BlurImpl::SimdTranspose => self.blur_plane_transpose(plane),
            #[cfg(feature = "unsafe-simd")]
            BlurImpl::UnsafeSimd => self.blur_plane_unsafe_simd(plane),
        }
    }

    fn blur_plane_scalar(&mut self, plane: &[f32]) -> Vec<f32> {
        let mut out = vec![0f32; self.width * self.height];
        self.scalar_kernel
            .horizontal_pass(plane, &mut self.scalar_temp, self.width);
        self.scalar_kernel
            .vertical_pass_chunked::<128, 32>(&self.scalar_temp, &mut out, self.width, self.height);
        out
    }

    fn blur_plane_simd(&mut self, plane: &[f32]) -> Vec<f32> {
        self.simd.blur_single_plane(plane, self.width, self.height)
    }

    fn blur_plane_transpose(&mut self, plane: &[f32]) -> Vec<f32> {
        self.transpose
            .blur_single_plane(plane, self.width, self.height)
    }

    #[cfg(feature = "unsafe-simd")]
    fn blur_plane_unsafe_simd(&mut self, plane: &[f32]) -> Vec<f32> {
        self.unsafe_simd
            .blur_single_plane(plane, self.width, self.height)
    }
}
