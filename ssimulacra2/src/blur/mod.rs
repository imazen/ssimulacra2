mod gaussian;

#[cfg(feature = "blur-libblur")]
mod libblur_backend;

#[cfg(feature = "blur-transpose")]
mod transpose_gaussian;

use gaussian::RecursiveGaussian;

#[cfg(feature = "blur-libblur")]
use libblur_backend::LibblurBackend;

#[cfg(feature = "blur-transpose")]
use transpose_gaussian::TransposeGaussian;

/// Structure handling image blur.
///
/// This struct contains the necessary buffers and the kernel used for blurring
/// (currently a recursive approximation of the Gaussian filter).
///
/// Backend selection (compile-time via features):
/// - `blur-accurate` (default): f64 IIR - most accurate, verified against C++
/// - `blur-libblur`: External libblur crate - ~5-6x faster, 0.001-1.1% less accurate
///
/// Note that the width and height of the image passed to [blur][Self::blur] needs to exactly
/// match the width and height of this instance. If you reduce the image size (e.g. via
/// downscaling), [`shrink_to`][Self::shrink_to] can be used to resize the internal buffers.
pub struct Blur {
    #[cfg(all(not(feature = "blur-libblur"), not(feature = "blur-transpose")))]
    kernel: RecursiveGaussian,
    #[cfg(all(not(feature = "blur-libblur"), not(feature = "blur-transpose")))]
    temp: Vec<f32>,
    #[cfg(feature = "blur-libblur")]
    backend: LibblurBackend,
    #[cfg(feature = "blur-transpose")]
    transpose: TransposeGaussian,
    width: usize,
    height: usize,
}

impl Blur {
    /// Create a new [Blur] for images of the given width and height.
    /// This pre-allocates the necessary buffers.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        #[cfg(all(not(feature = "blur-libblur"), not(feature = "blur-transpose")))]
        {
            Blur {
                kernel: RecursiveGaussian,
                temp: vec![0.0f32; width * height],
                width,
                height,
            }
        }
        #[cfg(feature = "blur-libblur")]
        {
            Blur {
                backend: LibblurBackend::new(width, height),
                width,
                height,
            }
        }
        #[cfg(feature = "blur-transpose")]
        {
            Blur {
                transpose: TransposeGaussian::new(width, height),
                width,
                height,
            }
        }
    }

    /// Truncates the internal buffers to fit images of the given width and height.
    ///
    /// This will [truncate][Vec::truncate] the internal buffers
    /// without affecting the allocated memory.
    pub fn shrink_to(&mut self, width: usize, height: usize) {
        #[cfg(all(not(feature = "blur-libblur"), not(feature = "blur-transpose")))]
        {
            self.temp.truncate(width * height);
        }
        #[cfg(feature = "blur-libblur")]
        {
            self.backend.shrink_to(width, height);
        }
        #[cfg(feature = "blur-transpose")]
        {
            self.transpose.shrink_to(width, height);
        }
        self.width = width;
        self.height = height;
    }

    /// Blur the given image.
    pub fn blur(&mut self, img: &[Vec<f32>; 3]) -> [Vec<f32>; 3] {
        [
            self.blur_plane(&img[0]),
            self.blur_plane(&img[1]),
            self.blur_plane(&img[2]),
        ]
    }

    #[cfg(all(not(feature = "blur-libblur"), not(feature = "blur-transpose")))]
    fn blur_plane(&mut self, plane: &[f32]) -> Vec<f32> {
        let mut out = vec![0f32; self.width * self.height];
        self.kernel
            .horizontal_pass(plane, &mut self.temp, self.width);
        self.kernel
            .vertical_pass_chunked::<128, 32>(&self.temp, &mut out, self.width, self.height);
        out
    }

    #[cfg(feature = "blur-libblur")]
    fn blur_plane(&mut self, plane: &[f32]) -> Vec<f32> {
        self.backend.blur_plane(plane).unwrap()
    }

    #[cfg(feature = "blur-transpose")]
    fn blur_plane(&mut self, plane: &[f32]) -> Vec<f32> {
        self.transpose
            .blur_single_plane(plane, self.width, self.height)
    }
}
