/// Libblur backend - fastest but least accurate
///
/// Based on PR #28 by gembleman: ~5-6x faster but 0.001-1.1% accuracy loss
use crate::Ssimulacra2Error;

#[cfg(feature = "inaccurate-libblur")]
use libblur::{BlurImage, BlurImageMut, EdgeMode, FastBlurChannels, ThreadingPolicy};
#[cfg(feature = "inaccurate-libblur")]
use std::borrow::Cow;

pub struct LibblurBackend {
    width: usize,
    height: usize,
}

impl LibblurBackend {
    pub fn new(width: usize, height: usize) -> Self {
        LibblurBackend { width, height }
    }

    pub fn shrink_to(&mut self, width: usize, height: usize) {
        self.width = width;
        self.height = height;
    }

    #[cfg(feature = "inaccurate-libblur")]
    pub fn blur_plane(&mut self, plane: &[f32]) -> Result<Vec<f32>, Ssimulacra2Error> {
        // Tuning sigma to match C++ reference (which uses Charalampidis, not standard Gaussian)
        const KERNEL_SIZE: u32 = 11;
        const SIGMA: f32 = 1.2; // Testing smaller values

        let mut out = vec![0f32; self.width * self.height];

        let src_image = BlurImage {
            data: Cow::Borrowed(plane),
            width: self.width as u32,
            height: self.height as u32,
            stride: self.width as u32,
            channels: FastBlurChannels::Plane,
        };

        let mut dst_image = BlurImageMut::borrow(
            &mut out,
            self.width as u32,
            self.height as u32,
            FastBlurChannels::Plane,
        );

        #[cfg(feature = "rayon")]
        let threading = ThreadingPolicy::Adaptive;
        #[cfg(not(feature = "rayon"))]
        let threading = ThreadingPolicy::Single;

        libblur::gaussian_blur_f32(
            &src_image,
            &mut dst_image,
            KERNEL_SIZE,
            SIGMA,
            EdgeMode::Reflect, // Try Reflect instead of Clamp
            threading,
        )
        .map_err(|_| Ssimulacra2Error::GaussianBlurError)?;

        Ok(out)
    }

    #[cfg(not(feature = "inaccurate-libblur"))]
    pub fn blur_plane(&mut self, _plane: &[f32]) -> Result<Vec<f32>, Ssimulacra2Error> {
        unreachable!("libblur backend not compiled")
    }
}
