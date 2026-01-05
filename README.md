# fast-ssim2

[![docs.rs](https://img.shields.io/docsrs/fast-ssim2?style=for-the-badge)](https://docs.rs/fast-ssim2)
[![Crates.io](https://img.shields.io/crates/v/fast-ssim2?style=for-the-badge)](https://crates.io/crates/fast-ssim2)
[![LICENSE](https://img.shields.io/crates/l/fast-ssim2?style=for-the-badge)](https://github.com/imazen/ssimulacra2/blob/main/LICENSE)

Rust implementation of the [SSIMULACRA2](https://github.com/cloudinary/ssimulacra2) perceptual image quality metric.

## Attribution

This crate is a fork of [rust-av/ssimulacra2](https://github.com/rust-av/ssimulacra2). Thank you to the rust-av team for creating the original Rust implementation - their clean, well-structured code made it straightforward to build upon.

### What's different

The upstream crate provides a correct, readable scalar implementation. This fork experiments with SIMD acceleration and adds some API conveniences:

- SIMD-accelerated blur and XYB conversion (via `wide` crate and x86 intrinsics)
- Precomputed reference API for batch comparisons (`Ssimulacra2Reference`)
- `imgref` type support via `ToLinearRgb` trait

These changes come with trade-offs: higher MSRV (1.89.0 vs 1.65.0), more complex code, and platform-specific behavior. If you prefer simplicity and stability, the upstream crate may be a better fit.

### Contributing back

We'd be happy to upstream any of these changes if the rust-av team is interested. The SIMD work is substantial and may not align with their goals, but we're open to collaboration.

## Installation

```toml
[dependencies]
fast-ssim2 = "0.6"
```

## Features

- **`simd`** (default): Safe SIMD via the `wide` crate
- **`unsafe-simd`** (default): x86_64 intrinsics (faster on supported hardware)
- **`imgref`**: Support for `imgref` image types
- **`rayon`**: Parallel computation

## Usage

### With imgref

```rust
use fast_ssim2::compute_ssimulacra2;
use imgref::ImgVec;

let source: ImgVec<[u8; 3]> = /* your image */;
let distorted: ImgVec<[u8; 3]> = /* distorted version */;

let score = compute_ssimulacra2(source.as_ref(), distorted.as_ref())?;
// 100 = identical, lower = more different
```

### With yuvxyb types

```rust
use fast_ssim2::{compute_frame_ssimulacra2, Rgb, TransferCharacteristic, ColorPrimaries};

let source = Rgb::new(data, width, height,
    TransferCharacteristic::SRGB, ColorPrimaries::BT709)?;
let distorted = Rgb::new(/* ... */)?;

let score = compute_frame_ssimulacra2(source, distorted)?;
```

### Precomputed reference (for batch comparisons)

```rust
use fast_ssim2::Ssimulacra2Reference;

let reference = Ssimulacra2Reference::new(source)?;

for distorted in variants {
    let score = reference.compare(distorted)?;
}
```

## Score interpretation

| Score | Meaning |
|-------|---------|
| 100 | Identical |
| 90+ | Imperceptible difference |
| 70-90 | Minor difference |
| 50-70 | Noticeable |
| < 50 | Significant degradation |

## License

BSD-2-Clause (same as upstream)

## Notes

Developed with assistance from Claude (Anthropic). Tested against the C++ reference implementation.
