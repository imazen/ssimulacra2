# fast-ssim2

[![docs.rs](https://img.shields.io/docsrs/fast-ssim2?style=for-the-badge)](https://docs.rs/fast-ssim2)
[![Crates.io](https://img.shields.io/crates/v/fast-ssim2?style=for-the-badge)](https://crates.io/crates/fast-ssim2)
[![LICENSE](https://img.shields.io/crates/l/fast-ssim2?style=for-the-badge)](https://github.com/imazen/ssimulacra2/blob/main/LICENSE)

Fast Rust implementation of the [SSIMULACRA2](https://github.com/cloudinary/ssimulacra2) perceptual image quality metric with SIMD acceleration.

## Fork Notice & Attribution

This crate is a fork of [rust-av/ssimulacra2](https://github.com/rust-av/ssimulacra2), originally created by the rust-av team. We are grateful for their foundational work.

### Why a fork?

We needed to iterate faster on performance optimizations and API improvements for our image processing pipelines. Rather than wait for upstream review cycles, we decided to maintain a separate crate while keeping the door open for upstreaming improvements.

### Differences from upstream

| Feature | rust-av/ssimulacra2 | fast-ssim2 |
|---------|---------------------|------------|
| Precomputed reference API | ❌ | ✅ `Ssimulacra2Reference` |
| `imgref` support | ❌ | ✅ `ToLinearRgb` trait |
| Safe SIMD (wide crate) | ✅ | ✅ |
| Unsafe x86 SIMD | ✅ | ✅ (via `safe_unaligned_simd`) |
| Re-exported yuvxyb types | Partial | ✅ Full |
| MSRV | 1.62.0 | 1.89.0 |

### Upstreaming

We welcome upstreaming of any improvements back to rust-av/ssimulacra2. If you're from the rust-av team and would like to incorporate any of these changes, please reach out or open a PR - we're happy to contribute back.

## Installation

```toml
[dependencies]
fast-ssim2 = "0.6"
```

## Features

- **`simd`** (default): Safe SIMD via the `wide` crate
- **`unsafe-simd`** (default): Faster x86_64 intrinsics with safe memory access
- **`imgref`**: Support for `imgref` image types (`ImgRef<[u8; 3]>`, etc.)
- **`rayon`**: Parallel computation support

## Quick Start

### With imgref (recommended)

```rust
use fast_ssim2::compute_ssimulacra2;
use imgref::{Img, ImgVec};

let source: ImgVec<[u8; 3]> = /* load your image */;
let distorted: ImgVec<[u8; 3]> = /* load distorted image */;

let score = compute_ssimulacra2(source.as_ref(), distorted.as_ref())?;
println!("SSIMULACRA2 score: {:.2}", score);
// 100 = identical, 0 = very different, negative = severely degraded
```

### With yuvxyb types

```rust
use fast_ssim2::{compute_frame_ssimulacra2, Rgb, TransferCharacteristic, ColorPrimaries};

let source = Rgb::new(source_data, width, height,
    TransferCharacteristic::SRGB, ColorPrimaries::BT709)?;
let distorted = Rgb::new(distorted_data, width, height,
    TransferCharacteristic::SRGB, ColorPrimaries::BT709)?;

let score = compute_frame_ssimulacra2(source, distorted)?;
```

## Precomputed Reference API (~1.7x speedup)

When comparing multiple distorted images against the same reference, precompute once and reuse:

```rust
use fast_ssim2::Ssimulacra2Reference;

// Precompute reference data once
let reference = Ssimulacra2Reference::new(source_image)?;

// Compare against many distorted images - ~1.7x faster per comparison
for distorted in variants {
    let score = reference.compare(distorted)?;
    println!("Score: {:.2}", score);
}
```

### Performance

| Image Size | Full Compute | Precomputed | Speedup |
|------------|--------------|-------------|---------|
| 256×256 | 154ms | 87ms | 1.76x |
| 512×512 | 252ms | 152ms | 1.66x |
| 1024×1024 | 610ms | 355ms | 1.72x |

Run benchmark: `cargo run --release --example precompute_benchmark`

## Supported Input Types

With the `imgref` feature:

| Type | Color Space | Conversion |
|------|-------------|------------|
| `ImgRef<[u8; 3]>` | sRGB (gamma) | `/255` + linearize |
| `ImgRef<[u16; 3]>` | sRGB (gamma) | `/65535` + linearize |
| `ImgRef<[f32; 3]>` | Linear RGB | none |
| `ImgRef<u8>` | sRGB grayscale | linearize + expand |
| `ImgRef<f32>` | Linear grayscale | expand to RGB |

Custom types can implement the `ToLinearRgb` trait.

## Score Interpretation

| Score | Quality |
|-------|---------|
| 100 | Identical |
| 90+ | Imperceptible difference |
| 70-90 | Slight difference |
| 50-70 | Noticeable difference |
| < 50 | Significant degradation |
| < 0 | Severe degradation |

## Minimum Supported Rust Version (MSRV)

1.89.0. MSRV increases are considered semver-compatible.

## License

BSD-2-Clause (same as upstream)

## AI-Generated Code Notice

Portions of this crate were developed with assistance from Claude (Anthropic). The code has been tested for correctness against the C++ reference implementation. Review critical paths before production use.
