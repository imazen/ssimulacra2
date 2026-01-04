# ssimulacra2

[![docs.rs](https://img.shields.io/docsrs/ssimulacra2?style=for-the-badge)](https://docs.rs/ssimulacra2)
[![Crates.io](https://img.shields.io/crates/v/ssimulacra2?style=for-the-badge)](https://crates.io/crates/ssimulacra2)
[![LICENSE](https://img.shields.io/crates/l/ssimulacra2?style=for-the-badge)](https://github.com/rust-av/ssimulacra2/blob/main/LICENSE)

Rust implementation of the [SSIMULACRA2 metric](https://github.com/cloudinary/ssimulacra2).

## imazen Fork Additions

This fork adds `Ssim2Reference` for precomputed reference comparisons, providing **~1.7x speedup** when comparing multiple distorted images against the same reference.

**Upstream**: [rust-av/ssimulacra2](https://github.com/rust-av/ssimulacra2)
**Branch**: `precompute-reference`

### Use Case

Simulated annealing or other optimization workflows where you need to compare hundreds/thousands of encoder variations against the same source image.

### API

```rust
use ssimulacra2::Ssim2Reference;
use yuvxyb::{Rgb, TransferCharacteristic, ColorPrimaries};

// Precompute reference data once
let reference = Rgb::new(reference_data, width, height,
    TransferCharacteristic::SRGB, ColorPrimaries::BT709)?;
let precomputed = Ssim2Reference::new(reference)?;

// Compare against multiple distorted images - ~1.7x faster per comparison
for distorted_data in variants {
    let distorted = Rgb::new(distorted_data, width, height,
        TransferCharacteristic::SRGB, ColorPrimaries::BT709)?;
    let score = precomputed.compare(distorted)?;
    println!("SSIMULACRA2: {}", score);
}
```

### Performance

| Image Size | Full Compute | Precomputed | Speedup |
|------------|--------------|-------------|---------|
| 256×256 | 154ms | 87ms | 1.76x |
| 512×512 | 252ms | 152ms | 1.66x |
| 1024×1024 | 610ms | 355ms | 1.72x |

Run benchmark: `cargo run --release --example precompute_benchmark`

### What's Precomputed

For each of 6 scales:
- Downscaled LinearRgb reference image
- XYB planar representation
- `mu1 = blur(reference)` - mean
- `sigma1_sq = blur(reference²)` - variance

Each comparison only computes distorted-side data and cross-terms (`sigma12`).

### Compatibility

- ✅ **Non-breaking**: Purely additive API
- ✅ **Identical results**: Tests verify exact match with `compute_frame_ssimulacra2()`
- ✅ **Works with upstream**: Can be upstreamed without changes

## Minimum supported Rust version (MSRV)

This crates requires a Rust version of 1.62.0 or higher. Increases in MSRV will result in a semver PATCH version increase.

