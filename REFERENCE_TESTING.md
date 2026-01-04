# SSIMULACRA2 Reference Testing

This document explains how to verify the Rust implementation against the C++ reference.

## Overview

The Rust ssimulacra2 implementation is tested against reference values from the C++ implementation to ensure correctness. This uses a two-step process:

1. **Capture**: Generate test images and capture scores from C++ ssimulacra2
2. **Test**: Run tests that compare Rust scores against captured C++ scores

## Prerequisites

### Build C++ SSIMULACRA2

```bash
# Clone the C++ reference implementation
git clone https://github.com/cloudinary/ssimulacra2.git
cd ssimulacra2

# Build (requires CMake, libhwy-dev, libjpeg-dev, libpng-dev)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Verify binary works
./build/ssimulacra2 --help
```

The binary will be at `./build/ssimulacra2`.

## Capturing Reference Data

Run the capture tool to generate reference data:

```bash
cd /path/to/rust-ssimulacra2

# Set path to C++ binary
export SSIMULACRA2_BIN=/path/to/cloudinary/ssimulacra2/build/ssimulacra2

# Capture reference data (generates src/reference_data.rs)
cargo run --release --example capture_cpp_reference
```

This will:
1. Generate ~100+ synthetic test images (gradients, noise, patterns, etc.)
2. Save them as PNGs in `/tmp/ssimulacra2_reference/`
3. Call the C++ binary to get reference scores
4. Generate `src/reference_data.rs` with all reference values

### Test Patterns Generated

| Pattern | Variations | Purpose |
|---------|-----------|---------|
| Perfect match | 4 sizes | Should score 100.0 |
| Uniform + shift | 5 shifts × 4 sizes | Test uniform color distortion |
| Gradients | H/V/Diag × 4 sizes | Test smooth transitions |
| Checkerboard | 3 cell sizes × 4 sizes | Test high-frequency patterns |
| Random noise | 3 seeds × 4 sizes | Test noise handling |
| Edges | Vertical/Horizontal × 4 sizes | Test sharp transitions |
| Distorted pairs | 2 pairs | Test non-identical images |

**Total**: ~100 test cases

## Running Reference Tests

Once reference data is captured:

```bash
# Run reference parity tests
cargo test --release --test reference_parity

# Expected output:
# All 100 reference tests passed! Max error: 1.23e-6
```

### Tolerance

Tests use **strict tolerance** (< 0.001 absolute error) to detect:
- Algorithmic regressions
- Platform-specific floating-point differences
- Precision bugs

If tests fail:
1. Check for recent code changes
2. Verify C++ binary version matches expectations
3. Re-capture reference data if algorithm intentionally changed

## Continuous Integration

### GitHub Actions Example

```yaml
name: Reference Parity Tests

on: [push, pull_request]

jobs:
  parity:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install C++ dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y cmake libhwy-dev libjpeg-dev libpng-dev

      - name: Build C++ ssimulacra2
        run: |
          git clone https://github.com/cloudinary/ssimulacra2.git /tmp/ssimulacra2
          cd /tmp/ssimulacra2
          cmake -B build -DCMAKE_BUILD_TYPE=Release
          cmake --build build

      - name: Capture reference data
        run: |
          export SSIMULACRA2_BIN=/tmp/ssimulacra2/build/ssimulacra2
          cargo run --release --example capture_cpp_reference

      - name: Run parity tests
        run: cargo test --release --test reference_parity
```

## Comparison to Current Tests

### Before (Current)

```rust
#[test]
fn test_ssimulacra2() {
    let result = compute_frame_ssimulacra2(source, distorted).unwrap();
    let expected = 17.398_505_f64;
    assert!(
        // SOMETHING is WEIRD with Github CI where it gives different results
        (result - expected).abs() < 0.25f64,  // ❌ Loose tolerance!
        "Result {result:.6} not equal to expected {expected:.6}",
    );
}
```

**Problems**:
- Single test case
- Value not verified against C++ (self-generated)
- Loose tolerance (0.25 = 1.4% error)
- CI instability noted

### After (With Reference Data)

```rust
#[test]
fn test_reference_parity() {
    for case in REFERENCE_CASES {
        let score = compute_frame_ssimulacra2(source, distorted).unwrap();
        assert!(
            (score - case.expected_score).abs() < 0.001,  // ✅ Strict!
            "{}: expected {}, got {}",
            case.name,
            case.expected_score,
            score
        );
    }
}
```

**Benefits**:
- 100+ test cases
- All verified against C++ reference
- Strict tolerance (0.001)
- Clear test names
- Auto-generated (reproducible)

## Troubleshooting

### "ssimulacra2 binary not found"

Set `SSIMULACRA2_BIN`:
```bash
export SSIMULACRA2_BIN=/path/to/ssimulacra2
```

Or ensure binary is in `PATH`:
```bash
export PATH=/path/to/build:$PATH
cargo run --example capture_cpp_reference
```

### "Could not parse score from output"

The C++ binary output format might have changed. Check:
```bash
$SSIMULACRA2_BIN source.png distorted.png
```

Expected format: Last number on each line is the score.

### Test failures after capturing

If tests fail immediately after capturing:
1. This indicates Rust implementation differs from C++
2. Check recent code changes
3. Compare specific failing cases to isolate issue

### Platform differences

Floating-point operations may differ slightly across platforms. If you see small differences (< 1e-5):
- Acceptable for different architectures (x86 vs ARM)
- Re-capture on target platform for CI

## Maintenance

### When to Re-capture

Re-capture reference data when:
- ✅ C++ ssimulacra2 updates to new version
- ✅ Algorithm intentionally changes
- ✅ Platform changes (x86 → ARM, etc.)

Do NOT re-capture when:
- ❌ Tests are failing (fix the bug first!)
- ❌ Just to make tests pass (indicates regression)

### Version Tracking

Document reference data version in `reference_data.rs` header:
```rust
//! Generated from: cloudinary/ssimulacra2 v2.1
//! Date: 2026-01-03
//! Platform: x86_64-unknown-linux-gnu
```
