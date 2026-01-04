# Note on libblur Backend

## Summary

The `inaccurate-libblur` feature was explored as a performance optimization but is **not recommended for production use** due to incompatibility with the C++ SSIMULACRA2 reference implementation.

## Background

The libblur crate provides fast Gaussian blur using standard convolution. An earlier experiment attempted to integrate it as an alternative to our Charalampidis 2016 recursive IIR filter.

## Technical Findings

### Algorithm Differences

The C++ SSIMULACRA2 reference uses:
- Algorithm: Charalampidis 2016 recursive IIR approximation
- Effective sigma: ~1.5 (implicit in the IIR coefficients)

The libblur crate uses:
- Algorithm: Standard Gaussian convolution
- Configurable sigma parameter

**Key insight:** These are fundamentally different mathematical operations. Even with careful sigma tuning, they produce different blur characteristics.

### Compatibility Testing

When tested against 66 C++ reference test cases:

| Backend | Max Error vs C++ | Tests Passed | Notes |
|---------|------------------|--------------|-------|
| baseline (f64 IIR) | 0.955 | 66/66 (100%) | Reference-compatible |
| transpose (f32 IIR) | 0.926 | 66/66 (100%) | Faster, compatible |
| libblur | 971.27 | 42/66 (64%) | Incompatible |

The large error occurs because different blur algorithms affect the entire SSIMULACRA2 pipeline:
1. Different blur → different intermediate values
2. Error compounds through multi-scale analysis
3. Final scores diverge significantly from reference

### Performance Measurements

Tested configurations (1448×1080 blur operation):

| Configuration | Time | Speedup vs Baseline |
|---------------|------|---------------------|
| baseline (f64 IIR) | ~85ms | 1.0x |
| transpose (f32 IIR) | ~34ms | 2.5x |
| libblur | ~13ms | 6.5x |

While libblur is indeed faster, the speedup comes at the cost of changing the fundamental metric behavior.

## Recommendation

**Use `blur-transpose` as default** (current choice as of 2026):
- 2.5x faster than baseline
- Maintains compatibility with C++ reference
- Uses same Charalampidis IIR algorithm, just with better memory layout

**Optional: `blur-simd`** for maximum performance:
- 3.6x faster than baseline
- Still uses Charalampidis IIR algorithm
- Slight accuracy trade-off (1.24 max error) but maintains algorithm compatibility

**Avoid: `inaccurate-libblur`** for SSIMULACRA2 metric:
- Breaks compatibility with reference implementation
- Scores not comparable across implementations
- May be acceptable for other use cases where compatibility isn't required

## Context on Earlier Exploration

The initial libblur integration was explored with limited test coverage. The incompatibility wasn't apparent until comprehensive reference testing was added. This is a good reminder that perceptual metrics require careful validation against reference implementations.

The libblur crate itself is well-implemented and useful for general-purpose Gaussian blur. The incompatibility is specific to SSIMULACRA2's requirement for a particular blur algorithm.

## For Maintainers

If someone proposes using libblur again:
1. Ask: "Does it use the same blur algorithm as the C++ reference?"
2. Require: Full reference parity testing (all 66 test cases)
3. Verify: Max error < 1.5 vs C++ reference

The performance gain is real, but algorithm compatibility is non-negotiable for a standardized metric.

---

**Testing Date:** January 2026
**Test Suite:** 66 C++ reference cases from libjxl SSIMULACRA2 v2.1
**Libblur Version:** 0.17.5
