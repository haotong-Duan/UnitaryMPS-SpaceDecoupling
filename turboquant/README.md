# TurboQuant for MPS

Post-training quantization of a trained MPS following Zandieh et al., *TurboQuant*
(ICLR 2026), Algorithm 1 (MSE-optimal branch).

## Files

| file | role |
|---|---|
| `tq_codebook.m` | Lloyd-Max centroids for the rotated-coordinate distribution, cached per bit-width; also returns the predicted per-coordinate distortion |
| `tq_fwht.m` | orthonormal fast Walsh-Hadamard transform |
| `tq_signs.m` | reproducible random sign flips (only the seed is stored) |
| `tq_compress.m` | block-wise quantization of a real vector |
| `tq_decompress.m` | reconstruction |
| `quantize_mps_turboquant.m` | applies the above to every MPS core |
| `mps_logZ.m` | log Z by transfer-matrix contraction, no canonicalization |
| `mps_nll.m` | NLL of a non-normalized MPS |
| `quantize_mps_uniform.m` | baseline: mode-2 2x2 rotation + max-abs uniform quantization |
| `tq_selftest.m` | checks measured distortion against theory |
| `main_mps_quant_turboquant.m` | full experiment |

## Usage

```matlab
addpath('turboquant');
tq_selftest;                                    % sanity check
[mps, stats] = quantize_mps_turboquant(mps, 6); % 6 bits per parameter
nll = mps_nll(mps);                             % handles Z ~= 1 itself
```

`stats.bits_per_par` includes the fp32 block scales (32 bits per 1024 coordinates,
i.e. 0.031 bits/parameter) and the stored seed.

## Why not re-canonicalize

`left_canonical()` costs one SVD per bond (784 SVDs of up to 400x400) and drives
every bond dimension to `max_bondim`, so calling it once per bit-width makes the
sweep take tens of minutes and hundreds of MB per copy. `mps_nll` instead computes
`log Z` directly in O(N D^3) and leaves the cores untouched. Canonicalization is
only needed for sampling, and `generate_sample_half` already does it internally.
