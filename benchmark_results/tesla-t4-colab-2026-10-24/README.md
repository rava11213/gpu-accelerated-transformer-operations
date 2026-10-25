# Tesla T4 benchmark — 2026-10-24

These measurements were collected in a Google Colab T4 runtime from commit
`89f9846`. The shape sweep uses 10 warmup iterations and 100 timed iterations;
the matched framework comparison uses 20 warmup iterations and 200 timed
iterations. Timings use CUDA events and exclude allocation and host/device
transfer.

## Environment

- GPU: NVIDIA Tesla T4, 15,360 MiB
- Driver: 580.82.07
- CUDA compiler: CUDA 13.0 (`sm_75`)
- Build flags: `-O3 -std=c++17 --use_fast_math -lineinfo`

## Headline comparisons

At a 4096 x 768 rowwise shape:

| Operation | Custom CUDA | PyTorch | Custom / PyTorch |
|---|---:|---:|---:|
| Softmax | 0.1788 ms | 0.1160 ms | 1.54x slower |
| LayerNorm | 0.1620 ms | 0.1096 ms | 1.48x slower |
| Residual + LayerNorm | 0.2168 ms | 0.2662 ms | **1.23x faster** |

For 1024 x 1024 x 1024 matrix multiplication, 32 x 32 shared-memory tiling
with 2 x 2 register tiles reduced kernel time from 4.2320 ms to 1.5170 ms, a
**2.79x speedup** over the naive custom kernel. PyTorch/cuBLAS remained faster
at 0.6436 ms in the separately timed framework comparison.

Every tested kernel passed its CPU-reference correctness check. See
`sweep.csv` for the full shape sweep and `framework_comparison.csv` for matched
custom/PyTorch measurements.

The CPU reference is intentionally scalar and exists for validation, not as an
optimized CPU-library baseline. CPU-vs-GPU ratios should therefore not be used
as the headline performance claim.

