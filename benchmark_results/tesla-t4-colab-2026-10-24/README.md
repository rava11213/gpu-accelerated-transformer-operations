# Tesla T4 benchmark — 2026-10-24

These measurements were collected in a Google Colab T4 runtime from commit
`5e5ecd8`. All custom CUDA results use 10 warmup iterations and 100 timed
iterations unless the sweep script specifies otherwise. Timings use CUDA events
and exclude allocation and host/device transfer.

## Environment

- GPU: NVIDIA Tesla T4, 15,360 MiB
- Driver: 580.82.07
- CUDA compiler: CUDA 13.0 (`sm_75`)
- Build flags: `-O3 -std=c++17 --use_fast_math -lineinfo`

## Headline comparisons

At a 4096 x 768 rowwise shape:

| Operation | Custom CUDA | PyTorch | Custom / PyTorch |
|---|---:|---:|---:|
| Softmax | 0.1577 ms | 0.1168 ms | 1.35x slower |
| LayerNorm | 0.1366 ms | 0.1296 ms | 1.05x slower |
| Residual + LayerNorm | 0.2093 ms | 0.2698 ms | **1.29x faster** |

For 1024 x 1024 x 1024 matrix multiplication, shared-memory tiling reduced
kernel time from 4.2204 ms to 2.9959 ms, a **1.41x speedup** over the naive
custom kernel. PyTorch/cuBLAS remained faster at 0.6130 ms.

Every tested kernel passed its CPU-reference correctness check. See
`sweep.csv` for the full shape sweep and `framework_comparison.csv` for matched
custom/PyTorch measurements.

The CPU reference is intentionally scalar and exists for validation, not as an
optimized CPU-library baseline. CPU-vs-GPU ratios should therefore not be used
as the headline performance claim.

