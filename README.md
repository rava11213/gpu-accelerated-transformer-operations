# GPU-Accelerated Transformer Operations

Educational CUDA implementations of three transformer building blocks:

- tiled, row-major FP32 matrix multiplication;
- numerically stable row-wise softmax;
- row-wise layer normalization with learnable scale and bias.

The executable validates GPU output against plain CPU references and reports CUDA-event kernel latency plus CPU wall-clock latency. A separate script measures the equivalent PyTorch CUDA operations. The custom kernels favor readable optimization techniques—shared-memory tiling, coalesced access, warp shuffles, and fused reductions—rather than claiming to outperform production cuBLAS/cuDNN kernels.

## Requirements

- Linux or Windows/WSL with an NVIDIA GPU
- CUDA Toolkit 12.x (CUDA 11.8 also works)
- GNU Make and a C++17 host compiler
- Optional: Python 3 and CUDA-enabled PyTorch for framework baselines
- Optional: NVIDIA Nsight Systems (`nsys`) and Nsight Compute (`ncu`)

macOS cannot compile or run modern NVIDIA CUDA code.

## Build and validate

```bash
make CUDA_ARCH=89       # 80=A100, 86=RTX 30, 89=RTX 40, 90=H100
make test
```

With a recent toolkit, `make` alone uses `-arch=native`. Run custom shapes:

```bash
./build/transformer_ops --op all --m 1024 --n 1024 --k 1024 \
  --rows 4096 --cols 768 --warmup 10 --iterations 100 --check
```

`--check` computes CPU references once. It is intentionally omitted from timing loops. Use smaller matrix dimensions for the CPU matmul check if validation takes too long.

## PyTorch baseline

```bash
python3 benchmarks/pytorch_baseline.py --m 1024 --n 1024 --k 1024 \
  --rows 4096 --cols 768 --iterations 100
```

Compare only runs on the same GPU, power state, precision, shapes, warmup, and software stack. The output reports kernel execution time; it excludes allocation and host/device transfer time.

## Profile

```bash
make profile
ncu --set full --kernel-name regex:'(matmul|softmax|layernorm)_kernel' \
  ./build/transformer_ops --op all --iterations 20
```

The Nsight Systems report is written under `results/`. In Nsight Compute, inspect global-memory load/store efficiency, achieved occupancy, shared-memory usage, warp stalls, and FLOP throughput. Profile without `--check` so CPU reference work does not clutter the trace.

## Notes on performance claims

Speedup depends heavily on hardware and shape. Do not hard-code a “20×” claim: capture the executable output and Nsight report on the target GPU, then report the exact configuration. PyTorch matmul normally dispatches to highly tuned cuBLAS and is expected to beat this teaching kernel for many shapes.

