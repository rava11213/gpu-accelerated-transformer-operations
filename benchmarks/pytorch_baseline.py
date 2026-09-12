#!/usr/bin/env python3
"""PyTorch eager baselines using CUDA events (kernel time, not allocation time)."""
import argparse
import torch


def timed(fn, warmup, iterations):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iterations): fn()
    end.record(); end.synchronize()
    return start.elapsed_time(end) / iterations


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=512); p.add_argument("--n", type=int, default=512)
    p.add_argument("--k", type=int, default=512); p.add_argument("--rows", type=int, default=1024)
    p.add_argument("--cols", type=int, default=768); p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=100); a = p.parse_args()
    if not torch.cuda.is_available(): raise SystemExit("CUDA-enabled PyTorch and an NVIDIA GPU are required")
    d = "cuda"; x = torch.randn(a.rows, a.cols, device=d); gamma = torch.randn(a.cols, device=d); beta = torch.randn(a.cols, device=d)
    left = torch.randn(a.m, a.k, device=d); right = torch.randn(a.k, a.n, device=d)
    cases = {"matmul": lambda: left @ right, "softmax": lambda: torch.softmax(x, dim=-1),
             "layernorm": lambda: torch.nn.functional.layer_norm(x, (a.cols,), gamma, beta)}
    print(f"Device: {torch.cuda.get_device_name()}")
    for name, fn in cases.items(): print(f"{name:<12} PyTorch {timed(fn, a.warmup, a.iterations):.4f} ms")


if __name__ == "__main__": main()

