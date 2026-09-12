#pragma once

#include <cuda_runtime.h>
#include <cfloat>

namespace transformer_ops {

constexpr int TILE = 16;
constexpr int BLOCK = 256;

// C[M,N] = A[M,K] * B[K,N], row-major. Shared-memory tiling cuts
// redundant global reads; adjacent threads read adjacent elements.
__global__ void matmul_kernel(const float* __restrict__ a,
                              const float* __restrict__ b,
                              float* __restrict__ c, int m, int n, int k) {
  __shared__ float as[TILE][TILE];
  __shared__ float bs[TILE][TILE + 1];
  const int row = blockIdx.y * TILE + threadIdx.y;
  const int col = blockIdx.x * TILE + threadIdx.x;
  float sum = 0.0f;

  for (int tile = 0; tile < (k + TILE - 1) / TILE; ++tile) {
    const int ak = tile * TILE + threadIdx.x;
    const int bk = tile * TILE + threadIdx.y;
    as[threadIdx.y][threadIdx.x] = (row < m && ak < k) ? a[row * k + ak] : 0.0f;
    bs[threadIdx.y][threadIdx.x] = (bk < k && col < n) ? b[bk * n + col] : 0.0f;
    __syncthreads();
#pragma unroll
    for (int i = 0; i < TILE; ++i) sum = fmaf(as[threadIdx.y][i], bs[i][threadIdx.x], sum);
    __syncthreads();
  }
  if (row < m && col < n) c[row * n + col] = sum;
}

__device__ __forceinline__ float warp_sum(float v) {
  for (int offset = 16; offset; offset >>= 1) v += __shfl_down_sync(0xffffffff, v, offset);
  return v;
}

__device__ __forceinline__ float warp_max(float v) {
  for (int offset = 16; offset; offset >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
  return v;
}

// One block per row. The two reductions stay on-chip and the final reads and
// writes are coalesced for the common case of wide transformer rows.
__global__ void softmax_kernel(const float* __restrict__ x,
                               float* __restrict__ y, int rows, int cols) {
  __shared__ float warp_values[32];
  const int row = blockIdx.x;
  if (row >= rows) return;
  float local_max = -FLT_MAX;
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    local_max = fmaxf(local_max, x[row * cols + col]);
  local_max = warp_max(local_max);
  if ((threadIdx.x & 31) == 0) warp_values[threadIdx.x >> 5] = local_max;
  __syncthreads();
  float row_max = (threadIdx.x < (blockDim.x + 31) / 32) ? warp_values[threadIdx.x] : -FLT_MAX;
  if (threadIdx.x < 32) row_max = warp_max(row_max);
  if (threadIdx.x == 0) warp_values[0] = row_max;
  __syncthreads();
  row_max = warp_values[0];

  float local_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    local_sum += expf(x[row * cols + col] - row_max);
  local_sum = warp_sum(local_sum);
  if ((threadIdx.x & 31) == 0) warp_values[threadIdx.x >> 5] = local_sum;
  __syncthreads();
  float row_sum = (threadIdx.x < (blockDim.x + 31) / 32) ? warp_values[threadIdx.x] : 0.0f;
  if (threadIdx.x < 32) row_sum = warp_sum(row_sum);
  if (threadIdx.x == 0) warp_values[0] = row_sum;
  __syncthreads();
  row_sum = warp_values[0];
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    y[row * cols + col] = expf(x[row * cols + col] - row_max) / row_sum;
}

__global__ void layernorm_kernel(const float* __restrict__ x,
                                 const float* __restrict__ gamma,
                                 const float* __restrict__ beta,
                                 float* __restrict__ y, int rows, int cols,
                                 float epsilon) {
  __shared__ float warp_values[32];
  __shared__ float mean_shared, inv_std_shared;
  const int row = blockIdx.x;
  if (row >= rows) return;
  float sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) sum += x[row * cols + col];
  sum = warp_sum(sum);
  if ((threadIdx.x & 31) == 0) warp_values[threadIdx.x >> 5] = sum;
  __syncthreads();
  float total = (threadIdx.x < (blockDim.x + 31) / 32) ? warp_values[threadIdx.x] : 0.0f;
  if (threadIdx.x < 32) total = warp_sum(total);
  if (threadIdx.x == 0) mean_shared = total / cols;
  __syncthreads();

  const float mean = mean_shared;
  float sq_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float d = x[row * cols + col] - mean;
    sq_sum = fmaf(d, d, sq_sum);
  }
  sq_sum = warp_sum(sq_sum);
  if ((threadIdx.x & 31) == 0) warp_values[threadIdx.x >> 5] = sq_sum;
  __syncthreads();
  float sq_total = (threadIdx.x < (blockDim.x + 31) / 32) ? warp_values[threadIdx.x] : 0.0f;
  if (threadIdx.x < 32) sq_total = warp_sum(sq_total);
  if (threadIdx.x == 0) inv_std_shared = rsqrtf(sq_total / cols + epsilon);
  __syncthreads();
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    y[row * cols + col] = (x[row * cols + col] - mean) * inv_std_shared * gamma[col] + beta[col];
}

inline void launch_matmul(const float* a, const float* b, float* c, int m, int n, int k,
                          cudaStream_t stream = nullptr) {
  dim3 block(TILE, TILE), grid((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);
  matmul_kernel<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
}

inline void launch_softmax(const float* x, float* y, int rows, int cols,
                           cudaStream_t stream = nullptr) {
  softmax_kernel<<<rows, BLOCK, 0, stream>>>(x, y, rows, cols);
}

inline void launch_layernorm(const float* x, const float* gamma, const float* beta,
                             float* y, int rows, int cols, float epsilon = 1e-5f,
                             cudaStream_t stream = nullptr) {
  layernorm_kernel<<<rows, BLOCK, 0, stream>>>(x, gamma, beta, y, rows, cols, epsilon);
}

}  // namespace transformer_ops

