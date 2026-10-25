#pragma once

#include <cuda_runtime.h>
#include <cfloat>

namespace transformer_ops {

constexpr int TILE = 16;
constexpr int MATMUL_TILE = 32;
constexpr int BLOCK = 256;

// Deliberately simple baseline: one thread computes one output and rereads
// both operands from global memory. It makes the benefit of tiling measurable.
__global__ void matmul_naive_kernel(const float* __restrict__ a,
                                    const float* __restrict__ b,
                                    float* __restrict__ c, int m, int n, int k) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || col >= n) return;
  float sum = 0.0f;
  for (int p = 0; p < k; ++p) sum = fmaf(a[row * k + p], b[p * n + col], sum);
  c[row * n + col] = sum;
}

// C[M,N] = A[M,K] * B[K,N], row-major. Shared-memory tiling cuts
// redundant global reads; adjacent threads read adjacent elements.
__global__ void matmul_kernel(const float* __restrict__ a,
                              const float* __restrict__ b,
                              float* __restrict__ c, int m, int n, int k) {
  __shared__ float as[MATMUL_TILE][MATMUL_TILE];
  __shared__ float bs[MATMUL_TILE][MATMUL_TILE + 1];
  const int local_row = threadIdx.y * 2;
  const int local_col = threadIdx.x * 2;
  const int row = blockIdx.y * MATMUL_TILE + local_row;
  const int col = blockIdx.x * MATMUL_TILE + local_col;
  float sum00 = 0.0f, sum01 = 0.0f, sum10 = 0.0f, sum11 = 0.0f;

  for (int tile = 0; tile < (k + MATMUL_TILE - 1) / MATMUL_TILE; ++tile) {
    const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
#pragma unroll
    for (int index = thread_id; index < MATMUL_TILE * MATMUL_TILE;
         index += TILE * TILE) {
      const int shared_row = index / MATMUL_TILE;
      const int shared_col = index % MATMUL_TILE;
      const int global_k_col = tile * MATMUL_TILE + shared_col;
      const int global_k_row = tile * MATMUL_TILE + shared_row;
      as[shared_row][shared_col] =
          (blockIdx.y * MATMUL_TILE + shared_row < m && global_k_col < k)
              ? a[(blockIdx.y * MATMUL_TILE + shared_row) * k + global_k_col] : 0.0f;
      bs[shared_row][shared_col] =
          (global_k_row < k && blockIdx.x * MATMUL_TILE + shared_col < n)
              ? b[global_k_row * n + blockIdx.x * MATMUL_TILE + shared_col] : 0.0f;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < MATMUL_TILE; ++i) {
      const float a0 = as[local_row][i], a1 = as[local_row + 1][i];
      const float b0 = bs[i][local_col], b1 = bs[i][local_col + 1];
      sum00 = fmaf(a0, b0, sum00);
      sum01 = fmaf(a0, b1, sum01);
      sum10 = fmaf(a1, b0, sum10);
      sum11 = fmaf(a1, b1, sum11);
    }
    __syncthreads();
  }
  if (row < m && col < n) c[row * n + col] = sum00;
  if (row < m && col + 1 < n) c[row * n + col + 1] = sum01;
  if (row + 1 < m && col < n) c[(row + 1) * n + col] = sum10;
  if (row + 1 < m && col + 1 < n) c[(row + 1) * n + col + 1] = sum11;
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
  extern __shared__ float exponentials[];
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
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = expf(x[row * cols + col] - row_max);
    exponentials[col] = value;
    local_sum += value;
  }
  local_sum = warp_sum(local_sum);
  if ((threadIdx.x & 31) == 0) warp_values[threadIdx.x >> 5] = local_sum;
  __syncthreads();
  float row_sum = (threadIdx.x < (blockDim.x + 31) / 32) ? warp_values[threadIdx.x] : 0.0f;
  if (threadIdx.x < 32) row_sum = warp_sum(row_sum);
  if (threadIdx.x == 0) warp_values[0] = row_sum;
  __syncthreads();
  row_sum = warp_values[0];
  const float inv_sum = 1.0f / row_sum;
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    y[row * cols + col] = exponentials[col] * inv_sum;
}

__global__ void layernorm_kernel(const float* __restrict__ x,
                                 const float* __restrict__ gamma,
                                 const float* __restrict__ beta,
                                 float* __restrict__ y, int rows, int cols,
                                 float epsilon) {
  __shared__ float warp_values[32], warp_squares[32];
  __shared__ float mean_shared, inv_std_shared;
  const int row = blockIdx.x;
  if (row >= rows) return;
  float sum = 0.0f, square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = x[row * cols + col];
    sum += value;
    square_sum = fmaf(value, value, square_sum);
  }
  sum = warp_sum(sum);
  square_sum = warp_sum(square_sum);
  if ((threadIdx.x & 31) == 0) {
    warp_values[threadIdx.x >> 5] = sum;
    warp_squares[threadIdx.x >> 5] = square_sum;
  }
  __syncthreads();
  float total = (threadIdx.x < (blockDim.x + 31) / 32) ? warp_values[threadIdx.x] : 0.0f;
  float square_total = (threadIdx.x < (blockDim.x + 31) / 32) ? warp_squares[threadIdx.x] : 0.0f;
  if (threadIdx.x < 32) {
    total = warp_sum(total);
    square_total = warp_sum(square_total);
  }
  if (threadIdx.x == 0) {
    const float mean = total / cols;
    mean_shared = mean;
    inv_std_shared = rsqrtf(fmaxf(square_total / cols - mean * mean, 0.0f) + epsilon);
  }
  __syncthreads();
  const float mean = mean_shared;
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    y[row * cols + col] = (x[row * cols + col] - mean) * inv_std_shared * gamma[col] + beta[col];
}

// Fuses the residual addition and layer normalization used around transformer
// sublayers. The sum is written once for the residual stream while statistics
// and normalization are computed without an intermediate add kernel.
__global__ void residual_layernorm_kernel(const float* __restrict__ x,
                                          const float* __restrict__ residual,
                                          const float* __restrict__ gamma,
                                          const float* __restrict__ beta,
                                          float* __restrict__ y,
                                          float* __restrict__ residual_out,
                                          int rows, int cols, float epsilon) {
  __shared__ float warp_values[32], warp_squares[32];
  __shared__ float mean_shared, inv_std_shared;
  const int row = blockIdx.x;
  if (row >= rows) return;

  float sum = 0.0f, square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int i = row * cols + col;
    const float value = x[i] + residual[i];
    residual_out[i] = value;
    sum += value;
    square_sum = fmaf(value, value, square_sum);
  }
  sum = warp_sum(sum);
  square_sum = warp_sum(square_sum);
  if ((threadIdx.x & 31) == 0) {
    warp_values[threadIdx.x >> 5] = sum;
    warp_squares[threadIdx.x >> 5] = square_sum;
  }
  __syncthreads();
  float total = threadIdx.x < (blockDim.x + 31) / 32 ? warp_values[threadIdx.x] : 0.0f;
  float square_total = threadIdx.x < (blockDim.x + 31) / 32 ? warp_squares[threadIdx.x] : 0.0f;
  if (threadIdx.x < 32) {
    total = warp_sum(total);
    square_total = warp_sum(square_total);
  }
  if (threadIdx.x == 0) {
    const float mean = total / cols;
    mean_shared = mean;
    inv_std_shared = rsqrtf(fmaxf(square_total / cols - mean * mean, 0.0f) + epsilon);
  }
  __syncthreads();
  const float mean = mean_shared;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int i = row * cols + col;
    y[i] = (residual_out[i] - mean) * inv_std_shared * gamma[col] + beta[col];
  }
}

inline void launch_matmul(const float* a, const float* b, float* c, int m, int n, int k,
                          cudaStream_t stream = nullptr) {
  dim3 block(TILE, TILE), grid((n + MATMUL_TILE - 1) / MATMUL_TILE,
                              (m + MATMUL_TILE - 1) / MATMUL_TILE);
  matmul_kernel<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
}

inline void launch_matmul_naive(const float* a, const float* b, float* c,
                                int m, int n, int k, cudaStream_t stream = nullptr) {
  dim3 block(TILE, TILE), grid((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);
  matmul_naive_kernel<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
}

inline void launch_softmax(const float* x, float* y, int rows, int cols,
                           cudaStream_t stream = nullptr) {
  softmax_kernel<<<rows, BLOCK, size_t(cols) * sizeof(float), stream>>>(x, y, rows, cols);
}

inline void launch_layernorm(const float* x, const float* gamma, const float* beta,
                             float* y, int rows, int cols, float epsilon = 1e-5f,
                             cudaStream_t stream = nullptr) {
  layernorm_kernel<<<rows, BLOCK, 0, stream>>>(x, gamma, beta, y, rows, cols, epsilon);
}


inline void launch_residual_layernorm(const float* x, const float* residual,
                                      const float* gamma, const float* beta,
                                      float* y, float* residual_out, int rows, int cols,
                                      float epsilon = 1e-5f,
                                      cudaStream_t stream = nullptr) {
  residual_layernorm_kernel<<<rows, BLOCK, 0, stream>>>(
      x, residual, gamma, beta, y, residual_out, rows, cols, epsilon);
}

}  // namespace transformer_ops
