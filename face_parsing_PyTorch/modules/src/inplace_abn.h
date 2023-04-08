#pragma once

#include <ATen/ATen.h>

#include <vector>

std::vector<at::Tensor> mean_var_cpu(at::Tensor x);
std::vector<at::Tensor> mean_var_cuda(at::Tensor x);
std::vector<at::Tensor> mean_var_cuda_h(at::Tensor x);

at::Tensor forward_cpu(at::Tensor x, at::Tensor mean, at::Tensor var, at::Tensor weight, at::Tensor bias,
                       bool affine, float eps);
at::Tensor forward_cuda(at::Tensor x, at::Tensor mean, at::Tensor var, at::Tensor weight, at::Tensor bias,
                        bool affine, float eps);
at::Tensor forward_cuda_h(at::Tensor x, at::Tensor mean, at::Tensor var, at::Tensor weight, at::Tensor bias,
                          bool affine, float eps);

std::vector<at::Tensor> edz_eydz_cpu(at::Tensor z, at::Tensor dz, at::Tensor weight, at::Tensor bias,
                                     bool affine, float eps);
std::vector<at::Tensor> edz_eydz_cuda(at::Tensor z, at::Tensor dz, at::Tensor weight, at::Tensor bias,
                                      bool affine, float eps);
std::vector<at::Tensor> edz_eydz_cuda_h(at::Tensor z, at::Tensor dz, at::Tensor weight, at::Tensor bias,
                                        bool affine, float eps);

at::Tensor backward_cpu(at::Tensor z, at::Tensor dz, at::Tensor var, at::Tensor weight, at::Tensor bias,
                                     at::Tensor edz, at::Tensor eydz, bool affine, float eps);
at::Tensor backward_cuda(at::Tensor z, at::Tensor dz, at::Tensor var, at::Tensor weight, at::Tensor bias,
                                      at::Tensor edz, at::Tensor eydz, bool affine, float eps);
at::Tensor backward_cuda_h(at::Tensor z, at::Tensor dz, at::Tensor var, at::Tensor weight, at::Tensor bias,
                                        at::Tensor edz, at::Tensor eydz, bool affine, float eps);

void leaky_relu_backward_cpu(at::Tensor z, at::Tensor dz, float slope);
void leaky_relu_backward_cuda(at::Tensor z, at::Tensor dz, float slope);
void leaky_relu_backward_cuda_h(at::Tensor z, at::Tensor dz, float slope);

void elu_backward_cpu(at::Tensor z, at::Tensor dz);
void elu_backward_cuda(at::Tensor z, at::Tensor dz);

static void get_dims(at::Tensor x, int64_t& num, int64_t& chn, int64_t& sp) {
  num = x.size(0);
  chn = x.size(1);
  sp = 1;
  for (int64_t i = 2; i < x.ndimension(); ++i)
    sp *= x.size(i);
}

/*
 * Specialized CUDA reduction functions for BN
 */
#ifdef __CUDACC__

#include "utils/cuda.cuh"

template <typename T, typename Op>
__device__ T reduce(Op op, int plane, int N, int S) {
  T sum = (T)0;
  for (int batch = 0; batch < N; ++batch) {
    for (int x = threadIdx.x; x < S; x += blockDim.x) {
      sum += op(batch, plane, x);
    }
  }

  // sum over NumThreads within a warp
  sum = warpSum(sum);

  // 'transpose', and reduce within warp again
  __shared__ T shared[32];
  __syncthreads();
  if (threadIdx.x % WARP_SIZE == 0) {
    shared[threadIdx.x / WARP_SIZE] = sum;
  }
  if (threadIdx.x >= blockDim.x / WARP_SIZE && threadIdx.x < WARP_SIZE) {
    // zero out the other entries in shared
    shared[threadIdx.x] = (T)0;
  }
  __syncthreads();
  if (threadIdx.x / WARP_SIZE == 0) {
    sum = warpSum(shared[threadIdx.x]);
    if (threadIdx.x == 0) {
      shared[0] = sum;
    }
  }
  __syncthreads();

  // Everyone picks it up, should be broadcast into the whole gradInput
  return shared[0];
}
#endif
