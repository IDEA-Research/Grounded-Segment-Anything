#pragma once

#include <ATen/ATen.h>

/*
 * Functions to share code between CPU and GPU
 */

#ifdef __CUDACC__
// CUDA versions

#define HOST_DEVICE __host__ __device__
#define INLINE_HOST_DEVICE __host__ __device__ inline
#define FLOOR(x) floor(x)

#if __CUDA_ARCH__ >= 600
// Recent compute capabilities have block-level atomicAdd for all data types, so we use that
#define ACCUM(x,y) atomicAdd_block(&(x),(y))
#else
// Older architectures don't have block-level atomicAdd, nor atomicAdd for doubles, so we defer to atomicAdd for float
// and use the known atomicCAS-based implementation for double
template<typename data_t>
__device__ inline data_t atomic_add(data_t *address, data_t val) {
  return atomicAdd(address, val);
}

template<>
__device__ inline double atomic_add(double *address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

#define ACCUM(x,y) atomic_add(&(x),(y))
#endif // #if __CUDA_ARCH__ >= 600

#else
// CPU versions

#define HOST_DEVICE
#define INLINE_HOST_DEVICE inline
#define FLOOR(x) std::floor(x)
#define ACCUM(x,y) (x) += (y)

#endif // #ifdef __CUDACC__