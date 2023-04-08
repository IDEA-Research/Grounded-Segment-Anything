#pragma once

/*
 * General settings and functions
 */
const int WARP_SIZE = 32;
const int MAX_BLOCK_SIZE = 1024;

static int getNumThreads(int nElem) {
  int threadSizes[6] = {32, 64, 128, 256, 512, MAX_BLOCK_SIZE};
  for (int i = 0; i < 6; ++i) {
    if (nElem <= threadSizes[i]) {
      return threadSizes[i];
    }
  }
  return MAX_BLOCK_SIZE;
}

/*
 * Reduction utilities
 */
template <typename T>
__device__ __forceinline__ T WARP_SHFL_XOR(T value, int laneMask, int width = warpSize,
                                           unsigned int mask = 0xffffffff) {
#if CUDART_VERSION >= 9000
  return __shfl_xor_sync(mask, value, laneMask, width);
#else
  return __shfl_xor(value, laneMask, width);
#endif
}

__device__ __forceinline__ int getMSB(int val) { return 31 - __clz(val); }

template<typename T>
struct Pair {
  T v1, v2;
  __device__ Pair() {}
  __device__ Pair(T _v1, T _v2) : v1(_v1), v2(_v2) {}
  __device__ Pair(T v) : v1(v), v2(v) {}
  __device__ Pair(int v) : v1(v), v2(v) {}
  __device__ Pair &operator+=(const Pair<T> &a) {
    v1 += a.v1;
    v2 += a.v2;
    return *this;
  }
};

template<typename T>
static __device__ __forceinline__ T warpSum(T val) {
#if __CUDA_ARCH__ >= 300
  for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    val += WARP_SHFL_XOR(val, 1 << i, WARP_SIZE);
  }
#else
  __shared__ T values[MAX_BLOCK_SIZE];
  values[threadIdx.x] = val;
  __threadfence_block();
  const int base = (threadIdx.x / WARP_SIZE) * WARP_SIZE;
  for (int i = 1; i < WARP_SIZE; i++) {
    val += values[base + ((i + threadIdx.x) % WARP_SIZE)];
  }
#endif
  return val;
}

template<typename T>
static __device__ __forceinline__ Pair<T> warpSum(Pair<T> value) {
  value.v1 = warpSum(value.v1);
  value.v2 = warpSum(value.v2);
  return value;
}