#pragma once

#include <ATen/ATen.h>

// Define AT_CHECK for old version of ATen where the same function was called AT_ASSERT
#ifndef AT_CHECK
#define AT_CHECK AT_ASSERT
#endif

#define CHECK_CUDA(x) AT_CHECK((x).type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) AT_CHECK(!(x).type().is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) AT_CHECK((x).is_contiguous(), #x " must be contiguous")

#define CHECK_CUDA_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_CPU_INPUT(x) CHECK_CPU(x); CHECK_CONTIGUOUS(x)