/*!
**************************************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from
*https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/

#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA
Tensor ms_deform_attn_cuda_forward(const Tensor &value,
                                   const Tensor &spatial_shapes,
                                   const Tensor &level_start_index,
                                   const Tensor &sampling_loc,
                                   const Tensor &attn_weight,
                                   const int im2col_step);

void ms_deform_attn_cuda_backward(
    const Tensor &value, const Tensor &spatial_shapes,
    const Tensor &level_start_index, const Tensor &sampling_loc,
    const Tensor &attn_weight, const Tensor &grad_output, Tensor &grad_value,
    Tensor &grad_sampling_loc, Tensor &grad_attn_weight, const int im2col_step);

#endif

Tensor ms_deform_attn_forward(const Tensor &value, const Tensor &spatial_shapes,
                              const Tensor &level_start_index,
                              const Tensor &sampling_loc,
                              const Tensor &attn_weight,
                              const int im2col_step) {
  if (value.type().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(value)
    CHECK_CUDA_INPUT(spatial_shapes)
    CHECK_CUDA_INPUT(level_start_index)
    CHECK_CUDA_INPUT(sampling_loc)
    CHECK_CUDA_INPUT(attn_weight)
    return ms_deform_attn_cuda_forward(value, spatial_shapes, level_start_index,
                                       sampling_loc, attn_weight, im2col_step);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

void ms_deform_attn_backward(const Tensor &value, const Tensor &spatial_shapes,
                             const Tensor &level_start_index,
                             const Tensor &sampling_loc,
                             const Tensor &attn_weight,
                             const Tensor &grad_output, Tensor &grad_value,
                             Tensor &grad_sampling_loc,
                             Tensor &grad_attn_weight, const int im2col_step) {
  if (value.type().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(value)
    CHECK_CUDA_INPUT(spatial_shapes)
    CHECK_CUDA_INPUT(level_start_index)
    CHECK_CUDA_INPUT(sampling_loc)
    CHECK_CUDA_INPUT(attn_weight)
    CHECK_CUDA_INPUT(grad_output)
    CHECK_CUDA_INPUT(grad_value)
    CHECK_CUDA_INPUT(grad_sampling_loc)
    CHECK_CUDA_INPUT(grad_attn_weight)
    ms_deform_attn_cuda_backward(value, spatial_shapes, level_start_index,
                                 sampling_loc, attn_weight, grad_output,
                                 grad_value, grad_sampling_loc,
                                 grad_attn_weight, im2col_step);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  } else {
    AT_ERROR("Not implemented on the CPU");
  }
}