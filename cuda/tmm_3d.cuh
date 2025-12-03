#pragma once

#include "../tensor_storage/blco.h"

template <typename T, typename S>
T* tmm_3d_cuda(const Blco_Tensor<T, S>& sparse_tensor,
               int mode,
               int block_size = 256);

template <typename T, typename S>
T* tucker_compute_core_3d_cuda(const Blco_Tensor<T, S>& sparse_tensor,
                               int block_size = 256);
