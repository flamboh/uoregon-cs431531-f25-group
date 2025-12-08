#pragma once

#include "../tensor_storage/blco.h"
#include "tmm_nd.cuh"

template <typename T, typename S>
T* tmm_3d_cuda(const Blco_Tensor<T, S>& sparse_tensor,
               int mode,
               int block_size = 256,
               bool log_timings = true);

template <typename T, typename S>
T* tucker_compute_core_3d_cuda(const Blco_Tensor<T, S>& sparse_tensor,
                               int block_size = 256);

template <typename T, typename S>
T* contract_n_minus_one_modes_3d_cuda(const Blco_Tensor<T, S>& sparse_tensor,
                                      int block_size,
                                      int mode);
