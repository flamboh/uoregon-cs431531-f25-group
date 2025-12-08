#pragma once

#include "../tensor_storage/blco.h"

// Maximum tensor modes supported by the CUDA ND kernels. Raise if higher ranks are needed.
constexpr int kMaxTensorModes = 16;

template <typename T, typename S>
T* tmm_nd_cuda(const Blco_Tensor<T, S>& sparse_tensor,
               int mode,
               int block_size = 256,
               bool log_timings = true);

template <typename T, typename S>
T* tucker_compute_core_nd_cuda(const Blco_Tensor<T, S>& sparse_tensor,
                               int block_size = 256);

template <typename T, typename S>
T* contract_n_minus_one_modes_nd_cuda(const Blco_Tensor<T, S>& sparse_tensor,
                                      int block_size,
                                      int mode);
