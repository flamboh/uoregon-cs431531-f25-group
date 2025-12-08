#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "cuda_utils.cuh"
#include "tmm_nd.cuh"
#include "../tensor_storage/tensor_utils.h"

namespace {

template <typename T>
struct dependent_false : std::false_type {};

template <typename T>
__device__ inline void atomicAddTyped(T* addr, T val) {
    if constexpr (std::is_same_v<T, double>) {
        atomicAdd(addr, val);
    } else if constexpr (std::is_same_v<T, float>) {
        atomicAdd(addr, val);
    } else if constexpr (std::is_same_v<T, int>) {
        atomicAdd(addr, static_cast<int>(val));
    } else if constexpr (std::is_same_v<T, long long> || std::is_same_v<T, long>) {
        atomicAdd(reinterpret_cast<unsigned long long*>(addr),
                  static_cast<unsigned long long>(val));
    } else {
        static_assert(dependent_false<T>::value, "Unsupported type for atomicAddTyped");
    }
}

uint64_t safe_product(const std::vector<int>& dims) {
    if (dims.empty()) {
        throw std::invalid_argument("Tensor must have at least one dimension");
    }
    uint64_t result = 1;
    for (int dim : dims) {
        if (dim <= 0) {
            throw std::invalid_argument("Tensor dimensions must be positive");
        }
        if (result > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(dim)) {
            throw std::overflow_error("Tensor size exceeds 64-bit capacity");
        }
        result *= static_cast<uint64_t>(dim);
    }
    return result;
}

uint64_t safe_int_pow(uint64_t base, int exp) {
    if (exp < 0) {
        throw std::invalid_argument("Exponent must be non-negative");
    }
    uint64_t result = 1;
    for (int i = 0; i < exp; ++i) {
        if (base != 0 && result > std::numeric_limits<uint64_t>::max() / base) {
            throw std::overflow_error("Rank power exceeds 64-bit capacity");
        }
        result *= base;
    }
    return result;
}

std::vector<int> compute_bit_widths(const std::vector<int>& dims) {
    std::vector<int> widths(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        int width = 0;
        int value = dims[i] - 1;
        while (value > 0) {
            ++width;
            value >>= 1;
        }
        widths[i] = width;
    }
    return widths;
}

std::vector<uint64_t> build_output_strides(const std::vector<int>& dims) {
    const int rank_n = static_cast<int>(dims.size());
    std::vector<uint64_t> strides(rank_n, 1);
    for (int i = rank_n - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * static_cast<uint64_t>(dims[i + 1]);
    }
    return strides;
}

void validate_rank_count(int rank_n) {
    if (rank_n <= 0) {
        throw std::invalid_argument("Tensor rank must be positive");
    }
    if (rank_n > kMaxTensorModes) {
        throw std::runtime_error("Tensor rank exceeds kMaxTensorModes; raise the constant to continue");
    }
}

template <typename T>
std::vector<T*> copy_factor_matrices_to_device(const std::vector<T*>& fmats,
                                               const std::vector<int>& dims,
                                               int rank) {
    const int rank_n = static_cast<int>(dims.size());
    if (static_cast<int>(fmats.size()) < rank_n) {
        throw std::runtime_error("Factor matrix list is smaller than tensor rank");
    }

    std::vector<T*> device_ptrs(rank_n, nullptr);
    try {
        for (int i = 0; i < rank_n; ++i) {
            const size_t elems = static_cast<size_t>(dims[i]) * rank;
            CUDA_CHECK(cudaMalloc(&device_ptrs[i], elems * sizeof(T)));
            CUDA_CHECK(cudaMemcpy(device_ptrs[i],
                                  fmats[i],
                                  elems * sizeof(T),
                                  cudaMemcpyHostToDevice));
        }
    } catch (...) {
        for (T*& ptr : device_ptrs) {
            if (ptr) {
                cudaFree(ptr);
                ptr = nullptr;
            }
        }
        throw;
    }

    return device_ptrs;
}

template <typename T>
void free_factor_matrices_from_device(std::vector<T*>& device_ptrs) {
    for (T*& ptr : device_ptrs) {
        if (ptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    }
}

template <typename T>
T** copy_factor_pointer_array_to_device(const std::vector<T*>& device_ptrs) {
    if (device_ptrs.empty()) {
        return nullptr;
    }
    T** d_array = nullptr;
    CUDA_CHECK(cudaMalloc(&d_array, device_ptrs.size() * sizeof(T*)));
    CUDA_CHECK(cudaMemcpy(d_array,
                          device_ptrs.data(),
                          device_ptrs.size() * sizeof(T*),
                          cudaMemcpyHostToDevice));
    return d_array;
}

} // namespace

__device__ inline uint64_t linearize_rank_coords(const int* coords, int count, int rank) {
    uint64_t idx = 0;
    for (int i = 0; i < count; ++i) {
        idx = idx * static_cast<uint64_t>(rank) + static_cast<uint64_t>(coords[i]);
    }
    return idx;
}

template <typename T>
__global__ void tmm_kernel_nd_sparse(int mode,
                                     BLCO_BLOCK_GPU<T>* input_tensor,
                                     uint64_t nnz,
                                     const uint64_t* masks,
                                     const int* bit_widths,
                                     const int* dims,
                                     const uint64_t* output_strides,
                                     int rank_n,
                                     int num_blocks,
                                     int rank,
                                     const T* __restrict__ fmat,
                                     T* output_tensor,
                                     int warp_width = 32) {
    const uint64_t global_idx =
        static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x % warp_width;
    const unsigned full_mask = 0xFFFFFFFFu;

    bool active = global_idx < nnz;
    T thread_val = 0;
    int coords[kMaxTensorModes];
    for (int i = 0; i < kMaxTensorModes; ++i) {
        coords[i] = 0;
    }

    if (active) {
        const int block_index = find_block_index(input_tensor, num_blocks);
        if (block_index < 0) {
            active = false;
        } else {
            const uint64_t lin_index =
                extract_linear_index(input_tensor, static_cast<int>(global_idx), block_index);
            thread_val = extract_value(input_tensor, block_index);
            const int block = input_tensor[block_index].idx;

            for (int m = 0; m < rank_n; ++m) {
                coords[m] = extract_mode_nd(lin_index,
                                            m + 1,
                                            masks,
                                            bit_widths,
                                            rank_n,
                                            block);
            }
        }
    }

    unsigned active_mask = __ballot_sync(full_mask, active);
    unsigned long long group_mask = static_cast<unsigned long long>(active_mask);
    const int target_idx = mode - 1;

    for (int m = 0; m < rank_n && group_mask != 0; ++m) {
        if (m == target_idx) {
            continue;
        }
        const int current_coord = coords[m];
        unsigned long long matches_mask = 0;
        for (int lane_idx = 0; lane_idx < warp_width; ++lane_idx) {
            const int neighbor_coord = __shfl_sync(full_mask, coords[m], lane_idx, warp_width);
            const int neighbor_active = __shfl_sync(full_mask, active ? 1 : 0, lane_idx, warp_width);
            const bool matches = (neighbor_active != 0) && (neighbor_coord == current_coord);
            matches_mask |= (static_cast<unsigned long long>(matches) << lane_idx);
        }
        group_mask &= matches_mask;
    }

    bool leader = false;
    if (group_mask != 0) {
        const int first_lane = __ffsll(group_mask) - 1;
        leader = active && (lane == first_lane);
    }

    const int contracted_dim = dims[target_idx];
    const int mode_coord = coords[target_idx];

    for (int r = 0; r < rank; ++r) {
        const size_t f_idx = static_cast<size_t>(r) * contracted_dim + mode_coord;
        const T fmat_val = fmat[f_idx];
        const T contrib = fmat_val * thread_val;

        T reduction_sum = 0;
        for (unsigned long long temp = group_mask; temp != 0; temp &= (temp - 1)) {
            const int lane_index = __ffsll(temp) - 1;
            reduction_sum += __shfl_sync(full_mask, contrib, lane_index, warp_width);
        }

        if (!leader) {
            continue;
        }

        uint64_t output_index = 0;
        for (int dim = 0; dim < rank_n; ++dim) {
            const uint64_t stride = output_strides[dim];
            const int coord = (dim == target_idx) ? r : coords[dim];
            output_index += stride * static_cast<uint64_t>(coord);
        }
        atomicAddTyped(&output_tensor[output_index], reduction_sum);
    }
}

template <typename T>
__global__ void tucker_core_kernel_nd_sparse(BLCO_BLOCK_GPU<T>* input_tensor,
                                             uint64_t nnz,
                                             const uint64_t* masks,
                                             const int* bit_widths,
                                             const int* dims,
                                             int rank_n,
                                             int num_blocks,
                                             int rank,
                                             const T* const* __restrict__ factors,
                                             T* output_tensor) {
    const uint64_t global_idx =
        static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    bool active = global_idx < nnz;

    int coords[kMaxTensorModes];
    for (int i = 0; i < kMaxTensorModes; ++i) {
        coords[i] = 0;
    }
    T thread_val = 0;

    if (active) {
        const int block_index = find_block_index(input_tensor, num_blocks);
        if (block_index < 0) {
            active = false;
        } else {
            const uint64_t lin_index =
                extract_linear_index(input_tensor, static_cast<int>(global_idx), block_index);
            thread_val = extract_value(input_tensor, block_index);
            const int block = input_tensor[block_index].idx;
            for (int m = 0; m < rank_n; ++m) {
                coords[m] = extract_mode_nd(lin_index,
                                            m + 1,
                                            masks,
                                            bit_widths,
                                            rank_n,
                                            block);
            }
        }
    }

    if (!active) {
        return;
    }

    int rank_coords[kMaxTensorModes];
    for (int i = 0; i < rank_n; ++i) {
        rank_coords[i] = 0;
    }

    bool done = false;
    while (!done) {
        T contrib = thread_val;
        for (int mode_idx = 0; mode_idx < rank_n; ++mode_idx) {
            const T* factor = factors[mode_idx];
            const size_t idx = static_cast<size_t>(rank_coords[mode_idx]) * dims[mode_idx] + coords[mode_idx];
            contrib *= factor[idx];
        }

        uint64_t core_index = 0;
        uint64_t stride = 1;
        for (int mode_idx = rank_n - 1; mode_idx >= 0; --mode_idx) {
            core_index += stride * static_cast<uint64_t>(rank_coords[mode_idx]);
            stride *= static_cast<uint64_t>(rank);
        }
        atomicAddTyped(&output_tensor[core_index], contrib);

        for (int idx = rank_n - 1; idx >= 0; --idx) {
            rank_coords[idx]++;
            if (rank_coords[idx] < rank) {
                break;
            }
            rank_coords[idx] = 0;
            if (idx == 0) {
                done = true;
            }
        }
    }
}

template <typename T>
__global__ void multimode_contraction_kernel_nd_sparse(BLCO_BLOCK_GPU<T>* input_tensor,
                                                       uint64_t nnz,
                                                       const uint64_t* masks,
                                                       const int* bit_widths,
                                                       const int* dims,
                                                       int rank_n,
                                                       int num_blocks,
                                                       int rank,
                                                       const T* const* __restrict__ factors,
                                                       T* output_Ym,
                                                       int uncontracted_mode,
                                                       uint64_t output_cols,
                                                       int store_size) {
    extern __shared__ unsigned char smem_raw[];
    T* block_store = reinterpret_cast<T*>(smem_raw);

    if (store_size > 0) {
        for (int i = threadIdx.x; i < store_size; i += blockDim.x) {
            block_store[i] = 0;
        }
    }
    __syncthreads();

    const int uncontracted_idx = uncontracted_mode - 1;
    const uint64_t global_idx =
        static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    bool active = global_idx < nnz;

    int coords[kMaxTensorModes];
    for (int i = 0; i < kMaxTensorModes; ++i) {
        coords[i] = 0;
    }
    T thread_val = 0;

    if (active) {
        const int block_index = find_block_index(input_tensor, num_blocks);
        if (block_index < 0) {
            active = false;
        } else {
            const uint64_t lin_index =
                extract_linear_index(input_tensor, static_cast<int>(global_idx), block_index);
            thread_val = extract_value(input_tensor, block_index);
            const int block = input_tensor[block_index].idx;
            for (int m = 0; m < rank_n; ++m) {
                coords[m] = extract_mode_nd(lin_index,
                                            m + 1,
                                            masks,
                                            bit_widths,
                                            rank_n,
                                            block);
            }
        }
    }

    if (!active) {
        return;
    }

    int contracted_modes[kMaxTensorModes];
    int contracted_count = 0;
    for (int i = 0; i < rank_n; ++i) {
        if (i == uncontracted_idx) {
            continue;
        }
        contracted_modes[contracted_count++] = i;
    }

    const int row_index = coords[uncontracted_idx];

    if (contracted_count == 0) {
        const uint64_t out_index = static_cast<uint64_t>(row_index) * output_cols;
        if (store_size > 0 && out_index < static_cast<uint64_t>(store_size)) {
            atomicAddTyped(&block_store[out_index], thread_val);
        } else {
            atomicAddTyped(&output_Ym[out_index], thread_val);
        }
        __syncthreads();
        if (store_size > 0) {
            for (int i = threadIdx.x; i < store_size; i += blockDim.x) {
                const T val = block_store[i];
                if (val != static_cast<T>(0)) {
                    atomicAddTyped(&output_Ym[i], val);
                }
            }
        }
        return;
    }

    int rank_coords[kMaxTensorModes];
    for (int i = 0; i < contracted_count; ++i) {
        rank_coords[i] = 0;
    }

    bool done = false;
    while (!done) {
        T contrib = thread_val;
        for (int i = 0; i < contracted_count; ++i) {
            const int mode_idx = contracted_modes[i];
            const T* factor = factors[mode_idx];
            const size_t idx = static_cast<size_t>(rank_coords[i]) * dims[mode_idx] + coords[mode_idx];
            contrib *= factor[idx];
        }
        const uint64_t col_index = linearize_rank_coords(rank_coords, contracted_count, rank);
        const uint64_t out_index = static_cast<uint64_t>(row_index) * output_cols + col_index;

        if (store_size > 0 && out_index < static_cast<uint64_t>(store_size)) {
            atomicAddTyped(&block_store[out_index], contrib);
        } else {
            atomicAddTyped(&output_Ym[out_index], contrib);
        }

        for (int i = contracted_count - 1; i >= 0; --i) {
            rank_coords[i]++;
            if (rank_coords[i] < rank) {
                break;
            }
            rank_coords[i] = 0;
            if (i == 0) {
                done = true;
            }
        }
    }

    __syncthreads();
    if (store_size > 0) {
        for (int i = threadIdx.x; i < store_size; i += blockDim.x) {
            const T val = block_store[i];
            if (val != static_cast<T>(0)) {
                atomicAddTyped(&output_Ym[i], val);
            }
        }
    }
}

template <typename T, typename S>
T* tmm_nd_cuda(const Blco_Tensor<T, S>& sparse_tensor,
               int mode,
               int block_size,
               bool log_timings) {
    auto total_start = std::chrono::high_resolution_clock::now();
    std::vector<int> dims = sparse_tensor.get_dims();
    const int rank_n = static_cast<int>(dims.size());
    validate_rank_count(rank_n);
    if (mode < 1 || mode > rank_n) {
        throw std::invalid_argument("tmm_nd_cuda mode is out of bounds");
    }

    const auto blco_tensor = sparse_tensor.get_blco();
    const int num_blocks = static_cast<int>(blco_tensor.size());
    const int rank = sparse_tensor.get_factor_rank();
    if (rank <= 0) {
        throw std::runtime_error("Factor rank must be positive");
    }
    const uint64_t nnz = sparse_tensor.get_nnz();
    std::vector<uint64_t> masks = sparse_tensor.get_bitmasks();
    if (static_cast<int>(masks.size()) < rank_n) {
        throw std::runtime_error("Bitmask count is smaller than tensor rank");
    }
    std::vector<int> bit_widths = compute_bit_widths(dims);
    std::vector<T*> fmats = sparse_tensor.get_fmats();
    if (static_cast<int>(fmats.size()) < rank_n) {
        throw std::runtime_error("Factor matrices missing for one or more modes");
    }

    std::vector<int> output_dims = dims;
    output_dims[mode - 1] = rank;
    const uint64_t output_elems = safe_product(output_dims);
    if (nnz == 0) {
        T* zero_output = static_cast<T*>(calloc(static_cast<size_t>(output_elems), sizeof(T)));
        return zero_output;
    }

    auto upload_start = std::chrono::high_resolution_clock::now();
    BLCO_BLOCK_GPU<T>* d_blocks = nullptr;
    blocks_to_gpu(d_blocks, blco_tensor, num_blocks);

    int* d_dims = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dims, sizeof(int) * rank_n));
    CUDA_CHECK(cudaMemcpy(d_dims, dims.data(), sizeof(int) * rank_n, cudaMemcpyHostToDevice));

    int* d_bit_widths = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bit_widths, sizeof(int) * rank_n));
    CUDA_CHECK(cudaMemcpy(d_bit_widths,
                          bit_widths.data(),
                          sizeof(int) * rank_n,
                          cudaMemcpyHostToDevice));

    uint64_t* d_masks = nullptr;
    CUDA_CHECK(cudaMalloc(&d_masks, sizeof(uint64_t) * rank_n));
    CUDA_CHECK(cudaMemcpy(d_masks,
                          masks.data(),
                          sizeof(uint64_t) * rank_n,
                          cudaMemcpyHostToDevice));

    std::vector<uint64_t> output_strides = build_output_strides(output_dims);
    uint64_t* d_output_strides = nullptr;
    CUDA_CHECK(cudaMalloc(&d_output_strides, sizeof(uint64_t) * rank_n));
    CUDA_CHECK(cudaMemcpy(d_output_strides,
                          output_strides.data(),
                          sizeof(uint64_t) * rank_n,
                          cudaMemcpyHostToDevice));

    std::vector<T*> d_fmats = copy_factor_matrices_to_device(fmats, dims, rank);

    T* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_output, static_cast<size_t>(output_elems) * sizeof(T)));
    CUDA_CHECK(cudaMemset(d_output, 0, static_cast<size_t>(output_elems) * sizeof(T)));
    auto upload_end = std::chrono::high_resolution_clock::now();
    const double upload_ms =
        std::chrono::duration<double, std::milli>(upload_end - upload_start).count();

    if (block_size <= 0) {
        block_size = 256;
    }
    block_size = std::min(block_size, 1024);
    const uint64_t threads_per_block = static_cast<uint64_t>(block_size);
    const uint64_t grid_x = (nnz + threads_per_block - 1) / threads_per_block;

    const T* d_fmat_launch = d_fmats[mode - 1];
    const int warp_width = 32;

    cudaEvent_t kernel_start, kernel_stop;
    CUDA_CHECK(cudaEventCreate(&kernel_start));
    CUDA_CHECK(cudaEventCreate(&kernel_stop));
    CUDA_CHECK(cudaEventRecord(kernel_start));
    tmm_kernel_nd_sparse<T><<<dim3(static_cast<unsigned>(grid_x)),
                              dim3(static_cast<unsigned>(threads_per_block))>>>(
        mode,
        d_blocks,
        nnz,
        d_masks,
        d_bit_widths,
        d_dims,
        d_output_strides,
        rank_n,
        num_blocks,
        rank,
        d_fmat_launch,
        d_output,
        warp_width);
    CUDA_CHECK(cudaEventRecord(kernel_stop));
    CUDA_CHECK(cudaEventSynchronize(kernel_stop));
    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop));

    T* host_output = static_cast<T*>(malloc(static_cast<size_t>(output_elems) * sizeof(T)));
    cudaEvent_t download_start, download_stop;
    CUDA_CHECK(cudaEventCreate(&download_start));
    CUDA_CHECK(cudaEventCreate(&download_stop));
    CUDA_CHECK(cudaEventRecord(download_start));
    CUDA_CHECK(cudaMemcpy(host_output,
                          d_output,
                          static_cast<size_t>(output_elems) * sizeof(T),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(download_stop));
    CUDA_CHECK(cudaEventSynchronize(download_stop));
    float download_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&download_ms, download_start, download_stop));

    free_blocks_from_gpu(d_blocks, num_blocks);
    free_factor_matrices_from_device(d_fmats);
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_dims));
    CUDA_CHECK(cudaFree(d_masks));
    CUDA_CHECK(cudaFree(d_bit_widths));
    CUDA_CHECK(cudaFree(d_output_strides));

    CUDA_CHECK(cudaEventDestroy(kernel_start));
    CUDA_CHECK(cudaEventDestroy(kernel_stop));
    CUDA_CHECK(cudaEventDestroy(download_start));
    CUDA_CHECK(cudaEventDestroy(download_stop));

    if (log_timings) {
        std::cout << "TMM GPU timings (ms) upload=" << upload_ms
                  << " kernel=" << kernel_ms
                  << " download=" << download_ms << "\n";
    }
    auto total_end = std::chrono::high_resolution_clock::now();
    const double total_ms =
        std::chrono::duration<double, std::milli>(total_end - total_start).count();
    if (log_timings) {
        std::cout << "TMM total wall time (ms): " << total_ms << "\n";
    }

    return host_output;
}

template <typename T, typename S>
T* tucker_compute_core_nd_cuda(const Blco_Tensor<T, S>& sparse_tensor,
                               int block_size) {
    auto total_start = std::chrono::high_resolution_clock::now();
    std::vector<int> dims = sparse_tensor.get_dims();
    const int rank_n = static_cast<int>(dims.size());
    validate_rank_count(rank_n);

    const auto blco_tensor = sparse_tensor.get_blco();
    const int num_blocks = static_cast<int>(blco_tensor.size());
    const int rank = sparse_tensor.get_factor_rank();
    if (rank <= 0) {
        throw std::runtime_error("Factor rank must be positive");
    }
    const uint64_t nnz = sparse_tensor.get_nnz();
    std::vector<uint64_t> masks = sparse_tensor.get_bitmasks();
    if (static_cast<int>(masks.size()) < rank_n) {
        throw std::runtime_error("Bitmask count is smaller than tensor rank");
    }
    std::vector<int> bit_widths = compute_bit_widths(dims);
    std::vector<T*> fmats = sparse_tensor.get_fmats();
    if (static_cast<int>(fmats.size()) < rank_n) {
        throw std::runtime_error("Factor matrices missing for one or more modes");
    }

    const uint64_t output_elems = safe_int_pow(static_cast<uint64_t>(rank), rank_n);
    if (nnz == 0) {
        T* zero_output = static_cast<T*>(calloc(static_cast<size_t>(output_elems), sizeof(T)));
        return zero_output;
    }

    auto upload_start = std::chrono::high_resolution_clock::now();
    BLCO_BLOCK_GPU<T>* d_blocks = nullptr;
    blocks_to_gpu(d_blocks, blco_tensor, num_blocks);

    int* d_dims = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dims, sizeof(int) * rank_n));
    CUDA_CHECK(cudaMemcpy(d_dims, dims.data(), sizeof(int) * rank_n, cudaMemcpyHostToDevice));

    int* d_bit_widths = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bit_widths, sizeof(int) * rank_n));
    CUDA_CHECK(cudaMemcpy(d_bit_widths,
                          bit_widths.data(),
                          sizeof(int) * rank_n,
                          cudaMemcpyHostToDevice));

    uint64_t* d_masks = nullptr;
    CUDA_CHECK(cudaMalloc(&d_masks, sizeof(uint64_t) * rank_n));
    CUDA_CHECK(cudaMemcpy(d_masks,
                          masks.data(),
                          sizeof(uint64_t) * rank_n,
                          cudaMemcpyHostToDevice));

    std::vector<T*> d_fmats = copy_factor_matrices_to_device(fmats, dims, rank);
    T** d_factor_ptrs = copy_factor_pointer_array_to_device(d_fmats);

    T* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_output, static_cast<size_t>(output_elems) * sizeof(T)));
    CUDA_CHECK(cudaMemset(d_output, 0, static_cast<size_t>(output_elems) * sizeof(T)));
    auto upload_end = std::chrono::high_resolution_clock::now();
    const double upload_ms =
        std::chrono::duration<double, std::milli>(upload_end - upload_start).count();

    if (block_size <= 0) {
        block_size = 256;
    }
    block_size = std::min(block_size, 1024);
    const uint64_t threads_per_block = static_cast<uint64_t>(block_size);
    const uint64_t grid_x = (nnz + threads_per_block - 1) / threads_per_block;

    cudaEvent_t kernel_start, kernel_stop;
    CUDA_CHECK(cudaEventCreate(&kernel_start));
    CUDA_CHECK(cudaEventCreate(&kernel_stop));
    CUDA_CHECK(cudaEventRecord(kernel_start));
    tucker_core_kernel_nd_sparse<T><<<dim3(static_cast<unsigned>(grid_x)),
                                      dim3(static_cast<unsigned>(threads_per_block))>>>(
        d_blocks,
        nnz,
        d_masks,
        d_bit_widths,
        d_dims,
        rank_n,
        num_blocks,
        rank,
        reinterpret_cast<const T* const*>(d_factor_ptrs),
        d_output);
    CUDA_CHECK(cudaEventRecord(kernel_stop));
    CUDA_CHECK(cudaEventSynchronize(kernel_stop));
    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop));

    T* host_output = static_cast<T*>(malloc(static_cast<size_t>(output_elems) * sizeof(T)));
    cudaEvent_t download_start, download_stop;
    CUDA_CHECK(cudaEventCreate(&download_start));
    CUDA_CHECK(cudaEventCreate(&download_stop));
    CUDA_CHECK(cudaEventRecord(download_start));
    CUDA_CHECK(cudaMemcpy(host_output,
                          d_output,
                          static_cast<size_t>(output_elems) * sizeof(T),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(download_stop));
    CUDA_CHECK(cudaEventSynchronize(download_stop));
    float download_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&download_ms, download_start, download_stop));

    free_blocks_from_gpu(d_blocks, num_blocks);
    if (d_factor_ptrs) {
        CUDA_CHECK(cudaFree(d_factor_ptrs));
    }
    free_factor_matrices_from_device(d_fmats);
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_dims));
    CUDA_CHECK(cudaFree(d_masks));
    CUDA_CHECK(cudaFree(d_bit_widths));

    CUDA_CHECK(cudaEventDestroy(kernel_start));
    CUDA_CHECK(cudaEventDestroy(kernel_stop));
    CUDA_CHECK(cudaEventDestroy(download_start));
    CUDA_CHECK(cudaEventDestroy(download_stop));

    std::cout << "Core GPU timings (ms) upload=" << upload_ms
              << " kernel=" << kernel_ms
              << " download=" << download_ms << "\n";
    auto total_end = std::chrono::high_resolution_clock::now();
    const double total_ms =
        std::chrono::duration<double, std::milli>(total_end - total_start).count();
    std::cout << "Core total wall time (ms): " << total_ms << "\n";

    return host_output;
}

template <typename T, typename S>
T* contract_n_minus_one_modes_nd_cuda(const Blco_Tensor<T, S>& sparse_tensor,
                                      int block_size,
                                      int mode) {
    std::vector<int> dims = sparse_tensor.get_dims();
    const int rank_n = static_cast<int>(dims.size());
    validate_rank_count(rank_n);
    if (mode < 1 || mode > rank_n) {
        throw std::invalid_argument("contract_n_minus_one_modes_nd_cuda mode out of bounds");
    }

    const auto blco_tensor = sparse_tensor.get_blco();
    const int num_blocks = static_cast<int>(blco_tensor.size());
    const int rank = sparse_tensor.get_factor_rank();
    if (rank <= 0) {
        throw std::runtime_error("Factor rank must be positive");
    }
    const uint64_t nnz = sparse_tensor.get_nnz();
    std::vector<uint64_t> masks = sparse_tensor.get_bitmasks();
    if (static_cast<int>(masks.size()) < rank_n) {
        throw std::runtime_error("Bitmask count is smaller than tensor rank");
    }
    std::vector<int> bit_widths = compute_bit_widths(dims);
    std::vector<T*> fmats = sparse_tensor.get_fmats();
    if (static_cast<int>(fmats.size()) < rank_n) {
        throw std::runtime_error("Factor matrices missing for one or more modes");
    }

    const uint64_t output_cols = safe_int_pow(static_cast<uint64_t>(rank), rank_n - 1);
    const uint64_t output_size = static_cast<uint64_t>(dims[mode - 1]) * output_cols;
    if (nnz == 0) {
        T* zero_output = static_cast<T*>(calloc(static_cast<size_t>(output_size), sizeof(T)));
        return zero_output;
    }

    auto upload_start = std::chrono::high_resolution_clock::now();
    BLCO_BLOCK_GPU<T>* d_blocks = nullptr;
    blocks_to_gpu(d_blocks, blco_tensor, num_blocks);

    int* d_dims = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dims, sizeof(int) * rank_n));
    CUDA_CHECK(cudaMemcpy(d_dims, dims.data(), sizeof(int) * rank_n, cudaMemcpyHostToDevice));

    int* d_bit_widths = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bit_widths, sizeof(int) * rank_n));
    CUDA_CHECK(cudaMemcpy(d_bit_widths,
                          bit_widths.data(),
                          sizeof(int) * rank_n,
                          cudaMemcpyHostToDevice));

    uint64_t* d_masks = nullptr;
    CUDA_CHECK(cudaMalloc(&d_masks, sizeof(uint64_t) * rank_n));
    CUDA_CHECK(cudaMemcpy(d_masks,
                          masks.data(),
                          sizeof(uint64_t) * rank_n,
                          cudaMemcpyHostToDevice));

    std::vector<T*> d_fmats = copy_factor_matrices_to_device(fmats, dims, rank);
    T** d_factor_ptrs = copy_factor_pointer_array_to_device(d_fmats);

    T* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_output, static_cast<size_t>(output_size) * sizeof(T)));
    CUDA_CHECK(cudaMemset(d_output, 0, static_cast<size_t>(output_size) * sizeof(T)));
    auto upload_end = std::chrono::high_resolution_clock::now();
    const double upload_ms =
        std::chrono::duration<double, std::milli>(upload_end - upload_start).count();

    if (block_size <= 0) {
        block_size = 256;
    }
    block_size = std::min(block_size, 1024);
    const uint64_t threads_per_block = static_cast<uint64_t>(block_size);
    const uint64_t grid_x = (nnz + threads_per_block - 1) / threads_per_block;

    const size_t max_shared = getMaxSharedMemory();
    const uint64_t max_elements = max_shared / sizeof(T);
    const uint64_t store_capacity = std::min(output_size, max_elements);
    const int store_size = static_cast<int>(store_capacity);
    const size_t shared_bytes = static_cast<size_t>(store_size) * sizeof(T);

    cudaEvent_t kernel_start, kernel_stop;
    CUDA_CHECK(cudaEventCreate(&kernel_start));
    CUDA_CHECK(cudaEventCreate(&kernel_stop));
    CUDA_CHECK(cudaEventRecord(kernel_start));
    multimode_contraction_kernel_nd_sparse<T><<<dim3(static_cast<unsigned>(grid_x)),
                                                dim3(static_cast<unsigned>(threads_per_block)),
                                                shared_bytes>>>(
        d_blocks,
        nnz,
        d_masks,
        d_bit_widths,
        d_dims,
        rank_n,
        num_blocks,
        rank,
        reinterpret_cast<const T* const*>(d_factor_ptrs),
        d_output,
        mode,
        output_cols,
        store_size);
    CUDA_CHECK(cudaEventRecord(kernel_stop));
    CUDA_CHECK(cudaEventSynchronize(kernel_stop));
    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop));

    T* host_output = static_cast<T*>(malloc(static_cast<size_t>(output_size) * sizeof(T)));
    cudaEvent_t download_start, download_stop;
    CUDA_CHECK(cudaEventCreate(&download_start));
    CUDA_CHECK(cudaEventCreate(&download_stop));
    CUDA_CHECK(cudaEventRecord(download_start));
    CUDA_CHECK(cudaMemcpy(host_output,
                          d_output,
                          static_cast<size_t>(output_size) * sizeof(T),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(download_stop));
    CUDA_CHECK(cudaEventSynchronize(download_stop));
    float download_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&download_ms, download_start, download_stop));

    free_blocks_from_gpu(d_blocks, num_blocks);
    if (d_factor_ptrs) {
        CUDA_CHECK(cudaFree(d_factor_ptrs));
    }
    free_factor_matrices_from_device(d_fmats);
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_dims));
    CUDA_CHECK(cudaFree(d_masks));
    CUDA_CHECK(cudaFree(d_bit_widths));

    CUDA_CHECK(cudaEventDestroy(kernel_start));
    CUDA_CHECK(cudaEventDestroy(kernel_stop));
    CUDA_CHECK(cudaEventDestroy(download_start));
    CUDA_CHECK(cudaEventDestroy(download_stop));

    std::cout << "Contraction GPU timings (ms) upload=" << upload_ms
              << " kernel=" << kernel_ms
              << " download=" << download_ms << "\n";

    return host_output;
}

#define INSTANTIATE_TMM_ND(TTYPE, STYPE) \
    template TTYPE* tmm_nd_cuda<TTYPE, STYPE>(const Blco_Tensor<TTYPE, STYPE>&, int, int, bool); \
    template TTYPE* tucker_compute_core_nd_cuda<TTYPE, STYPE>(const Blco_Tensor<TTYPE, STYPE>&, int); \
    template TTYPE* contract_n_minus_one_modes_nd_cuda<TTYPE, STYPE>(const Blco_Tensor<TTYPE, STYPE>&, int, int);

INSTANTIATE_TMM_ND(int, uint64_t)
INSTANTIATE_TMM_ND(float, uint64_t)
INSTANTIATE_TMM_ND(long int, uint64_t)
INSTANTIATE_TMM_ND(int, __uint128_t)
INSTANTIATE_TMM_ND(float, __uint128_t)
INSTANTIATE_TMM_ND(long int, __uint128_t)

#undef INSTANTIATE_TMM_ND
