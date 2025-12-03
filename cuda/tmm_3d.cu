#include <cuda_runtime.h>
#include <type_traits>
#include <algorithm>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <cstdlib>

#include "cuda_utils.cuh"
#include "tmm_3d.cuh"

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

} // namespace

template <typename T>
__global__ void tmm_kernel_3d_sparse(int mode,
                                     BLCO_BLOCK_GPU<T>* input_tensor,
                                     uint64_t nnz,
                                     const uint64_t* masks,
                                     const T* fmat,
                                     const int* dims,
                                     int num_blocks,
                                     int rank,
                                     T* output_tensor,
                                     int warp_width = 32)
{
    const uint64_t global_idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x % warp_width;

    bool active = global_idx < nnz;
    const int contracted_dim = dims[mode - 1];
    T thread_val = 0;

    int m1_index = 0;
    int m2_index = 0;
    int m3_index = 0;
    const int nt_mode1_list[3] = {2, 1, 1};
    const int nt_mode2_list[3] = {3, 3, 2};
    int i_mode = 0;
    int nt_mode_1 = -1;
    int nt_mode_2 = -1;

    if (active) {
        const int bl_index = find_block_index(input_tensor, num_blocks);
        const uint64_t lin_index = extract_linear_index(input_tensor, static_cast<int>(global_idx), bl_index);
        thread_val = extract_value(input_tensor, bl_index);
        const int block = input_tensor[bl_index].idx;

        const int bit_widths[3] = {
            ceiling_log2(dims[0]),
            ceiling_log2(dims[1]),
            ceiling_log2(dims[2])
        };

        m1_index = extract_mode_nd(lin_index, 1, masks, bit_widths, 3, block);
        m2_index = extract_mode_nd(lin_index, 2, masks, bit_widths, 3, block);
        m3_index = extract_mode_nd(lin_index, 3, masks, bit_widths, 3, block);

        const int indices[3] = {m1_index, m2_index, m3_index};
        i_mode = indices[mode - 1];

        nt_mode_1 = indices[nt_mode1_list[mode - 1] - 1];
        nt_mode_2 = indices[nt_mode2_list[mode - 1] - 1];
    }

    unsigned long long nt_mask_1 = 0;
    unsigned long long nt_mask_2 = 0;

    const unsigned full_mask = 0xFFFFFFFFu;
    for (int k = 0; k < warp_width; ++k) {
        const int neighbor_nt = __shfl_sync(full_mask, nt_mode_1, k, warp_width);
        const bool matches = neighbor_nt == nt_mode_1;
        nt_mask_1 |= (static_cast<unsigned long long>(matches) << k);
    }
    for (int k = 0; k < warp_width; ++k) {
        const int neighbor_nt = __shfl_sync(full_mask, nt_mode_2, k, warp_width);
        const bool matches = neighbor_nt == nt_mode_2;
        nt_mask_2 |= (static_cast<unsigned long long>(matches) << k);
    }

    const unsigned long long group_mask = nt_mask_1 & nt_mask_2;
    bool leader = false;
    int first_lane = -1;
    if (group_mask != 0) {
        first_lane = __ffsll(group_mask) - 1;
        leader = active && (lane == first_lane);
    }

    for (int i = 0; i < rank; ++i) {
        const int fmat_flat_index = i * contracted_dim + i_mode;
        const T fmat_val = fmat[fmat_flat_index];
        const T contrib = fmat_val * thread_val;
        T reduction_sum = 0;

        for (unsigned long long temp = group_mask; temp != 0; temp &= (temp - 1)) {
            const int lane_index = __ffsll(temp) - 1;
            reduction_sum += __shfl_sync(full_mask, contrib, lane_index, warp_width);
        }

        if (!leader) {
            continue;
        }

        int new_coords[3];
        new_coords[mode - 1] = i;
        new_coords[nt_mode1_list[mode - 1] - 1] = nt_mode_1;
        new_coords[nt_mode2_list[mode - 1] - 1] = nt_mode_2;

        int local_dims[3] = {dims[0], dims[1], dims[2]};
        local_dims[mode - 1] = rank;
        const int output_index =
            new_coords[0] * local_dims[1] * local_dims[2] +
            new_coords[1] * local_dims[2] +
            new_coords[2];
        atomicAddTyped(&output_tensor[output_index], reduction_sum);
    }
}

template <typename T>
__global__ void tucker_core_kernel_3d_sparse(BLCO_BLOCK_GPU<T>* input_tensor,
                                             uint64_t nnz,
                                             const uint64_t* masks,
                                             const T* d_U1,
                                             const T* d_U2,
                                             const T* d_U3,
                                             const int* dims,
                                             int num_blocks,
                                             int rank,
                                             T* output_tensor)
{
    const uint64_t global_idx =
        static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int local_idx = threadIdx.x;

    extern __shared__ __align__(sizeof(T)) unsigned char smem[];
    T* block_tensor = reinterpret_cast<T*>(smem);
    const int output_size = rank * rank * rank;

    for (int i = local_idx; i < output_size; i += blockDim.x) {
        block_tensor[i] = 0;
    }
    __syncthreads();

    const bool active = global_idx < nnz;
    T thread_val = 0;
    int m1_index = 0;
    int m2_index = 0;
    int m3_index = 0;

    if (active) {
        const int bl_index = find_block_index(input_tensor, num_blocks);
        const uint64_t lin_index =
            extract_linear_index(input_tensor, static_cast<int>(global_idx), bl_index);
        thread_val = extract_value(input_tensor, bl_index);
        const int block = input_tensor[bl_index].idx;

        const int bit_widths[3] = {
            ceiling_log2(dims[0]),
            ceiling_log2(dims[1]),
            ceiling_log2(dims[2])
        };

        m1_index = extract_mode_nd(lin_index, 1, masks, bit_widths, 3, block);
        m2_index = extract_mode_nd(lin_index, 2, masks, bit_widths, 3, block);
        m3_index = extract_mode_nd(lin_index, 3, masks, bit_widths, 3, block);
    }

    if (active) {
        const int dim0 = dims[0];
        const int dim1 = dims[1];
        const int dim2 = dims[2];

        for (int r1 = 0; r1 < rank; ++r1) {
            const T u1_val = d_U1[r1 * dim0 + m1_index];
            for (int r2 = 0; r2 < rank; ++r2) {
                const T u2_val = d_U2[r2 * dim1 + m2_index];
                for (int r3 = 0; r3 < rank; ++r3) {
                    const T u3_val = d_U3[r3 * dim2 + m3_index];
                    const T contrib = thread_val * u1_val * u2_val * u3_val;
                    const int out_idx = r1 * rank * rank + r2 * rank + r3;
                    atomicAddTyped(&block_tensor[out_idx], contrib);
                }
            }
        }
    }

    __syncthreads();

    for (int i = local_idx; i < output_size; i += blockDim.x) {
        const T val = block_tensor[i];
        if (val != static_cast<T>(0)) {
            atomicAddTyped(&output_tensor[i], val);
        }
    }
}

template <typename T, typename S>
T* tmm_3d_cuda(const Blco_Tensor<T, S>& sparse_tensor,
               int mode,
               int block_size)
{
    if (mode < 1 || mode > 3) {
        throw std::invalid_argument("tmm_3d_cuda mode must be in [1,3]");
    }

    std::vector<int> dims = sparse_tensor.get_dims();
    if (dims.size() != 3) {
        throw std::runtime_error("tmm_3d_cuda expects a 3D tensor");
    }

    const auto blco_tensor = sparse_tensor.get_blco();
    const int num_blocks = static_cast<int>(blco_tensor.size());
    const int rank = sparse_tensor.get_factor_rank();
    const uint64_t nnz = sparse_tensor.get_nnz();
    std::vector<uint64_t> masks = sparse_tensor.get_bitmasks();
    std::vector<T*> fmats = sparse_tensor.get_fmats();

    if (fmats.size() < 3) {
        throw std::runtime_error("Expected three factor matrices for tmm_3d_cuda");
    }

    std::vector<int> output_dims = dims;
    output_dims[mode - 1] = rank;
    const size_t output_elems =
        static_cast<size_t>(output_dims[0]) * output_dims[1] * output_dims[2];

    if (nnz == 0) {
        T* empty_output = static_cast<T*>(calloc(output_elems, sizeof(T)));
        return empty_output;
    }

    BLCO_BLOCK_GPU<T>* d_blocks = nullptr;
    blocks_to_gpu(d_blocks, blco_tensor, num_blocks);

    int* d_dims = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dims, sizeof(int) * 3));
    CUDA_CHECK(cudaMemcpy(d_dims, dims.data(), sizeof(int) * 3, cudaMemcpyHostToDevice));

    uint64_t* d_masks = nullptr;
    CUDA_CHECK(cudaMalloc(&d_masks, sizeof(uint64_t) * 3));
    CUDA_CHECK(cudaMemcpy(d_masks, masks.data(), sizeof(uint64_t) * 3, cudaMemcpyHostToDevice));

    T* d_fmats[3] = {nullptr, nullptr, nullptr};
    for (int i = 0; i < 3; ++i) {
        const size_t elems = static_cast<size_t>(dims[i]) * rank;
        CUDA_CHECK(cudaMalloc(&d_fmats[i], elems * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(d_fmats[i], fmats[i], elems * sizeof(T), cudaMemcpyHostToDevice));
    }

    T* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_output, output_elems * sizeof(T)));
    CUDA_CHECK(cudaMemset(d_output, 0, output_elems * sizeof(T)));

    if (block_size <= 0) {
        block_size = 256;
    }
    block_size = std::min(block_size, 1024);
    const uint64_t threads_per_block = static_cast<uint64_t>(block_size);
    const uint64_t grid_x = (nnz + threads_per_block - 1) / threads_per_block;

    const T* d_fmat_launch = d_fmats[mode - 1];
    const int warp_width = 32;
    tmm_kernel_3d_sparse<T><<<dim3(static_cast<unsigned>(grid_x)),
                              dim3(static_cast<unsigned>(threads_per_block))>>>(
        mode,
        d_blocks,
        nnz,
        d_masks,
        d_fmat_launch,
        d_dims,
        num_blocks,
        rank,
        d_output,
        warp_width);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    T* host_output = static_cast<T*>(malloc(output_elems * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(host_output,
                          d_output,
                          output_elems * sizeof(T),
                          cudaMemcpyDeviceToHost));

    free_blocks_from_gpu(d_blocks, num_blocks);
    for (int i = 0; i < 3; ++i) {
        CUDA_CHECK(cudaFree(d_fmats[i]));
    }
    CUDA_CHECK(cudaFree(d_dims));
    CUDA_CHECK(cudaFree(d_masks));
    CUDA_CHECK(cudaFree(d_output));

    return host_output;
}

template <typename T, typename S>
T* tucker_compute_core_3d_cuda(const Blco_Tensor<T, S>& sparse_tensor,
                               int block_size)
{
    std::vector<int> dims = sparse_tensor.get_dims();
    if (dims.size() != 3) {
        throw std::runtime_error("tucker_compute_core_3d_cuda expects a 3D tensor");
    }

    const auto blco_tensor = sparse_tensor.get_blco();
    const int num_blocks = static_cast<int>(blco_tensor.size());
    const int rank = sparse_tensor.get_factor_rank();
    const uint64_t nnz = sparse_tensor.get_nnz();
    std::vector<uint64_t> masks = sparse_tensor.get_bitmasks();
    std::vector<T*> fmats = sparse_tensor.get_fmats();

    if (fmats.size() < 3) {
        throw std::runtime_error("Expected three factor matrices for core computation");
    }

    const size_t output_elems = static_cast<size_t>(rank) * rank * rank;
    if (nnz == 0) {
        T* empty_output = static_cast<T*>(calloc(output_elems, sizeof(T)));
        return empty_output;
    }

    BLCO_BLOCK_GPU<T>* d_blocks = nullptr;
    blocks_to_gpu(d_blocks, blco_tensor, num_blocks);

    int* d_dims = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dims, sizeof(int) * 3));
    CUDA_CHECK(cudaMemcpy(d_dims, dims.data(), sizeof(int) * 3, cudaMemcpyHostToDevice));

    uint64_t* d_masks = nullptr;
    CUDA_CHECK(cudaMalloc(&d_masks, sizeof(uint64_t) * 3));
    CUDA_CHECK(cudaMemcpy(d_masks, masks.data(), sizeof(uint64_t) * 3, cudaMemcpyHostToDevice));

    T* d_fmats[3] = {nullptr, nullptr, nullptr};
    for (int i = 0; i < 3; ++i) {
        const size_t elems = static_cast<size_t>(dims[i]) * rank;
        CUDA_CHECK(cudaMalloc(&d_fmats[i], elems * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(d_fmats[i], fmats[i], elems * sizeof(T), cudaMemcpyHostToDevice));
    }

    T* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_output, output_elems * sizeof(T)));
    CUDA_CHECK(cudaMemset(d_output, 0, output_elems * sizeof(T)));

    if (block_size <= 0) {
        block_size = 256;
    }
    block_size = std::min(block_size, 1024);
    const uint64_t threads_per_block = static_cast<uint64_t>(block_size);
    const uint64_t grid_x = (nnz + threads_per_block - 1) / threads_per_block;

    const size_t max_shared = getMaxSharedMemory();
    const size_t shared_bytes =
        std::min(max_shared, output_elems * sizeof(T));

    tucker_core_kernel_3d_sparse<T><<<dim3(static_cast<unsigned>(grid_x)),
                                      dim3(static_cast<unsigned>(threads_per_block)),
                                      shared_bytes>>>(
        d_blocks,
        nnz,
        d_masks,
        d_fmats[0],
        d_fmats[1],
        d_fmats[2],
        d_dims,
        num_blocks,
        rank,
        d_output);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    T* host_output = static_cast<T*>(malloc(output_elems * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(host_output,
                          d_output,
                          output_elems * sizeof(T),
                          cudaMemcpyDeviceToHost));

    free_blocks_from_gpu(d_blocks, num_blocks);
    for (int i = 0; i < 3; ++i) {
        CUDA_CHECK(cudaFree(d_fmats[i]));
    }
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_dims));
    CUDA_CHECK(cudaFree(d_masks));

    return host_output;
}

#define INSTANTIATE_TMM(TTYPE, STYPE) \
    template TTYPE* tmm_3d_cuda<TTYPE, STYPE>(const Blco_Tensor<TTYPE, STYPE>&, int, int); \
    template TTYPE* tucker_compute_core_3d_cuda<TTYPE, STYPE>(const Blco_Tensor<TTYPE, STYPE>&, int);

INSTANTIATE_TMM(int, uint64_t)
INSTANTIATE_TMM(float, uint64_t)
INSTANTIATE_TMM(long int, uint64_t)
INSTANTIATE_TMM(int, __uint128_t)
INSTANTIATE_TMM(float, __uint128_t)
INSTANTIATE_TMM(long int, __uint128_t)

#undef INSTANTIATE_TMM
