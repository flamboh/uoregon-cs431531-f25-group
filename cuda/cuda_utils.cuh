#pragma once

#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../tensor_storage/blco.h"
#include "../tensor_storage/tensor_utils.h"

using namespace std;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

// Return the per-block shared memory capacity for the active device.
inline size_t getMaxSharedMemory() {
    int deviceId = 0;
    CUDA_CHECK(cudaGetDevice(&deviceId));

    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, deviceId));
    return props.sharedMemPerBlock;
}

// Copy BLCO CPU blocks into GPU memory (structure-of-arrays layout).
template<typename T>
inline void blocks_to_gpu(BLCO_BLOCK_GPU<T>*& d_block_arr,
                          const vector<BLCO_BLOCK_CPU<T>>& tensor,
                          int num_blocks)
{
    d_block_arr = nullptr;
    if (num_blocks <= 0) return;

    vector<T*> d_values_ptrs;
    vector<uint64_t*> d_indexes_ptrs;

    try {
        CUDA_CHECK(cudaMalloc(&d_block_arr, num_blocks * sizeof(BLCO_BLOCK_GPU<T>)));
        vector<BLCO_BLOCK_GPU<T>> h_arr_for_gpu(num_blocks);

        for (int i = 0; i < num_blocks; ++i) {
            const int num_elements = tensor[i].size;
            h_arr_for_gpu[i].idx = tensor[i].idx;
            h_arr_for_gpu[i].size = num_elements;

            if (num_elements > 0) {
                T* d_values = nullptr;
                uint64_t* d_indexes = nullptr;

                CUDA_CHECK(cudaMalloc(&d_values, num_elements * sizeof(T)));
                CUDA_CHECK(cudaMalloc(&d_indexes, num_elements * sizeof(uint64_t)));

                d_values_ptrs.push_back(d_values);
                d_indexes_ptrs.push_back(d_indexes);

                CUDA_CHECK(cudaMemcpy(d_values,
                                      tensor[i].values.data(),
                                      num_elements * sizeof(T),
                                      cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_indexes,
                                      tensor[i].indexes.data(),
                                      num_elements * sizeof(uint64_t),
                                      cudaMemcpyHostToDevice));

                h_arr_for_gpu[i].values = d_values;
                h_arr_for_gpu[i].indexes = d_indexes;
            } else {
                h_arr_for_gpu[i].values = nullptr;
                h_arr_for_gpu[i].indexes = nullptr;
            }
        }

        CUDA_CHECK(cudaMemcpy(d_block_arr,
                              h_arr_for_gpu.data(),
                              num_blocks * sizeof(BLCO_BLOCK_GPU<T>),
                              cudaMemcpyHostToDevice));
    } catch (...) {
        for (T* ptr : d_values_ptrs) {
            cudaFree(ptr);
        }
        for (uint64_t* ptr : d_indexes_ptrs) {
            cudaFree(ptr);
        }
        if (d_block_arr) {
            cudaFree(d_block_arr);
            d_block_arr = nullptr;
        }
        throw;
    }
}

// Free all BLCO GPU blocks previously allocated with blocks_to_gpu.
template<typename T>
inline void free_blocks_from_gpu(BLCO_BLOCK_GPU<T>* gpu_block_arr, int num_blocks)
{
    if (!gpu_block_arr || num_blocks <= 0) return;

    vector<BLCO_BLOCK_GPU<T>> h_blocks(num_blocks);
    CUDA_CHECK(cudaMemcpy(h_blocks.data(),
                          gpu_block_arr,
                          num_blocks * sizeof(BLCO_BLOCK_GPU<T>),
                          cudaMemcpyDeviceToHost));

    for (int b = 0; b < num_blocks; ++b) {
        if (h_blocks[b].values)  CUDA_CHECK(cudaFree(h_blocks[b].values));
        if (h_blocks[b].indexes) CUDA_CHECK(cudaFree(h_blocks[b].indexes));
    }

    CUDA_CHECK(cudaFree(gpu_block_arr));
}

inline pair<int,int> determine_dimensions_no_smem(uint64_t non_zeros, int wf_sz = 64)
{
    pair<int,int> dims;

    if(non_zeros <= 320){
        dims.first = 1;
        int mul = non_zeros / wf_sz;
        if(mul * wf_sz < non_zeros) mul++;
        dims.second = wf_sz * mul;
    }
    else{
        dims.first = non_zeros / 320;
        if(non_zeros % 320 != 0) dims.first++;
        dims.second = 320;
    }

    return dims;
}

template<typename T>
inline pair<int,int> determine_dimensions_smem(uint64_t non_zeros, int ui, int tensor_rank, int wf_sz = 64)
{
    pair<int,int> dims;

    if(non_zeros <= 1024){
        dims.first = 1;
        int mul = non_zeros / wf_sz;
        if(mul * wf_sz < non_zeros) mul++;
        dims.second = wf_sz * mul;
    }
    else{
        dims.first = non_zeros / 1024;
        if(non_zeros % 1024 != 0) dims.first++;
        dims.second = 1024;
    }

    size_t max_shared_mem = getMaxSharedMemory();
    int wf_per_block = dims.second / wf_sz;
    bool enough_mem = ui * tensor_rank * wf_per_block * sizeof(T) <= max_shared_mem;

    if(!enough_mem){
        int c1 = ui * tensor_rank * sizeof(T);
        for(int i = 0; i < wf_per_block; i++){
            if(--wf_per_block * c1 < static_cast<int>(max_shared_mem)) break;
        }

        dims.second = wf_per_block * wf_sz;
        dims.first = non_zeros / dims.second;
        if(non_zeros % dims.second != 0) dims.first++;
    }

    return dims;
}

__device__ inline int ceiling_log2(int x)
{
    if (x == 1) return 0;
    int res = 0;
    while (x) {
        x >>= 1;
        ++res;
    }
    return res;
}

template<typename T>
__device__ inline int find_block_index(BLCO_BLOCK_GPU<T>* tensor, int num_blocks)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int prefix_sum = 0;

    for(int i = 0; i < num_blocks; i++){
        if(idx < prefix_sum + tensor[i].size){
            return i;
        }
        prefix_sum += tensor[i].size;
    }
    return -1;
}

template<typename T>
__device__ inline uint64_t extract_linear_index(BLCO_BLOCK_GPU<T>* tensor, int thread_id, int block_idx)
{
    if(block_idx == -1) return 0;

    int prefix_sum = 0;
    for(int i = 0; i < block_idx; i++){
        prefix_sum += tensor[i].size;
    }

    return tensor[block_idx].indexes[thread_id - prefix_sum];
}

template<typename T>
__device__ inline T extract_value(BLCO_BLOCK_GPU<T>* tensor, int block_idx)
{
    if(block_idx == -1) return 0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int prefix_sum = 0;
    for(int i = 0; i < block_idx; i++){
        prefix_sum += tensor[i].size;
    }

    return static_cast<T>(tensor[block_idx].values[idx - prefix_sum]);
}

__device__ inline int extract_mode_nd(uint64_t linear_idx, int mode,
                                      const uint64_t* bitmasks,
                                      const int* bit_widths,
                                      int rank,
                                      int block)
{
    int shift = 0;
    for (int m = 0; m < mode - 1; m++) {
        shift += bit_widths[m];
    }

    uint64_t mask = bitmasks[mode - 1];
    uint64_t output = (linear_idx >> shift) & mask;

    int total_bits = 0;
    for (int m = 0; m < rank; m++) total_bits += bit_widths[m];

    if (total_bits > 64) {
        int mode_end = shift + bit_widths[mode - 1];

        if (mode_end > 64) {
            int overlap_bits = mode_end - 64;
            output |= static_cast<uint64_t>(block)
                      << (bit_widths[mode - 1] - overlap_bits);
        }
    }

    return static_cast<int>(output);
}
