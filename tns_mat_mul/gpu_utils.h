#pragma once 

#include <hip/hip_runtime.h>
#include "../tensor_storage/blco.h"
#include <hip/amd_detail/amd_hip_runtime.h>
#include <iostream>
#include <vector>
#include <stdexcept>

// Helper macro for checking HIP runtime API calls
#define HIP_CHECK(call)                                                \
    do {                                                               \
        hipError_t err = call;                                         \
        if (err != hipSuccess) {                                       \
            std::cerr << "HIP Error: " << hipGetErrorString(err)       \
                      << " at " << __FILE__ << ":" << __LINE__         \
                      << " in function " << #call << std::endl;        \
            throw std::runtime_error("HIP API call failed.");          \
        }                                                              \
    } while (0)


//Get maximum shared memory
inline size_t getMaxSharedMemory() {
    int deviceId;
    HIP_CHECK(hipGetDevice(&deviceId));

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, deviceId));

    // The key property for kernel launch limitation:
    return props.sharedMemPerBlock;
}

inline double get_gpu_memory_capacity() {
    int deviceCount = 0;
    hipGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cout << "No HIP-compatible devices found.\n";
        return -1.0;
    }

    // Assuming we check device 0
    int device = 0;
    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, device);

    // props.totalGlobalMem gives the memory size in bytes
    unsigned long long total_bytes = props.totalGlobalMem;

    // Convert bytes to GB for readability
    double total_gb = static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0);

    // You can also check available memory at runtime
    size_t free, total;
    hipMemGetInfo(&free, &total);
    double free_gb = static_cast<double>(free) / (1024.0 * 1024.0 * 1024.0);
    return free_gb;
}

inline void print_amd_gpu_model() {
    int deviceCount = 0;
    hipGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cout << "No HIP-compatible AMD devices found.\n";
        return;
    }

    // Loop through all found devices
    for (int i = 0; i < 1; i++) {
        hipDeviceProp_t props;
        hipGetDeviceProperties(&props, i);

        // The name of the GPU is stored in props.name
        std::cout << "Device: " << props.name << "\n";
        std::cout << "Arch: " << props.gcnArchName << "\n";
        std::cout << "Shared Memory: " << props.sharedMemPerBlock << "bytes\n";
        std::cout << "Currently Free VRAM: " << get_gpu_memory_capacity() << " GB\n";
    }
}

//======================================================================
// Move BLCO tensor blocks from CPU to GPU memory
//======================================================================
template<typename T>
inline void blocks_to_gpu(BLCO_BLOCK_GPU<T>*& d_block_arr,
                   const std::vector<BLCO_BLOCK_CPU<T>>& tensor,
                   int num_blocks)
{
    // Initialize output pointer to nullptr in case of early exit
    d_block_arr = nullptr;
    if (num_blocks <= 0) {
        return;
    }

    // Pointers for device memory allocated *inside* the loop
    std::vector<T*> d_values_ptrs;
    std::vector<uint64_t*> d_indexes_ptrs;

    try {
        // 1. Allocate main GPU array of BLCO_BLOCK_GPU structs
        HIP_CHECK(hipMalloc(&d_block_arr, num_blocks * sizeof(BLCO_BLOCK_GPU<T>)));

        // Temporary host-side struct array (with GPU pointers inside)
        std::vector<BLCO_BLOCK_GPU<T>> h_arr_for_gpu(num_blocks);

        // 2. Process and Copy each CPU block into GPU memory
        for (int i = 0; i < num_blocks; ++i) {
            int num_elements = tensor[i].size;
            h_arr_for_gpu[i].idx = tensor[i].idx;
            h_arr_for_gpu[i].size  = num_elements;

            if (num_elements > 0) {
                T* d_values;
                uint64_t* d_indexes;

                // Allocate device arrays for this block
                HIP_CHECK(hipMalloc(&d_values,  num_elements * sizeof(T)));
                HIP_CHECK(hipMalloc(&d_indexes, num_elements * sizeof(uint64_t)));

                // Store pointers for cleanup in case of a later failure
                d_values_ptrs.push_back(d_values);
                d_indexes_ptrs.push_back(d_indexes);

                // Copy block contents from CPU → GPU
                HIP_CHECK(hipMemcpy(d_values,  tensor[i].values.data(),  
                                    num_elements * sizeof(T), hipMemcpyHostToDevice));
                HIP_CHECK(hipMemcpy(d_indexes, tensor[i].indexes.data(), 
                                    num_elements * sizeof(uint64_t), hipMemcpyHostToDevice));

                h_arr_for_gpu[i].values  = d_values;
                h_arr_for_gpu[i].indexes = d_indexes;
            } else {
                // Empty block → null pointers
                h_arr_for_gpu[i].values = nullptr;
                h_arr_for_gpu[i].indexes = nullptr;
            }
        }

        // 3. Copy the struct array (with device pointers) into GPU memory
        HIP_CHECK(hipMemcpy(d_block_arr, h_arr_for_gpu.data(),
                            num_blocks * sizeof(BLCO_BLOCK_GPU<T>),
                            hipMemcpyHostToDevice));

    } catch (const std::exception& e) {
        // Cleanup all memory allocated up to the point of failure
        // Free memory for individual blocks
        for (T* ptr : d_values_ptrs) {
            hipFree(ptr); // Removed HIP_CHECK since we are already handling an exception
        }
        for (uint64_t* ptr : d_indexes_ptrs) {
            hipFree(ptr);
        }
        
        // Free the main array if it was successfully allocated
        if (d_block_arr != nullptr) {
            hipFree(d_block_arr);
        }
        
        // Ensure the caller's pointer is null after failure
        d_block_arr = nullptr;
        // Re-throw the exception to signal failure to the caller
        throw;
    }
}

//======================================================================
// Free BLCO tensor blocks from GPU memory
//======================================================================
template<typename T>
inline void free_blocks_from_gpu(BLCO_BLOCK_GPU<T>* gpu_block_arr, int num_blocks)
{
    if (!gpu_block_arr || num_blocks <= 0) return;

    // Copy GPU struct array back to CPU so we can access the device pointers
    std::vector<BLCO_BLOCK_GPU<T>> h_blocks(num_blocks);
    HIP_CHECK(hipMemcpy(h_blocks.data(), gpu_block_arr,
              num_blocks * sizeof(BLCO_BLOCK_GPU<T>),
              hipMemcpyDeviceToHost));

    // Free device memory for each block’s values/indexes
    for (int b = 0; b < num_blocks; ++b) {
        if (h_blocks[b].values)  HIP_CHECK(hipFree(h_blocks[b].values));
        if (h_blocks[b].indexes) HIP_CHECK(hipFree(h_blocks[b].indexes));
    }

    // Free the array of structs itself
    HIP_CHECK(hipFree(gpu_block_arr));
}


// ----------------------------
// Grid/Block Dimension Helpers
// ----------------------------

// Compute grid/block dimensions when NOT using shared memory (smem).
inline std::pair<int,int> determine_dimensions_no_smem(uint64_t non_zeros, int wf_sz = 64)
{
    std::pair<int,int> dims;

    if(non_zeros <= 320){
        dims.first = 1;  // only one block
        int mul = non_zeros / wf_sz;
        if(mul * wf_sz < non_zeros) mul++;  // round up to next multiple of wf_sz
        dims.second = wf_sz * mul;          // threads per block
    }
    else{
        dims.first = non_zeros / 320;       // number of blocks
        if(non_zeros % 320 != 0) dims.first++;
        dims.second = 320;                  // threads per block fixed at 320
    }

    return dims;
}

// Compute grid/block dimensions when using shared memory (smem).
template<typename T>
inline std::pair<int,int> determine_dimensions_smem(uint64_t non_zeros, int ui, int tensor_rank, int wf_sz = 64)
{
    std::pair<int,int> dims;

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

    // Check shared memory usage
    size_t max_shared_mem = getMaxSharedMemory();
    int wf_per_block = dims.second / wf_sz;
    bool enough_mem = ui * tensor_rank * wf_per_block * sizeof(T) <= max_shared_mem;

    if(!enough_mem){
        // Reduce wf_per_block until memory fits
        int c1 = ui * tensor_rank * sizeof(T);
        for(int i = 0; i < wf_per_block; i++){
            if(--wf_per_block * c1 < max_shared_mem) break;
        }

        // Adjust block dimensions
        dims.second = wf_per_block * wf_sz;
        dims.first = non_zeros / dims.second;
        if(non_zeros % dims.second != 0) dims.first++;
    }

    return dims;
}

// ----------------------------
// Device-Side Utilities
// ----------------------------

// Compute ceil(log2(x)) at runtime on device.
__device__ inline int ceiling_log2(int x) 
{
    if (x == 1) return 0;
    int res = 0;
    while (x) {
        x >>= 1;   // divide by 2 each step
        ++res;
    }
    return res;
}

// Find which block of the BLCO tensor the current thread's index falls into.
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
    return -1; // invalid
}

// Given a thread's global ID and the block index, return its linearized index
// from the block's index array.
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

// Same as above but retrieves the nonzero *value* instead of the index.
template<typename T>
__device__ inline T extract_value(BLCO_BLOCK_GPU<T>* tensor, int block_idx)
{
    if(block_idx == -1) return 0;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int prefix_sum = 0;
    for(int i = 0; i < block_idx; i++){
        prefix_sum += tensor[i].size;
    }
    
    // NOTE: The return type must match the function template type T
    // Since the tensor values are of type T, we cast the array access result to T.
    return (T)tensor[block_idx].values[idx - prefix_sum];
}

// Extract the coordinate for a given mode (1-based index) from a linearized index.
__device__ inline int extract_mode_nd(uint64_t linear_idx, int mode, 
    const uint64_t* bitmasks, const int* bit_widths, int rank, int block)
{
    // Compute total shift by summing bit widths of all previous modes
    int shift = 0;
    for (int m = 0; m < mode - 1; m++) {
        shift += bit_widths[m];
    }

    // Extract using mask and shift
    uint64_t mask = bitmasks[mode - 1];
    uint64_t output = (linear_idx >> shift) & mask;

    // Handle 64-bit overflow case (rare but possible for very large tensors)
    int total_bits = 0;
    for (int m = 0; m < rank; m++) total_bits += bit_widths[m];

    if (total_bits > 64) {
        // If this mode extends beyond 64 bits, use block info to reconstruct high bits
        int extra_bits = total_bits - 64;

        // Compute how many bits this mode contributes beyond 64
        int mode_start = shift;
        int mode_end = shift + bit_widths[mode - 1];

        if (mode_end > 64) {
            int overlap_bits = mode_end - 64;
            output |= static_cast<uint64_t>(block) << (bit_widths[mode - 1] - overlap_bits);
        }
    }

    return static_cast<int>(output);
}

