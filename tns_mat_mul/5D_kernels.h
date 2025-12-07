#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_runtime.h>
#include "../tensor_storage/blco.h"
#include "gpu_utils.h"

//======================================================================
// Tensor Matrix Multiply
//======================================================================
template<typename T>
__global__ void tmm_kernel_5D_sparse(int mode, BLCO_BLOCK_GPU<T>* input_tensor, uint64_t nnz, uint64_t* masks, 
T* fmat, int* dims, int num_blocks, int rank, T* output_tensor, int wavefront_size = 64) 
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int block_idx = threadIdx.x;
    int wavefront_idx = threadIdx.x % wavefront_size; 

    bool active = true;
    if (global_idx >= nnz) active = false; 
    // contracted_dim is the size of the contracted dimension
    int contracted_dim = dims[mode - 1]; 
    T thread_val = 0; //Default if thread is inactive

    // Mode indexes: Now 5
    int m1_index; int m2_index; int m3_index; int m4_index; int m5_index;
    
    // Lookup tables for 5 modes -> 4 non-target modes (1-based mode number)
    const int nt_mode1_list[5] = {2, 1, 1, 1, 1}; 
    const int nt_mode2_list[5] = {3, 3, 2, 2, 2}; 
    const int nt_mode3_list[5] = {4, 4, 4, 3, 3}; 
    const int nt_mode4_list[5] = {5, 5, 5, 5, 4}; 
    
    // Non-target mode variables: Now 4
    int i_mode = -1; 
    int nt_mode_1 = -1; int nt_mode_2 = -1; int nt_mode_3 = -1; int nt_mode_4 = -1; // NEW nt_mode_4
    
    if(active){
        //Determine linear index, value, and bit widths
        int bl_index = find_block_index(input_tensor,num_blocks);
        uint64_t lin_index = extract_linear_index(input_tensor,global_idx,bl_index);
        thread_val = extract_value(input_tensor,bl_index);
        int block = input_tensor[bl_index].idx;
        
        // Bit widths for all 5 dimensions
        int m1_bits = ceiling_log2(dims[0]), m2_bits = ceiling_log2(dims[1]), 
            m3_bits = ceiling_log2(dims[2]), m4_bits = ceiling_log2(dims[3]),
            m5_bits = ceiling_log2(dims[4]); // NEW m5_bits
        int bit_widths[5] = {m1_bits,m2_bits,m3_bits,m4_bits,m5_bits}; // Array size is 5

        // Extract indices (5 total modes)
        m1_index = extract_mode_nd(lin_index, 1, masks, bit_widths, 5, block);
        m2_index = extract_mode_nd(lin_index, 2, masks, bit_widths, 5, block);
        m3_index = extract_mode_nd(lin_index, 3, masks, bit_widths, 5, block);
        m4_index = extract_mode_nd(lin_index, 4, masks, bit_widths, 5, block);
        m5_index = extract_mode_nd(lin_index, 5, masks, bit_widths, 5, block); // NEW m5_index
        
        // Indices array size is 5
        int indices[5] = {m1_index,m2_index,m3_index,m4_index,m5_index};
        i_mode = indices[mode - 1];

        // Determine the four non-target modes (Must subtract 1 for 0-based index)
        nt_mode_1 = indices[nt_mode1_list[mode - 1] - 1];
        nt_mode_2 = indices[nt_mode2_list[mode - 1] - 1];
        nt_mode_3 = indices[nt_mode3_list[mode - 1] - 1];
        nt_mode_4 = indices[nt_mode4_list[mode - 1] - 1]; // NEW nt_mode_4
    }

    // Non-target mode masks used for thread reduction: Now 4
    unsigned long long nt_mask_1 = 0;
    unsigned long long nt_mask_2 = 0;
    unsigned long long nt_mask_3 = 0; 
    unsigned long long nt_mask_4 = 0; // New mask for 4th NT mode
    unsigned long long group_mask;
    unsigned int full_mask = (1u << wavefront_size) - 1u;
    
    // Create thread masks (Now 4 loops)
    for (int k = 0; k < wavefront_size; ++k) {
        int neighbor_nt = __shfl(nt_mode_1, k, wavefront_size);
        bool matches = neighbor_nt == nt_mode_1;
        nt_mask_1 |= ((1ULL & matches) << k);
    }
    for (int k = 0; k < wavefront_size; ++k) {
        int neighbor_nt = __shfl(nt_mode_2, k, wavefront_size);
        bool matches = neighbor_nt == nt_mode_2;
        nt_mask_2 |= ((1ULL & matches) << k);
    }
    for (int k = 0; k < wavefront_size; ++k) {
        int neighbor_nt = __shfl(nt_mode_3, k, wavefront_size);
        bool matches = neighbor_nt == nt_mode_3;
        nt_mask_3 |= ((1ULL & matches) << k);
    }
    for (int k = 0; k < wavefront_size; ++k) {
        int neighbor_nt = __shfl(nt_mode_4, k, wavefront_size);
        bool matches = neighbor_nt == nt_mode_4;
        nt_mask_4 |= ((1ULL & matches) << k);
    }

    // Group mask now combines 4 masks
    group_mask = nt_mask_1 & nt_mask_2 & nt_mask_3 & nt_mask_4; 
    bool leader = (wavefront_idx == (__ffsll(group_mask) - 1)) & active;

    for(int i = 0; i < rank; i++){
        //Determine each indice's contribution
        int fmat_flat_index = i * contracted_dim + i_mode;
        T fmat_val = fmat[fmat_flat_index];
        T contrib = fmat_val * thread_val;
        T reduction_sum = 0;
        
        //Group reduction (maintains the serial O(N) loop error)
        for (unsigned long long temp_mask = group_mask; temp_mask != 0; temp_mask &= (temp_mask - 1)){
            int lane_index = __ffsll(temp_mask) - 1;
            reduction_sum += __shfl(contrib, lane_index, wavefront_size);
        }

        // Determine the output index (5D indexing)
        int new_coords[5];
        
        // Populate new_coords array
        new_coords[mode - 1] = i; // Contracted mode gets the rank index 'i'
        
        // Non-contracted modes get their original values (1-based mode number from lists)
        new_coords[nt_mode1_list[mode - 1] - 1] = nt_mode_1;
        new_coords[nt_mode2_list[mode - 1] - 1] = nt_mode_2;
        new_coords[nt_mode3_list[mode - 1] - 1] = nt_mode_3;
        new_coords[nt_mode4_list[mode - 1] - 1] = nt_mode_4; // NEW nt_mode_4 coord

        //New dimensions
        int local_dims[5];
        for (int d = 0; d < 5; ++d) {
            local_dims[d] = dims[d];
        }
        local_dims[mode - 1] = rank;

        // 5D Row-Major Indexing:
        int output_index = new_coords[0] * local_dims[1] * local_dims[2] * local_dims[3] * local_dims[4] + 
                           new_coords[1] * local_dims[2] * local_dims[3] * local_dims[4] + 
                           new_coords[2] * local_dims[3] * local_dims[4] + 
                           new_coords[3] * local_dims[4] + 
                           new_coords[4];

        T* output_address = &(output_tensor[output_index]);

        //Group leaders atomically add the group contribution
        if(leader){
            if constexpr (std::is_same_v<T, double>) {
                atomicAdd_f64(output_address, reduction_sum);
            } else if constexpr (std::is_same_v<T, float>) {
                atomicAdd(output_address, reduction_sum);
            } else if constexpr (std::is_same_v<T, int>){
                atomicAdd(output_address, reduction_sum);
            }
        }
    }

    return;
}

template<typename T, typename S>
T* tmm_5D(const Blco_Tensor<T,S>& sparse_tensor, int mode, int block_size, bool print = true)
{
    const std::vector<BLCO_BLOCK_CPU<T>> blco_tensor = sparse_tensor.get_blco();

    // 1. Dims and Rank
    std::vector<int> dims = sparse_tensor.get_dims(); 
    const int decomp_rank = sparse_tensor.get_factor_rank();
    const int non_zeros = sparse_tensor.get_nnz();
    const int RANK_DIMS = 5;

    // Safety check for mode
    if (mode < 1 || mode > RANK_DIMS) {
        std::cerr << "Error: Invalid mode specified (" << mode << ") for " 
                  << RANK_DIMS << "-dimensional tensor." << std::endl;
        return nullptr;
    }

    // Masks
    std::vector<uint64_t> masks = sparse_tensor.get_bitmasks();

    // Fmats
    std::vector<T*> fmats = sparse_tensor.get_fmats();
    T* mode_1_fmat = fmats[0];
    T* mode_2_fmat = fmats[1];
    T* mode_3_fmat = fmats[2];
    T* mode_4_fmat = fmats[3]; 
    T* mode_5_fmat = fmats[4]; 

    // Number of blocks
    int num_blocks = blco_tensor.size();

    // Device pointers
    BLCO_BLOCK_GPU<T>* d_input_tensor;
    uint64_t* d_masks;
    T* d_fmat_1; T* d_fmat_2; T* d_fmat_3; T* d_fmat_4; T* d_fmat_5;
    int* device_dims;

    // Allocate shared data
    HIP_CHECK(hipMalloc(&d_input_tensor, sizeof(BLCO_BLOCK_GPU<T>) * num_blocks));
    
    // Allocate all 5 factor matrices (Size: Dim[i] * decomp_rank)
    HIP_CHECK(hipMalloc(&d_fmat_1, sizeof(T) * dims[0] * decomp_rank));
    HIP_CHECK(hipMalloc(&d_fmat_2, sizeof(T) * dims[1] * decomp_rank));
    HIP_CHECK(hipMalloc(&d_fmat_3, sizeof(T) * dims[2] * decomp_rank));
    HIP_CHECK(hipMalloc(&d_fmat_4, sizeof(T) * dims[3] * decomp_rank));
    HIP_CHECK(hipMalloc(&d_fmat_5, sizeof(T) * dims[4] * decomp_rank)); 
    
    // Allocate space for 5 dimensions
    HIP_CHECK(hipMalloc(&device_dims, sizeof(int) * RANK_DIMS)); 
    // Allocate space for 5 masks
    HIP_CHECK(hipMalloc(&d_masks, sizeof(uint64_t) * RANK_DIMS)); 

    // Copy host data to GPU
    blocks_to_gpu(d_input_tensor, blco_tensor, num_blocks);
    
    HIP_CHECK(hipMemcpy(d_fmat_1, mode_1_fmat, sizeof(T) * dims[0] * decomp_rank, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_fmat_2, mode_2_fmat, sizeof(T) * dims[1] * decomp_rank, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_fmat_3, mode_3_fmat, sizeof(T) * dims[2] * decomp_rank, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_fmat_4, mode_4_fmat, sizeof(T) * dims[3] * decomp_rank, hipMemcpyHostToDevice)); 
    HIP_CHECK(hipMemcpy(d_fmat_5, mode_5_fmat, sizeof(T) * dims[4] * decomp_rank, hipMemcpyHostToDevice)); // New 5th copy
    HIP_CHECK(hipMemcpy(device_dims, dims.data(), sizeof(int) * RANK_DIMS, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_masks, masks.data(), sizeof(uint64_t) * RANK_DIMS, hipMemcpyHostToDevice));

    std::pair<int,int> dimensions;
    dimensions.second = block_size;
    dimensions.first = (non_zeros + block_size - 1) / block_size;
    T* host_output_tensor;
    T* d_output_tensor;
    uint64_t output_size;
    T* d_fmat_launch = nullptr; 

    // --- Output Tensor Allocation and Factor Matrix Selection (5 Cases) ---
    
    // Calculate the total number of elements in the output tensor
    output_size = (uint64_t)decomp_rank; // Start with the rank
    for (int i = 0; i < RANK_DIMS; ++i) {
        if (i != mode - 1) { // Multiply by all non-contracted dimensions
            output_size *= dims[i];
        }
    }

    // Select the factor matrix to pass to the kernel
    if (mode == 1)      d_fmat_launch = d_fmat_1;
    else if (mode == 2) d_fmat_launch = d_fmat_2;
    else if (mode == 3) d_fmat_launch = d_fmat_3;
    else if (mode == 4) d_fmat_launch = d_fmat_4;
    else if (mode == 5) d_fmat_launch = d_fmat_5;
    
    // Allocate and initialize output memory
    host_output_tensor = (T*)calloc(output_size, sizeof(T));
    HIP_CHECK(hipMalloc(&d_output_tensor, sizeof(T) * output_size));
    HIP_CHECK(hipMemcpy(d_output_tensor, host_output_tensor, sizeof(T) * output_size, hipMemcpyHostToDevice));

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    
    // Record start
    HIP_CHECK(hipEventRecord(start, 0));

    // Launch the 5D kernel
    hipLaunchKernelGGL(
        tmm_kernel_5D_sparse<T>,  // Assuming this is the 5D kernel
        dim3(dimensions.first), dim3(dimensions.second), 0, 0, mode, d_input_tensor, non_zeros,
        d_masks, d_fmat_launch, device_dims, num_blocks, decomp_rank, d_output_tensor
    );

    // Record stop
    HIP_CHECK(hipEventRecord(stop, 0));
    HIP_CHECK(hipEventSynchronize(stop));

    // Compute elapsed time in ms
    float milliseconds = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));

    if(print) std::cout << "Kernel Duration: " << milliseconds << " ms\n";
    
    // Copy result back
    HIP_CHECK(hipMemcpy(host_output_tensor, d_output_tensor, sizeof(T) * output_size, hipMemcpyDeviceToHost));

    // Free device memory (Now 5 factor matrices)
    free_blocks_from_gpu(d_input_tensor,num_blocks);
    HIP_CHECK(hipFree(d_output_tensor));
    HIP_CHECK(hipFree(d_fmat_1));
    HIP_CHECK(hipFree(d_fmat_2));
    HIP_CHECK(hipFree(d_fmat_3));
    HIP_CHECK(hipFree(d_fmat_4));
    HIP_CHECK(hipFree(d_fmat_5));
    HIP_CHECK(hipFree(device_dims)); 
    HIP_CHECK(hipFree(d_masks)); 

    return host_output_tensor;
}


//======================================================================
// Core tensor calculations
//======================================================================
template<typename T>
__global__ void tucker_core_kernel_5D_sparse(
    BLCO_BLOCK_GPU<T>* input_tensor, uint64_t nnz, uint64_t* masks, 
    T* d_U1, T* d_U2, T* d_U3, T* d_U4, T* d_U5, // Added U5
    int* dims, int num_blocks, int rank, T* output_tensor, 
    int shmem_size, int wavefront_size = 64) 
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int block_idx = threadIdx.x;

    // The output core tensor G is R1 x R2 x R3 x R4 x R5
    const int R1 = rank;
    const int R2 = rank;
    const int R3 = rank;
    const int R4 = rank;
    const int R5 = rank;
    
    // Size of block tensor is R1 * R2 * R3 * R4 * R5
    extern __shared__ __align__(sizeof(T)) unsigned char smem_raw[];
    T* block_tensor = reinterpret_cast<T*>(smem_raw);

    // Initialize block tensor to all 0's
    const int output_size = R1 * R2 * R3 * R4 * R5; // Updated output size for 5D
    for(int i = block_idx; i < output_size; i += blockDim.x){
        block_tensor[i] = 0;
    }
    
    bool active = true;
    if (global_idx >= nnz) active = false; 
    
    T thread_val = 0; // The value of the non-zero element x_i1,i2,i3,i4,i5
    int m1_index = -1; // i1
    int m2_index = -1; // i2
    int m3_index = -1; // i3
    int m4_index = -1; // i4
    int m5_index = -1; // i5 

    if(active){
        // Determine linear index, value, and bit widths
        int bl_index = find_block_index(input_tensor,num_blocks);
        uint64_t lin_index = extract_linear_index(input_tensor,global_idx,bl_index);
        thread_val = extract_value(input_tensor,bl_index);
        int block = input_tensor[bl_index].idx;
        
        // Bit widths for all 5 dimensions
        int m1_bits = ceiling_log2(dims[0]);
        int m2_bits = ceiling_log2(dims[1]);
        int m3_bits = ceiling_log2(dims[2]);
        int m4_bits = ceiling_log2(dims[3]);
        int m5_bits = ceiling_log2(dims[4]); // New bit width for D5
        
        const int RANK_DIMS = 5; // Total number of dimensions
        int bit_widths[RANK_DIMS] = {m1_bits, m2_bits, m3_bits, m4_bits, m5_bits}; 

        // Extract indices (i1, i2, i3, i4, i5) from the sparse tensor
        m1_index = extract_mode_nd(lin_index, 1, masks, bit_widths, RANK_DIMS, block);
        m2_index = extract_mode_nd(lin_index, 2, masks, bit_widths, RANK_DIMS, block);
        m3_index = extract_mode_nd(lin_index, 3, masks, bit_widths, RANK_DIMS, block);
        m4_index = extract_mode_nd(lin_index, 4, masks, bit_widths, RANK_DIMS, block);
        m5_index = extract_mode_nd(lin_index, 5, masks, bit_widths, RANK_DIMS, block); // New Index Extraction
    }
    
    // --- Core Tensor Calculation Loops (5 Nested Loops) ---
        
    for(int r1 = 0; r1 < R1; r1++){
        T u1_val = d_U1[r1 * dims[0] + m1_index * active];
        
        for(int r2 = 0; r2 < R2; r2++){
            T u2_val = d_U2[r2 * dims[1] + m2_index * active];
            
            for(int r3 = 0; r3 < R3; r3++){
                T u3_val = d_U3[r3 * dims[2] + m3_index * active];
                
                for(int r4 = 0; r4 < R4; r4++){
                    T u4_val = d_U4[r4 * dims[3] + m4_index * active]; 
                    
                    // New innermost loop for r5
                    for(int r5 = 0; r5 < R5; r5++){
                        T u5_val = d_U5[r5 * dims[4] + m5_index * active];

                        // Total contribution to G(r1, r2, r3, r4, r5)
                        T contrib = thread_val * u1_val * u2_val * u3_val * u4_val * u5_val * active;
                        
                        // Core Tensor Row-Major Indexing (R1 x R2 x R3 x R4 x R5)
                        // Index = r1*R2*R3*R4*R5 + r2*R3*R4*R5 + r3*R4*R5 + r4*R5 + r5
                        int output_index = r1 * R2 * R3 * R4 * R5 + 
                                           r2 * R3 * R4 * R5 + 
                                           r3 * R4 * R5 + 
                                           r4 * R5 + 
                                           r5;
                        
                        T* output_address;
                        
                        if(output_index < shmem_size) output_address = &(block_tensor[output_index]);
                        else output_address = &(output_tensor[output_index]);

                        // Atomic add to shared memory
                        if constexpr (std::is_same_v<T, double>) {
                            atomicAdd_f64(output_address, contrib);
                        } else if constexpr (std::is_same_v<T, float>) {
                            safeAtomicAdd(output_address, contrib);
                        } else if constexpr (std::is_same_v<T, int>){
                            atomicAdd(output_address, contrib);
                        }
                    } // End r5
                } // End r4
            } // End r3
        } // End r2
    } // End r1

    __builtin_amdgcn_s_barrier();

    for(int i = block_idx; i < shmem_size; i += blockDim.x){
        T output = block_tensor[i];
        T* output_address = &(output_tensor[i]);
        if constexpr (std::is_same_v<T, double>) {
            atomicAdd_f64(output_address, output);
        } else if constexpr (std::is_same_v<T, float>) {
            atomicAdd(output_address, output);
        } else if constexpr (std::is_same_v<T, int>){
            atomicAdd(output_address, output);
        }
    }

    return;
}

// Core tensor calculation launch
template<typename T, typename S>
T* tucker_compute_core_5D(const Blco_Tensor<T,S>& sparse_tensor, int block_size)
{
    const std::vector<BLCO_BLOCK_CPU<T>> blco_tensor = sparse_tensor.get_blco();

    // Rows, cols, rank, and non zeros
    std::vector<int> dims = sparse_tensor.get_dims();
    const int decomp_rank = sparse_tensor.get_factor_rank();
    const int non_zeros = sparse_tensor.get_nnz();
    const int RANK_DIMS = 5;

    // Masks
    std::vector<uint64_t> masks = sparse_tensor.get_bitmasks();

    // Fmats (All three needed for core tensor calculation)
    std::vector<T*> fmats = sparse_tensor.get_fmats();
    T* mode_1_fmat = fmats[0];
    T* mode_2_fmat = fmats[1];
    T* mode_3_fmat = fmats[2];
    T* mode_4_fmat = fmats[3];
    T* mode_5_fmat = fmats[4];

    // Number of blocks
    int num_blocks = blco_tensor.size();

    // Device pointers
    BLCO_BLOCK_GPU<T>* d_input_tensor;
    uint64_t* d_masks;
    T* d_fmat_1; T* d_fmat_2; T* d_fmat_3; T* d_fmat_4; T* d_fmat_5;
    int* device_dims;

    // Allocate
    HIP_CHECK(hipMalloc(&d_input_tensor, sizeof(BLCO_BLOCK_GPU<T>) * num_blocks));
    HIP_CHECK(hipMalloc(&d_fmat_1, sizeof(T) * dims[0] * decomp_rank));
    HIP_CHECK(hipMalloc(&d_fmat_2, sizeof(T) * dims[1] * decomp_rank));
    HIP_CHECK(hipMalloc(&d_fmat_3, sizeof(T) * dims[2] * decomp_rank));
    HIP_CHECK(hipMalloc(&d_fmat_4, sizeof(T) * dims[3] * decomp_rank));
    HIP_CHECK(hipMalloc(&d_fmat_5, sizeof(T) * dims[4] * decomp_rank));
    HIP_CHECK(hipMalloc(&device_dims, sizeof(int) * RANK_DIMS));
    HIP_CHECK(hipMalloc(&d_masks, sizeof(uint64_t) * RANK_DIMS));

    // Copy host data to GPU
    blocks_to_gpu(d_input_tensor, blco_tensor, num_blocks);
    HIP_CHECK(hipMemcpy(d_fmat_1, mode_1_fmat, sizeof(T) * dims[0] * decomp_rank, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_fmat_2, mode_2_fmat, sizeof(T) * dims[1] * decomp_rank, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_fmat_3, mode_3_fmat, sizeof(T) * dims[2] * decomp_rank, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_fmat_4, mode_4_fmat, sizeof(T) * dims[3] * decomp_rank, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_fmat_5, mode_5_fmat, sizeof(T) * dims[3] * decomp_rank, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_dims, dims.data(), sizeof(int) * RANK_DIMS, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_masks, masks.data(), sizeof(uint64_t) * RANK_DIMS, hipMemcpyHostToDevice));

    std::pair<int,int> dimensions;
    dimensions.second = block_size;
    dimensions.first = (non_zeros + block_size - 1) / block_size;
    T* host_output_tensor;
    T* d_output_tensor;
    int output_size;

    // --- Core Tensor Output Size Calculation ---
    // The Core Tensor G has dimensions R1 x R2 x R3 x R4 x R5.
    // Assuming R1 = R2 = R3 = R4 = R5 = decomp_rank.
    output_size = decomp_rank * decomp_rank * decomp_rank * decomp_rank * decomp_rank;

    // Allocate and initialize output memory for the Core Tensor G
    host_output_tensor = (T*)calloc(output_size, sizeof(T));
    HIP_CHECK(hipMalloc(&d_output_tensor, sizeof(T) * output_size));
    HIP_CHECK(hipMemcpy(d_output_tensor, host_output_tensor, sizeof(T) * output_size, hipMemcpyHostToDevice));
    
    size_t max_shmem = getMaxSharedMemory();
    size_t shared_mem_bytes = std::min(sizeof(T) * output_size, max_shmem);
    int store_size = shared_mem_bytes / sizeof(T);

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    
    // Record start
    HIP_CHECK(hipEventRecord(start, 0));

    // Launch the Core Tensor kernel (Note: Removed 'mode' and replaced single fmat with all three)
    hipLaunchKernelGGL(
        tucker_core_kernel_5D_sparse<T>,  
        dim3(dimensions.first), dim3(dimensions.second), 
        shared_mem_bytes, // Shared memory size 
        0, // Stream
        d_input_tensor, non_zeros, d_masks, 
        d_fmat_1, d_fmat_2, d_fmat_3, d_fmat_4, d_fmat_5, 
        device_dims, num_blocks, decomp_rank, d_output_tensor, 
        store_size
    );

    // Record stop
    HIP_CHECK(hipEventRecord(stop, 0));
    HIP_CHECK(hipEventSynchronize(stop));

    float milliseconds = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));

    std::cout << "Kernel Duration: " << milliseconds << " ms\n";
    
    // Copy result back
    HIP_CHECK(hipMemcpy(host_output_tensor, d_output_tensor, sizeof(T) * output_size, hipMemcpyDeviceToHost));

    // Free device memory
    free_blocks_from_gpu(d_input_tensor,num_blocks);
    HIP_CHECK(hipFree(d_output_tensor));
    HIP_CHECK(hipFree(d_fmat_1));
    HIP_CHECK(hipFree(d_fmat_2));
    HIP_CHECK(hipFree(d_fmat_3));
    HIP_CHECK(hipFree(d_fmat_4));
    HIP_CHECK(hipFree(d_fmat_5));
    HIP_CHECK(hipFree(device_dims)); 
    HIP_CHECK(hipFree(d_masks)); 

    return host_output_tensor;
}