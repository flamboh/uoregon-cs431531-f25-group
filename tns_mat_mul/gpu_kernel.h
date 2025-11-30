#include <hip/hip_runtime.h>
#include "../tensor_storage/blco.h"
#include <hip/amd_detail/amd_hip_runtime.h>

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
size_t getMaxSharedMemory() {
    int deviceId;
    HIP_CHECK(hipGetDevice(&deviceId));

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, deviceId));

    // The key property for kernel launch limitation:
    return props.sharedMemPerBlock;
}

//======================================================================
// Move BLCO tensor blocks from CPU to GPU memory
//======================================================================
template<typename T>
void blocks_to_gpu(BLCO_BLOCK_GPU<T>*& d_block_arr,
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
            HIP_CHECK(hipFree(ptr));
        }
        for (uint64_t* ptr : d_indexes_ptrs) {
            HIP_CHECK(hipFree(ptr));
        }
        
        // Free the main array if it was successfully allocated
        if (d_block_arr != nullptr) {
            HIP_CHECK(hipFree(d_block_arr));
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
void free_blocks_from_gpu(BLCO_BLOCK_GPU<T>* gpu_block_arr, int num_blocks)
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
// Decides how many thread blocks (dims.first) and how many threads per block (dims.second)
// should be launched based on the number of nonzeros and a default wavefront size.
//
// Strategy: 
//  - If the tensor is small (≤ 320 nonzeros), launch just 1 block with enough threads
//    to cover all nonzeros, rounded up to a multiple of wf_sz.
//  - Otherwise, assign 320 threads per block and compute how many blocks are needed.
std::pair<int,int> determine_dimensions_no_smem(uint64_t non_zeros, int wf_sz = 64)
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
// Adds an extra check to ensure the shared memory requirement fits within MAX_SHARED_MEM.
//
// Parameters:
//   - non_zeros: number of nonzero entries in tensor
//   - ui: number of unique target indices per wavefront
//   - tensor_rank: rank (width) of the factor matrices
//   - wf_sz: wavefront size (default 64 on AMD GPUs)
//
// Logic:
//  - Similar to above, but with a max threads-per-block of 1024.
//  - Check whether smem usage (ui * rank * wf_per_block * sizeof(T)) fits within GPU limits.
//  - If not, reduce wf_per_block until memory fits.
template<typename T>
std::pair<int,int> determine_dimensions_smem(uint64_t non_zeros, int ui, int tensor_rank, int wf_sz = 64)
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
// Used to determine how many bits are required to represent dimensions.
__device__ int ceiling_log2(int x) 
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
// Iterates through block sizes until the prefix sum exceeds the thread's global index.
template<typename T>
__device__ int find_block_index(BLCO_BLOCK_GPU<T>* tensor, int num_blocks)
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
__device__ uint64_t extract_linear_index(BLCO_BLOCK_GPU<T>* tensor, int thread_id, int block_idx)
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
__device__ uint64_t extract_value(BLCO_BLOCK_GPU<T>* tensor, int block_idx)
{
    if(block_idx == -1) return 0;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int prefix_sum = 0;
    for(int i = 0; i < block_idx; i++){
        prefix_sum += tensor[i].size;
    }
    
    return tensor[block_idx].values[idx - prefix_sum];
}

// Extract the coordinate for a given mode (1-based index) from a linearized index.
// Supports arbitrary N-dimensional tensors by using per-mode bitmasks and bit widths.
//
// Parameters:
//   linear_idx  - linearized index of the tensor element
//   mode        - which mode to extract (1-based)
//   bitmasks    - array of mode-specific bitmasks
//   bit_widths  - array of bit widths per mode
//   rank        - total number of tensor modes
//   block       - block index (for cases where bits exceed 64)
//
__device__ int extract_mode_nd(uint64_t linear_idx, int mode, 
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

//======================================================================
// 3D Functions
//======================================================================
template<typename T>
__global__ void tmm_kernel_3D_sparse(int mode, BLCO_BLOCK_GPU<T>* input_tensor, uint64_t nnz, uint64_t* masks, 
T* fmat, int* dims, int num_blocks, int rank, T* output_tensor, int wavefront_size = 64) 
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int block_idx = threadIdx.x;
    int wavefront_idx = threadIdx.x % wavefront_size; 

    bool active = true;
    if (global_idx >= nnz) active = false; 
    int contracted_dim = dims[mode - 1];
    T thread_val = 0; //Default if thread is inactive

    int m1_index; int m2_index; int m3_index; //Mode indexes
    // Lookup tables for mode → nt_mode1, nt_mode2
    const int nt_mode1_list[3] = {2, 1, 1};
    const int nt_mode2_list[3] = {3, 3, 2};
    int i_mode = -1; int nt_mode_1 = -1; int nt_mode_2 = -1;
    if(active){
        //Determine linear index, value, and bit widths
        int bl_index = find_block_index(input_tensor,num_blocks);
        uint64_t lin_index = extract_linear_index(input_tensor,global_idx,bl_index);
        thread_val = extract_value(input_tensor,bl_index);
        int block = input_tensor[bl_index].idx;
        int m1_bits = ceiling_log2(dims[0]), m2_bits = ceiling_log2(dims[1]), m3_bits = ceiling_log2(dims[2]);
        int bit_widths[3] = {m1_bits,m2_bits,m3_bits};

        //Extract indices
        m1_index = extract_mode_nd(lin_index, 1, masks, bit_widths, 3, block);
        m2_index = extract_mode_nd(lin_index, 2, masks, bit_widths, 3, block);
        m3_index = extract_mode_nd(lin_index, 3, masks, bit_widths, 3, block);
        int indices[3] = {m1_index,m2_index,m3_index};
        i_mode = indices[mode - 1];

        //Determine the two non target nodes
        nt_mode_1 = indices[nt_mode1_list[mode - 1] - 1];
        nt_mode_2 = indices[nt_mode2_list[mode - 1] - 1];
    }

    //Non target mode masks used for thread reduction
    unsigned long long nt_mask_1 = 0;
    unsigned long long nt_mask_2 = 0;
    unsigned long long group_mask;
    unsigned int full_mask = (1u << wavefront_size) - 1u;
    
    //Create thread masks
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
    group_mask = nt_mask_1 & nt_mask_2;
    bool leader = (wavefront_idx == (__ffsll(group_mask) - 1)) & active;

    for(int i = 0; i < rank; i++){
        //Determine each indice's contribution
        int fmat_flat_index = i * contracted_dim + i_mode;
        T fmat_val = fmat[fmat_flat_index];
        T contrib = fmat_val * thread_val;
        T reduction_sum = 0;
        
        //Group reduction
        for (unsigned long long temp_mask = group_mask; temp_mask != 0; temp_mask &= (temp_mask - 1)){
            int lane_index = __ffsll(temp_mask) - 1;
            reduction_sum += __shfl(contrib, lane_index, wavefront_size);
        }

        //Determine the output index
        int new_coords[3];
        new_coords[mode - 1] = i;
        new_coords[nt_mode1_list[mode - 1] - 1] = nt_mode_1;
        new_coords[nt_mode2_list[mode - 1] - 1] = nt_mode_2;
        
        //Determine new dimensions
        int local_dims[3];
        for (int d = 0; d < 3; ++d) {
            local_dims[d] = dims[d];
        }
        local_dims[mode - 1] = rank;

        int output_index = new_coords[0] * local_dims[1] * local_dims[2] + new_coords[1] * local_dims[2] + new_coords[2];
        T* output_address = &(output_tensor[output_index]);

        //Group leaders atomically add the group contribution
        if(leader){
            if constexpr (std::is_same_v<T, double>) {
                atomicAdd_f64(output_address, reduction_sum);
            } else if constexpr (std::is_same_v<T, float>) {
                safeAtomicAdd(output_address, reduction_sum);
            } else if constexpr (std::is_same_v<T, int>){
                atomicAdd(output_address, reduction_sum);
            }
        }
    }

    return;
}

template<typename T, typename S>
T* tmm_3D(const Blco_Tensor<T,S>& sparse_tensor, int mode)
{
    const std::vector<BLCO_BLOCK_CPU<T>> blco_tensor = sparse_tensor.get_blco();

    //Rows, cols, rank, and non zeros
    std::vector<int> dims = sparse_tensor.get_dims();
    const int decomp_rank = sparse_tensor.get_factor_rank();
    const int non_zeros = sparse_tensor.get_nnz();

    //Masks
    std::vector<uint64_t> masks = sparse_tensor.get_bitmasks();

    //Fmats
    std::vector<T*> fmats = sparse_tensor.get_fmats();
    T* mode_1_fmat = fmats[0];
    T* mode_2_fmat = fmats[1];
    T* mode_3_fmat = fmats[2];

    //Number of blocks
    int num_blocks = blco_tensor.size();
    bool blocked = num_blocks > 1;

    // Device pointers
    BLCO_BLOCK_GPU<T>* d_input_tensor;
    uint64_t* d_masks;
    T* d_fmat_1; T* d_fmat_2; T* d_fmat_3;
    int* device_dims;

    // Allocate
    HIP_CHECK(hipMalloc(&d_input_tensor, sizeof(BLCO_BLOCK_GPU<T>) * num_blocks));
    HIP_CHECK(hipMalloc(&d_fmat_1, sizeof(T) * dims[0] * decomp_rank));
    HIP_CHECK(hipMalloc(&d_fmat_2, sizeof(T) * dims[1] * decomp_rank));
    HIP_CHECK(hipMalloc(&d_fmat_3, sizeof(T) * dims[2] * decomp_rank));
    HIP_CHECK(hipMalloc(&device_dims, sizeof(int) * 3));
    HIP_CHECK(hipMalloc(&d_masks, sizeof(uint64_t) * 3));

    // Copy host data to GPU
    blocks_to_gpu(d_input_tensor, blco_tensor, num_blocks);
    HIP_CHECK(hipMemcpy(d_fmat_1, mode_1_fmat, sizeof(T) * dims[0] * decomp_rank, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_fmat_2, mode_2_fmat, sizeof(T) * dims[1] * decomp_rank, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_fmat_3, mode_3_fmat, sizeof(T) * dims[2] * decomp_rank, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_dims, dims.data(), sizeof(int) * 3, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_masks, masks.data(), sizeof(uint64_t) * 3, hipMemcpyHostToDevice));

    std::pair<int,int> dimensions = determine_dimensions_no_smem(non_zeros);
    T* host_output_tensor;
    T* d_output_tensor;
    int output_size;
    T* d_fmat_launch = nullptr; // Pointer to the fmat being used for launch

    if (mode == 1) {
        output_size = decomp_rank * dims[1] * dims[2] * dims[3];
        d_fmat_launch = d_fmat_1;
    } else if (mode == 2) {
        output_size = dims[0] * decomp_rank * dims[2] * dims[3];
        d_fmat_launch = d_fmat_2;
    } else if (mode == 3) {
        output_size = dims[0] * dims[1] * decomp_rank * dims[3];
        d_fmat_launch = d_fmat_3;
    } else {
        return nullptr;
    }

    // Allocate and initialize output memory for the selected mode
    host_output_tensor = (T*)calloc(output_size, sizeof(T));
    HIP_CHECK(hipMalloc(&d_output_tensor, sizeof(T) * output_size));
    HIP_CHECK(hipMemcpy(d_output_tensor, host_output_tensor, sizeof(T) * output_size, hipMemcpyHostToDevice));

    // Launch the 3D kernel
    hipLaunchKernelGGL(
        tmm_kernel_3D_sparse<T>,  // Assuming this is the new 4D kernel
        dim3(dimensions.first), dim3(dimensions.second), 0, 0, mode, d_input_tensor, non_zeros,
        d_masks, d_fmat_launch, device_dims, num_blocks, decomp_rank, d_output_tensor
    );

    // Free device memory
    free_blocks_from_gpu(d_input_tensor,num_blocks);
    HIP_CHECK(hipFree(d_output_tensor));
    HIP_CHECK(hipFree(d_fmat_1));
    HIP_CHECK(hipFree(d_fmat_2));
    HIP_CHECK(hipFree(d_fmat_3));

    return host_output_tensor;
}

// Core tensor calculation kernel
template<typename T>
__global__ void tucker_core_kernel_3D_sparse(
    BLCO_BLOCK_GPU<T>* input_tensor, uint64_t nnz, uint64_t* masks, 
    T* d_U1, T* d_U2, T* d_U3, // Factor matrices for all 3 modes (U1, U2, U3)
    int* dims, int num_blocks, int rank, T* output_tensor, int wavefront_size = 64) 
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int block_idx = threadIdx.x;
    int wavefront_idx = threadIdx.x % wavefront_size; 

    //Size of block tensor is R1 * R2 * R3
    extern __shared__ __align__(sizeof(T)) unsigned char smem_raw[];
    T* block_tensor = reinterpret_cast<T*>(smem_raw);

    //Initialize block tensor to all 0's
    const int output_size = rank * rank * rank;
    for(int i = block_idx; i < output_size; i += blockDim.x){
        block_tensor[i] = 0;
    }
    
    // Decomp rank is the size of R1, R2, R3 (assumed equal to 'rank' parameter)
    const int R1 = rank;
    const int R2 = rank;
    const int R3 = rank;
    
    bool active = true;
    if (global_idx >= nnz) active = false; 
    
    T thread_val = 0; // The value of the non-zero element x_i1,i2,i3
    int m1_index = -1; // i1
    int m2_index = -1; // i2
    int m3_index = -1; // i3

    // The non-target grouping logic and variables are removed entirely.

    if(active){
        // Determine linear index, value, and bit widths
        int bl_index = find_block_index(input_tensor,num_blocks);
        uint64_t lin_index = extract_linear_index(input_tensor,global_idx,bl_index);
        thread_val = extract_value(input_tensor,bl_index);
        int block = input_tensor[bl_index].idx;
        
        // Bit widths for all 3 dimensions
        int m1_bits = ceiling_log2(dims[0]), m2_bits = ceiling_log2(dims[1]), m3_bits = ceiling_log2(dims[2]);
        int bit_widths[3] = {m1_bits,m2_bits,m3_bits};

        // Extract indices (i1, i2, i3) from the sparse tensor
        m1_index = extract_mode_nd(lin_index, 1, masks, bit_widths, 3, block);
        m2_index = extract_mode_nd(lin_index, 2, masks, bit_widths, 3, block);
        m3_index = extract_mode_nd(lin_index, 3, masks, bit_widths, 3, block);
    }
    
    // --- Core Tensor Calculation Loop ---
        
    for(int r1 = 0; r1 < R1; r1++){
        // U1(i1, r1) = d_U1[r1 * I1 + i1] (Assuming column-major Factor Matrix storage, but using your linear index calculation: i * contracted_dim + i_mode)
        T u1_val = d_U1[m1_index * R1 + r1 * active];
        
        for(int r2 = 0; r2 < R2; r2++){
            T u2_val = d_U2[m2_index * R2 + r2 * active];
            
            for(int r3 = 0; r3 < R3; r3++){
                T u3_val = d_U3[m3_index * R3 + r3 * active];

                // Total contribution to G(r1, r2, r3)
                T contrib = thread_val * u1_val * u2_val * u3_val * active;
                
                // Core Tensor Row-Major Indexing (R1 x R2 x R3)
                // Index = r1 * R2 * R3 + r2 * R3 + r3
                int output_index = r1 * R2 * R3 + r2 * R3 + r3;
                T* output_address = &(block_tensor[output_index]);

                if constexpr (std::is_same_v<T, double>) {
                    atomicAdd_f64(output_address, contrib);
                } else if constexpr (std::is_same_v<T, float>) {
                    // Using the corrected function name for float atomic add
                    atomicAdd(output_address, contrib);
                } else if constexpr (std::is_same_v<T, int>){
                    atomicAdd(output_address, contrib);
                }
            }
        }
    }

    for(int i = block_idx; i < output_size; i += blockDim.x){
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
T* tucker_compute_core_3D(const Blco_Tensor<T,S>& sparse_tensor)
{
    const std::vector<BLCO_BLOCK_CPU<T>> blco_tensor = sparse_tensor.get_blco();

    // Rows, cols, rank, and non zeros
    std::vector<int> dims = sparse_tensor.get_dims();
    const int decomp_rank = sparse_tensor.get_factor_rank();
    const int non_zeros = sparse_tensor.get_nnz();
    const int RANK_DIMS = 3;

    // Masks
    std::vector<uint64_t> masks = sparse_tensor.get_bitmasks();

    // Fmats (All three needed for core tensor calculation)
    std::vector<T*> fmats = sparse_tensor.get_fmats();
    T* mode_1_fmat = fmats[0];
    T* mode_2_fmat = fmats[1];
    T* mode_3_fmat = fmats[2];

    // Number of blocks
    int num_blocks = blco_tensor.size();

    // Device pointers
    BLCO_BLOCK_GPU<T>* d_input_tensor;
    uint64_t* d_masks;
    T* d_fmat_1; T* d_fmat_2; T* d_fmat_3;
    int* device_dims;

    // Allocate
    HIP_CHECK(hipMalloc(&d_input_tensor, sizeof(BLCO_BLOCK_GPU<T>) * num_blocks));
    HIP_CHECK(hipMalloc(&d_fmat_1, sizeof(T) * dims[0] * decomp_rank));
    HIP_CHECK(hipMalloc(&d_fmat_2, sizeof(T) * dims[1] * decomp_rank));
    HIP_CHECK(hipMalloc(&d_fmat_3, sizeof(T) * dims[2] * decomp_rank));
    HIP_CHECK(hipMalloc(&device_dims, sizeof(int) * RANK_DIMS));
    HIP_CHECK(hipMalloc(&d_masks, sizeof(uint64_t) * RANK_DIMS));

    // Copy host data to GPU
    blocks_to_gpu(d_input_tensor, blco_tensor, num_blocks);
    HIP_CHECK(hipMemcpy(d_fmat_1, mode_1_fmat, sizeof(T) * dims[0] * decomp_rank, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_fmat_2, mode_2_fmat, sizeof(T) * dims[1] * decomp_rank, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_fmat_3, mode_3_fmat, sizeof(T) * dims[2] * decomp_rank, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_dims, dims.data(), sizeof(int) * RANK_DIMS, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_masks, masks.data(), sizeof(uint64_t) * RANK_DIMS, hipMemcpyHostToDevice));

    std::pair<int,int> dimensions = determine_dimensions_no_smem(non_zeros);
    T* host_output_tensor;
    T* d_output_tensor;
    int output_size;

    // --- Core Tensor Output Size Calculation ---
    // The Core Tensor G has dimensions R1 x R2 x R3.
    // Assuming R1 = R2 = R3 = decomp_rank.
    output_size = decomp_rank * decomp_rank * decomp_rank;

    // Allocate and initialize output memory for the Core Tensor G
    host_output_tensor = (T*)calloc(output_size, sizeof(T));
    HIP_CHECK(hipMalloc(&d_output_tensor, sizeof(T) * output_size));
    HIP_CHECK(hipMemcpy(d_output_tensor, host_output_tensor, sizeof(T) * output_size, hipMemcpyHostToDevice));
    
    size_t max_shmem = getMaxSharedMemory();
    size_t shared_mem_bytes = std::min(sizeof(T) * output_size, max_shmem);

    // Launch the Core Tensor kernel (Note: Removed 'mode' and replaced single fmat with all three)
    hipLaunchKernelGGL(
        tucker_core_kernel_3D_sparse<T>,  
        dim3(dimensions.first), dim3(dimensions.second), 
        shared_mem_bytes, // Shared memory size 
        0, // Stream
        d_input_tensor, non_zeros, d_masks, 
        d_fmat_1, d_fmat_2, d_fmat_3, // All three factor matrices passed
        device_dims, num_blocks, decomp_rank, d_output_tensor
    );
    
    // Copy result back
    HIP_CHECK(hipMemcpy(host_output_tensor, d_output_tensor, sizeof(T) * output_size, hipMemcpyDeviceToHost));

    // Free device memory
    free_blocks_from_gpu(d_input_tensor,num_blocks);
    HIP_CHECK(hipFree(d_output_tensor));
    HIP_CHECK(hipFree(d_fmat_1));
    HIP_CHECK(hipFree(d_fmat_2));
    HIP_CHECK(hipFree(d_fmat_3));
    HIP_CHECK(hipFree(device_dims)); 
    HIP_CHECK(hipFree(d_masks)); 

    return host_output_tensor;
}

//======================================================================
// 4D Functions
//======================================================================
template<typename T>
__global__ void tmm_kernel_4D_sparse(int mode, BLCO_BLOCK_GPU<T>* input_tensor, uint64_t nnz, uint64_t* masks, 
T* fmat, int* dims, int num_blocks, int rank, T* output_tensor, int wavefront_size = 64) 
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int block_idx = threadIdx.x;
    int wavefront_idx = threadIdx.x % wavefront_size; 

    bool active = true;
    if (global_idx >= nnz) active = false; 
    // rank is 4, contracted_dim is the size of the contracted dimension
    int contracted_dim = dims[mode - 1]; 
    T thread_val = 0; //Default if thread is inactive

    // Mode indexes: Now 4
    int m1_index; int m2_index; int m3_index; int m4_index; 
    
    // Lookup tables for 4 modes -> 3 non-target modes (1-based mode number)
    const int nt_mode1_list[4] = {2, 1, 1, 1}; // First non-target mode
    const int nt_mode2_list[4] = {3, 3, 2, 2}; // Second non-target mode
    const int nt_mode3_list[4] = {4, 4, 4, 3}; // Third non-target mode
    
    // Non-target mode variables: Now 3
    int i_mode = -1; int nt_mode_1 = -1; int nt_mode_2 = -1; int nt_mode_3 = -1;
    
    if(active){
        //Determine linear index, value, and bit widths
        int bl_index = find_block_index(input_tensor,num_blocks);
        uint64_t lin_index = extract_linear_index(input_tensor,global_idx,bl_index);
        thread_val = extract_value(input_tensor,bl_index);
        int block = input_tensor[bl_index].idx;
        
        // Bit widths for all 4 dimensions
        int m1_bits = ceiling_log2(dims[0]), m2_bits = ceiling_log2(dims[1]), 
            m3_bits = ceiling_log2(dims[2]), m4_bits = ceiling_log2(dims[3]);
        int bit_widths[4] = {m1_bits,m2_bits,m3_bits,m4_bits}; // Array size is 4

        // Extract indices (4 total modes, rank 4)
        m1_index = extract_mode_nd(lin_index, 1, masks, bit_widths, 4, block);
        m2_index = extract_mode_nd(lin_index, 2, masks, bit_widths, 4, block);
        m3_index = extract_mode_nd(lin_index, 3, masks, bit_widths, 4, block);
        m4_index = extract_mode_nd(lin_index, 4, masks, bit_widths, 4, block);
        
        // Indices array size is 4
        int indices[4] = {m1_index,m2_index,m3_index,m4_index};
        i_mode = indices[mode - 1];

        // Determine the three non-target modes (Must subtract 1 for 0-based index)
        nt_mode_1 = indices[nt_mode1_list[mode - 1] - 1];
        nt_mode_2 = indices[nt_mode2_list[mode - 1] - 1];
        nt_mode_3 = indices[nt_mode3_list[mode - 1] - 1];
    }

    // Non-target mode masks used for thread reduction: Now 3
    unsigned long long nt_mask_1 = 0;
    unsigned long long nt_mask_2 = 0;
    unsigned long long nt_mask_3 = 0; // New mask for 3rd NT mode
    unsigned long long group_mask;
    unsigned int full_mask = (1u << wavefront_size) - 1u;
    
    // Create thread masks (Now 3 loops)
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
    for (int k = 0; k < wavefront_size; ++k) { // New loop for 3rd NT mode
        int neighbor_nt = __shfl(nt_mode_3, k, wavefront_size);
        bool matches = neighbor_nt == nt_mode_3;
        nt_mask_3 |= ((1ULL & matches) << k);
    }

    // Group mask now combines 3 masks
    group_mask = nt_mask_1 & nt_mask_2 & nt_mask_3; 
    bool leader = (wavefront_idx == (__ffsll(group_mask) - 1)) & active;

    for(int i = 0; i < rank; i++){
        //Determine each indice's contribution
        int fmat_flat_index = i * contracted_dim + i_mode;
        T fmat_val = fmat[fmat_flat_index];
        T contrib = fmat_val * thread_val;
        T reduction_sum = 0;
        
        //Group reduction
        for (unsigned long long temp_mask = group_mask; temp_mask != 0; temp_mask &= (temp_mask - 1)){
            int lane_index = __ffsll(temp_mask) - 1;
            reduction_sum += __shfl(contrib, lane_index, wavefront_size);
        }

        // Determine the output index (4D indexing)
        int new_coords[4];
        new_coords[mode - 1] = i; 
        new_coords[nt_mode1_list[mode - 1] - 1] = nt_mode_1;
        new_coords[nt_mode2_list[mode - 1] - 1] = nt_mode_2;
        new_coords[nt_mode3_list[mode - 1] - 1] = nt_mode_3;
        
        //Determine new dimensions
        int local_dims[4];
        for (int d = 0; d < 4; ++d) {
            local_dims[d] = dims[d];
        }
        local_dims[mode - 1] = rank;
        
        int output_index = new_coords[0] * local_dims[1] * local_dims[2] * local_dims[3] + new_coords[1] * local_dims[2] * 
                        local_dims[3] + new_coords[2] * local_dims[3] + new_coords[3];
                           
        T* output_address = &(output_tensor[output_index]);

        //Group leaders atomically add the group contribution
        if(leader){
            if constexpr (std::is_same_v<T, double>) {
                atomicAdd_f64(output_address, reduction_sum);
            } else if constexpr (std::is_same_v<T, float>) {
                // Using the corrected function name for float atomic add
                atomicAdd(output_address, reduction_sum);
            } else if constexpr (std::is_same_v<T, int>){
                atomicAdd(output_address, reduction_sum);
            }
        }
    }

    return;
}

template<typename T, typename S>
T* tmm_4D(const Blco_Tensor<T,S>& sparse_tensor, int mode)
{
    const std::vector<BLCO_BLOCK_CPU<T>> blco_tensor = sparse_tensor.get_blco();

    // Rows, cols, rank, and non zeros
    // dims is now expected to have 4 elements (D0, D1, D2, D3)
    std::vector<int> dims = sparse_tensor.get_dims(); 
    const int decomp_rank = sparse_tensor.get_factor_rank();
    const int non_zeros = sparse_tensor.get_nnz();
    const int RANK_DIMS = 4; // Constant for 4 dimensions

    // Masks (Now 4)
    std::vector<uint64_t> masks = sparse_tensor.get_bitmasks();

    // Fmats (Now 4)
    std::vector<T*> fmats = sparse_tensor.get_fmats();
    T* mode_1_fmat = fmats[0];
    T* mode_2_fmat = fmats[1];
    T* mode_3_fmat = fmats[2];
    T* mode_4_fmat = fmats[3]; // New 4th factor matrix

    // Number of blocks
    int num_blocks = blco_tensor.size();
    bool blocked = num_blocks > 1;

    // Device pointers (Now 4 factor matrices)
    BLCO_BLOCK_GPU<T>* d_input_tensor;
    uint64_t* d_masks;
    T* d_fmat_1; T* d_fmat_2; T* d_fmat_3; T* d_fmat_4; // New d_fmat_4
    int* device_dims;

    // Allocate memory on the GPU
    HIP_CHECK(hipMalloc(&d_input_tensor, sizeof(BLCO_BLOCK_GPU<T>) * num_blocks));
    
    // Allocate all 4 factor matrices (Size: Dim[i] * decomp_rank)
    HIP_CHECK(hipMalloc(&d_fmat_1, sizeof(T) * dims[0] * decomp_rank));
    HIP_CHECK(hipMalloc(&d_fmat_2, sizeof(T) * dims[1] * decomp_rank));
    HIP_CHECK(hipMalloc(&d_fmat_3, sizeof(T) * dims[2] * decomp_rank));
    HIP_CHECK(hipMalloc(&d_fmat_4, sizeof(T) * dims[3] * decomp_rank));
    
    // Allocate space for 4 dimensions
    HIP_CHECK(hipMalloc(&device_dims, sizeof(int) * RANK_DIMS)); 
    // Allocate space for 4 masks
    HIP_CHECK(hipMalloc(&d_masks, sizeof(uint64_t) * RANK_DIMS)); 

    // Copy host data to GPU
    blocks_to_gpu(d_input_tensor, blco_tensor, num_blocks);
    HIP_CHECK(hipMemcpy(d_fmat_1, mode_1_fmat, sizeof(T) * dims[0] * decomp_rank, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_fmat_2, mode_2_fmat, sizeof(T) * dims[1] * decomp_rank, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_fmat_3, mode_3_fmat, sizeof(T) * dims[2] * decomp_rank, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_fmat_4, mode_4_fmat, sizeof(T) * dims[3] * decomp_rank, hipMemcpyHostToDevice)); // New copy
    HIP_CHECK(hipMemcpy(device_dims, dims.data(), sizeof(int) * RANK_DIMS, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_masks, masks.data(), sizeof(uint64_t) * RANK_DIMS, hipMemcpyHostToDevice));

    std::pair<int,int> dimensions = determine_dimensions_no_smem(non_zeros);
    T* host_output_tensor;
    T* d_output_tensor;
    uint64_t output_size;
    T* d_fmat_launch = nullptr; // Pointer to the fmat being used for launch

    // --- Output Tensor Allocation and Kernel Launch (4 Cases) ---

    if (mode == 1) {
        output_size = (uint64_t)decomp_rank * dims[1] * dims[2] * dims[3];
        d_fmat_launch = d_fmat_1;
    } else if (mode == 2) {
        output_size = (uint64_t)dims[0] * decomp_rank * dims[2] * dims[3];
        d_fmat_launch = d_fmat_2;
    } else if (mode == 3) {
        output_size = (uint64_t)dims[0] * dims[1] * decomp_rank * dims[3];
        d_fmat_launch = d_fmat_3;
    } else if (mode == 4) { // New mode 4 case
        output_size = (uint64_t)dims[0] * dims[1] * dims[2] * decomp_rank;
        d_fmat_launch = d_fmat_4;
    } else {
        return nullptr;
    }

    // Allocate and initialize output memory for the selected mode
    host_output_tensor = (T*)calloc(output_size, sizeof(T));
    HIP_CHECK(hipMalloc(&d_output_tensor, sizeof(T) * output_size));
    HIP_CHECK(hipMemcpy(d_output_tensor, host_output_tensor, sizeof(T) * output_size, hipMemcpyHostToDevice));

    // Launch the 4D kernel
    hipLaunchKernelGGL(
        tmm_kernel_4D_sparse<T>,  // Assuming this is the new 4D kernel
        dim3(dimensions.first), dim3(dimensions.second), 0, 0, mode, d_input_tensor, non_zeros,
        d_masks, d_fmat_launch, device_dims, num_blocks, decomp_rank, d_output_tensor
    );
    
    // Copy result back
    HIP_CHECK(hipMemcpy(host_output_tensor, d_output_tensor, sizeof(T) * output_size, hipMemcpyDeviceToHost));

    // Free device memory (Now 4 factor matrices)
    free_blocks_from_gpu(d_input_tensor,num_blocks);
    HIP_CHECK(hipFree(d_output_tensor));
    HIP_CHECK(hipFree(d_fmat_1));
    HIP_CHECK(hipFree(d_fmat_2));
    HIP_CHECK(hipFree(d_fmat_3));
    HIP_CHECK(hipFree(d_fmat_4));
    HIP_CHECK(hipFree(device_dims)); 
    HIP_CHECK(hipFree(d_masks)); 

    return host_output_tensor;
}

// Core tensor calculation (4D)
template<typename T>
__global__ void tucker_core_kernel_4D_sparse(
    BLCO_BLOCK_GPU<T>* input_tensor, uint64_t nnz, uint64_t* masks, 
    T* d_U1, T* d_U2, T* d_U3, T* d_U4, // Added U4
    int* dims, int num_blocks, int rank, T* output_tensor, int wavefront_size = 64) 
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int block_idx = threadIdx.x;

    // The output core tensor G is R1 x R2 x R3 x R4
    const int R1 = rank;
    const int R2 = rank;
    const int R3 = rank;
    const int R4 = rank; // New Rank Dimension
    
    // Size of block tensor is R1 * R2 * R3 * R4
    extern __shared__ __align__(sizeof(T)) unsigned char smem_raw[];
    T* block_tensor = reinterpret_cast<T*>(smem_raw);

    // Initialize block tensor to all 0's
    const int output_size = R1 * R2 * R3 * R4; // Updated output size for 4D
    for(int i = block_idx; i < output_size; i += blockDim.x){
        block_tensor[i] = 0;
    }
    
    bool active = true;
    if (global_idx >= nnz) active = false; 
    
    T thread_val = 0; // The value of the non-zero element x_i1,i2,i3,i4
    int m1_index = -1; // i1
    int m2_index = -1; // i2
    int m3_index = -1; // i3
    int m4_index = -1; // i4 (New Index)

    if(active){
        // Determine linear index, value, and bit widths
        int bl_index = find_block_index(input_tensor,num_blocks);
        uint64_t lin_index = extract_linear_index(input_tensor,global_idx,bl_index);
        thread_val = extract_value(input_tensor,bl_index);
        int block = input_tensor[bl_index].idx;
        
        // Bit widths for all 4 dimensions
        int m1_bits = ceiling_log2(dims[0]);
        int m2_bits = ceiling_log2(dims[1]);
        int m3_bits = ceiling_log2(dims[2]);
        int m4_bits = ceiling_log2(dims[3]); // New bit width for D4
        int bit_widths[4] = {m1_bits,m2_bits,m3_bits,m4_bits}; // Size 4
        const int RANK_DIMS = 4; // Total number of dimensions

        // Extract indices (i1, i2, i3, i4) from the sparse tensor
        m1_index = extract_mode_nd(lin_index, 1, masks, bit_widths, RANK_DIMS, block);
        m2_index = extract_mode_nd(lin_index, 2, masks, bit_widths, RANK_DIMS, block);
        m3_index = extract_mode_nd(lin_index, 3, masks, bit_widths, RANK_DIMS, block);
        m4_index = extract_mode_nd(lin_index, 4, masks, bit_widths, RANK_DIMS, block); // New Index Extraction
    }
    
    // --- Core Tensor Calculation Loops (4 Nested Loops) ---
        
    for(int r1 = 0; r1 < R1; r1++){
        T u1_val = d_U1[m1_index * R1 + r1 * active];
        
        for(int r2 = 0; r2 < R2; r2++){
            T u2_val = d_U2[m2_index * R2 + r2 * active];
            
            for(int r3 = 0; r3 < R3; r3++){
                T u3_val = d_U3[m3_index * R3 + r3 * active];
                
                // New innermost loop for r4
                for(int r4 = 0; r4 < R4; r4++){
                    T u4_val = d_U4[m4_index * R4 + r4 * active]; // New Factor Matrix value

                    // Total contribution to G(r1, r2, r3, r4)
                    T contrib = thread_val * u1_val * u2_val * u3_val * u4_val * active;
                    
                    // Core Tensor Row-Major Indexing (R1 x R2 x R3 x R4)
                    // Index = r1 * R2*R3*R4 + r2 * R3*R4 + r3 * R4 + r4
                    int output_index = r1 * R2 * R3 * R4 + 
                                       r2 * R3 * R4 + 
                                       r3 * R4 + 
                                       r4;
                    
                    T* output_address = &(block_tensor[output_index]);

                    // Atomic add to shared memory
                    if constexpr (std::is_same_v<T, double>) {
                        atomicAdd_f64(output_address, contrib);
                    } else if constexpr (std::is_same_v<T, float>) {
                        atomicAdd(output_address, contrib);
                    } else if constexpr (std::is_same_v<T, int>){
                        atomicAdd(output_address, contrib);
                    }
                } // End r4
            } // End r3
        } // End r2
    } // End r1

    // Global Atomic Write (Shared Memory Reduction)
    for(int i = block_idx; i < output_size; i += blockDim.x){
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
T* tucker_compute_core_4D(const Blco_Tensor<T,S>& sparse_tensor)
{
    const std::vector<BLCO_BLOCK_CPU<T>> blco_tensor = sparse_tensor.get_blco();

    // Rows, cols, rank, and non zeros
    std::vector<int> dims = sparse_tensor.get_dims();
    const int decomp_rank = sparse_tensor.get_factor_rank();
    const int non_zeros = sparse_tensor.get_nnz();
    const int RANK_DIMS = 4;

    // Masks
    std::vector<uint64_t> masks = sparse_tensor.get_bitmasks();

    // Fmats (All three needed for core tensor calculation)
    std::vector<T*> fmats = sparse_tensor.get_fmats();
    T* mode_1_fmat = fmats[0];
    T* mode_2_fmat = fmats[1];
    T* mode_3_fmat = fmats[2];
    T* mode_4_fmat = fmats[3];

    // Number of blocks
    int num_blocks = blco_tensor.size();

    // Device pointers
    BLCO_BLOCK_GPU<T>* d_input_tensor;
    uint64_t* d_masks;
    T* d_fmat_1; T* d_fmat_2; T* d_fmat_3; T* d_fmat_4;
    int* device_dims;

    // Allocate
    HIP_CHECK(hipMalloc(&d_input_tensor, sizeof(BLCO_BLOCK_GPU<T>) * num_blocks));
    HIP_CHECK(hipMalloc(&d_fmat_1, sizeof(T) * dims[0] * decomp_rank));
    HIP_CHECK(hipMalloc(&d_fmat_2, sizeof(T) * dims[1] * decomp_rank));
    HIP_CHECK(hipMalloc(&d_fmat_3, sizeof(T) * dims[2] * decomp_rank));
    HIP_CHECK(hipMalloc(&d_fmat_4, sizeof(T) * dims[3] * decomp_rank));
    HIP_CHECK(hipMalloc(&device_dims, sizeof(int) * RANK_DIMS));
    HIP_CHECK(hipMalloc(&d_masks, sizeof(uint64_t) * RANK_DIMS));

    // Copy host data to GPU
    blocks_to_gpu(d_input_tensor, blco_tensor, num_blocks);
    HIP_CHECK(hipMemcpy(d_fmat_1, mode_1_fmat, sizeof(T) * dims[0] * decomp_rank, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_fmat_2, mode_2_fmat, sizeof(T) * dims[1] * decomp_rank, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_fmat_3, mode_3_fmat, sizeof(T) * dims[2] * decomp_rank, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_fmat_4, mode_4_fmat, sizeof(T) * dims[3] * decomp_rank, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(device_dims, dims.data(), sizeof(int) * RANK_DIMS, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_masks, masks.data(), sizeof(uint64_t) * RANK_DIMS, hipMemcpyHostToDevice));

    std::pair<int,int> dimensions = determine_dimensions_no_smem(non_zeros);
    T* host_output_tensor;
    T* d_output_tensor;
    int output_size;

    // --- Core Tensor Output Size Calculation ---
    // The Core Tensor G has dimensions R1 x R2 x R3 x R4.
    // Assuming R1 = R2 = R3 = R4 = decomp_rank.
    output_size = decomp_rank * decomp_rank * decomp_rank * decomp_rank;

    // Allocate and initialize output memory for the Core Tensor G
    host_output_tensor = (T*)calloc(output_size, sizeof(T));
    HIP_CHECK(hipMalloc(&d_output_tensor, sizeof(T) * output_size));
    HIP_CHECK(hipMemcpy(d_output_tensor, host_output_tensor, sizeof(T) * output_size, hipMemcpyHostToDevice));
    
    size_t max_shmem = getMaxSharedMemory();
    size_t shared_mem_bytes = std::min(sizeof(T) * output_size, max_shmem);

    // Launch the Core Tensor kernel (Note: Removed 'mode' and replaced single fmat with all three)
    hipLaunchKernelGGL(
        tucker_core_kernel_4D_sparse<T>,  
        dim3(dimensions.first), dim3(dimensions.second), 
        shared_mem_bytes, // Shared memory size 
        0, // Stream
        d_input_tensor, non_zeros, d_masks, 
        d_fmat_1, d_fmat_2, d_fmat_3, d_fmat_4, // All three factor matrices passed
        device_dims, num_blocks, decomp_rank, d_output_tensor
    );
    
    // Copy result back
    HIP_CHECK(hipMemcpy(host_output_tensor, d_output_tensor, sizeof(T) * output_size, hipMemcpyDeviceToHost));

    // Free device memory
    free_blocks_from_gpu(d_input_tensor,num_blocks);
    HIP_CHECK(hipFree(d_output_tensor));
    HIP_CHECK(hipFree(d_fmat_1));
    HIP_CHECK(hipFree(d_fmat_2));
    HIP_CHECK(hipFree(d_fmat_3));
    HIP_CHECK(hipFree(d_fmat_4));
    HIP_CHECK(hipFree(device_dims)); 
    HIP_CHECK(hipFree(d_masks)); 

    return host_output_tensor;
}

//======================================================================
// 5D Functions
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
    const int nt_mode1_list[5] = {2, 1, 1, 1, 1}; // First non-target mode
    const int nt_mode2_list[5] = {3, 3, 2, 2, 2}; // Second non-target mode
    const int nt_mode3_list[5] = {4, 4, 4, 3, 3}; // Third non-target mode
    const int nt_mode4_list[5] = {5, 5, 5, 5, 4}; // Fourth non-target mode (NEW)
    
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
    for (int k = 0; k < wavefront_size; ++k) { // NEW loop for 4th NT mode
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
        int new_coords[5]; // Array size is 5
        
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
        // index = i0*D1*D2*D3*D4 + i1*D2*D3*D4 + i2*D3*D4 + i3*D4 + i4
        int output_index = new_coords[0] * local_dims[1] * local_dims[2] * local_dims[3] * local_dims[4] + 
                           new_coords[1] * local_dims[2] * local_dims[3] * local_dims[4] + 
                           new_coords[2] * local_dims[3] * local_dims[4] + 
                           new_coords[3] * local_dims[4] + 
                           new_coords[4]; // NEW term for D4
                           
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
T* tmm_5D(const Blco_Tensor<T,S>& sparse_tensor, int mode)
{
    const std::vector<BLCO_BLOCK_CPU<T>> blco_tensor = sparse_tensor.get_blco();

    // 1. Dims and Rank
    std::vector<int> dims = sparse_tensor.get_dims(); 
    const int decomp_rank = sparse_tensor.get_factor_rank();
    const int non_zeros = sparse_tensor.get_nnz();
    const int RANK_DIMS = 5; // Updated constant for 5 dimensions

    // Safety check for mode
    if (mode < 1 || mode > RANK_DIMS) {
        std::cerr << "Error: Invalid mode specified (" << mode << ") for " 
                  << RANK_DIMS << "-dimensional tensor." << std::endl;
        return nullptr;
    }

    // Masks (Now 5)
    std::vector<uint64_t> masks = sparse_tensor.get_bitmasks();

    // Fmats (Now 5)
    std::vector<T*> fmats = sparse_tensor.get_fmats();
    T* mode_1_fmat = fmats[0];
    T* mode_2_fmat = fmats[1];
    T* mode_3_fmat = fmats[2];
    T* mode_4_fmat = fmats[3]; 
    T* mode_5_fmat = fmats[4]; // New 5th factor matrix

    // Number of blocks
    int num_blocks = blco_tensor.size();

    // Device pointers (Now 5 factor matrices)
    BLCO_BLOCK_GPU<T>* d_input_tensor;
    uint64_t* d_masks;
    T* d_fmat_1; T* d_fmat_2; T* d_fmat_3; T* d_fmat_4; T* d_fmat_5; // New d_fmat_5
    int* device_dims;

    // Allocate shared data
    HIP_CHECK(hipMalloc(&d_input_tensor, sizeof(BLCO_BLOCK_GPU<T>) * num_blocks));
    
    // Allocate all 5 factor matrices (Size: Dim[i] * decomp_rank)
    HIP_CHECK(hipMalloc(&d_fmat_1, sizeof(T) * dims[0] * decomp_rank));
    HIP_CHECK(hipMalloc(&d_fmat_2, sizeof(T) * dims[1] * decomp_rank));
    HIP_CHECK(hipMalloc(&d_fmat_3, sizeof(T) * dims[2] * decomp_rank));
    HIP_CHECK(hipMalloc(&d_fmat_4, sizeof(T) * dims[3] * decomp_rank));
    HIP_CHECK(hipMalloc(&d_fmat_5, sizeof(T) * dims[4] * decomp_rank)); // New 5th allocation
    
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

    std::pair<int,int> dimensions = determine_dimensions_no_smem(non_zeros);
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

    // Launch the 5D kernel
    hipLaunchKernelGGL(
        tmm_kernel_5D_sparse<T>,  // Assuming this is the 5D kernel
        dim3(dimensions.first), dim3(dimensions.second), 0, 0, mode, d_input_tensor, non_zeros,
        d_masks, d_fmat_launch, device_dims, num_blocks, decomp_rank, d_output_tensor
    );
    
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


// Core tensor calculation 5D
template<typename T>
__global__ void tucker_core_kernel_5D_sparse(
    BLCO_BLOCK_GPU<T>* input_tensor, uint64_t nnz, uint64_t* masks, 
    T* d_U1, T* d_U2, T* d_U3, T* d_U4, T* d_U5, // Added U5
    int* dims, int num_blocks, int rank, T* output_tensor, int wavefront_size = 64) 
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int block_idx = threadIdx.x;

    // The output core tensor G is R1 x R2 x R3 x R4 x R5
    const int R1 = rank;
    const int R2 = rank;
    const int R3 = rank;
    const int R4 = rank;
    const int R5 = rank; // New Rank Dimension
    
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
    int m5_index = -1; // i5 (New Index)

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
        T u1_val = d_U1[m1_index * R1 + r1 * active];
        
        for(int r2 = 0; r2 < R2; r2++){
            T u2_val = d_U2[m2_index * R2 + r2 * active];
            
            for(int r3 = 0; r3 < R3; r3++){
                T u3_val = d_U3[m3_index * R3 + r3 * active];
                
                for(int r4 = 0; r4 < R4; r4++){
                    T u4_val = d_U4[m4_index * R4 + r4 * active]; 
                    
                    // New innermost loop for r5
                    for(int r5 = 0; r5 < R5; r5++){
                        T u5_val = d_U5[m5_index * R5 + r5 * active];

                        // Total contribution to G(r1, r2, r3, r4, r5)
                        T contrib = thread_val * u1_val * u2_val * u3_val * u4_val * u5_val * active;
                        
                        // Core Tensor Row-Major Indexing (R1 x R2 x R3 x R4 x R5)
                        // Index = r1*R2*R3*R4*R5 + r2*R3*R4*R5 + r3*R4*R5 + r4*R5 + r5
                        int output_index = r1 * R2 * R3 * R4 * R5 + 
                                           r2 * R3 * R4 * R5 + 
                                           r3 * R4 * R5 + 
                                           r4 * R5 + 
                                           r5;
                        
                        T* output_address = &(block_tensor[output_index]);

                        // Atomic add to shared memory
                        if constexpr (std::is_same_v<T, double>) {
                            atomicAdd_f64(output_address, contrib);
                        } else if constexpr (std::is_same_v<T, float>) {
                            atomicAdd(output_address, contrib);
                        } else if constexpr (std::is_same_v<T, int>){
                            atomicAdd(output_address, contrib);
                        }
                    } // End r5
                } // End r4
            } // End r3
        } // End r2
    } // End r1

    // Global Atomic Write (Shared Memory Reduction)
    for(int i = block_idx; i < output_size; i += blockDim.x){
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
T* tucker_compute_core_5D(const Blco_Tensor<T,S>& sparse_tensor)
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

    std::pair<int,int> dimensions = determine_dimensions_no_smem(non_zeros);
    T* host_output_tensor;
    T* d_output_tensor;
    int output_size;

    // --- Core Tensor Output Size Calculation ---
    // The Core Tensor G has dimensions R1 x R2 x R3 x R4.
    // Assuming R1 = R2 = R3 = R4 = decomp_rank.
    output_size = decomp_rank * decomp_rank * decomp_rank * decomp_rank * decomp_rank;

    // Allocate and initialize output memory for the Core Tensor G
    host_output_tensor = (T*)calloc(output_size, sizeof(T));
    HIP_CHECK(hipMalloc(&d_output_tensor, sizeof(T) * output_size));
    HIP_CHECK(hipMemcpy(d_output_tensor, host_output_tensor, sizeof(T) * output_size, hipMemcpyHostToDevice));
    
    size_t max_shmem = getMaxSharedMemory();
    size_t shared_mem_bytes = std::min(sizeof(T) * output_size, max_shmem);

    // Launch the Core Tensor kernel (Note: Removed 'mode' and replaced single fmat with all three)
    hipLaunchKernelGGL(
        tucker_core_kernel_4D_sparse<T>,  
        dim3(dimensions.first), dim3(dimensions.second), 
        shared_mem_bytes, // Shared memory size 
        0, // Stream
        d_input_tensor, non_zeros, d_masks, 
        d_fmat_1, d_fmat_2, d_fmat_3, d_fmat_4, // All three factor matrices passed
        device_dims, num_blocks, decomp_rank, d_output_tensor
    );
    
    // Copy result back
    HIP_CHECK(hipMemcpy(host_output_tensor, d_output_tensor, sizeof(T) * output_size, hipMemcpyDeviceToHost));

    // Free device memory
    free_blocks_from_gpu(d_input_tensor,num_blocks);
    HIP_CHECK(hipFree(d_output_tensor));
    HIP_CHECK(hipFree(d_fmat_1));
    HIP_CHECK(hipFree(d_fmat_2));
    HIP_CHECK(hipFree(d_fmat_3));
    HIP_CHECK(hipFree(d_fmat_4));
    HIP_CHECK(hipFree(device_dims)); 
    HIP_CHECK(hipFree(d_masks)); 

    return host_output_tensor;
}