#include <vector>
#include <iostream>
#include <stdexcept>
#include <string>
#include <numeric>
#include <algorithm>
#include <limits>
#include <cmath>
#include <random>
#include <type_traits>
#include "../tensor_storage/blco.h"

//======================================================================
// Tensor Matrix Multiply
//======================================================================

//3D implementation
template<typename T, typename S>
std::vector<T> tmm_3D_cpu(int mode, const std::vector<NNZ_Entry<T>>& sparse_tensor, 
const std::vector<int>& dims, T* fmat, int decomp_rank)
{
    // --- 1. Validation and Dimension Setup ---
    if (dims.size() != 3 || mode < 1 || mode > 3) {
        throw std::invalid_argument("TMM CPU requires 3 dimensions and mode 0, 1, or 2.");
    }
    const int I1 = dims[0];
    const int I2 = dims[1];
    const int I3 = dims[2];
    const int R = decomp_rank; // New dimension (Rank)

    // Original dimension being contracted (e.g., I1 for mode 0)
    const int I_mode = dims[mode - 1]; 

    // --- 2. Determine Output Tensor Dimensions and Initialize ---
    int B_dims[3];
    B_dims[0] = (mode == 1) ? R : I1;
    B_dims[1] = (mode == 2) ? R : I2;
    B_dims[2] = (mode == 3) ? R : I3;

    const int B_I1 = B_dims[0];
    const int B_I2 = B_dims[1];
    const int B_I3 = B_dims[2];

    const size_t output_size = (size_t)B_I1 * B_I2 * B_I3;
    std::vector<T> output_tensor(output_size, T{0}); // Use T{0} for template robustness

    // --- 3. Perform TMM using Sparse Iteration ---
    // The core loop iterates over the non-zero entries of A.
    for (const auto& entry : sparse_tensor) {
        if (entry.indices.size() != 3) continue;

        const int i1 = entry.indices[0]; 
        const int i2 = entry.indices[1]; 
        const int i3 = entry.indices[2];
        const T a_val = entry.value;

        // Ensure indices are within bounds
        if (i1 < 0 || i1 >= I1 || i2 < 0 || i2 >= I2 || i3 < 0 || i3 >= I3) {
            continue; 
        }

        // --- Core TMM Update ---
        for (int r = 0; r < R; ++r) {
            
            // --- A. Get F Index ---
            // F is I_mode x R, Row-Major. F(row, col) = row * R + col.
            // row is the index from the input tensor A (i1, i2, or i3).
            // col is the new dimension index (r).
            int F_row_idx; 
            switch (mode) {
                case 1: F_row_idx = i1; break;
                case 2: F_row_idx = i2; break;
                case 3: F_row_idx = i3; break;
                default: F_row_idx = 0; // Should not happen due to validation
            }
        
            // FIX: The index calculation now assumes Column-Major storage (r * I_mode + i_mode)
            const size_t F_idx = (size_t)r * I_mode + F_row_idx; 
            const T f_val = fmat[F_idx];
            
            // --- B. Get B Index ---
            // The output tensor B is B_I1 x B_I2 x B_I3, Row-Major.
            // B(idx0, idx1, idx2) = idx0 * (B_I2 * B_I3) + idx1 * B_I3 + idx2
            
            // The indices of B are the original indices (i1, i2, i3) with 'mode' replaced by 'r'.
            int B_idx0 = (mode == 1) ? r : i1;
            int B_idx1 = (mode == 2) ? r : i2;
            int B_idx2 = (mode == 3) ? r : i3;

            const size_t B_idx = (size_t)B_idx0 * (B_I2 * B_I3) + B_idx1 * B_I3 + B_idx2;
            
            // TMM Update: B(...) += A(...) * F(...)
            output_tensor[B_idx] += a_val * f_val;
        }
    }

    return output_tensor;
}

//4D implementation
template<typename T, typename S>
std::vector<T> tmm_4D_cpu(int mode, const std::vector<NNZ_Entry<T>>& sparse_tensor, 
const std::vector<int>& dims, T* fmat, int decomp_rank)
{
    // --- 1. Validation and Dimension Setup ---
    if (dims.size() != 4 || mode < 1 || mode > 4) {
        throw std::invalid_argument("TMM CPU requires exactly 4 dimensions and mode 1, 2, 3, or 4.");
    }
    const int I1 = dims[0];
    const int I2 = dims[1];
    const int I3 = dims[2];
    const int I4 = dims[3];
    const int R = decomp_rank; // New dimension (Rank)

    // Original dimension being contracted (e.g., I1 for mode 1)
    const int I_mode = dims[mode - 1]; 

    // --- 2. Determine Output Tensor Dimensions and Initialize ---
    int B_dims[4];
    B_dims[0] = (mode == 1) ? R : I1;
    B_dims[1] = (mode == 2) ? R : I2;
    B_dims[2] = (mode == 3) ? R : I3;
    B_dims[3] = (mode == 4) ? R : I4;

    const int B_I1 = B_dims[0];
    const int B_I2 = B_dims[1];
    const int B_I3 = B_dims[2];
    const int B_I4 = B_dims[3];

    // Total size: B_I1 * B_I2 * B_I3 * B_I4
    const size_t output_size = (size_t)B_I1 * B_I2 * B_I3 * B_I4;
    std::vector<T> output_tensor(output_size, T{0}); 

    // --- 3. Perform TMM using Sparse Iteration ---
    // The core loop iterates over the non-zero entries of A.
    for (const auto& entry : sparse_tensor) {
        if (entry.indices.size() != 4) continue;

        const int i1 = entry.indices[0]; 
        const int i2 = entry.indices[1]; 
        const int i3 = entry.indices[2];
        const int i4 = entry.indices[3]; // New 4th index
        const T a_val = entry.value;

        // Ensure indices are within bounds
        if (i1 < 0 || i1 >= I1 || i2 < 0 || i2 >= I2 || i3 < 0 || i3 >= I3 || i4 < 0 || i4 >= I4) {
            continue; 
        }

        // --- Core TMM Update ---
        for (int r = 0; r < R; ++r) {
            
            // --- A. Get F Index (Row-Major F: I_mode x R) ---
            // The row of F is the index of the dimension being contracted.
            int F_row_idx; 
            switch (mode) {
                case 1: F_row_idx = i1; break;
                case 2: F_row_idx = i2; break;
                case 3: F_row_idx = i3; break;
                case 4: F_row_idx = i4; break; // New case for 4th dimension
                default: F_row_idx = 0; 
            }
            const size_t F_idx = (size_t)r * I_mode + F_row_idx; 
            const T f_val = fmat[F_idx];
            
            // --- B. Get B Index (Row-Major B: B_I1 x B_I2 x B_I3 x B_I4) ---
            
            // The indices of B are the original indices (i1, i2, i3, i4) with 'mode' replaced by 'r'.
            int B_idx0 = (mode == 1) ? r : i1;
            int B_idx1 = (mode == 2) ? r : i2;
            int B_idx2 = (mode == 3) ? r : i3;
            int B_idx3 = (mode == 4) ? r : i4; // New 4th output index

            // Row-Major Index Calculation for 4D:
            // idx = idx0 * (D2*D3*D4) + idx1 * (D3*D4) + idx2 * (D4) + idx3
            const size_t B_idx = (size_t)B_idx0 * (B_I2 * B_I3 * B_I4) + 
                                 B_idx1 * (B_I3 * B_I4) + 
                                 B_idx2 * B_I4 + 
                                 B_idx3;
            
            // TMM Update: B(...) += A(...) * F(...)
            output_tensor[B_idx] += a_val * f_val;
        }
    }

    return output_tensor;
}

//5D implementation
template<typename T, typename S>
std::vector<T> tmm_5D_cpu(int mode, const std::vector<NNZ_Entry<T>>& sparse_tensor, 
                                   const std::vector<int>& dims, T* fmat, int decomp_rank)
{
    // --- 1. Validation and Dimension Setup (No change) ---
    if (dims.size() != 5 || mode < 1 || mode > 5) {
        throw std::invalid_argument("TMM CPU requires exactly 5 dimensions and mode 1, 2, 3, 4, or 5.");
    }
    const int I1 = dims[0]; //... I5 and R remain the same
    const int I2 = dims[1];
    const int I3 = dims[2];
    const int I4 = dims[3];
    const int I5 = dims[4]; 
    const int R = decomp_rank; 
    const int I_mode = dims[mode - 1]; 

    // --- 2. Determine Output Tensor Dimensions and Initialize ---
    int B_dims[5];
    B_dims[0] = (mode == 1) ? R : I1;
    B_dims[1] = (mode == 2) ? R : I2;
    B_dims[2] = (mode == 3) ? R : I3;
    B_dims[3] = (mode == 4) ? R : I4;
    B_dims[4] = (mode == 5) ? R : I5;

    const int B_I1 = B_dims[0];
    const int B_I2 = B_dims[1];
    const int B_I3 = B_dims[2];
    const int B_I4 = B_dims[3];
    const int B_I5 = B_dims[4];

    // Pre-calculate products for index calculation efficiency
    const size_t P1 = (size_t)B_I2 * B_I3 * B_I4 * B_I5;
    const size_t P2 = (size_t)B_I3 * B_I4 * B_I5;
    const size_t P3 = (size_t)B_I4 * B_I5;
    const size_t P4 = (size_t)B_I5;

    // Total size of the output tensor B
    const size_t output_size = (size_t)B_I1 * B_I2 * B_I3 * B_I4 * B_I5;
    std::vector<T> output_tensor(output_size, T{0}); // Final result

    // --- 3. Parallel TMM using OpenMP ---
    
    #pragma omp parallel
    {
        // 🔑 Private accumulation tensor for each thread, initialized to zero
        std::vector<T> B_local(output_size, T{0}); 

        // Schedule the outer loop across all threads
        #pragma omp for nowait
        for (size_t k = 0; k < sparse_tensor.size(); ++k) {
            const auto& entry = sparse_tensor[k];
            if (entry.indices.size() != 5) continue;

            const int i1 = entry.indices[0]; 
            const int i2 = entry.indices[1]; 
            const int i3 = entry.indices[2];
            const int i4 = entry.indices[3];
            const int i5 = entry.indices[4];
            const T a_val = entry.value;

            // Ensure indices are within bounds (same as original)
            if (i1 < 0 || i1 >= I1 || i2 < 0 || i2 >= I2 || i3 < 0 || i3 >= I3 || i4 < 0 || i4 >= I4 || i5 < 0 || i5 >= I5) {
                continue; 
            }

            // --- Core TMM Update (Inner Loop) ---
            for (int r = 0; r < R; ++r) {
                
                // --- A. Get F Index ---
                int F_row_idx; 
                switch (mode) {
                    case 1: F_row_idx = i1; break;
                    case 2: F_row_idx = i2; break;
                    case 3: F_row_idx = i3; break;
                    case 4: F_row_idx = i4; break;
                    case 5: F_row_idx = i5; break;
                    default: F_row_idx = 0; 
                }
                const size_t F_idx = (size_t)r * I_mode + F_row_idx;
                const T f_val = fmat[F_idx];
                
                // --- B. Get B Index ---
                int B_idx0 = (mode == 1) ? r : i1;
                int B_idx1 = (mode == 2) ? r : i2;
                int B_idx2 = (mode == 3) ? r : i3;
                int B_idx3 = (mode == 4) ? r : i4;
                int B_idx4 = (mode == 5) ? r : i5;

                // Row-Major Index Calculation using pre-calculated products:
                const size_t B_idx = (size_t)B_idx0 * P1 + 
                                     B_idx1 * P2 + 
                                     B_idx2 * P3 + 
                                     B_idx3 * P4 + 
                                     B_idx4;
                
                // TMM Update into thread-local copy
                B_local[B_idx] += a_val * f_val;
            }
        } // End of OpenMP 'for' loop

        // 🔑 Reduction: Merge B_local results back into the shared output_tensor
        #pragma omp critical
        {
            for (size_t i = 0; i < output_size; ++i) {
                output_tensor[i] += B_local[i];
            }
        }
    } // End of OpenMP 'parallel' block

    return output_tensor;
}

//======================================================================
// Core Tensor Calculation (Tucker Decomposition)
//======================================================================
template<typename T, typename S>
std::vector<T> core_tensor_3D_cpu(const std::vector<NNZ_Entry<T>>& sparse_tensor, const std::vector<int>& dims, 
const std::vector<T*>& fmats, int decomp_rank)
{
    // --- 1. Validation and Dimension Setup ---
    if (dims.size() != 3 || fmats.size() != 3) {
        throw std::invalid_argument("Core Tensor 3D CPU requires 3 dimensions and 3 factor matrices.");
    }
    const int I1 = dims[0];
    const int I2 = dims[1];
    const int I3 = dims[2];
    const int R = decomp_rank; // R1 = R2 = R3 = R

    // --- 2. Determine Output Tensor Size and Initialize ---
    // Output G is R x R x R. Stored in Row-Major order.
    // Index: G[r1 * R*R + r2 * R + r3]
    const size_t output_size = (size_t)R * R * R;
    std::vector<T> core_tensor_G(output_size, T{0}); 

    // --- 3. Perform Simultaneous Contractions ---
    // The calculation is G(r1, r2, r3) += A(i1, i2, i3) * F1^T(r1, i1) * F2^T(r2, i2) * F3^T(r3, i3)
    
    // We iterate ONLY over the non-zero entries (i1, i2, i3) of the input A.
    for (const auto& entry : sparse_tensor) {
        if (entry.indices.size() != 3) continue;

        const int i1 = entry.indices[0]; 
        const int i2 = entry.indices[1]; 
        const int i3 = entry.indices[2];
        const T a_val = entry.value;

        // Ensure indices are within bounds
        if (i1 < 0 || i1 >= I1 || i2 < 0 || i2 >= I2 || i3 < 0 || i3 >= I3) {
            continue; 
        }

        // --- Core Accumulation Loop ---
        for (int r1 = 0; r1 < R; ++r1) {        // New index for 1st dimension
            for (int r2 = 0; r2 < R; ++r2) {    // New index for 2nd dimension
                for (int r3 = 0; r3 < R; ++r3) { // New index for 3rd dimension

                    // --- A. Get Factor Matrix Terms ---
                    // Fk is Ik x R, Row-Major. Fk^T(rk, ik) is Fk(ik, rk).
                    // Index = ik * R + rk
                    
                    // FIX: Column-Major Indexing: r1 * I1 + i1
                    const size_t F1_idx = (size_t)r1 * I1 + i1; 
                    const T f1_val = fmats[0][F1_idx];

                    // F2 is I2 x R matrix, stored Column-Major. Access F2(i2, r2)
                    // FIX: Column-Major Indexing: r2 * I2 + i2
                    const size_t F2_idx = (size_t)r2 * I2 + i2;
                    const T f2_val = fmats[1][F2_idx];
                    
                    // F3 is I3 x R matrix, stored Column-Major. Access F3(i3, r3)
                    // FIX: Column-Major Indexing: r3 * I3 + i3
                    const size_t F3_idx = (size_t)r3 * I3 + i3;
                    const T f3_val = fmats[2][F3_idx];

                    // --- B. Get G Index (Row-Major G: R x R x R) ---
                    // G(r1, r2, r3) corresponds to index = r1 * (R * R) + r2 * R + r3
                    const size_t G_idx = (size_t)r1 * (R * R) + r2 * R + r3;
                    
                    // Accumulation: G(...) += A(...) * F1^T(...) * F2^T(...) * F3^T(...)
                    core_tensor_G[G_idx] += a_val * f1_val * f2_val * f3_val;
                }
            }
        }
    }

    return core_tensor_G;
}

template<typename T, typename S>
std::vector<T> core_tensor_4D_cpu(const std::vector<NNZ_Entry<T>>& sparse_tensor, const std::vector<int>& dims, 
                                 const std::vector<T*>& fmats, int decomp_rank)
{
    // --- 1. Validation and Dimension Setup ---
    if (dims.size() != 4 || fmats.size() != 4) {
        throw std::invalid_argument("Core Tensor 4D CPU requires 4 dimensions and 4 factor matrices.");
    }
    const int I1 = dims[0];
    const int I2 = dims[1];
    const int I3 = dims[2];
    const int I4 = dims[3];
    const int R = decomp_rank;

    // --- 2. Determine Output Tensor Size and Initialize ---
    const size_t R_sq = (size_t)R * R;
    const size_t R_cub = R_sq * R;
    const size_t output_size = R_cub * R; // R^4
    std::vector<T> core_tensor_G(output_size, T{0}); 

    // --- 3. Perform Simultaneous Contractions ---
    for (const auto& entry : sparse_tensor) {
        if (entry.indices.size() != 4) continue;

        const int i1 = entry.indices[0]; 
        const int i2 = entry.indices[1]; 
        const int i3 = entry.indices[2];
        const int i4 = entry.indices[3];
        const T a_val = entry.value;

        // Ensure indices are within bounds
        if (i1 < 0 || i1 >= I1 || i2 < 0 || i2 >= I2 || i3 < 0 || i3 >= I3 || i4 < 0 || i4 >= I4) {
            continue; 
        }

        // --- Core Accumulation Loop ---
        for (int r1 = 0; r1 < R; ++r1) {
            for (int r2 = 0; r2 < R; ++r2) {
                for (int r3 = 0; r3 < R; ++r3) {
                    for (int r4 = 0; r4 < R; ++r4) {

                        const T f1_val = fmats[0][(size_t)r1 * I1 + i1];
                        const T f2_val = fmats[1][(size_t)r2 * I2 + i2];
                        const T f3_val = fmats[2][(size_t)r3 * I3 + i3];
                        const T f4_val = fmats[3][(size_t)r4 * I4+ i4];

                        // --- B. Get G Index (Row-Major G: R x R x R x R) ---
                        // Index = r1 * R^3 + r2 * R^2 + r3 * R + r4
                        const size_t G_idx = r1 * R_cub + r2 * R_sq + r3 * R + r4;
                        
                        // Accumulation
                        core_tensor_G[G_idx] += a_val * f1_val * f2_val * f3_val * f4_val;
                    }
                }
            }
        }
    }

    return core_tensor_G;
}

template<typename T, typename S>
std::vector<T> core_tensor_5D_cpu(const std::vector<NNZ_Entry<T>>& sparse_tensor, 
                                          const std::vector<int>& dims, 
                                          const std::vector<T*>& fmats, int decomp_rank)
{
    // --- 1. Validation and Dimension Setup (No change) ---
    if (dims.size() != 5 || fmats.size() != 5) {
        throw std::invalid_argument("Core Tensor 5D CPU requires 5 dimensions and 5 factor matrices.");
    }
    const int I1 = dims[0]; //... I5 and R remain the same
    const int I2 = dims[1];
    const int I3 = dims[2];
    const int I4 = dims[3];
    const int I5 = dims[4];
    const int R = decomp_rank;

    // --- 2. Determine Output Tensor Size and Initialize (No change) ---
    const size_t R_sq = (size_t)R * R;
    const size_t R_cub = R_sq * R;
    const size_t R_4 = R_cub * R;
    const size_t output_size = R_4 * R; // R^5
    
    // The final result vector
    std::vector<T> core_tensor_G(output_size, T{0}); 

    // --- 3. Parallel Contractions using OpenMP Reduction ---
    
    // Create a thread-local accumulation structure for reduction.
    // Each thread will accumulate results into its own temporary copy (G_local).
    // The reduction clause 'reduction(+:core_tensor_G)' is not available 
    // for std::vector<T>, so we must manually manage it using a thread-local array
    // and a critical section or atomic update (or a better library approach like TBB).
    
    // For simplicity and correctness with standard OpenMP:
    // We can use a per-thread private copy of the entire output tensor G_local.
    
    #pragma omp parallel
    {
        // 🔑 Private accumulation tensor for each thread, initialized to zero
        std::vector<T> G_local(output_size, T{0}); 

        // Schedule the outer loop across all threads
        #pragma omp for nowait
        for (size_t k = 0; k < sparse_tensor.size(); ++k) {
            const auto& entry = sparse_tensor[k];
            if (entry.indices.size() != 5) continue;

            const int i1 = entry.indices[0]; 
            const int i2 = entry.indices[1]; 
            const int i3 = entry.indices[2];
            const int i4 = entry.indices[3]; 
            const int i5 = entry.indices[4];
            const T a_val = entry.value;

            // Ensure indices are within bounds (No change)
            if (i1 < 0 || i1 >= I1 || i2 < 0 || i2 >= I2 || i3 < 0 || i3 >= I3 || i4 < 0 || i4 >= I4 || i5 < 0 || i5 >= I5) {
                continue; 
            }

            // --- Core Accumulation Loop (Writes to G_local) ---
            for (int r1 = 0; r1 < R; ++r1) {
                for (int r2 = 0; r2 < R; ++r2) {
                    for (int r3 = 0; r3 < R; ++r3) {
                        for (int r4 = 0; r4 < R; ++r4) {
                            for (int r5 = 0; r5 < R; ++r5) {

                                // --- A. Get Factor Matrix Terms ---
                                const T f1_val = fmats[0][(size_t)r1 * I1 + i1];
                                const T f2_val = fmats[1][(size_t)r2 * I2 + i2];
                                const T f3_val = fmats[2][(size_t)r3 * I3 + i3];
                                const T f4_val = fmats[3][(size_t)r4 * I4+ i4];
                                const T f5_val = fmats[4][(size_t)r5 * I5+ i5];

                                // --- B. Get G Index (Row-Major G: R^5) ---
                                const size_t G_idx = r1 * R_4 + r2 * R_cub + r3 * R_sq + r4 * R + r5;
                                
                                // Accumulation into thread-local copy
                                G_local[G_idx] += a_val * f1_val * f2_val * f3_val * f4_val * f5_val;
                            }
                        }
                    }
                }
            }
        } // End of OpenMP 'for' loop

        // 🔑 Reduction: Merge G_local results back into the shared core_tensor_G
        // Only one thread can perform this merge operation at a time.
        #pragma omp critical
        {
            for (size_t i = 0; i < output_size; ++i) {
                core_tensor_G[i] += G_local[i];
            }
        }
    } // End of OpenMP 'parallel' block

    return core_tensor_G;
}


//======================================================================
// Output comparison functions
//======================================================================

template<typename T>
void print_3d_tensor_from_flat(T* arr, const std::vector<int>& j_dims)
{
    if (j_dims.size() < 3) {
        std::cout << "Error: Cannot print as 3D tensor. Dimension count is less than 3." << std::endl;
        return;
    }
    
    // For a 3D tensor (J1 x J2 x J3)
    const int J1 = j_dims[0];
    const int J2 = j_dims[1];
    const int J3 = j_dims[2];

    for (int i = 0; i < J1; ++i) { // Outer dimension (Slice)
        std::cout << "Slice " << i << " (Mode 1):" << std::endl;
        for (int j = 0; j < J2; ++j) { // Row dimension
            for (int k = 0; k < J3; ++k) { // Column dimension
                // Row-major index calculation: i * (J2*J3) + j * J3 + k
                uint64_t index = (uint64_t)i * J2 * J3 + (uint64_t)j * J3 + (uint64_t)k;
                
                // Print the value with some padding
                std::cout << arr[index] << "\t";
            }
            std::cout << std::endl; // Newline after each row
        }
        std::cout << std::endl; // Space between slices
    }
}


// Compare tensor arrays (single mode TMM)
template<typename T>
bool compare_tmm_arrays(T* arr1, T* arr2, std::vector<int> dims, int decomp_rank, int mode)
{
    // 1. Determine the contracted dimensions (J1, J2, J3)
    std::vector<int> J_dims = dims;
    if (mode >= 1 && mode <= dims.size()) {
        J_dims[mode - 1] = decomp_rank;
    } else {
        std::cerr << "Error: Invalid mode (" << mode << ") provided for comparison." << std::endl;
        return false;
    }

    const int J1 = J_dims[0];
    const int J2 = J_dims[1];
    const int J3 = J_dims[2];

    // 2. Calculate the total size
    uint64_t size = (uint64_t)J1 * J2 * J3;
    
    // Safety check for size calculation
    if (size == 0) {
        std::cerr << "Error: Calculated output size is zero. Check dimensions and rank." << std::endl;
        return false;
    }

    // 3. Compare elements and print on mismatch
    for(uint64_t i = 0; i < size; i++){
        if(arr1[i] != arr2[i]) return false;
    }
    
    return true;
}

//Compare tensor arrays (single mode TMM)
template<typename T>
float compare_tmm_arrays_float(T* arr1, T* arr2, std::vector<int> dims, int decomp_rank, int mode)
{
    // The output tensor size is the product of all dimensions, with the mode-th dimension replaced by decomp_rank (R).
    uint64_t size = (uint64_t)decomp_rank;
    for(int i = 0; i < dims.size(); i++){
        if(i != mode - 1) size *= dims[i]; // FIX: added semicolon and correct logic
    }
    
    float diff = 0.0f;
    for(uint64_t i = 0; i < size; i++){
        diff += std::abs(arr1[i] - arr2[i]);
    }

    // Return mean absolute difference
    return diff / size;
}

//Compare tensor arrays (multimode TMM)
template<typename T>
bool compare_multimode_contraction_arrays(T* arr1, T* arr2, std::vector<int> dims, int decomp_rank, int mode)
{
    // The output tensor size is the product of all dimensions, with the mode-th dimension replaced by decomp_rank (R).
    uint64_t size = (uint64_t)dims[mode - 1];
    for(int i = 0; i < dims.size(); i++){
        if(i != mode - 1) size *= decomp_rank;
    }

    for(uint64_t i = 0; i < size; i++){
        if(arr1[i] != arr2[i]) return false;
    }
    return true;
}

//Compare tensor arrays (multimode TMM)
template<typename T>
float compare_multimode_contraction_arrays_float(T* arr1, T* arr2, std::vector<int> dims, int decomp_rank, int mode)
{
    // The output tensor size is the product of all dimensions, with the mode-th dimension replaced by decomp_rank (R).
    uint64_t size = (uint64_t)dims[mode - 1];
    for(int i = 0; i < dims.size(); i++){
        if(i != mode - 1) size *= decomp_rank;
    }
    
    float diff = 0.0f;
    for(uint64_t i = 0; i < size; i++){
        diff += std::abs(arr1[i] - arr2[i]);
    }

    // Return mean absolute difference
    return diff / size;
}

//Compare core tensor arrays for ints and long ints
template<typename T>
bool compare_ct_arrays(T* arr1, T* arr2, std::vector<int> dims, int decomp_rank)
{
    // The core tensor size is R^N, where N = dims.size() and R = decomp_rank.
    uint64_t size = 1;
    for(size_t i = 0; i < dims.size(); i++){ // FIX: simplified R^N calculation
        size *= decomp_rank;
    }

    for(uint64_t i = 0; i < size; i++){
        if(arr1[i] != arr2[i]) return false;
    }
    return true;
}

//Compare core tensor arrays for ints and long ints
template<typename T>
float compare_ct_arrays_float(T* arr1, T* arr2, std::vector<int> dims, int decomp_rank)
{
    // The core tensor size is R^N, where N = dims.size() and R = decomp_rank.
    uint64_t size = 1;
    for(size_t i = 0; i < dims.size(); i++){ // FIX: simplified R^N calculation
        size *= decomp_rank;
    }

    float diff = 0.0f;
    for(uint64_t i = 0; i < size; i++){
        diff += std::abs(arr1[i] - arr2[i]);
    }

    // Return mean absolute difference
    return diff / size;
}