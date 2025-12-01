#include "../tensor_storage/blco.h"
#include "../tensor_storage/tensor_utils.h"


template<typename T>
T* tensor_matrix_mul(
    const std::vector<NNZ_Entry<T>>& entries, 
    const std::vector<int>& dims, 
    const T* flattened_mat, // Changed to const T* as it's an input matrix
    int mode,               // m in the signature, represents 1-based mode index
    int new_dim_size        // n in the signature, represents the new rank J
) {
    if (mode <= 0 || mode > dims.size()) {
        throw std::out_of_range("Invalid mode index. Must be between 1 and rank.");
    }

    int rank = dims.size();
    int mode_idx = mode - 1; // 0-based index for the contracted mode
    int contracted_dim = dims[mode_idx]; // I_mode: The size of the dimension being contracted

    // Factor matrix U is assumed to be (new_dim_size x contracted_dim)
    // U's flat index: row * contracted_dim + col => j * I_mode + i_mode

    // 1. Determine Output Dimensions and Total Size
    std::vector<int> new_dims = dims; 
    new_dims[mode_idx] = new_dim_size; // Replace old dimension with new rank

    long long total_size = 1;
    for (int dim : new_dims) {
        total_size *= dim;
    }

    // 2. Allocate and Initialize the Output Flattened Array Y (Dense)
    // Use '()' for zero-initialization.
    T* Y = new T[total_size](); 

    // 3. Calculate Strides for the Result Tensor Y (Row-Major Order)
    // Stride[i] is the jump size when index i increases by 1.
    std::vector<long long> strides(rank);
    long long current_stride = 1;
    for (int r = rank - 1; r >= 0; --r) {
        strides[r] = current_stride;
        current_stride *= new_dims[r];
    }
    
    // 4. Perform TTM: Y_k = sum_{i_mode} X_i * U_{k_mode, i_mode}
    for (const auto& entry : entries) {
        if (entry.indices.size() != rank) {
            throw std::runtime_error("NNZ entry indices size does not match tensor rank.");
        }
        
        T x_val = entry.value;
        int i_mode = entry.indices[mode_idx]; // Index of the contracted mode in X

        if (i_mode < 0 || i_mode >= contracted_dim) {
             throw std::runtime_error("Index out of bounds for contracted mode.");
        }

        // Loop over the new dimension J (index 'j')
        for (int j = 0; j < new_dim_size; j++) {
            
            // a. Get the factor matrix value U[j][i_mode]
            // Flat index = row * columns + col
            long long U_flat_index = (long long)j * contracted_dim + i_mode;
            T u_val = flattened_mat[U_flat_index];
            
            // b. Calculate the contribution
            T contribution = x_val * u_val;
            
            // c. Calculate the flat index in the output array Y
            long long Y_flat_index = 0;
            for (int r = 0; r < rank; r++) {
                int index_r;
                if (r == mode_idx) {
                    // Use the new index 'j' for the contracted mode
                    index_r = j;
                } else {
                    // Use the original index from the NNZ entry
                    index_r = entry.indices[r];
                }
                
                if (index_r < 0 || index_r >= new_dims[r]) {
                    // Should theoretically not happen if input indices are valid
                    throw std::runtime_error("Internal error: Calculated index out of bounds for Y.");
                }
                
                Y_flat_index += (long long)index_r * strides[r];
            }
            // d. Accumulate the contribution in the dense array Y
            if (Y_flat_index < total_size) {
                Y[Y_flat_index] += contribution;
            }
        }
    }

    return Y;
}

template<typename T>
bool compare_arrays(T* arr1, T* arr2, int size)
{
    for(int i = 0; i < size; i++){
        if(arr1[i] != arr2[i]) return false;
    }
    return true;
}
