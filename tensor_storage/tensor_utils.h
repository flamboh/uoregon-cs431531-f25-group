#ifndef UTILS_H
#define UTILS_H

// ==========================
// Standard Library Includes
// ==========================
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <omp.h>


// ==========================
// Structs
// ==========================

// Represents a single nonzero entry in a sparse tensor.
// Stores the coordinate (i, j, k) and its value.
template<typename T>
struct NNZ_Entry {
    int rank;
    std::vector<int> indices; // Vector indices
    T value; // nonzero value for entries
};

// ==========================
// Template / Inline Definitions
// ==========================

// --- Math Utilities ---

// floor(log2(x)) for integer-like types
template<typename S>
int floor_log2(S x) {
    int res = -1;
    while (x) {
        x >>= 1;   // shift right until zero
        ++res;
    }
    return res;
}

// ceil(log2(x)) for integer-like types
template<typename S>
int ceiling_log2(S x) {
    if (x == 1) return 0;
    int res = 0;
    while (x) {
        x >>= 1;
        ++res;
    }
    return res;
}

// --- Sparse Tensor Utilities ---

// Generate a random sparse tensor with approximate block structure.
// Parameters:
//   - rows, cols, depth: tensor dimensions
//   - target nnz: Number of non zeros
//   - min_val, max_val: value range for nonzeros
//   - block_size: size of each "dense block"
//   - max_blocks: stop condition for block attempts
//
// Uses random number generators to select starting coordinates and fill small blocks.
template<typename T>
std::vector<NNZ_Entry<T>> generate_block_sparse_tensor_nd(
    const std::vector<int>& dims,   // tensor dimensions [D0, D1, D2, ...]
    uint64_t target_nnz,            // target number of non-zero entries
    T min_val, T max_val,           // value range
    int block_size,                 // edge size of dense sub-blocks
    int max_blocks,                 // cap for random block sampling
    float dropout_rate = 0.5f       // per-entry dropout probability
) {
    int rank = dims.size();
    if (rank < 2)
        throw std::invalid_argument("Tensor rank must be >= 2.");
    if (target_nnz == 0)
        return {}; // Return empty if no non-zeros are requested
    if (min_val > max_val)
        throw std::invalid_argument("Invalid value range.");

    // Note: Removed the calculation of total_entries and the density check.
    
    std::vector<NNZ_Entry<T>> entries;
    // Reserve space based on the requested target_nnz
    entries.reserve(target_nnz);

    std::mt19937 rng(std::random_device{}());

    // Distributions
    std::uniform_real_distribution<float> dropout_dist(0.0f, 1.0f);
    // std::uniform_int_distribution<int> block_pos_dist(0, 1000000); // Not used in the original loop logic
    
    auto generate_value = [&]() -> T {
        if constexpr (std::is_integral_v<T>) {
            // Need to handle the case where T is integral but T is not the same as int
            // Using a generic way to handle integral/floating point
            using value_type = T;
            std::uniform_int_distribution<value_type> dist(min_val, max_val);
            return dist(rng);
        } else if constexpr (std::is_floating_point_v<T>) {
            using value_type = T;
            std::uniform_real_distribution<value_type> dist(min_val, max_val);
            return dist(rng);
        } else {
            throw std::invalid_argument("Unsupported type for value generation.");
        }
    };

    // Compute stride = block spacing between start positions
    int stride = block_size * 2;

    // Generate blocks until target nonzeros reached
    uint64_t nnz_count = 0;
    int blocks_attempted = 0;

    while (nnz_count < target_nnz && blocks_attempted < max_blocks) {
        blocks_attempted++;

        // Random block start per dimension
        std::vector<int> block_start(rank);
        for (int r = 0; r < rank; ++r) {
            // Calculate the maximum start position index based on stride
            // Ensure the start position allows for a full block_size extent without exceeding dims[r]
            int max_start_index = dims[r] - block_size;
            int limit = std::max(max_start_index / stride, 0);

            // The original logic was potentially flawed in calculating the limit, 
            // relying on (dims[r] - block_size) / stride. Keeping the spirit but ensuring a valid distribution.
            // If limit is 0, the only possible start is 0.
            std::uniform_int_distribution<int> start_dist(0, limit);
            block_start[r] = start_dist(rng) * stride;
        }

        // Recursive n-dimensional block filling (iterative implementation)
        std::vector<int> idx(rank, 0);
        bool done = false;

        while (!done && nnz_count < target_nnz) {
            // Compute the actual coordinates for this entry
            std::vector<int> coord(rank);
            for (int r = 0; r < rank; ++r)
                coord[r] = block_start[r] + idx[r];

            // Check bounds (redundant if block_start calculation is perfect, but safer to keep)
            bool in_bounds = true;
            for (int r = 0; r < rank; ++r)
                if (coord[r] >= dims[r]) { in_bounds = false; break; }

            // If entry is in bounds and passes dropout, add it
            if (in_bounds && dropout_dist(rng) > dropout_rate) {
                entries.push_back({rank, coord, generate_value()});
                ++nnz_count;
            }

            // Increment n-dimensional index inside the block
            for (int r = rank - 1; r >= 0; --r) {
                idx[r]++;
                if (idx[r] < block_size)
                    break; // Moved to the next entry in the block
                idx[r] = 0; // Dimension wraps around
                if (r == 0) done = true; // Block finished
            }
        }
    }
    
    // Crucial step: If we overshot the target_nnz in the last block, truncate the vector.
    if (entries.size() > target_nnz) {
        entries.resize(target_nnz);
    }

    return entries;
}

// Read a tensor from a file (1-indexed format).
// Each line: i j k value
// Converts to 0-indexed internally.
// Read a tensor from a file (1-indexed format).
// Each line: index_1 index_2 ... index_rank value
// Converts to 0-indexed internally.
template<typename T>
std::vector<NNZ_Entry<T>> read_tensor_file(const std::string &filename, int rank, size_t maxLines = 0) 
{
    std::ifstream file(filename);
    std::vector<NNZ_Entry<T>> entries;

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return entries;
    }

    size_t count = 0;
    while (count < maxLines || maxLines == 0) {
        std::vector<int> indices(rank);
        T value;
        bool read_success = true;

        // 1. Read 'rank' number of indices
        for (int r = 0; r < rank; ++r) {
            // Check if reading the index failed (e.g., end of file)
            if (!(file >> indices[r])) {
                read_success = false;
                break;
            }
        }

        // 2. Read the value (only if all indices were successfully read)
        if (read_success) {
            if (!(file >> value)) {
                read_success = false;
            }
        }

        // If reading was successful for all (rank + 1) items, process the entry
        if (read_success) {
            // Shift to 0-index internally
            for (int r = 0; r < rank; ++r) {
                indices[r] -= 1; 
            }
            
            entries.push_back({rank, indices, value});
            count++;
        } else {
            // Break if file read failed (reached EOF or corrupt line)
            break; 
        }
    }

    return entries;
}

// Return true if a given entry exists in the tensor.
template<typename T>
bool find_entry(std::vector<NNZ_Entry<T>> entry_vec, std::vector<int> dims, T val) {
    for (size_t i = 0; i < entry_vec.size(); i++) {
        if(dims == entry_vec[i].indices && val == entry_vec[i].value) return true;
    }
    return false;
}

// Print all entries in a sparse tensor
template<typename T>
void print_entry_vec(const std::vector<NNZ_Entry<T>>& entry_vec) {
    for (size_t i = 0; i < entry_vec.size(); ++i) {
        int rank = entry_vec[i].indices.size();
        std::cout << "indices: ";
        for(int j = 0; j < rank; j++){
            std::cout << entry_vec[i].indices[j] << " ";
        }
        std::cout << "val: " << entry_vec[i].value << "\n";
    }
}

// Print a dense 2D matrix with column width formatting
template<typename T>
void print_matrix(T** matrix, int rows, int cols, int width) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j)
            std::cout << std::setw(width) << matrix[i][j] << " ";
        std::cout << "\n";
    }
}

// ==========================
// Non-template Implementations
// ==========================

// Print x least-significant bits of a 128-bit value
inline void print_lsb_bits(__uint128_t value, int x) {
    if (x < 1 || x > 128) {
        std::cerr << "Error: x must be between 1 and 128\n";
        return;
    }
    for (int i = x - 1; i >= 0; --i) {
        int bit = (value >> i) & 1;
        std::cout << bit;
    }
    std::cout << std::endl;
}

// Print x least-significant bits of a 64-bit value
inline void print_uint64(uint64_t value, int x) {
    if (x < 1 || x > 64) {
        std::cerr << "Error: x must be between 1 and 64\n";
        return;
    }
    for (int i = x - 1; i >= 0; --i) {
        int bit = (value >> i) & 1;
        std::cout << bit;
    }
    std::cout << std::endl;
}

// Convert 128-bit unsigned integer to scientific notation string
inline std::string uint128_to_sci_string(__uint128_t value, int precision) {
    if (value == 0) return "0.0e+0";

    __uint128_t temp = value;
    int exponent = 0;
    while (temp >= 10) {
        temp /= 10;
        ++exponent;
    }

    __uint128_t scale = 1;
    for (int i = 0; i < precision; ++i) scale *= 10;
    __uint128_t digits = (value * scale) / __uint128_t(std::pow(10, exponent));

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision)
        << static_cast<double>(digits) / scale << "e+" << exponent;
    return oss.str();
}


#endif
