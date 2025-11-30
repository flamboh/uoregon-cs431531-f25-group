#ifndef BLCO_H
#define BLCO_H

#include <vector>
#include <utility>
#include <unordered_map>
#include <chrono>
#include <omp.h> 
#include <numeric>
#include "tensor.h"

//======================================================================
// BLCO_BLOCK_CPU: Represents one block of a BLCO tensor stored on CPU
// - block: block identifier
// - size: number of NNZ in the block
// - indexes: vector of encoded BLCO indices
// - values: vector of corresponding NNZ values
//======================================================================
template<typename T>
struct BLCO_BLOCK_CPU {
    int idx;
    int size;
    std::vector<uint64_t> indexes;
    std::vector<T> values;
};

//======================================================================
// BLCO_BLOCK_GPU: Same structure but ready for GPU (device pointers)
//======================================================================
template<typename T>
struct BLCO_BLOCK_GPU {
    int idx;
    int size;
    uint64_t* indexes;
    T* values;
};


//======================================================================
// BLCO_Tensor_3D
//======================================================================
// Blocked Linearized Coordinate (BLCO) tensor format for 3D tensors
// Extends ALTO_Tensor_3D<T,S> with:
//   - Index re-encoding for GPU efficiency (shift/mask rather than scatter)
//   - Blocking step: splits tensor into manageable chunks
//   - Support for both 64-bit and 128-bit indexing
//   - Functions for extracting coordinates from BLCO indices
//   - Debug utilities
//
// Notes: 
//   * Assumes dimensions < 2^31 (due to use of int)
//   * Optimized for GPU out-of-memory streaming
//======================================================================
template<typename T, typename S>
class Blco_Tensor : public Tensor<T, S>
{
protected:
    int blocks_populated; //The number of blocks which are populated    
    bool blocks_needed; //If 64 bits isn't enough to represent all of the indexes properly                 
    std::vector<int> bit_widths; // Bit-widths needed for each mode
    std::vector<int> block_modes; // Modes whose indices are represented by the block number
    std::vector<uint64_t> bitmasks; // Masks defining bit placement for each mode
    std::vector<int> populated_blocks; //Indexes of all the different blocks which are populated
    std::vector<BLCO_BLOCK_CPU<T>> blco_tensor; // BLCO representation: vector of blocks

    //------------------------------------------------------------------
    // Utility: sort two parallel vectors by the first one
    //------------------------------------------------------------------
    void sort_pair_by_first(std::pair<std::vector<S>, std::vector<T>>& p) 
    {
        std::vector<size_t> indices(p.first.size());
        std::iota(indices.begin(), indices.end(), 0);  

        std::sort(indices.begin(), indices.end(),
                  [&](size_t i, size_t j) { return p.first[i] < p.first[j]; });

        std::vector<S> sorted_first;
        std::vector<T> sorted_second;
        for (size_t i : indices) {
            sorted_first.push_back(p.first[i]);
            sorted_second.push_back(p.second[i]);
        }

        p.first = std::move(sorted_first);
        p.second = std::move(sorted_second);
    }

    void determine_bit_widths()
    {
        int bit_sum = 0;
        for(int i = 0; i < this->rank; i++){
            int bits_needed = ceiling_log2(this->dims[i]);
            bit_widths.push_back(bits_needed);
            bit_sum += bits_needed;
        }
        if(bit_sum > 64) blocks_needed = true;
        else blocks_needed = false;
    }

    //------------------------------------------------------------------
    // Create BLCO masks (unlike ALTO, assigns bits sequentially)
    //------------------------------------------------------------------

    void create_blco_masks()
    {
        if (std::accumulate(bit_widths.begin(), bit_widths.end(), 0) == 0) return;

        int total_bits = 0;
        for (int i = 0; i < this->rank; ++i) {
            uint64_t mask;
            int w = bit_widths[i];
            if (w >= 64) {
                mask = ~uint64_t(0); // all ones if the mode uses >=64 bits
            } else {
                mask = (uint64_t(1) << w) - 1;
            }
            bitmasks.push_back(mask);

            total_bits += w;
            if (total_bits > 64) {
                // mode i (0-based) crosses into block bits.
                // store mode index in 1-based convention if you use that elsewhere.
                block_modes.push_back(i + 1);
            }
        }
    }


    //------------------------------------------------------------------
    // - Uses masks to place bits for row/col/depth
    // - If >64 bits are needed, overflow goes to higher bits
    //------------------------------------------------------------------
    // indices is expected length == this->rank
    S index_conversion(const std::vector<int>& indices) 
    {
        int shift = 0;
        S val = 0;

        for (int i = 0; i < this->rank; ++i) {
            // bounds & mask the coordinate for this mode
            S indice = static_cast<S>(indices[i]) & static_cast<S>(bitmasks[i]); // single mask

            // place the bits of indice starting at 'shift'
            val |= (indice << shift);

            // advance shift by this mode's bit width
            shift += bit_widths[i];
        }

        return val;
    }


    //------------------------------------------------------------------
    // Build intermediate tensor representation (BLCO indices + values)
    //------------------------------------------------------------------
    std::pair<std::vector<S>, std::vector<T>> create_intermediate_tensor(std::vector<NNZ_Entry<T>> entries)
    {
        std::pair<std::vector<S>, std::vector<T>> p1;
        size_t n = this->nnz_entries;
        p1.first.resize(n);
        p1.second.resize(n);

        // #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            p1.first[i]  = index_conversion(entries[i].indices);
            p1.second[i] = entries[i].value;
        }
        return p1;
    }

    //------------------------------------------------------------------
    // Create BLCO tensor (split into blocks if >64 bits)
    //------------------------------------------------------------------
    void create_blco_tensor(const std::vector<NNZ_Entry<T>>& entries)
    {
        // Create intermediate (encoded indices, values)
        auto encoded = create_intermediate_tensor(entries);
        const auto& linear_indices = encoded.first;
        const auto& values = encoded.second;

        // Single-block case (≤ 64 bits total)
        if (!blocks_needed) {
            populated_blocks.push_back(0);
            BLCO_BLOCK_CPU<T> block;
            block.idx   = 0;
            block.size  = this->nnz_entries;
        
            // Convert 128-bit indices to 64-bit before assigning
            std::vector<uint64_t> linear_indices_64;
            linear_indices_64.reserve(linear_indices.size());
            for (auto idx : linear_indices)
                linear_indices_64.push_back(static_cast<uint64_t>(idx));
        
            block.indexes = std::move(linear_indices_64);
            block.values  = values;
        
            blco_tensor.push_back(std::move(block));
            return;
        }

        // ----------------------------------------------------------------------
        // Multi-block case (> 64 bits total)
        // ----------------------------------------------------------------------

        // Masks for splitting 128-bit encoded index
        const __uint128_t lower_mask = ((__uint128_t)1 << 64) - 1;  // low 64 bits = 1s
        const __uint128_t upper_mask = ~lower_mask;                  // bits >=64

        // Process each nonzero entry
        for (int i = 0; i < this->nnz_entries; ++i) {
            __uint128_t full_index = static_cast<__uint128_t>(linear_indices[i]);
            uint64_t short_index   = static_cast<uint64_t>(full_index & lower_mask);
            int block_id           = static_cast<int>((full_index & upper_mask) >> 64);
            const T& val           = values[i];

            // Find existing block position or insertion point
            auto pos_it = std::lower_bound(populated_blocks.begin(), populated_blocks.end(), block_id);
            size_t pos  = std::distance(populated_blocks.begin(), pos_it);

            // Create block if not found
            if (pos_it == populated_blocks.end() || *pos_it != block_id) {
                populated_blocks.insert(populated_blocks.begin() + pos, block_id);
                BLCO_BLOCK_CPU<T> new_block;
                new_block.idx = block_id;
                new_block.size = 0;
                blco_tensor.insert(blco_tensor.begin() + pos, std::move(new_block));
            }

            // Append to corresponding block
            blco_tensor[pos].indexes.push_back(short_index);
            blco_tensor[pos].values.push_back(val);
            blco_tensor[pos].size = static_cast<int>(blco_tensor[pos].indexes.size());
        }
    }


public:
    //------------------------------------------------------------------
    // Constructors: build from NNZ list
    //------------------------------------------------------------------
    Blco_Tensor(const std::vector<NNZ_Entry<T>>& entry_vec, std::vector<int> dims, int decomp_rank = 10) : Tensor<T,S>(entry_vec, dims, decomp_rank)
    {
        determine_bit_widths();
        create_blco_masks();
        create_blco_tensor(entry_vec);
    }

    //------------------------------------------------------------------
    // Decode BLCO index → coordinate (64-bit version)
    //------------------------------------------------------------------
    int get_mode_idx_blco(uint64_t blco_index, int mode, int block) const
    {
        // mode is 1-based; convert to 0-based
        int m = mode - 1;
        int shift = 0;
        for (int i = 0; i < m; ++i) shift += bit_widths[i];

        uint64_t mask = bitmasks[m];

        if (shift >= 64) {
            // all bits are in the block (upper part)
            int upper_shift = shift - 64;
            // extract bits_from_block starting at upper_shift
            uint64_t val = static_cast<uint64_t>((static_cast<uint64_t>(block) >> upper_shift) & mask);
            return static_cast<int>(val);
        } else {
            int bits_in_low = std::min(64 - shift, bit_widths[m]);
            uint64_t low_part = (blco_index >> shift) & ((bits_in_low == 64) ? ~uint64_t(0) : ((uint64_t(1) << bits_in_low) - 1));

            if (bits_in_low == bit_widths[m]) {
                return static_cast<int>(low_part);
            } else {
                // need remaining bits from block
                int bits_from_block = bit_widths[m] - bits_in_low;
                uint64_t block_mask_small = (bits_from_block == 64) ? ~uint64_t(0) : ((uint64_t(1) << bits_from_block) - 1);
                uint64_t upper_part = static_cast<uint64_t>(block & block_mask_small);
                uint64_t combined = (upper_part << bits_in_low) | low_part;
                return static_cast<int>(combined & mask);
            }
        }
    }


    //------------------------------------------------------------------
    // Find a block by ID (returns index or -1)
    //------------------------------------------------------------------
    int find_block(int target_block)
    {
        int low = 0;
        int high = populated_blocks.size() - 1;

        while (low <= high) {
            int mid = low + (high - low) / 2;

            if (populated_blocks[mid] == target_block) {
                return mid; 
            } else if (populated_blocks[mid] < target_block) {
                low = mid + 1; 
            } else {
                high = mid - 1; 
            }
        }
        return -1;
    }

    //------------------------------------------------------------------
    // Getters
    //------------------------------------------------------------------
    const std::vector<BLCO_BLOCK_CPU<T>>& get_blco() const {return blco_tensor;}
    std::vector<uint64_t> get_bitmasks() const {return bitmasks;}

    //------------------------------------------------------------------
    // Copy GPU result vector back into factor matrix (for MTTKRP output)
    //------------------------------------------------------------------
    void copy_vector_to_fmat(T* v1, int mode) const
    {
        for(int i = 0; i < this->dims[mode - 1] * this->factor_rank; i++){
            this->fmats[mode - 1][i] = v1[i];
        }
    }

    //------------------------------------------------------------------
    // Estimate number of distinct indexes per GPU wavefront
    // Helps in tuning GPU kernels
    //------------------------------------------------------------------
    int determine_indexes_per_wavefront(int mode) const
    {
        int max_indexes = 0;

        if(!blocks_needed){
            BLCO_BLOCK_CPU<T> b1 = blco_tensor[0];
            std::vector<int> indexes;
            for(int i = 0; i < b1.indexes.size(); i++){
                if(i != 0 && i % 64 == 0){
                    if(indexes.size() > max_indexes) max_indexes = indexes.size();
                    indexes.clear();
                }
                int idx = get_mode_idx_blco(b1.indexes[i], mode, 0);
                if (std::find(indexes.begin(), indexes.end(), idx) == indexes.end()) 
                    indexes.push_back(idx);
            }
            return std::max(max_indexes, (int)indexes.size());
        }
        else{
            int block_num = -1;
            for (const auto &block : blco_tensor) {
                std::vector<int> indexes;
                block_num++;
                for (size_t i = 0; i < block.indexes.size(); i++) {
                    if (i != 0 && i % 64 == 0) {
                        if (indexes.size() > max_indexes) max_indexes = indexes.size();
                        indexes.clear();
                    }
                    int idx = get_mode_idx_blco(block.indexes[i],mode, block);
                    if (std::find(indexes.begin(), indexes.end(), idx) == indexes.end()) {
                        indexes.push_back(idx);
                    }
                }
                if (indexes.size() > max_indexes) max_indexes = indexes.size();
            }
            return max_indexes;
        }
    }
    //------------------------------------------------------------------
    // Debug utilities: print decoded indices and values
    //------------------------------------------------------------------
    void debug_linear_indices()
    {
        int block_num;
        for (auto &b1 : blco_tensor) {
            block_num = b1.idx;
            for(int j = 0; j < b1.indexes.size(); j++){
                std::vector<int> indices;
                for(int k = 0; k < this->rank; k++){
                    std::cout << "Mode " << k + 1 << " : " << get_mode_idx_blco(b1.indexes[j], block_num, k + 1);
                }
                std::cout << ", val=" << b1.values[j] << "\n";
            }
        }
    }
};

#endif