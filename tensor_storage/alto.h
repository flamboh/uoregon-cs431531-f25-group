#ifndef ALTO_H
#define ALTO_H

#include <unordered_map>
#include <unordered_set>
#include <omp.h>
#include "tensor.h"

//======================================================================
// ALTOEntry: Represents one nonzero entry in ALTO format
//======================================================================
// linear_index = compact ALTO-encoded coordinate (bitmask-based)
// value        = stored tensor value at that location
template<typename T, typename S>
struct ALTOEntry {
    S linear_index;  
    T value;            
};

//======================================================================
// Alto_Tensor_3D
//======================================================================
// Inherits from Tensor_3D<T,S> and extends it with ALTO storage.
//
// Key features:
//   - Encodes (i,j,k, ...) coordinates into a *linearized ALTO index*
//     using adaptive bitmask assignment.
//   - Stores nonzeros in a sorted vector (by ALTO index).
//   - Partitions NNZs for parallel algorithms.
//======================================================================
template<typename T, typename S>
class Alto_Tensor : public Tensor<T, S>{
protected:
    int num_threads;                        // Number of threads for parallel MTTKRP                        
    std::vector<S> bitmasks;                // Stores bitmasks for each mode
    std::vector<int> partitions;            // Partition boundaries for NNZ distribution
    std::vector<ALTOEntry<T,S>> alto_tensor;// Compact ALTO representation of the tensor

    //------------------------------------------------------------------
    // Choose "ideal" number of threads for parallel algorithm based on NNZ size
    //------------------------------------------------------------------
    int ideal_threads()
    {
        const int thresholds[] = {250, 500, 750, 1000, 2000, 20000};
        const int max_threads = 8;
    
        for (int t = 0; t < sizeof(thresholds) / sizeof(thresholds[0]); ++t) {
            if (this->nnz_entries < thresholds[t]) return t + 2;
        }
        return max_threads;
    }

    //------------------------------------------------------------------
    // Partition NNZs evenly across threads
    //------------------------------------------------------------------
    void set_partitions()
    {
        int partition_size = this->nnz_entries / num_threads;
        if(this->nnz_entries % num_threads != 0) partition_size++;

        int sum = 0;
        for(int i = 0; i < num_threads; i++){
            sum += partition_size;
            if(sum > this->nnz_entries) sum = this->nnz_entries;
            partitions.push_back(sum);
        }
    }

    //------------------------------------------------------------------
    // For a given mode and block (thread partition), find minimum index
    //------------------------------------------------------------------
    int determine_block_offset(int mode, int block_index)
    {
        int start = (block_index > 0) ? partitions[block_index - 1] : 0;
        int end = partitions[block_index];
        int min = std::max(this->dims);

        for(int i = start; i < end; i++){
            int idx = get_mode_idx(alto_tensor[i].linear_index, mode);
            if(idx < min) min = idx;
        }
        return min;
    }

    //------------------------------------------------------------------
    // For a given mode and block (thread partition), find maximum index
    //------------------------------------------------------------------
    int determine_block_limit(int mode, int block_index)
    {
        int start = (block_index > 0) ? partitions[block_index - 1] : 0;
        int end = partitions[block_index];
        int max = 0;

        for(int i = start; i < end; i++){
            int idx = get_mode_idx(alto_tensor[i].linear_index, mode);
            if(idx > max) max = idx;
        }
        return max;
    }

    //------------------------------------------------------------------
    // Find which partition/block a given NNZ index belongs to
    //------------------------------------------------------------------
    int determine_block(int index)
    {
        int previous = 0;
        for(int i = 0; i < partitions.size(); i++){
            if(previous < index && index < partitions[i]) return i;
            previous = partitions[i];
        }
        return -1;
    }

    //------------------------------------------------------------------
    // Mark entries that lie on "fiber boundaries" spanning multiple blocks
    // These require atomics to prevent race conditions.
    //------------------------------------------------------------------
    void set_boundaries(int mode)
    {
        std::unordered_map<int, std::unordered_set<int>> fiber_blocks; 
        S boundary_mask = S(1) << this->num_bits;

        // Map: fiber index → set of blocks that contain it
        for (int i = 0; i < alto_tensor.size(); ++i) {
            S lin_idx = alto_tensor[i].linear_index;
            int idx = get_mode_idx(lin_idx, mode);
            int block = determine_block(i);
            fiber_blocks[idx].insert(block);
        }

        // Mark boundary entries
        for (int i = 0; i < alto_tensor.size(); ++i) {
            S lin_idx = alto_tensor[i].linear_index;
            int idx = get_mode_idx(lin_idx, mode);
            int block = determine_block(i);

            if (fiber_blocks[idx].size() > 1) {
                alto_tensor[i].linear_index |= boundary_mask;
            }
        }
    }

    //------------------------------------------------------------------
    // Reset boundary flag bit in linear index
    //------------------------------------------------------------------
    void reset_boundaries()
    {
        S mask = ~(S(1) << this->num_bits);
        for(int i = 0; i < alto_tensor.size(); i++){
            alto_tensor[i].linear_index &= mask;
        }
    }

    //------------------------------------------------------------------
    // Helper: Pick largest mode (bit allocation priority)
    //------------------------------------------------------------------
    int largest_mode(std::vector<int> bits)
    {
        int max_mode = 1; //Find the mode with the largest number of bits left
        bool all_zeros = true;
        for(int i = 1; i < bits.size(); i++){
            if(all_zeros && bits[i] > 0) all_zeros = false;
            if((bits[i] > bits[max_mode - 1]) || (bits[i] == bits[max_mode - 1] && this->dims[i] > this->dims[max_mode - 1])) max_mode = i + 1;
        }
        return !all_zeros * max_mode;
    }

    //------------------------------------------------------------------
    // Create ALTO bitmasks for each mode (rows, cols, depth)
    // Assigns bits in interleaved order based on largest_mode()
    //------------------------------------------------------------------
    void create_masks()
    {
        std::vector<int> bits;
        for(int i = 0; i < this->rank; i++){
            bits.push_back(ceiling_log2(this->dims[i]));
        }

        for(int i = 0; i < this->rank; i++) bitmasks.push_back(S(0));

        S mask = S(1) << (this->num_bits - 1);
        int sum_of_bits = std::accumulate(bits.begin(), bits.end(), 0);
        while (sum_of_bits > 0 && mask != 0) {
            int l1 = largest_mode(bits);
            bits[l1 - 1]--;
            bitmasks[l1 - 1] |= mask;
            mask >>= 1;
            sum_of_bits--;
        }
    }

    //------------------------------------------------------------------
    // Translate (i,j,k,...) → ALTO linearized index using masks
    //------------------------------------------------------------------
    S translate_idx(std::vector<int> indices) {
        if (std::accumulate(this->dims.begin(), this->dims.end(), 0) == 0) return 0;
    
        S val = 0;
        for (int i = 0; i < this->num_bits; ++i) {
            S mask = static_cast<S>(1) << i;
            for(int i = 0; i < this->rank; i++){
                if(mask & bitmasks[i]) { if(indices[i] & 1ULL) val |= mask; indices[i] >>= 1;}
            }
        }
        return val;
    }

    //------------------------------------------------------------------
    // Build ALTO tensor from vector of NNZ entries
    //------------------------------------------------------------------
    void create_alto_vector(const std::vector<NNZ_Entry<T>>& tensor_vec)
    {
        alto_tensor.clear();
        alto_tensor.resize(tensor_vec.size());

        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(tensor_vec.size()); i++) {
            ALTOEntry<T,S> entry;
            entry.linear_index = translate_idx(tensor_vec[i].indices);
            entry.value = tensor_vec[i].value;
            alto_tensor[i] = entry;
        }

        std::sort(alto_tensor.begin(), alto_tensor.end(),
            [](const ALTOEntry<T,S>& a, const ALTOEntry<T,S>& b) {
                return a.linear_index < b.linear_index;
            });
    }

public:
    //------------------------------------------------------------------
    // Constructor
    //------------------------------------------------------------------
    Alto_Tensor(const std::vector<NNZ_Entry<T>>& entry_vec, std::vector<int> dims) : Tensor<T,S>(entry_vec, dims)
    {
        create_masks();
        create_alto_vector(entry_vec);
        num_threads = ideal_threads();
        set_partitions();
    }

    //------------------------------------------------------------------
    // Extract coordinate from ALTO index
    //------------------------------------------------------------------
    int get_mode_idx(S alto_idx, int mode) 
    {
        S mask = bitmasks[mode - 1];

        int coord = 0, bit_pos = 0;
        int length = sizeof(S) * 8;

        for (int i = 0; i < length; ++i) {
            if ((mask >> i) & static_cast<S>(1)) {
                coord |= ((alto_idx >> i) & static_cast<S>(1)) << bit_pos;
                ++bit_pos;
            }
        }
        return coord;
    }

    //------------------------------------------------------------------
    // Getters
    //------------------------------------------------------------------
    const std::vector<ALTOEntry<T,S>>& get_alto() const { return alto_tensor; }
    const std::vector<S> get_modemasks() const { return bitmasks;}
    
    //------------------------------------------------------------------
    // Debug utility: Print ALTO indices and decoded coordinates
    //------------------------------------------------------------------
    void debug_linear_indices(){
        for (auto& e : alto_tensor) {
            std::string sci = uint128_to_sci_string(e.linear_index,10);
            std::cout << "Index: " << sci;
            for(int i = 0; i < this->rank; i++){
                std::cout << ", mode = " << get_mode_idx(e.linear_index, i + 1) << " ";
            }
            std::cout << ", val=" << e.value << "\n";
        }
    }    
};

#endif


















