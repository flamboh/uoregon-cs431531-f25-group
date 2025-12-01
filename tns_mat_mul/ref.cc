#include "../tensor_storage/blco.h"

template<typename T, typename S>
Blco_Tensor<T, S> ttm_tucker(const Blco_Tensor<T, S>& X,
                              const std::vector<std::vector<T>>& factor_matrix,
                              int mode)
{
    // X: input sparse tensor
    // factor_matrix: matrix to multiply along 'mode' (rows = dim[mode], cols = rank)
    // mode: 1-based mode index (1,2,3 for a 3D tensor)

    int R = factor_matrix[0].size(); // new rank along this mode
    std::vector<int> new_dims = X.dims; 
    new_dims[mode - 1] = R; // replace mode dim with factor rank

    std::vector<NNZ_Entry<T>> new_entries;

    for (const auto& block : X.get_blco())
    {
        int block_id = block.idx;
        for (int i = 0; i < block.size; i++)
        {
            // Decode coordinates
            std::vector<int> coords(X.rank);
            for (int m = 0; m < X.rank; m++)
                coords[m] = X.get_mode_idx_blco(block.indexes[i], m + 1, block_id);

            T val = block.values[i];

            // Multiply along specified mode
            int mode_idx = coords[mode - 1];
            for (int r = 0; r < R; r++)
            {
                NNZ_Entry<T> new_entry;
                new_entry.indices = coords;
                new_entry.indices[mode - 1] = r; // new mode index
                new_entry.value = val * factor_matrix[mode_idx][r];
                new_entries.push_back(std::move(new_entry));
            }
        }
    }

    // Build new sparse tensor in BLCO format
    Blco_Tensor<T, S> Y(new_entries, new_dims);
    return Y;
}
