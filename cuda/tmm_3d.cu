#include <stdexcept>
#include <vector>

#include "tmm_3d.cuh"
#include "tmm_nd.cuh"

namespace {

void validate_3d_mode(int mode) {
    if (mode < 1 || mode > 3) {
        throw std::invalid_argument("3D mode must be in [1,3]");
    }
}

void validate_3d_dims(const std::vector<int>& dims) {
    if (dims.size() != 3) {
        throw std::runtime_error("Expected a 3D tensor");
    }
}

} // namespace

template <typename T, typename S>
T* tmm_3d_cuda(const Blco_Tensor<T, S>& sparse_tensor,
               int mode,
               int block_size,
               bool log_timings) {
    validate_3d_mode(mode);
    validate_3d_dims(sparse_tensor.get_dims());
    return tmm_nd_cuda<T, S>(sparse_tensor, mode, block_size, log_timings);
}

template <typename T, typename S>
T* tucker_compute_core_3d_cuda(const Blco_Tensor<T, S>& sparse_tensor,
                               int block_size) {
    validate_3d_dims(sparse_tensor.get_dims());
    return tucker_compute_core_nd_cuda<T, S>(sparse_tensor, block_size);
}

template <typename T, typename S>
T* contract_n_minus_one_modes_3d_cuda(const Blco_Tensor<T, S>& sparse_tensor,
                                      int block_size,
                                      int mode) {
    validate_3d_mode(mode);
    validate_3d_dims(sparse_tensor.get_dims());
    return contract_n_minus_one_modes_nd_cuda<T, S>(sparse_tensor, block_size, mode);
}

#define INSTANTIATE_TMM(TTYPE, STYPE) \
    template TTYPE* tmm_3d_cuda<TTYPE, STYPE>(const Blco_Tensor<TTYPE, STYPE>&, int, int, bool); \
    template TTYPE* tucker_compute_core_3d_cuda<TTYPE, STYPE>(const Blco_Tensor<TTYPE, STYPE>&, int); \
    template TTYPE* contract_n_minus_one_modes_3d_cuda<TTYPE, STYPE>(const Blco_Tensor<TTYPE, STYPE>&, int, int);

INSTANTIATE_TMM(int, uint64_t)
INSTANTIATE_TMM(float, uint64_t)
INSTANTIATE_TMM(long int, uint64_t)
INSTANTIATE_TMM(int, __uint128_t)
INSTANTIATE_TMM(float, __uint128_t)
INSTANTIATE_TMM(long int, __uint128_t)

#undef INSTANTIATE_TMM
