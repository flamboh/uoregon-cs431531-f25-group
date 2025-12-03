#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iomanip>

#include "tmm_3d.cuh"

#include "../tns_mat_mul/tmm_utils.h"

using std::string;
using std::vector;

namespace {

int host_ceiling_log2(int x) {
    if (x <= 1) return 0;
    int res = 0;
    while (x > 0) {
        x >>= 1;
        ++res;
    }
    return res;
}

} // namespace

struct LaunchConfig {
    int nnz;
    int mode;
    int decomp_rank;
    vector<int> dims;
};

LaunchConfig parse_args(int argc, char** argv) {
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0]
                  << " <nnz> <mode> <rank> <dim1> <dim2> <dim3> [block_size]\n";
        std::exit(1);
    }

    LaunchConfig cfg;
    cfg.nnz = std::atoi(argv[1]);
    cfg.mode = std::atoi(argv[2]);
    cfg.decomp_rank = std::atoi(argv[3]);
    cfg.dims = {std::atoi(argv[4]), std::atoi(argv[5]), std::atoi(argv[6])};
    return cfg;
}

template <typename T, typename S>
void run_ttms(const LaunchConfig& cfg, int block_size) {
    const double total_entries = static_cast<double>(cfg.dims[0]) * cfg.dims[1] * cfg.dims[2];
    const double freq = cfg.nnz / total_entries;
    auto mat_start = std::chrono::high_resolution_clock::now();
    vector<NNZ_Entry<T>> entries = generate_block_sparse_tensor_nd<T>(
        cfg.dims, freq, 0, 100, std::max(1, cfg.dims[0] / 20), 100);
    auto mat_end = std::chrono::high_resolution_clock::now();

    auto factor_start = std::chrono::high_resolution_clock::now();
    Blco_Tensor<T, S> blco(entries, cfg.dims, cfg.decomp_rank);
    auto factor_end = std::chrono::high_resolution_clock::now();
    const double mat_ms = std::chrono::duration<double, std::milli>(mat_end - mat_start).count();
    const double factor_ms = std::chrono::duration<double, std::milli>(factor_end - factor_start).count();
    std::cout << "Matricization time: " << mat_ms << " ms | Factor init time: "
              << factor_ms << " ms\n";
    vector<T*> fmats = blco.get_fmats();
    T* cpu_ttm = tensor_matrix_mul<T>(entries, cfg.dims, fmats[cfg.mode - 1], cfg.mode, cfg.decomp_rank);
    T* gpu_ttm = tmm_3d_cuda<T, S>(blco, cfg.mode, block_size);

    auto approx_equal = [](T lhs, T rhs) {
        if constexpr (std::is_floating_point_v<T>) {
            return std::abs(static_cast<double>(lhs) - static_cast<double>(rhs)) <= 0.01;
        } else {
            return lhs == rhs;
        }
    };

    vector<int> new_dims = cfg.dims;
    new_dims[cfg.mode - 1] = cfg.decomp_rank;
    uint64_t total = static_cast<uint64_t>(new_dims[0]) * new_dims[1] * new_dims[2];
    bool passed = true;
    for (uint64_t i = 0; i < total; ++i) {
        if (!approx_equal(cpu_ttm[i], gpu_ttm[i])) {
            passed = false;
            break;
        }
    }
    std::cout << "TMM check: " << (passed ? "PASS" : "FAIL")
              << " | dims=(" << cfg.dims[0] << "," << cfg.dims[1] << "," << cfg.dims[2] << ")"
              << " rank=" << cfg.decomp_rank
              << " nnz=" << cfg.nnz
              << "\n";

    if (!passed) {
        for (uint64_t i = 0; i < total; ++i) {
            if (!approx_equal(cpu_ttm[i], gpu_ttm[i])) {
                std::cout << std::setprecision(12)
                          << "Mismatch at index " << i
                          << " cpu=" << cpu_ttm[i]
                          << " gpu=" << gpu_ttm[i]
                          << " diff=" << static_cast<double>(cpu_ttm[i]) - static_cast<double>(gpu_ttm[i])
                          << "\n";
                break;
            }
        }
    }

    free(cpu_ttm);
    free(gpu_ttm);

    T* cuda_core = tucker_compute_core_3d_cuda<T, S>(blco, block_size);
    std::cout << "Core tensor computed (" << cfg.decomp_rank << "^3 values)\n";
    free(cuda_core);
}

int main(int argc, char** argv) {
    int block_size = 256;
    if (argc > 7) {
        block_size = std::atoi(argv[7]);
    }
    LaunchConfig cfg = parse_args(argc, argv);

    int bits = host_ceiling_log2(cfg.dims[0]) +
               host_ceiling_log2(cfg.dims[1]) +
               host_ceiling_log2(cfg.dims[2]);
    if (bits <= 64) {
        run_ttms<float, uint64_t>(cfg, block_size);
    } else {
        run_ttms<float, __uint128_t>(cfg, block_size);
    }
    return 0;
}
