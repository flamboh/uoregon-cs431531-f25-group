#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cstring>

#include "tmm_3d.cuh"
#include "../tns_mat_mul/cpu_tests.h"

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
                  << " <nnz> <mode|0=all> <rank> <dim1> <dim2> <dim3> [block_size]\n";
        std::exit(1);
    }

    LaunchConfig cfg;
    cfg.nnz = std::atoi(argv[1]);
    cfg.mode = std::atoi(argv[2]);
    if (cfg.mode < 0 || cfg.mode > 3) {
        std::cerr << "Error: mode must be 0 (all) or in [1,3].\n";
        std::exit(1);
    }
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
    vector<T*> blco_fmats = blco.get_fmats();
    std::vector<std::vector<T>> row_major_fmats(3);
    for (int mode_idx = 0; mode_idx < 3; ++mode_idx) {
        const int dim = cfg.dims[mode_idx];
        row_major_fmats[mode_idx].resize(static_cast<size_t>(dim) * cfg.decomp_rank);
        T* src = blco_fmats[mode_idx];
        T* dst = row_major_fmats[mode_idx].data();
        for (int r = 0; r < cfg.decomp_rank; ++r) {
            for (int d = 0; d < dim; ++d) {
                dst[static_cast<size_t>(d) * cfg.decomp_rank + r] =
                    src[static_cast<size_t>(r) * dim + d];
            }
        }
    }

    auto approx_equal = [](T lhs, T rhs) {
        if constexpr (std::is_floating_point_v<T>) {
            return std::abs(static_cast<double>(lhs) - static_cast<double>(rhs)) <= 0.5;
        } else {
            return lhs == rhs;
        }
    };

    std::vector<int> modes_to_run;
    if (cfg.mode == 0) {
        modes_to_run = {1, 2, 3};
    } else {
        modes_to_run = {cfg.mode};
    }

    double total_cpu_time = 0.0;
    double total_gpu_time = 0.0;

    struct TmmSummary {
        int mode;
        double cpu_ms;
        double gpu_ms;
        bool pass;
    };
    std::vector<TmmSummary> tmm_summaries;

    for (int target_mode : modes_to_run) {
        std::cout << "\n=== Mode " << target_mode << " TMM ===\n";
        std::cout << "[Mode " << target_mode
                  << "] Warmup TMM launch to stabilize GPU state...\n";
        T* warmup_out = tmm_3d_cuda<T, S>(blco, target_mode, block_size, false);
        free(warmup_out);

        auto cpu_tmm_start = std::chrono::high_resolution_clock::now();
        std::vector<T> cpu_ttm = tmm_3D_cpu<T, S>(
            target_mode, entries, cfg.dims, blco_fmats[target_mode - 1], cfg.decomp_rank);
        auto cpu_tmm_end = std::chrono::high_resolution_clock::now();
        const double cpu_tmm_ms =
            std::chrono::duration<double, std::milli>(cpu_tmm_end - cpu_tmm_start).count();
        std::cout << "[Mode " << target_mode << "] TMM CPU time: " << cpu_tmm_ms << " ms\n";

        auto gpu_tmm_start = std::chrono::high_resolution_clock::now();
        T* gpu_ttm = tmm_3d_cuda<T, S>(blco, target_mode, block_size);
        auto gpu_tmm_end = std::chrono::high_resolution_clock::now();
        const double gpu_tmm_ms =
            std::chrono::duration<double, std::milli>(gpu_tmm_end - gpu_tmm_start).count();

        vector<int> new_dims = cfg.dims;
        new_dims[target_mode - 1] = cfg.decomp_rank;
        uint64_t total = static_cast<uint64_t>(new_dims[0]) * new_dims[1] * new_dims[2];
        std::vector<T> gpu_ttm_vec(total);
        std::copy(gpu_ttm, gpu_ttm + total, gpu_ttm_vec.begin());
        bool tmm_pass = true;
        for (uint64_t i = 0; i < total; ++i) {
            if (!approx_equal(cpu_ttm[i], gpu_ttm_vec[i])) {
                tmm_pass = false;
                break;
            }
        }
        std::cout << "[Mode " << target_mode << "] TMM check: " << (tmm_pass ? "PASS" : "FAIL")
                  << " | dims=(" << cfg.dims[0] << "," << cfg.dims[1] << "," << cfg.dims[2] << ")"
                  << " rank=" << cfg.decomp_rank
                  << " nnz=" << cfg.nnz << "\n";
        if (!tmm_pass) {
            int printed = 0;
            for (uint64_t i = 0; i < total && printed < 3; ++i) {
                if (!approx_equal(cpu_ttm[i], gpu_ttm_vec[i])) {
                    std::cout << std::setprecision(12)
                              << "Mismatch at index " << i
                              << " cpu=" << cpu_ttm[i]
                              << " gpu=" << gpu_ttm_vec[i]
                              << " diff=" << static_cast<double>(cpu_ttm[i]) - static_cast<double>(gpu_ttm_vec[i])
                              << "\n";
                    ++printed;
                }
            }
        }
        std::cout << "[Mode " << target_mode << "] TMM GPU total wall time: "
                  << gpu_tmm_ms << " ms\n";
        total_cpu_time += cpu_tmm_ms;
        total_gpu_time += gpu_tmm_ms;
        tmm_summaries.push_back({target_mode, cpu_tmm_ms, gpu_tmm_ms, tmm_pass});
        free(gpu_ttm);
    }

    if (tmm_summaries.size() > 1) {
        std::cout << "\nTMM mode summary:\n";
        for (const auto& summary : tmm_summaries) {
            std::cout << "  Mode " << summary.mode
                      << ": CPU " << summary.cpu_ms << " ms | GPU "
                      << summary.gpu_ms << " ms | "
                      << (summary.pass ? "PASS" : "FAIL") << "\n";
        }
    }

    std::vector<T*> cpu_fmats_row_ptrs(3);
    for (int mode_idx = 0; mode_idx < 3; ++mode_idx) {
        cpu_fmats_row_ptrs[mode_idx] = row_major_fmats[mode_idx].data();
    }
    auto cpu_core_start = std::chrono::high_resolution_clock::now();
    std::vector<T> cpu_core = core_tensor_3D_cpu<T, S>(entries, cfg.dims, cpu_fmats_row_ptrs, cfg.decomp_rank);
    auto cpu_core_end = std::chrono::high_resolution_clock::now();
    const double cpu_core_ms =
        std::chrono::duration<double, std::milli>(cpu_core_end - cpu_core_start).count();
    std::cout << "Core CPU time: " << cpu_core_ms << " ms\n";
    auto gpu_core_start = std::chrono::high_resolution_clock::now();
    T* cuda_core = tucker_compute_core_3d_cuda<T, S>(blco, block_size);
    auto gpu_core_end = std::chrono::high_resolution_clock::now();
    const double gpu_core_ms =
        std::chrono::duration<double, std::milli>(gpu_core_end - gpu_core_start).count();
    size_t core_size = static_cast<size_t>(cfg.decomp_rank) * cfg.decomp_rank * cfg.decomp_rank;
    std::vector<T> gpu_core_vec(core_size);
    std::copy(cuda_core, cuda_core + core_size, gpu_core_vec.begin());
    bool core_pass = true;
    int printed_core = 0;
    for (size_t i = 0; i < core_size; ++i) {
        if (!approx_equal(cpu_core[i], gpu_core_vec[i])) {
            core_pass = false;
            if (printed_core < 3) {
                std::cout << std::setprecision(12)
                          << "Core mismatch at index " << i
                          << " cpu=" << cpu_core[i]
                          << " gpu=" << gpu_core_vec[i]
                          << " diff=" << static_cast<double>(cpu_core[i]) - static_cast<double>(gpu_core_vec[i])
                          << "\n";
                ++printed_core;
            }
        }
    }
    std::cout << "Core tensor check: " << (core_pass ? "PASS" : "FAIL")
              << " (" << cfg.decomp_rank << "^3 values)\n";
    free(cuda_core);

    total_cpu_time += cpu_core_ms;
    total_gpu_time += gpu_core_ms;

    std::cout << "\nTotal CPU compute time (TMM + core): "
              << total_cpu_time << " ms\n";
    std::cout << "Total GPU compute time (TMM + core): "
              << total_gpu_time << " ms\n";
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
        run_ttms<int, uint64_t>(cfg, block_size);
    } else {
        run_ttms<int, __uint128_t>(cfg, block_size);
    }
    return 0;
}
