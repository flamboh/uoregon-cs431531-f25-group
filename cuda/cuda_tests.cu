#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <cstring>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "tmm_nd.cuh"
#include "../tensor_storage/tensor_utils.h"

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

std::string dims_to_string(const std::vector<int>& dims) {
    std::ostringstream oss;
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i != 0) {
            oss << "x";
        }
        oss << dims[i];
    }
    return oss.str();
}

uint64_t product_u64(const std::vector<int>& dims) {
    uint64_t result = 1;
    for (int dim : dims) {
        result *= static_cast<uint64_t>(dim);
    }
    return result;
}

uint64_t power_u64(int base, int exp) {
    uint64_t result = 1;
    for (int i = 0; i < exp; ++i) {
        result *= static_cast<uint64_t>(base);
    }
    return result;
}

std::vector<uint64_t> build_row_major_strides(const std::vector<int>& dims) {
    const int rank_n = static_cast<int>(dims.size());
    std::vector<uint64_t> strides(rank_n, 1);
    for (int i = rank_n - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * static_cast<uint64_t>(dims[i + 1]);
    }
    return strides;
}

template <typename T>
std::vector<T> tmm_cpu_nd(int mode,
                          const std::vector<NNZ_Entry<T>>& entries,
                          const std::vector<int>& dims,
                          const T* fmat,
                          int rank) {
    if (mode < 1 || mode > static_cast<int>(dims.size())) {
        throw std::invalid_argument("CPU TMM mode out of bounds");
    }
    const int target_idx = mode - 1;
    const int contracted_dim = dims[target_idx];
    std::vector<int> output_dims = dims;
    output_dims[target_idx] = rank;
    const std::vector<uint64_t> output_strides = build_row_major_strides(output_dims);
    const uint64_t output_elems = product_u64(output_dims);
    std::vector<T> output(output_elems, static_cast<T>(0));

    for (const auto& entry : entries) {
        if (entry.indices.size() != dims.size()) {
            continue;
        }
        const int contracted_coord = entry.indices[target_idx];
        if (contracted_coord < 0 || contracted_coord >= contracted_dim) {
            continue;
        }
        std::vector<int> coords = entry.indices;
        for (int r = 0; r < rank; ++r) {
            const size_t f_idx = static_cast<size_t>(r) * contracted_dim + contracted_coord;
            const T f_val = fmat[f_idx];
            coords[target_idx] = r;
            uint64_t out_idx = 0;
            for (size_t i = 0; i < coords.size(); ++i) {
                out_idx += output_strides[i] * static_cast<uint64_t>(coords[i]);
            }
            output[out_idx] += entry.value * f_val;
        }
    }

    return output;
}

template <typename T>
std::vector<T> core_tensor_cpu_nd(const std::vector<NNZ_Entry<T>>& entries,
                                  const std::vector<int>& dims,
                                  const std::vector<T*>& row_major_fmats,
                                  int rank) {
    const int rank_n = static_cast<int>(dims.size());
    if (static_cast<int>(row_major_fmats.size()) != rank_n) {
        throw std::invalid_argument("Row-major factors size mismatch");
    }
    const uint64_t output_elems = power_u64(rank, rank_n);
    std::vector<T> output(output_elems, static_cast<T>(0));
    std::vector<uint64_t> rank_strides(rank_n, 1);
    for (int i = rank_n - 2; i >= 0; --i) {
        rank_strides[i] = rank_strides[i + 1] * static_cast<uint64_t>(rank);
    }

    for (const auto& entry : entries) {
        if (entry.indices.size() != dims.size()) {
            continue;
        }
        std::vector<int> rank_coords(rank_n, 0);
        bool done = false;
        while (!done) {
            T contrib = entry.value;
            for (int mode_idx = 0; mode_idx < rank_n; ++mode_idx) {
                const int coord = entry.indices[mode_idx];
                const T* factor = row_major_fmats[mode_idx];
                const size_t idx = static_cast<size_t>(coord) * rank + rank_coords[mode_idx];
                contrib *= factor[idx];
            }
            uint64_t out_idx = 0;
            for (int i = 0; i < rank_n; ++i) {
                out_idx += rank_strides[i] * static_cast<uint64_t>(rank_coords[i]);
            }
            output[out_idx] += contrib;

            for (int i = rank_n - 1; i >= 0; --i) {
                rank_coords[i]++;
                if (rank_coords[i] < rank) {
                    break;
                }
                rank_coords[i] = 0;
                if (i == 0) {
                    done = true;
                }
            }
        }
    }

    return output;
}

} // namespace

struct LaunchConfig {
    int nnz;
    int mode;
    int decomp_rank;
    int rank_n;
    vector<int> dims;
};

LaunchConfig parse_args(int argc, char** argv, int& block_size) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <nnz> <mode|0=all> <rank> <num_modes> <dim1> ... <dimN> [block_size]\n";
        std::exit(1);
    }

    LaunchConfig cfg;
    cfg.nnz = std::atoi(argv[1]);
    cfg.mode = std::atoi(argv[2]);
    cfg.decomp_rank = std::atoi(argv[3]);
    cfg.rank_n = std::atoi(argv[4]);
    if (cfg.rank_n < 2) {
        std::cerr << "Tensor rank must be at least 2.\n";
        std::exit(1);
    }
    const int expected_args = 5 + cfg.rank_n;
    if (argc < expected_args) {
        std::cerr << "Error: expected " << cfg.rank_n
                  << " dimension entries after num_modes.\n";
        std::exit(1);
    }
    cfg.dims.resize(cfg.rank_n);
    for (int i = 0; i < cfg.rank_n; ++i) {
        cfg.dims[i] = std::atoi(argv[5 + i]);
    }
    if (cfg.mode < 0 || cfg.mode > cfg.rank_n) {
        std::cerr << "Error: mode must be 0 (all) or in [1," << cfg.rank_n << "].\n";
        std::exit(1);
    }
    if (argc > expected_args) {
        block_size = std::atoi(argv[expected_args]);
    }
    return cfg;
}

template <typename T, typename S>
void run_ttms(const LaunchConfig& cfg, int block_size) {
    const double total_entries = static_cast<double>(product_u64(cfg.dims));
    const double density = static_cast<double>(cfg.nnz) / total_entries;
    auto mat_start = std::chrono::high_resolution_clock::now();
    vector<NNZ_Entry<T>> entries = generate_block_sparse_tensor_nd<T>(
        cfg.dims,
        static_cast<uint64_t>(cfg.nnz),
        0,
        100,
        std::max(1, cfg.dims[0] / 20),
        100);
    auto mat_end = std::chrono::high_resolution_clock::now();

    auto factor_start = std::chrono::high_resolution_clock::now();
    Blco_Tensor<T, S> blco(entries, cfg.dims, cfg.decomp_rank);
    auto factor_end = std::chrono::high_resolution_clock::now();
    const double mat_ms = std::chrono::duration<double, std::milli>(mat_end - mat_start).count();
    const double factor_ms = std::chrono::duration<double, std::milli>(factor_end - factor_start).count();
    std::cout << "Matricization time: " << mat_ms << " ms | Factor init time: "
              << factor_ms << " ms | density=" << density << "\n";
    vector<T*> blco_fmats = blco.get_fmats();
    std::vector<std::vector<T>> row_major_fmats(cfg.rank_n);
    for (int mode_idx = 0; mode_idx < cfg.rank_n; ++mode_idx) {
        const int dim = cfg.dims[mode_idx];
        row_major_fmats[mode_idx].resize(static_cast<size_t>(dim) * cfg.decomp_rank);
        T* src = blco_fmats[mode_idx];
        T* dst = row_major_fmats[mode_idx].data();
        for (int d = 0; d < dim; ++d) {
            for (int r = 0; r < cfg.decomp_rank; ++r) {
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
        modes_to_run.resize(cfg.rank_n);
        std::iota(modes_to_run.begin(), modes_to_run.end(), 1);
    } else {
        modes_to_run = {cfg.mode};
    }

    double total_cpu_tmm = 0.0;
    double total_gpu_tmm = 0.0;
    double total_upload_ms = 0.0;
    double total_kernel_ms = 0.0;
    double total_download_ms = 0.0;

    struct TmmSummary {
        int mode;
        double cpu_ms;
        double gpu_ms;
        double upload_ms;
        double kernel_ms;
        double download_ms;
        bool pass;
    };
    std::vector<TmmSummary> tmm_summaries;
    const std::string dims_str = dims_to_string(cfg.dims);

    for (int target_mode : modes_to_run) {
        std::cout << "\n=== Mode " << target_mode << " TMM ===\n";
        std::cout << "[Mode " << target_mode
                  << "] Warmup TMM launch to stabilize GPU state...\n";
        T* warmup_out = tmm_nd_cuda<T, S>(
            blco,
            target_mode,
            block_size,
            false,
            nullptr,
            nullptr,
            nullptr);
        free(warmup_out);

        auto cpu_tmm_start = std::chrono::high_resolution_clock::now();
        std::vector<T> cpu_ttm = tmm_cpu_nd<T>(
            target_mode,
            entries,
            cfg.dims,
            blco_fmats[target_mode - 1],
            cfg.decomp_rank);
        auto cpu_tmm_end = std::chrono::high_resolution_clock::now();
        const double cpu_tmm_ms =
            std::chrono::duration<double, std::milli>(cpu_tmm_end - cpu_tmm_start).count();
        std::cout << "[Mode " << target_mode << "] TMM CPU time: " << cpu_tmm_ms << " ms\n";

        auto gpu_tmm_start = std::chrono::high_resolution_clock::now();
        double upload_ms = 0.0;
        double kernel_ms = 0.0;
        double download_ms = 0.0;
        T* gpu_ttm = tmm_nd_cuda<T, S>(
            blco,
            target_mode,
            block_size,
            true,
            &upload_ms,
            &kernel_ms,
            &download_ms);
        auto gpu_tmm_end = std::chrono::high_resolution_clock::now();
        const double gpu_tmm_ms =
            std::chrono::duration<double, std::milli>(gpu_tmm_end - gpu_tmm_start).count();

        vector<int> new_dims = cfg.dims;
        new_dims[target_mode - 1] = cfg.decomp_rank;
        const uint64_t total_u64 = product_u64(new_dims);
        const size_t total = static_cast<size_t>(total_u64);
        std::vector<T> gpu_ttm_vec(total);
        std::copy(gpu_ttm, gpu_ttm + total, gpu_ttm_vec.begin());
        bool tmm_pass = true;
        for (size_t i = 0; i < total; ++i) {
            if (!approx_equal(cpu_ttm[i], gpu_ttm_vec[i])) {
                tmm_pass = false;
                break;
            }
        }
        std::cout << "[Mode " << target_mode << "] TMM check: " << (tmm_pass ? "PASS" : "FAIL")
                  << " | dims=" << dims_str
                  << " rank=" << cfg.decomp_rank
                  << " nnz=" << cfg.nnz << "\n";
        if (!tmm_pass) {
            int printed = 0;
            for (size_t i = 0; i < total && printed < 3; ++i) {
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
                  << gpu_tmm_ms << " ms (upload=" << upload_ms
                  << " ms, kernel=" << kernel_ms
                  << " ms, download=" << download_ms << " ms)\n";
        total_cpu_tmm += cpu_tmm_ms;
        total_gpu_tmm += gpu_tmm_ms;
        total_upload_ms += upload_ms;
        total_kernel_ms += kernel_ms;
        total_download_ms += download_ms;
        tmm_summaries.push_back({target_mode, cpu_tmm_ms, gpu_tmm_ms, upload_ms, kernel_ms, download_ms, tmm_pass});
        free(gpu_ttm);
    }

    if (tmm_summaries.size() > 1) {
        std::cout << "\nTMM mode summary:\n";
        for (const auto& summary : tmm_summaries) {
            std::cout << "  Mode " << summary.mode
                      << ": CPU " << summary.cpu_ms << " ms | GPU "
                      << summary.gpu_ms << " ms (upload " << summary.upload_ms
                      << " ms, kernel " << summary.kernel_ms
                      << " ms, download " << summary.download_ms << " ms) | "
                      << (summary.pass ? "PASS" : "FAIL") << "\n";
        }
    }

    if (!tmm_summaries.empty()) {
        const double inv_modes = 1.0 / static_cast<double>(tmm_summaries.size());
        const double avg_cpu_tmm = total_cpu_tmm * inv_modes;
        const double avg_gpu_tmm = total_gpu_tmm * inv_modes;
        const double avg_upload = total_upload_ms * inv_modes;
        const double avg_kernel = total_kernel_ms * inv_modes;
        const double avg_download = total_download_ms * inv_modes;
        std::cout << "\nAverage TMM timings (" << tmm_summaries.size() << " mode(s)):\n";
        std::cout << "  CPU: " << avg_cpu_tmm << " ms\n";
        std::cout << "  GPU: " << avg_gpu_tmm << " ms (upload " << avg_upload
                  << " ms, kernel " << avg_kernel
                  << " ms, download " << avg_download << " ms)\n";
    }

    std::vector<T*> cpu_fmats_row_ptrs(cfg.rank_n);
    for (int mode_idx = 0; mode_idx < cfg.rank_n; ++mode_idx) {
        cpu_fmats_row_ptrs[mode_idx] = row_major_fmats[mode_idx].data();
    }
    auto cpu_core_start = std::chrono::high_resolution_clock::now();
    std::vector<T> cpu_core =
        core_tensor_cpu_nd<T>(entries, cfg.dims, cpu_fmats_row_ptrs, cfg.decomp_rank);
    auto cpu_core_end = std::chrono::high_resolution_clock::now();
    const double cpu_core_ms =
        std::chrono::duration<double, std::milli>(cpu_core_end - cpu_core_start).count();
    std::cout << "Core CPU time: " << cpu_core_ms << " ms\n";
    auto gpu_core_start = std::chrono::high_resolution_clock::now();
    double core_upload_ms = 0.0;
    double core_kernel_ms = 0.0;
    double core_download_ms = 0.0;
    T* cuda_core = tucker_compute_core_nd_cuda<T, S>(
        blco,
        block_size,
        true,
        &core_upload_ms,
        &core_kernel_ms,
        &core_download_ms);
    auto gpu_core_end = std::chrono::high_resolution_clock::now();
    const double gpu_core_ms =
        std::chrono::duration<double, std::milli>(gpu_core_end - gpu_core_start).count();
    const uint64_t core_size_u64 = power_u64(cfg.decomp_rank, cfg.rank_n);
    const size_t core_size = static_cast<size_t>(core_size_u64);
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
              << " (" << cfg.decomp_rank << "^" << cfg.rank_n << " values)\n";
    free(cuda_core);

    std::cout << "\nCore timings:\n";
    std::cout << "  CPU: " << cpu_core_ms << " ms\n";
    std::cout << "  GPU: " << gpu_core_ms << " ms (upload " << core_upload_ms
              << " ms, kernel " << core_kernel_ms
              << " ms, download " << core_download_ms << " ms)\n";

    const double total_cpu_time = total_cpu_tmm + cpu_core_ms;
    const double total_gpu_time = total_gpu_tmm + gpu_core_ms;

    std::cout << "\nTotal CPU compute time (TMM + core): "
              << total_cpu_time << " ms\n";
    std::cout << "Total GPU compute time (TMM + core): "
              << total_gpu_time << " ms\n";
}

int main(int argc, char** argv) {
    int block_size = 256;
    LaunchConfig cfg = parse_args(argc, argv, block_size);

    int bits = 0;
    for (int dim : cfg.dims) {
        bits += host_ceiling_log2(dim);
    }
    if (bits <= 64) {
        run_ttms<int, uint64_t>(cfg, block_size);
    } else {
        run_ttms<int, __uint128_t>(cfg, block_size);
    }
    return 0;
}
