#include "gpu_kernel.h"
#include "tmm_utils.h"
#include "../tensor_storage/blco.h"
#include "../tensor_storage/tensor_utils.h"

template<typename T, typename S>
void test_sparse_tmm(int nnz, int mode, std::vector<int> dims, int decomp_rank = -1)
{
    int rank = dims.size();
    double total_entries = static_cast<double>(dims[0]);
    for(int i = 1; i < rank; i++) total_entries *= dims[i];
    double freq = static_cast<double>(nnz) / total_entries;

    int min_dim = *(std::min_element(dims.begin(), dims.end()));
    int block_size = 0.05 * min_dim;
    int max_blocks = 10 * (nnz / pow(block_size,rank));
    std::vector<NNZ_Entry<T>> test_vec = generate_block_sparse_tensor_nd<T>(dims,freq,0,100,block_size,max_blocks);

    int construction_rank;
    if(decomp_rank == -1) construction_rank = 10;
    else construction_rank = decomp_rank;
    Blco_Tensor<T,S> blco(test_vec,dims,construction_rank);

    std::vector<T*> fmats = blco.get_fmats();

    T* gpu_output = tmm_3D<T>(blco,mode);
    T* cpu_output = tensor_matrix_mul<T>(test_vec, dims, fmats[mode - 1], mode, construction_rank);

    std::vector<int> new_dims = dims; 
    new_dims[mode - 1] = decomp_rank; 
    uint64_t total_size = 1;
    for (int dim : new_dims) {
        total_size *= dim;
    }
    bool passed = compare_arrays<T>(gpu_output, cpu_output, total_size);

    if(passed) std::cout<<"tests passed!\n";
    else std::cout<<"tests failed!\n";

    free(gpu_output);
    free(cpu_output);
}


template<typename T, typename S>
void test_sparse_tmm_file(const std::string &filename, int nnz, int mode, std::vector<int> dims, int decomp_rank = -1)
{
    int rank = dims.size();
    double total_entries = static_cast<double>(dims[0]);
    for(int i = 1; i < rank; i++) total_entries *= dims[i];

    std::vector<NNZ_Entry<T>> test_vec = read_tensor_file<T>(filename,rank);

    int construction_rank;
    if(decomp_rank == -1) construction_rank = 10;
    else construction_rank = decomp_rank;
    Blco_Tensor<T,S> blco(test_vec,dims,construction_rank);

    std::vector<T*> fmats = blco.get_fmats();

    T* gpu_output;
    if(rank == 3) gpu_output = tmm_3D<T>(blco,mode);
    else if(rank == 4) gpu_output = tmm_4D<T>(blco,mode);
    else if(rank == 5) gpu_output = tmm_5D<T>(blco,mode);
    T* cpu_output = tensor_matrix_mul<T>(test_vec, dims, fmats[mode - 1], mode, construction_rank);

    std::vector<int> new_dims = dims;
    new_dims[mode - 1] = decomp_rank;

    // --- EFFECTIVE 5D TENSOR PRINTING ---

    // D0 is the slowest changing index (Slice of Slice of Slices index)
    const int D0 = new_dims[0];
    // D1 (Slice of Slices index)
    const int D1 = new_dims[1];
    // D2 is the next slice index (Slice index)
    const int D2 = new_dims[2];
    // D3 is the row index
    const int D3 = new_dims[3];
    // D4 is the column index
    const int D4 = new_dims[4]; // Corrected assignment for D4

    auto print_tensor = [&](const T* tensor_data, const char* name) {
        std::cout << "\n\n=== " << name << " (Dimensions: " 
                  << D0 << "x" << D1 << "x" << D2 << "x" << D3 << "x" << D4 << ") ===" << std::endl;

        // Loop 1 (Outer Loop - D0)
        for (int i = 0; i < D0; ++i) {
            std::cout << "\n--- Outer Slice i (D0) = " << i << " ---" << std::endl;

            // Loop 2 (Next Slice - D1)
            for (int j = 0; j < D1; ++j) {
                std::cout << "--- Sub-Slice j (D1) = " << j << " ---" << std::endl;

                // Loop 3 (Innermost Slice - D2)
                for (int k = 0; k < D2; ++k) {
                    std::cout << "--- Matrix Slice k (D2) = " << k << " ---" << std::endl;
                    
                    // Loop 4 (Rows - D3)
                    for (int l = 0; l < D3; ++l) {
                        // Loop 5 (Columns - D4)
                        for (int m = 0; m < D4; ++m) {
                            
                            // Linear Index Calculation (Row-Major Ordering):
                            // Index = i*(D1*D2*D3*D4) + j*(D2*D3*D4) + k*(D3*D4) + l*(D4) + m
                            uint64_t index = (uint64_t)i * D1 * D2 * D3 * D4 + 
                                             (uint64_t)j * D2 * D3 * D4 + 
                                             (uint64_t)k * D3 * D4 + 
                                             (uint64_t)l * D4 +
                                             (uint64_t)m;
                            
                            // Print the element, aligned for readability
                            std::cout << std::fixed << std::setprecision(4) 
                                      << std::setw(10) << tensor_data[index] << " ";
                        }
                        std::cout << "\n"; // Newline after each row (D3)
                    }
                }
            }
        }
        std::cout << "\n";
    };

    // Print both tensors using the lambda function
    print_tensor(gpu_output, "GPU Tensor Output");
    print_tensor(cpu_output, "CPU Tensor Output");

    // --- END OF TENSOR PRINTING ---

    uint64_t total_size = 1;
    for (int dim : new_dims) {
        total_size *= dim;
    }
    bool passed = compare_arrays<T>(gpu_output, cpu_output, total_size);

    if(passed) std::cout<<"tests passed\n";
    else std::cout<<"tests failed";

    free(gpu_output); 
    free(cpu_output);
}


int main(int argc, char* argv[]) {
    if (argc < 9 || argc > 12){
        std::cerr << "Usage: " << argv[0] 
                  << "<Filename> <NNZ> <Mode> <Decomposition rank> <Type> (three to five different dimensions)\n";
        return 1;
    }
    else if(std::string(argv[1]) == "--None"){
        int nnz = std::stoi(argv[2]);
        int rank = argc - 4;
        int mode = std::stoi(argv[3]);
        int decomp_rank;
        if(std::string(argv[4]) == "default") decomp_rank = 10;
        else decomp_rank = std::stoi(argv[4]);
        std::string type = std::string(argv[5]);
        std::vector<int> dimensions;
        for(int i = 6; i < argc; i++){
            dimensions.push_back(std::stoi(argv[i]));
        }

        int bits_needed = 0;
        for(int i = 0; i < rank; i++){
            bits_needed += ceiling_log2(dimensions[i]);
        }

        if(bits_needed <= 64){
            if(type == "int") test_sparse_tmm<int,uint64_t>(nnz, mode, dimensions, decomp_rank);
            else if(type == "float") test_sparse_tmm<float,uint64_t>(nnz, mode, dimensions, decomp_rank);
            else if(type == "long int") test_sparse_tmm<long int,uint64_t>(nnz, mode, dimensions, decomp_rank);
            else{ 
                std::cerr << "Unsupported type. The supported types are int, \
                float, long int, and long int\n";
                return 1;
            }
        }
        else{
            if(type == "int") test_sparse_tmm<int,__uint128_t>(nnz, mode, dimensions, decomp_rank);
            else if(type == "float") test_sparse_tmm<float,__uint128_t>(nnz, mode, dimensions, decomp_rank);
            else if(type == "long int") test_sparse_tmm<long int,__uint128_t>(nnz, mode, dimensions, decomp_rank);
            else{ 
                std::cerr << "Unsupported type. The supported types are int, \
                float, long int, and long int\n";
                return 1;
            }
        }
    }
    else{
        std::string filename = std::string(argv[1]);
        int nnz = std::stoi(argv[2]);
        int rank = argc - 4;
        int mode = std::stoi(argv[3]);
        int decomp_rank;
        if(std::string(argv[4]) == "default") decomp_rank = 10;
        else decomp_rank = std::stoi(argv[4]);
        std::string type = std::string(argv[5]);
        std::vector<int> dimensions;
        for(int i = 6; i < argc; i++){
            dimensions.push_back(std::stoi(argv[i]));
        }

        int bits_needed = 0;
        for(int i = 0; i < rank; i++){
            bits_needed += ceiling_log2(dimensions[i]);
        }

        if(bits_needed <= 64){
            if(type == "int") test_sparse_tmm_file<int,uint64_t>(filename, nnz, mode, dimensions, decomp_rank);
            else if(type == "float") test_sparse_tmm_file<float,uint64_t>(filename, nnz, mode, dimensions, decomp_rank);
            else if(type == "long int") test_sparse_tmm_file<long int,uint64_t>(filename, nnz, mode, dimensions, decomp_rank);
            else{ 
                std::cerr << "Unsupported type. The supported types are int, \
                float, long int, and long int\n";
                return 1;
            }
        }
        else{
            if(type == "int") test_sparse_tmm_file<int,__uint128_t>(filename, nnz, mode, dimensions, decomp_rank);
            else if(type == "float") test_sparse_tmm_file<float,__uint128_t>(filename, nnz, mode, dimensions, decomp_rank);
            else if(type == "long int") test_sparse_tmm_file<long int,__uint128_t>(filename, nnz, mode, dimensions, decomp_rank );
            else{ 
                std::cerr << "Unsupported type. The supported types are int, \
                float, long int, and long int\n";
                return 1;
            }
        }
    }

    return 0;
}