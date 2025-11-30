#include "gpu_kernel.h"
#include "tmm_utils.h"
#include "../tensor_storage/blco.h"
#include "../tensor_storage/tensor_utils.h"

template<typename T, typename S>
void test_sparse_core(int nnz, std::vector<int> dims, int decomp_rank = -1)
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

    T* gpu_output;
    if(rank == 3) gpu_output = tucker_compute_core_3D<T>(blco);
    // else if(rank == 4) gpu_output = tucker_compute_core_4D<T>(blco,mod);
    // else if(rank == 5) gpu_output = tucker_compute_core_5D<T>(blco,mode);

    // --- END OF TENSOR PRINTING ---

    free(gpu_output);
}


template<typename T, typename S>
void test_sparse_core_file(const std::string &filename, int nnz, std::vector<int> dims, int decomp_rank = -1)
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
    if(rank == 3) gpu_output = tucker_compute_core_3D<T>(blco);
    
    // D0 is the slowest changing index (slice index)
    const int D0 = construction_rank;
    // D1 is the row index
    const int D1 = construction_rank;
    // D2 is the column index
    const int D2 = construction_rank;

    auto print_tensor = [&](const T* tensor_data, const char* name) {
        std::cout << "\n\n=== " << name << " (Dimensions: " 
                  << D0 << "x" << D1 << "x" << D2 << ") ===" << std::endl;

        // Loop through the D0 dimension (slices)
        for (int i = 0; i < D0; ++i) {
            std::cout << "\n--- Slice i = " << i << " ---" << std::endl;
            // Loop through the D1 dimension (rows)
            for (int j = 0; j < D1; ++j) {
                // Loop through the D2 dimension (columns)
                for (int k = 0; k < D2; ++k) {
                    // Linear Index Calculation (Row-Major Ordering assumed):
                    // Index = i * (D1 * D2) + j * D2 + k
                    uint64_t index = (uint64_t)i * D1 * D2 + (uint64_t)j * D2 + (uint64_t)k;
                    
                    // Print the element, aligned for readability
                    std::cout << std::fixed << std::setprecision(4) 
                              << std::setw(10) << tensor_data[index] << " ";
                }
                std::cout << "\n"; // Newline after each row (D1)
            }
        }
        std::cout << "\n";
    };

    print_tensor(gpu_output, "GPU Tensor Output");
    
    free(gpu_output); 
}


int main(int argc, char* argv[]) {
    if (argc < 8 || argc > 11){
        std::cerr << "Usage: " << argv[0] 
                  << "<Filename> <NNZ> <Decomposition rank> <Type> (three to five different dimensions)\n";
        return 1;
    }
    else if(std::string(argv[1]) == "--None"){
        int nnz = std::stoi(argv[2]);
        int rank = argc - 4;
        int decomp_rank;
        if(std::string(argv[3]) == "default") decomp_rank = 10;
        else decomp_rank = std::stoi(argv[3]);
        std::string type = std::string(argv[4]);
        std::vector<int> dimensions;
        for(int i = 5; i < argc; i++){
            dimensions.push_back(std::stoi(argv[i]));
        }

        int bits_needed = 0;
        for(int i = 0; i < rank; i++){
            bits_needed += ceiling_log2(dimensions[i]);
        }

        if(bits_needed <= 64){
            if(type == "int") test_sparse_core<int,uint64_t>(nnz, dimensions, decomp_rank);
            else if(type == "float") test_sparse_core<float,uint64_t>(nnz, dimensions, decomp_rank);
            else if(type == "long int") test_sparse_core<long int,uint64_t>(nnz, dimensions, decomp_rank);
            else{ 
                std::cerr << "Unsupported type. The supported types are int, \
                float, long int, and long int\n";
                return 1;
            }
        }
        else{
            if(type == "int") test_sparse_core<int,__uint128_t>(nnz, dimensions, decomp_rank);
            else if(type == "float") test_sparse_core<float,__uint128_t>(nnz, dimensions, decomp_rank);
            else if(type == "long int") test_sparse_core<long int,__uint128_t>(nnz, dimensions, decomp_rank);
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
        int decomp_rank;
        if(std::string(argv[3]) == "default") decomp_rank = 10;
        else decomp_rank = std::stoi(argv[3]);
        std::string type = std::string(argv[4]);
        std::vector<int> dimensions;
        for(int i = 5; i < argc; i++){
            dimensions.push_back(std::stoi(argv[i]));
        }

        int bits_needed = 0;
        for(int i = 0; i < rank; i++){
            bits_needed += ceiling_log2(dimensions[i]);
        }

        if(bits_needed <= 64){
            if(type == "int") test_sparse_core_file<int,uint64_t>(filename, nnz, dimensions, decomp_rank);
            else if(type == "float") test_sparse_core_file<float,uint64_t>(filename, nnz, dimensions, decomp_rank);
            else if(type == "long int") test_sparse_core_file<long int,uint64_t>(filename, nnz, dimensions, decomp_rank);
            else{ 
                std::cerr << "Unsupported type. The supported types are int, \
                float, long int, and long int\n";
                return 1;
            }
        }
        else{
            if(type == "int") test_sparse_core_file<int,__uint128_t>(filename, nnz, dimensions, decomp_rank);
            else if(type == "float") test_sparse_core_file<float,__uint128_t>(filename, nnz, dimensions, decomp_rank);
            else if(type == "long int") test_sparse_core_file<long int,__uint128_t>(filename, nnz, dimensions, decomp_rank );
            else{ 
                std::cerr << "Unsupported type. The supported types are int, \
                float, long int, and long int\n";
                return 1;
            }
        }
    }

    return 0;
}