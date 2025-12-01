#include "tmm_utils.h"
#include "3D_kernels.h"
#include "4D_kernels.h"
#include "5D_kernels.h"
#include "../tensor_storage/blco.h"
#include "../tensor_storage/tensor_utils.h"

template<typename T, typename S>
void test_3D_kernels(std::vector<int> dims, int nnz, int block_size)
{
    int rank = dims.size();
    double total_entries = static_cast<double>(dims[0]);
    for(int i = 1; i < rank; i++) total_entries *= dims[i];
    double freq = static_cast<double>(nnz) / total_entries;

    int min_dim = *(std::min_element(dims.begin(), dims.end()));
    int block_size = 0.05 * min_dim;
    int max_blocks = 10 * (nnz / pow(block_size,rank));
    std::vector<NNZ_Entry<T>> test_vec = generate_block_sparse_tensor_nd<T>(dims,freq,0,100,block_size,max_blocks);

    int construction_rank = 10;
    Blco_Tensor<T,S> blco(test_vec,dims,construction_rank);

    std::cout << "Testing operations on " << dims[0] << " x " << dims[1] << " x " << dims[2] << " tensor with " << nnz << " non zeros\n";

    std::cout << "Testing mode 1 tensor matrix multiplication\n";
    auto start = std::chrono::high_resolution_clock::now();
    T* tmm_output_1 = tmm_3D<T>(blco,1,block_size);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Elapsed time for mode 1 TMM: " << duration << " ms\n\n";

    std::cout << "Testing mode 2 tensor matrix multiplication\n";
    auto start = std::chrono::high_resolution_clock::now();
    T* tmm_output_2 = tmm_3D<T>(blco,2,block_size);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Elapsed time for mode 2 TMM: " << duration << " ms\n\n";

    std::cout << "Testing mode 3 tensor matrix multiplication\n";
    auto start = std::chrono::high_resolution_clock::now();
    T* tmm_output_3 = tmm_3D<T>(blco,3,block_size);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Elapsed time for mode 3 TMM: " << duration << " ms\n\n";

    std::cout << "Testing core generation\n";
    auto start = std::chrono::high_resolution_clock::now();
    T* core_output = tucker_compute_core_3D<T>(blco,block_size);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Elapsed time for core generation: " << duration << " ms\n\n";

    std::cout << "Testing mode 1 contraction\n";
    auto start = std::chrono::high_resolution_clock::now();
    T* contraction_1 = tucker_compute_contraction_3D<T>(blco,block_size,1);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Elapsed time for mode 1 contraction: " << duration << " ms\n\n";

    std::cout << "Testing mode 2 contraction\n";
    auto start = std::chrono::high_resolution_clock::now();
    T* contraction_2 = tucker_compute_contraction_3D<T>(blco,block_size,2);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Elapsed time for mode 2 contraction: " << duration << " ms\n\n";

    std::cout << "Testing mode 3 contraction\n";
    auto start = std::chrono::high_resolution_clock::now();
    T* contraction_3 = tucker_compute_contraction_3D<T>(blco,block_size,3);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Elapsed time for mode 3 contraction: " << duration << " ms\n\n";

    free(tmm_output_1);
    free(tmm_output_2);
    free(tmm_output_3);
    free(core_output);
    free(contraction_1);
    free(contraction_2);
    free(contraction_3);
}

template<typename T, typename S>
void test_4D_kernels(std::vector<int> dims, int nnz, int block_size)
{
    int rank = dims.size();
    double total_entries = static_cast<double>(dims[0]);
    for(int i = 1; i < rank; i++) total_entries *= dims[i];
    double freq = static_cast<double>(nnz) / total_entries;

    int min_dim = *(std::min_element(dims.begin(), dims.end()));
    int block_size = 0.05 * min_dim;
    int max_blocks = 10 * (nnz / pow(block_size,rank));
    std::vector<NNZ_Entry<T>> test_vec = generate_block_sparse_tensor_nd<T>(dims,freq,0,100,block_size,max_blocks);

    int construction_rank = 10;
    Blco_Tensor<T,S> blco(test_vec,dims,construction_rank);

    std::cout << "Testing operations on " << dims[0] << " x " << dims[1] << " x " << dims[2] << 
    " x " << dims[3] << " tensor with " << nnz << " non zeros\n";

    std::cout << "Testing mode 1 tensor matrix multiplication\n";
    auto start = std::chrono::high_resolution_clock::now();
    T* tmm_output_1 = tmm_4D<T>(blco,1,block_size);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Elapsed time for mode 1 TMM: " << duration << " ms\n\n";

    std::cout << "Testing mode 2 tensor matrix multiplication\n";
    auto start = std::chrono::high_resolution_clock::now();
    T* tmm_output_2 = tmm_4D<T>(blco,2,block_size);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Elapsed time for mode 2 TMM: " << duration << " ms\n\n";

    std::cout << "Testing mode 3 tensor matrix multiplication\n";
    auto start = std::chrono::high_resolution_clock::now();
    T* tmm_output_3 = tmm_4D<T>(blco,3,block_size);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Elapsed time for mode 3 TMM: " << duration << " ms\n\n";

    std::cout << "Testing mode 4 tensor matrix multiplication\n";
    auto start = std::chrono::high_resolution_clock::now();
    T* tmm_output_4 = tmm_4D<T>(blco,4,block_size);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Elapsed time for mode 4 TMM: " << duration << " ms\n\n";

    std::cout << "Testing core generation\n";
    auto start = std::chrono::high_resolution_clock::now();
    T* core_output = tucker_compute_core_4D<T>(blco,block_size);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Elapsed time for core generation: " << duration << " ms\n\n";

    std::cout << "Testing mode 1 contraction\n";
    auto start = std::chrono::high_resolution_clock::now();
    T* contraction_1 = tucker_compute_contraction_4D<T>(blco,block_size,1);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Elapsed time for mode 1 contraction: " << duration << " ms\n\n";

    std::cout << "Testing mode 2 contraction\n";
    auto start = std::chrono::high_resolution_clock::now();
    T* contraction_2 = tucker_compute_contraction_4D<T>(blco,block_size,2);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Elapsed time for mode 2 contraction: " << duration << " ms\n\n";

    std::cout << "Testing mode 3 contraction\n";
    auto start = std::chrono::high_resolution_clock::now();
    T* contraction_3 = tucker_compute_contraction_4D<T>(blco,block_size,3);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Elapsed time for mode 3 contraction: " << duration << " ms\n\n";

    std::cout << "Testing mode 4 contraction\n";
    auto start = std::chrono::high_resolution_clock::now();
    T* contraction_4 = tucker_compute_contraction_4D<T>(blco,block_size,4);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Elapsed time for mode 4 contraction: " << duration << " ms\n\n";

    free(tmm_output_1);
    free(tmm_output_2);
    free(tmm_output_3);
    free(tmm_output_4);
    free(core_output);
    free(contraction_1);
    free(contraction_2);
    free(contraction_3);
    free(contraction_4);
}

template<typename T, typename S>
void test_5D_kernels(std::vector<int> dims, int nnz, int block_size)
{
    int rank = dims.size();
    double total_entries = static_cast<double>(dims[0]);
    for(int i = 1; i < rank; i++) total_entries *= dims[i];
    double freq = static_cast<double>(nnz) / total_entries;

    int min_dim = *(std::min_element(dims.begin(), dims.end()));
    int block_size = 0.05 * min_dim;
    int max_blocks = 10 * (nnz / pow(block_size,rank));
    std::vector<NNZ_Entry<T>> test_vec = generate_block_sparse_tensor_nd<T>(dims,freq,0,100,block_size,max_blocks);

    int construction_rank = 10;
    Blco_Tensor<T,S> blco(test_vec,dims,construction_rank);

    std::cout << "Testing operations on " << dims[0] << " x " << dims[1] << " x " << dims[2] << 
    " x " << dims[3] << " x " << dims[4] << " tensor with " << nnz << " non zeros\n";

    std::cout << "Testing mode 1 tensor matrix multiplication\n";
    auto start = std::chrono::high_resolution_clock::now();
    T* tmm_output_1 = tmm_5D<T>(blco,1,block_size);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Elapsed time for mode 1 TMM: " << duration << " ms\n\n";

    std::cout << "Testing mode 2 tensor matrix multiplication\n";
    auto start = std::chrono::high_resolution_clock::now();
    T* tmm_output_2 = tmm_5D<T>(blco,2,block_size);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Elapsed time for mode 2 TMM: " << duration << " ms\n\n";

    std::cout << "Testing mode 3 tensor matrix multiplication\n";
    auto start = std::chrono::high_resolution_clock::now();
    T* tmm_output_3 = tmm_5D<T>(blco,3,block_size);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Elapsed time for mode 3 TMM: " << duration << " ms\n\n";

    std::cout << "Testing mode 4 tensor matrix multiplication\n";
    auto start = std::chrono::high_resolution_clock::now();
    T* tmm_output_4 = tmm_5D<T>(blco,4,block_size);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Elapsed time for mode 4 TMM: " << duration << " ms\n\n";

    std::cout << "Testing mode 5 tensor matrix multiplication\n";
    auto start = std::chrono::high_resolution_clock::now();
    T* tmm_output_5 = tmm_5D<T>(blco,5,block_size);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Elapsed time for mode 5 TMM: " << duration << " ms\n\n";

    std::cout << "Testing core generation\n";
    auto start = std::chrono::high_resolution_clock::now();
    T* core_output = tucker_compute_core_4D<T>(blco,block_size);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Elapsed time for core generation: " << duration << " ms\n\n";

    std::cout << "Testing mode 1 contraction\n";
    auto start = std::chrono::high_resolution_clock::now();
    T* contraction_1 = tucker_compute_contraction_5D<T>(blco,block_size,1);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Elapsed time for mode 1 contraction: " << duration << " ms\n\n";

    std::cout << "Testing mode 2 contraction\n";
    auto start = std::chrono::high_resolution_clock::now();
    T* contraction_2 = tucker_compute_contraction_5D<T>(blco,block_size,2);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Elapsed time for mode 2 contraction: " << duration << " ms\n\n";

    std::cout << "Testing mode 3 contraction\n";
    auto start = std::chrono::high_resolution_clock::now();
    T* contraction_3 = tucker_compute_contraction_5D<T>(blco,block_size,3);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Elapsed time for mode 3 contraction: " << duration << " ms\n\n";

    std::cout << "Testing mode 4 contraction\n";
    auto start = std::chrono::high_resolution_clock::now();
    T* contraction_4 = tucker_compute_contraction_5D<T>(blco,block_size,4);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Elapsed time for mode 4 contraction: " << duration << " ms\n\n";

    std::cout << "Testing mode 5 contraction\n";
    auto start = std::chrono::high_resolution_clock::now();
    T* contraction_5 = tucker_compute_contraction_5D<T>(blco,block_size,5);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Elapsed time for mode 5 contraction: " << duration << " ms\n\n";

    free(tmm_output_1);
    free(tmm_output_2);
    free(tmm_output_3);
    free(tmm_output_4);
    free(tmm_output_5);
    free(core_output);
    free(contraction_1);
    free(contraction_2);
    free(contraction_3);
    free(contraction_4);
    free(contraction_5);
}

int main(int argc, char* argv[]) {
    if (argc < 8 || argc > 11){
        std::cerr << "Usage: " << argv[0] 
                  << "<NNZ> <Decomposition rank> <Block Size> <Type> (three to five different dimensions)\n";
        return 1;
    }
    else if (argc == 8){
        int nnz = std::stoi(argv[1]);
        int rank = argc - 5;
        int decomp_rank = std::stoi(argv[2]);
        int block_size = std::stoi(argv[3]);
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
            if(type == "int") test_3D_kernels<int,uint64_t>(dimensions, nnz, block_size);
            else if(type == "float") test_3D_kernels<float,uint64_t>(dimensions, nnz, block_size);
            else if(type == "long int") test_3D_kernels<long int,uint64_t>(dimensions, nnz, block_size);
            else{ 
                std::cerr << "Unsupported type. The supported types are int, \
                float, long int, and long int\n";
                return 1;
            }
        }
        else{
            if(type == "int") test_3D_kernels<int,__uint128_t>(dimensions, nnz, block_size);
            else if(type == "float") test_3D_kernels<float,__uint128_t>(dimensions, nnz, block_size);
            else if(type == "long int") test_3D_kernels<long int,__uint128_t>(dimensions, nnz, block_size);
            else{ 
                std::cerr << "Unsupported type. The supported types are int, \
                float, long int, and long int\n";
                return 1;
            }
        }
    }
    else if (argc == 9){
        int nnz = std::stoi(argv[1]);
        int rank = argc - 5;
        int decomp_rank = std::stoi(argv[2]);
        int block_size = std::stoi(argv[3]);
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
            if(type == "int") test_3D_kernels<int,uint64_t>(dimensions, nnz, block_size);
            else if(type == "float") test_3D_kernels<float,uint64_t>(dimensions, nnz, block_size);
            else if(type == "long int") test_3D_kernels<long int,uint64_t>(dimensions, nnz, block_size);
            else{ 
                std::cerr << "Unsupported type. The supported types are int, \
                float, long int, and long int\n";
                return 1;
            }
        }
        else{
            if(type == "int") test_3D_kernels<int,__uint128_t>(dimensions, nnz, block_size);
            else if(type == "float") test_3D_kernels<float,__uint128_t>(dimensions, nnz, block_size);
            else if(type == "long int") test_3D_kernels<long int,__uint128_t>(dimensions, nnz, block_size);
            else{ 
                std::cerr << "Unsupported type. The supported types are int, \
                float, long int, and long int\n";
                return 1;
            }
        }
    }
    else if (argc == 10){
        int nnz = std::stoi(argv[1]);
        int rank = argc - 5;
        int decomp_rank = std::stoi(argv[2]);
        int block_size = std::stoi(argv[3]);
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
            if(type == "int") test_3D_kernels<int,uint64_t>(dimensions, nnz, block_size);
            else if(type == "float") test_3D_kernels<float,uint64_t>(dimensions, nnz, block_size);
            else if(type == "long int") test_3D_kernels<long int,uint64_t>(dimensions, nnz, block_size);
            else{ 
                std::cerr << "Unsupported type. The supported types are int, \
                float, long int, and long int\n";
                return 1;
            }
        }
        else{
            if(type == "int") test_3D_kernels<int,__uint128_t>(dimensions, nnz, block_size);
            else if(type == "float") test_3D_kernels<float,__uint128_t>(dimensions, nnz, block_size);
            else if(type == "long int") test_3D_kernels<long int,__uint128_t>(dimensions, nnz, block_size);
            else{ 
                std::cerr << "Unsupported type. The supported types are int, \
                float, long int, and long int\n";
                return 1;
            }
        }
    }
}