#include "3D_kernels.h"
#include "4D_kernels.h"
#include "5D_kernels.h"
#include "cpu_tests.h"
#include "../tensor_storage/blco.h"
#include "../tensor_storage/tensor_utils.h"

template<typename T, typename S>
void test_3D_kernels(std::vector<int> dims, int nnz, int block_size, int construction_rank)
{
    int rank = dims.size();
    double total_entries = static_cast<double>(dims[0]);
    for(int i = 1; i < rank; i++) total_entries *= dims[i];
    double freq = static_cast<double>(nnz) / total_entries;

    int min_dim = *(std::min_element(dims.begin(), dims.end()));
    int cluster_size = 0.05 * min_dim;
    int max_blocks = 10 * (nnz / pow(cluster_size,rank));
    std::vector<NNZ_Entry<T>> test_vec = generate_block_sparse_tensor_nd<T>(dims,freq,0,100,cluster_size,max_blocks);

    Blco_Tensor<T,S> blco(test_vec,dims,construction_rank);
    std::vector<T*> fmats = blco.get_fmats();

    print_amd_gpu_model();
    std::cout<<"\n";

    std::cout << "Testing operations on " << dims[0] << " x " << dims[1] << " x " << dims[2] << " tensor with " << nnz << " non zeros\n\n";

    std::cout << "Starting warm-up phase (not timed)...\n";
    T* warm_up = tmm_3D<T>(blco, 1, block_size, false);
    free(warm_up);
    std::cout << "Warm-up complete.\n\n";

    std::cout << "Testing mode 1 tensor matrix multiplication\n";
    auto start = std::chrono::high_resolution_clock::now();
    T* tmm_output_1 = tmm_3D<T>(blco,1,block_size);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Total duration: " << static_cast<float>(duration) / 1000 << " ms\n\n";

    std::cout << "Testing mode 2 tensor matrix multiplication\n";
    start = std::chrono::high_resolution_clock::now();
    T* tmm_output_2 = tmm_3D<T>(blco,2,block_size);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Total duration: " << static_cast<float>(duration) / 1000 << " ms\n\n";

    std::cout << "Testing mode 3 tensor matrix multiplication\n";
    start = std::chrono::high_resolution_clock::now();
    T* tmm_output_3 = tmm_3D<T>(blco,3,block_size);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Total duration: " << static_cast<float>(duration) / 1000<< " ms\n\n";

    std::cout << "Testing core generation\n";
    start = std::chrono::high_resolution_clock::now();
    T* core_output = tucker_compute_core_3D<T>(blco,block_size);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Total duration: " << static_cast<float>(duration) / 1000 << " ms\n\n";

    std::cout << "Testing mode 1 contraction\n";
    start = std::chrono::high_resolution_clock::now();
    T* contraction_1 = contract_n_minus_one_modes_3D<T>(blco,block_size,1);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Total duration: " << static_cast<float>(duration) / 1000 << " ms\n\n";

    std::cout << "Testing mode 2 contraction\n";
    start = std::chrono::high_resolution_clock::now();
    T* contraction_2 = contract_n_minus_one_modes_3D<T>(blco,block_size,2);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Total duration: " << static_cast<float>(duration) / 1000 << " ms\n\n";

    std::cout << "Testing mode 3 contraction\n";
    start = std::chrono::high_resolution_clock::now();
    T* contraction_3 = contract_n_minus_one_modes_3D<T>(blco,block_size,3);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Total duration: " << static_cast<float>(duration) / 1000 << " ms\n\n";

    std::cout<<"Constructing CPU results:\n\n";
    std::vector<T> cpu_tmm_1 = tmm_3D_cpu<T,S>(1, test_vec, dims, fmats[0], construction_rank);
    std::vector<T> cpu_tmm_2 = tmm_3D_cpu<T,S>(2, test_vec, dims, fmats[1], construction_rank);
    std::vector<T> cpu_tmm_3 = tmm_3D_cpu<T,S>(3, test_vec, dims, fmats[2], construction_rank);
    std::vector<T> cpu_core_tensor = core_tensor_3D_cpu<T,S>(test_vec, dims, fmats, construction_rank);
    std::vector<T> cpu_contraction_1 = contract_tensor_3D_cpu<T,S>(1, test_vec, dims, fmats, construction_rank);
    std::vector<T> cpu_contraction_2 = contract_tensor_3D_cpu<T,S>(2, test_vec, dims, fmats, construction_rank);
    std::vector<T> cpu_contraction_3 = contract_tensor_3D_cpu<T,S>(3, test_vec, dims, fmats, construction_rank);
    
    std::cout<<"Comparing CPU and GPU results:\n";
    if(compare_tmm_arrays(tmm_output_1, cpu_tmm_1.data(), dims, construction_rank, 1)) std::cout<<"mode 1 tmm arrays match!\n";
    else std::cout<<"mode 1 tmm arrays do not match!\n";
    if(compare_tmm_arrays(tmm_output_2, cpu_tmm_2.data(), dims, construction_rank, 2)) std::cout<<"mode 2 tmm arrays match!\n";
    else std::cout<<"mode 2 tmm arrays do not match!\n";
    if(compare_tmm_arrays(tmm_output_3, cpu_tmm_3.data(), dims, construction_rank, 3)) std::cout<<"mode 3 tmm arrays match!\n";
    else std::cout<<"mode 3 tmm arrays do not match!\n";
    // if(compare_ct_arrays(core_output, cpu_core_tensor.data(), dims, construction_rank)) std::cout<<"core tensor arrays match!\n";
    // else std::cout<<"core tensor arrays don't match!\n";
    // if(compare_multimode_contraction_arrays(contraction_1, cpu_contraction_1.data(), dims, construction_rank, 1)) std::cout<<"mode 1 contraction arrays match!\n";
    // else std::cout<<"mode 1 contraction arrays do not match!\n";
    // if(compare_multimode_contraction_arrays(contraction_2, cpu_contraction_2.data(), dims, construction_rank, 2)) std::cout<<"mode 2 contraction arrays match!\n";
    // else std::cout<<"mode 2 contraction arrays do not match!\n";
    // if(compare_multimode_contraction_arrays(contraction_3, cpu_contraction_3.data(), dims, construction_rank, 3)) std::cout<<"mode 3 contraction arrays match!\n";
    // else std::cout<<"mode 3 contraction arrays do not match!\n";

    free(tmm_output_1);
    free(tmm_output_2);
    free(tmm_output_3);
    free(core_output);
    free(contraction_1);
    free(contraction_2);
    free(contraction_3);
}

template<typename T, typename S>
void test_4D_kernels(std::vector<int> dims, int nnz, int block_size, int construction_rank)
{
    int rank = dims.size();
    double total_entries = static_cast<double>(dims[0]);
    for(int i = 1; i < rank; i++) total_entries *= dims[i];
    double freq = static_cast<double>(nnz) / total_entries;

    int min_dim = *(std::min_element(dims.begin(), dims.end()));
    int cluster_size = 0.05 * min_dim;
    int max_blocks = 10 * (nnz / pow(cluster_size,rank));
    std::vector<NNZ_Entry<T>> test_vec = generate_block_sparse_tensor_nd<T>(dims,freq,0,100,cluster_size,max_blocks);

    Blco_Tensor<T,S> blco(test_vec,dims,construction_rank);
    std::vector<T*> fmats = blco.get_fmats();

    print_amd_gpu_model();
    std::cout<<"\n";

    std::cout << "Testing operations on " << dims[0] << " x " << dims[1] << " x " << dims[2] << 
    " x " << dims[3] << " tensor with " << nnz << " non zeros\n\n";

    std::cout << "Starting warm-up phase (not timed)...\n";
    T* warm_up = tmm_4D<T>(blco, 1, block_size, false);
    free(warm_up);
    std::cout << "Warm-up complete.\n\n";

    std::cout << "Testing mode 1 tensor matrix multiplication\n";
    auto start = std::chrono::high_resolution_clock::now();
    T* tmm_output_1 = tmm_4D<T>(blco,1,block_size);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Total duration: " << static_cast<float>(duration) / 1000 << " ms\n\n";

    std::cout << "Testing mode 2 tensor matrix multiplication\n";
    start = std::chrono::high_resolution_clock::now();
    T* tmm_output_2 = tmm_4D<T>(blco,2,block_size);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Total duration: " << static_cast<float>(duration) / 1000 << " ms\n\n";

    std::cout << "Testing mode 3 tensor matrix multiplication\n";
    start = std::chrono::high_resolution_clock::now();
    T* tmm_output_3 = tmm_4D<T>(blco,3,block_size);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Total duration: " << static_cast<float>(duration) / 1000 << " ms\n\n";

    std::cout << "Testing mode 4 tensor matrix multiplication\n";
    start = std::chrono::high_resolution_clock::now();
    T* tmm_output_4 = tmm_4D<T>(blco,4,block_size);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Total duration: " << static_cast<float>(duration) / 1000 << " ms\n\n";

    std::cout << "Testing core generation\n";
    start = std::chrono::high_resolution_clock::now();
    T* core_output = tucker_compute_core_4D<T>(blco,block_size);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Total duration: " << static_cast<float>(duration) / 1000 << " ms\n\n";

    std::cout << "Testing mode 1 contraction\n";
    start = std::chrono::high_resolution_clock::now();
    T* contraction_1 = contract_n_minus_one_modes_4D<T>(blco,block_size,1);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Total duration: " << static_cast<float>(duration) / 1000 << " ms\n\n";

    std::cout << "Testing mode 2 contraction\n";
    start = std::chrono::high_resolution_clock::now();
    T* contraction_2 = contract_n_minus_one_modes_4D<T>(blco,block_size,2);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Total duration: " << static_cast<float>(duration) / 1000 << " ms\n\n";

    std::cout << "Testing mode 3 contraction\n";
    start = std::chrono::high_resolution_clock::now();
    T* contraction_3 = contract_n_minus_one_modes_4D<T>(blco,block_size,3);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Total duration: " << static_cast<float>(duration) / 1000 << " ms\n\n";

    std::cout << "Testing mode 4 contraction\n";
    start = std::chrono::high_resolution_clock::now();
    T* contraction_4 = contract_n_minus_one_modes_4D<T>(blco,block_size,4);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Total duration: " << static_cast<float>(duration) / 1000 << " ms\n\n";

    std::cout<<"constructing CPU results\n\n";
    std::vector<T> cpu_tmm_1 = tmm_4D_cpu<T,S>(1, test_vec, dims, fmats[0], construction_rank);
    std::vector<T> cpu_tmm_2 = tmm_4D_cpu<T,S>(2, test_vec, dims, fmats[1], construction_rank);
    std::vector<T> cpu_tmm_3 = tmm_4D_cpu<T,S>(3, test_vec, dims, fmats[2], construction_rank);
    std::vector<T> cpu_tmm_4 = tmm_4D_cpu<T,S>(4, test_vec, dims, fmats[3], construction_rank);
    std::vector<T> cpu_core_tensor = core_tensor_4D_cpu<T,S>(test_vec, dims, fmats, construction_rank);
    std::vector<T> cpu_contraction_1 = contract_tensor_4D_cpu<T,S>(1, test_vec, dims, fmats, construction_rank);
    std::vector<T> cpu_contraction_2 = contract_tensor_4D_cpu<T,S>(2, test_vec, dims, fmats, construction_rank);
    std::vector<T> cpu_contraction_3 = contract_tensor_4D_cpu<T,S>(3, test_vec, dims, fmats, construction_rank);
    std::vector<T> cpu_contraction_4 = contract_tensor_4D_cpu<T,S>(4, test_vec, dims, fmats, construction_rank);
    
    std::cout<<"comparing CPU and GPU results\n";
    if(compare_tmm_arrays(tmm_output_1, cpu_tmm_1.data(), dims, construction_rank, 1)) std::cout<<"mode 1 tmm arrays match!\n";
    else std::cout<<"mode 1 tmm arrays do not match!\n";
    if(compare_tmm_arrays(tmm_output_2, cpu_tmm_2.data(), dims, construction_rank, 2)) std::cout<<"mode 2 tmm arrays match!\n";
    else std::cout<<"mode 2 tmm arrays do not match!\n";
    if(compare_tmm_arrays(tmm_output_3, cpu_tmm_3.data(), dims, construction_rank, 3)) std::cout<<"mode 3 tmm arrays match!\n";
    else std::cout<<"mode 3 tmm arrays do not match!\n";
    if(compare_tmm_arrays(tmm_output_4, cpu_tmm_4.data(), dims, construction_rank, 4)) std::cout<<"mode 4 tmm arrays match!\n";
    else std::cout<<"mode 4 tmm arrays do not match!\n";
    if(compare_ct_arrays(core_output, cpu_core_tensor.data(), dims, construction_rank)) std::cout<<"core tensor arrays match!\n";
    else std::cout<<"core tensor arrays don't match!\n";
    if(compare_multimode_contraction_arrays(contraction_1, cpu_contraction_1.data(), dims, construction_rank, 1)) std::cout<<"mode 1 contraction arrays match!\n";
    else std::cout<<"mode 1 contraction arrays do not match!\n";
    if(compare_multimode_contraction_arrays(contraction_2, cpu_contraction_2.data(), dims, construction_rank, 2)) std::cout<<"mode 2 contraction arrays match!\n";
    else std::cout<<"mode 2 contraction arrays do not match!\n";
    if(compare_multimode_contraction_arrays(contraction_3, cpu_contraction_3.data(), dims, construction_rank, 3)) std::cout<<"mode 3 contraction arrays match!\n";
    else std::cout<<"mode 3 contraction arrays do not match!\n";
    if(compare_multimode_contraction_arrays(contraction_4, cpu_contraction_4.data(), dims, construction_rank, 4)) std::cout<<"mode 4 contraction arrays match!\n";
    else std::cout<<"mode 4 contraction arrays do not match!\n";

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
void test_5D_kernels(std::vector<int> dims, int nnz, int block_size, int construction_rank)
{
    int rank = dims.size();
    double total_entries = static_cast<double>(dims[0]);
    for(int i = 1; i < rank; i++) total_entries *= dims[i];
    double freq = static_cast<double>(nnz) / total_entries;

    int min_dim = *(std::min_element(dims.begin(), dims.end()));
    int cluster_size = 0.05 * min_dim;
    int max_blocks = 10 * (nnz / pow(cluster_size,rank));
    std::vector<NNZ_Entry<T>> test_vec = generate_block_sparse_tensor_nd<T>(dims,freq,0,100,cluster_size,max_blocks);

    Blco_Tensor<T,S> blco(test_vec,dims,construction_rank);
    std::vector<T*> fmats = blco.get_fmats();

    print_amd_gpu_model();
    std::cout<<"\n";

    std::cout << "Testing operations on " << dims[0] << " x " << dims[1] << " x " << dims[2] << 
    " x " << dims[3] << " x " <<  dims[4] << " tensor with " << nnz << " non zeros\n\n";

    std::cout << "Starting warm-up phase (not timed)...\n";
    T* warm_up = tmm_5D<T>(blco, 1, block_size, false);
    free(warm_up);
    std::cout << "Warm-up complete.\n\n";

    std::cout << "Testing mode 1 tensor matrix multiplication\n";
    auto start = std::chrono::high_resolution_clock::now();
    T* tmm_output_1 = tmm_5D<T>(blco,1,block_size);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Total duration: " << static_cast<float>(duration) / 1000 << " ms\n\n";

    std::cout << "Testing mode 2 tensor matrix multiplication\n";
    start = std::chrono::high_resolution_clock::now();
    T* tmm_output_2 = tmm_5D<T>(blco,2,block_size);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Total duration: " << static_cast<float>(duration) / 1000 << " ms\n\n";

    std::cout << "Testing mode 3 tensor matrix multiplication\n";
    start = std::chrono::high_resolution_clock::now();
    T* tmm_output_3 = tmm_5D<T>(blco,3,block_size);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Total duration: " << static_cast<float>(duration) / 1000 << " ms\n\n";

    std::cout << "Testing mode 4 tensor matrix multiplication\n";
    start = std::chrono::high_resolution_clock::now();
    T* tmm_output_4 = tmm_5D<T>(blco,4,block_size);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Total duration: " << static_cast<float>(duration) / 1000 << " ms\n\n";

    std::cout << "Testing mode 5 tensor matrix multiplication\n";
    start = std::chrono::high_resolution_clock::now();
    T* tmm_output_5 = tmm_5D<T>(blco,5,block_size);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Total duration: " << static_cast<float>(duration) / 1000 << " ms\n\n";

    std::cout << "Testing core generation\n";
    start = std::chrono::high_resolution_clock::now();
    T* core_output = tucker_compute_core_4D<T>(blco,block_size);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Total duration: " << static_cast<float>(duration) / 1000 << " ms\n\n";

    std::cout << "Testing mode 1 contraction\n";
    start = std::chrono::high_resolution_clock::now();
    T* contraction_1 = contract_n_minus_one_modes_5D<T>(blco,block_size,1);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Total duration: " << static_cast<float>(duration) / 1000 << " ms\n\n";

    std::cout << "Testing mode 2 contraction\n";
    start = std::chrono::high_resolution_clock::now();
    T* contraction_2 = contract_n_minus_one_modes_5D<T>(blco,block_size,2);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Total duration: " << static_cast<float>(duration) / 1000 << " ms\n\n";

    std::cout << "Testing mode 3 contraction\n";
    start = std::chrono::high_resolution_clock::now();
    T* contraction_3 = contract_n_minus_one_modes_5D<T>(blco,block_size,3);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Total duration: " << static_cast<float>(duration) / 1000 << " ms\n\n";

    std::cout << "Testing mode 4 contraction\n";
    start = std::chrono::high_resolution_clock::now();
    T* contraction_4 = contract_n_minus_one_modes_5D<T>(blco,block_size,4);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Total duration: " << static_cast<float>(duration) / 1000 << " ms\n\n";

    std::cout << "Testing mode 5 contraction\n";
    start = std::chrono::high_resolution_clock::now();
    T* contraction_5 = contract_n_minus_one_modes_5D<T>(blco,block_size,5);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Total duration: " << static_cast<float>(duration) / 1000 << " ms\n\n";

    std::cout<<"constructing CPU results\n\n";
    std::vector<T> cpu_tmm_1 = tmm_5D_cpu<T,S>(1, test_vec, dims, fmats[0], construction_rank);
    std::vector<T> cpu_tmm_2 = tmm_5D_cpu<T,S>(2, test_vec, dims, fmats[1], construction_rank);
    std::vector<T> cpu_tmm_3 = tmm_5D_cpu<T,S>(3, test_vec, dims, fmats[2], construction_rank);
    std::vector<T> cpu_tmm_4 = tmm_5D_cpu<T,S>(4, test_vec, dims, fmats[3], construction_rank);
    std::vector<T> cpu_tmm_5 = tmm_5D_cpu<T,S>(5, test_vec, dims, fmats[3], construction_rank);
    std::vector<T> cpu_core_tensor = core_tensor_5D_cpu<T,S>(test_vec, dims, fmats, construction_rank);
    std::vector<T> cpu_contraction_1 = contract_tensor_5D_cpu<T,S>(1, test_vec, dims, fmats, construction_rank);
    std::vector<T> cpu_contraction_2 = contract_tensor_5D_cpu<T,S>(2, test_vec, dims, fmats, construction_rank);
    std::vector<T> cpu_contraction_3 = contract_tensor_5D_cpu<T,S>(3, test_vec, dims, fmats, construction_rank);
    std::vector<T> cpu_contraction_4 = contract_tensor_5D_cpu<T,S>(4, test_vec, dims, fmats, construction_rank);
    std::vector<T> cpu_contraction_5 = contract_tensor_5D_cpu<T,S>(5, test_vec, dims, fmats, construction_rank);
    
    std::cout<<"comparing CPU and GPU results\n";
    if(compare_tmm_arrays(tmm_output_1, cpu_tmm_1.data(), dims, construction_rank, 1)) std::cout<<"mode 1 tmm arrays match!\n";
    else std::cout<<"mode 1 tmm arrays do not match!\n";
    if(compare_tmm_arrays(tmm_output_2, cpu_tmm_2.data(), dims, construction_rank, 2)) std::cout<<"mode 2 tmm arrays match!\n";
    else std::cout<<"mode 2 tmm arrays do not match!\n";
    if(compare_tmm_arrays(tmm_output_3, cpu_tmm_3.data(), dims, construction_rank, 3)) std::cout<<"mode 3 tmm arrays match!\n";
    else std::cout<<"mode 3 tmm arrays do not match!\n";
    if(compare_tmm_arrays(tmm_output_4, cpu_tmm_4.data(), dims, construction_rank, 4)) std::cout<<"mode 4 tmm arrays match!\n";
    else std::cout<<"mode 4 tmm arrays do not match!\n";
    if(compare_tmm_arrays(tmm_output_5, cpu_tmm_5.data(), dims, construction_rank, 5)) std::cout<<"mode 5 tmm arrays match!\n";
    else std::cout<<"mode 5 tmm arrays do not match!\n";
    if(compare_ct_arrays(core_output, cpu_core_tensor.data(), dims, construction_rank)) std::cout<<"core tensor arrays match!\n";
    else std::cout<<"core tensor arrays don't match!\n";
    if(compare_multimode_contraction_arrays(contraction_1, cpu_contraction_1.data(), dims, construction_rank, 1)) std::cout<<"mode 1 contraction arrays match!\n";
    else std::cout<<"mode 1 contraction arrays do not match!\n";
    if(compare_multimode_contraction_arrays(contraction_2, cpu_contraction_2.data(), dims, construction_rank, 2)) std::cout<<"mode 2 contraction arrays match!\n";
    else std::cout<<"mode 2 contraction arrays do not match!\n";
    if(compare_multimode_contraction_arrays(contraction_3, cpu_contraction_3.data(), dims, construction_rank, 3)) std::cout<<"mode 3 contraction arrays match!\n";
    else std::cout<<"mode 3 contraction arrays do not match!\n";
    if(compare_multimode_contraction_arrays(contraction_4, cpu_contraction_4.data(), dims, construction_rank, 4)) std::cout<<"mode 4 contraction arrays match!\n";
    else std::cout<<"mode 4 contraction arrays do not match!\n";
    if(compare_multimode_contraction_arrays(contraction_5, cpu_contraction_5.data(), dims, construction_rank, 5)) std::cout<<"mode 5 contraction arrays match!\n";
    else std::cout<<"mode 5 contraction arrays do not match!\n";

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
            if(type == "int") test_3D_kernels<int,uint64_t>(dimensions, nnz, block_size, decomp_rank);
            else if(type == "float") test_3D_kernels<float,uint64_t>(dimensions, nnz, block_size, decomp_rank);
            else if(type == "long int") test_3D_kernels<long int,uint64_t>(dimensions, nnz, block_size, decomp_rank);
            else{ 
                std::cerr << "Unsupported type. The supported types are int, \
                float, long int, and long int\n";
                return 1;
            }
        }
        else{
            if(type == "int") test_3D_kernels<int,__uint128_t>(dimensions, nnz, block_size, decomp_rank);
            else if(type == "float") test_3D_kernels<float,__uint128_t>(dimensions, nnz, block_size, decomp_rank);
            else if(type == "long int") test_3D_kernels<long int,__uint128_t>(dimensions, nnz, block_size, decomp_rank);
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
            if(type == "int") test_4D_kernels<int,uint64_t>(dimensions, nnz, block_size, decomp_rank);
            else if(type == "float") test_4D_kernels<float,uint64_t>(dimensions, nnz, block_size, decomp_rank);
            else if(type == "long int") test_4D_kernels<long int,uint64_t>(dimensions, nnz, block_size,decomp_rank);
            else{ 
                std::cerr << "Unsupported type. The supported types are int, \
                float, long int, and long int\n";
                return 1;
            }
        }
        else{
            if(type == "int") test_4D_kernels<int,__uint128_t>(dimensions, nnz, block_size, decomp_rank);
            else if(type == "float") test_4D_kernels<float,__uint128_t>(dimensions, nnz, block_size, decomp_rank);
            else if(type == "long int") test_4D_kernels<long int,__uint128_t>(dimensions, nnz, block_size, decomp_rank);
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
            if(type == "int") test_5D_kernels<int,uint64_t>(dimensions, nnz, block_size, decomp_rank);
            else if(type == "float") test_5D_kernels<float,uint64_t>(dimensions, nnz, block_size, decomp_rank);
            else if(type == "long int") test_5D_kernels<long int,uint64_t>(dimensions, nnz, block_size, decomp_rank);
            else{ 
                std::cerr << "Unsupported type. The supported types are int, \
                float, long int, and long int\n";
                return 1;
            }
        }
        else{
            if(type == "int") test_5D_kernels<int,__uint128_t>(dimensions, nnz, block_size, decomp_rank);
            else if(type == "float") test_5D_kernels<float,__uint128_t>(dimensions, nnz, block_size, decomp_rank);
            else if(type == "long int") test_5D_kernels<long int,__uint128_t>(dimensions, nnz, block_size, decomp_rank);
            else{ 
                std::cerr << "Unsupported type. The supported types are int, \
                float, long int, and long int\n";
                return 1;
            }
        }
    }
}