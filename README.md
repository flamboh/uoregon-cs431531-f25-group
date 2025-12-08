# How to run CUDA module on Talapas
```
cd cuda
make
./cuda_tests 5000 0 10 3 100000 10 10 320
```
this will execute using a 3 dimensional 100000 x 10 x 10 filled with 5000 random non-zero entries, perform a tensor-matrix multiplication on all modes (0 represents all), and use a thread block size of 320.

For more usage details:
```
Usage: ./cuda_tests <nnz> <mode|0=all> <rank> <num_modes> <dim1> ... <dimN> [block_size]
```
