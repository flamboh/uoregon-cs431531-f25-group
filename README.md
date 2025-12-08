# Tucker Decomposition of Tensors
## how to use CUDA module on Talapas

### how to build
```
cd cuda
module load cuda
make
```
to run the outputted ./cuda_tests file, run an interactive job or use the cuda.srun to sbatch the job.

### how to run an interactive job
```
srun --account=cis431_531 --partition=interactivegpu --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=500m --gpus=1 --constraint=gpu-10gb --pty bash
```
### command to execute
```
./cuda_tests 5000 0 10 3 100000 10 10 320
```
this will execute using a 3 dimensional 100000 x 10 x 10 filled with 5000 random non-zero entries, perform a tensor-matrix multiplication on all modes (0 represents all), and use a thread block size of 320.

for more usage details:
```
Usage: ./cuda_tests <nnz> <mode|0=all> <rank> <num_modes> <dim1> ... <dimN> [block_size]
```
