#include <stdio.h>
#include <cuda_runtime.h>   // declares cudaFree, cudaMalloc, etc.

__global__ void add(int *a, int *b, int *c)
{
  int i = threadIdx.x;
  if (i < 5)
    c[i] = a[i] + b[i];
}

int main()
{
  int a[5] = {1, 2, 3, 4, 5};
  int b[5] = {10, 20, 30, 40, 50};
  int c[5];

  int *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

  cudaMalloc((void **)&d_a, 5 * sizeof(int));
  cudaMalloc((void **)&d_b, 5 * sizeof(int));
  cudaMalloc((void **)&d_c, 5 * sizeof(int));

  cudaMemcpy(d_a, a, 5 * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, 5 * sizeof(int), cudaMemcpyHostToDevice);

  add<<<1, 5>>>(d_a, d_b, d_c);

  cudaDeviceSynchronize();

  cudaMemcpy(c, d_c, 5 * sizeof(int), cudaMemcpyDeviceToHost);

  for (int i = 0; i < 5; i++)
    printf("%d ", c[i]);
  printf("\n");

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}
