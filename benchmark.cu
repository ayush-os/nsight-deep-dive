#include <iostream>
#include <cuda_runtime.h>

// CUDA Kernel: Vector Addition
// This is memory bound because we perform very little math per byte loaded.
__global__ void vectorAdd(const float* __restrict__ A, 
                          const float* __restrict__ B, 
                          float* __restrict__ C, 
                          int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    // 2^26 elements (~67 million), large enough to saturate DRAM bandwidth
    const int N = 1 << 26;
    size_t size = N * sizeof(float);

    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    // Allocate Host Memory
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    // Initialize Host Data
    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Allocate Device Memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy to Device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Kernel Launch Configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "Launching kernel with " << blocksPerGrid << " blocks..." << std::endl;

    // Warmup
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // The actual run you'll want to profile
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // Copy back and verify
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    if (h_C[0] != 3.0f) std::cerr << "Error in calculation!" << std::endl;

    std::cout << "Done." << std::endl;

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}