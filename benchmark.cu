#include <iostream>
#include <cuda_runtime.h>

// Matrix dimensions (N x N)
#define N 1024
#define BLOCK_SIZE 16

// Naive Matrix Multiplication Kernel
// Each thread computes one element of the output matrix C
__global__ void matMulNaive(const float* A, const float* B, float* C, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        float sum = 0.0f;
        for (int k = 0; k < size; ++k) {
            // A is accessed by row, B is accessed by column
            sum += A[row * size + k] * B[k * size + col];
        }
        C[row * size + col] = sum;
    }
}

int main() {
    size_t bytes = N * N * sizeof(float);

    // Host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel
    std::cout << "Launching Naive Matrix Multiplication..." << std::endl;
    matMulNaive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaDeviceSynchronize();

    // Copy back result
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    std::cout << "Done. Result[0]: " << h_C[0] << " (Expected: " << (float)N * 2.0f << ")" << std::endl;

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}