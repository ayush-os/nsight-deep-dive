#include <iostream>
#include <cuda_runtime.h>

#define TILE_SIZE 32

// CUDA Kernel: Tiled Matrix Multiplication
__global__ void matmul_tiled_kernel(float* A, float* B, float* C, int N) {
    // Shared memory for tiles of A and B
    __shared__ float as[TILE_SIZE][TILE_SIZE];
    __shared__ float bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // Identify the row and column of the C element to work on
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    // Loop over tiles required to compute the C element
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        
        // Coalesced loading from Global Memory to Shared Memory
        if (row < N && (t * TILE_SIZE + tx) < N)
            as[ty][tx] = A[row * N + (t * TILE_SIZE + tx)];
        else
            as[ty][tx] = 0.0f;

        if (col < N && (t * TILE_SIZE + ty) < N)
            bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        else
            bs[ty][tx] = 0.0f;

        // Synchronize to ensure the entire tile is loaded
        __syncthreads();

        // Compute partial product from the tile
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += as[ty][k] * bs[k][tx];
        }

        // Synchronize before loading the next tile
        __syncthreads();
    }

    // Write the final result back to Global Memory
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

void check_cuda_error(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Matrix size (N x N) - Use a multiple of TILE_SIZE for clean profiling
    const int N = 1024;
    size_t size = N * N * sizeof(float);

    // Host allocation
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Device allocation
    float *d_A, *d_B, *d_C;
    check_cuda_error(cudaMalloc(&d_A, size), "Malloc A");
    check_cuda_error(cudaMalloc(&d_B, size), "Malloc B");
    check_cuda_error(cudaMalloc(&d_C, size), "Malloc C");

    check_cuda_error(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice), "Copy A");
    check_cuda_error(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice), "Copy B");

    // Launch configuration
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    std::cout << "Launching Kernel: Matrix Size " << N << "x" << N << std::endl;
    
    // Warmup and launch
    matmul_tiled_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    check_cuda_error(cudaDeviceSynchronize(), "Kernel Exec");

    std::cout << "Done." << std::endl;

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}