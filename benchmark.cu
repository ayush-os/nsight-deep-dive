#include <cuda_runtime.h>
#include <mma.h>
#include <iostream>

using namespace nvcuda;

// Mapping the dimensions for a single WMMA operation
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void wmma_ker(half *a, half *b, float *c, int M, int N, int K) {
    // Determine the row and column of the "tile" this warp is handling
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // Create fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize accumulator with zeros
    wmma::fill_fragment(c_frag, 0.0f);

    // Load tiles from global memory
    // In a real optimized kernel, you'd use shared memory here
    wmma::load_matrix_sync(a_frag, a + (warpM * WMMA_M * K), K);
    wmma::load_matrix_sync(b_frag, b + (warpN * WMMA_N * K), K);

    // Perform Matrix Multiply-Accumulate
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store the result back to global memory
    wmma::store_matrix_sync(c + (warpM * WMMA_M * N + warpN * WMMA_N), c_frag, N, wmma::mem_row_major);
}

int main() {
    const int M = 1024, N = 1024, K = 1024;

    half *h_a, *h_b;
    float *h_c;
    half *d_a, *d_b;
    float *d_c;

    // Allocation
    h_a = (half*)malloc(M * K * sizeof(half));
    h_b = (half*)malloc(K * N * sizeof(half));
    h_c = (float*)malloc(M * N * sizeof(float));

    cudaMalloc(&d_a, M * K * sizeof(half));
    cudaMalloc(&d_b, K * N * sizeof(half));
    cudaMalloc(&d_c, M * N * sizeof(float));

    // Initialize data (simplistic for profiling)
    for (int i = 0; i < M * K; i++) h_a[i] = __float2half(1.0f);
    for (int i = 0; i < K * N; i++) h_b[i] = __float2half(1.0f);

    cudaMemcpy(d_a, h_a, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, K * N * sizeof(half), cudaMemcpyHostToDevice);

    // Setup execution configuration
    // 32 threads per warp; we want each warp to handle one 16x16 tile
    dim3 gridDim(M / WMMA_M, N / WMMA_N);
    dim3 blockDim(32, 1); 

    std::cout << "Launching Tensor Core Kernel..." << std::endl;
    wmma_ker<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    
    cudaDeviceSynchronize();
    std::cout << "Done." << std::endl;

    // Cleanup
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);

    return 0;
}