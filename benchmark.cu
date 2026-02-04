#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>

// Error checking macro
#define CUDA_CHECK(err) { if (err != cudaSuccess) { std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; exit(1); } }
#define CUBLAS_CHECK(err) { if (err != CUBLAS_STATUS_SUCCESS) { std::cerr << "cuBLAS Error at line " << __LINE__ << std::endl; exit(1); } }

int main() {
    // Matrix dimensions (M x K) * (K x N) = (M x N)
    const int M = 2048;
    const int N = 2048;
    const int K = 2048;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 1. Allocate Host Memory
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    std::vector<float> h_A(M * K, 1.1f);
    std::vector<float> h_B(K * N, 2.2f);
    std::vector<float> h_C(M * N, 0.0f);

    // 2. Allocate Device Memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    // 3. Initialize cuBLAS
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // 4. Copy data to Device
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));

    // 5. Warm-up run (to initialize cuBLAS internals)
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
    CUDA_CHECK(cudaDeviceSynchronize());

    // 6. Profiled run
    std::cout << "Starting profiled SGEMM..." << std::endl;
    cudaProfilerStart(); 
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaProfilerStop();
    std::cout << "Done." << std::endl;

    // 7. Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUBLAS_CHECK(cublasDestroy(handle));

    return 0;
}