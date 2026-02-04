#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// --- KERNEL 1: Unoptimized Global Memory Scan (Kogge-Stone) ---
// This is "unoptimized" because it hits Global Memory in a loop 
// and creates non-coalesced access patterns as 'stride' increases.
__global__ void scanGlobalNaive(float* g_idata, float* g_odata, int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    for (unsigned int stride = 1; stride < n; stride *= 2) {
        float val = 0;
        if (i >= stride) {
            val = g_idata[i - stride];
        }
        __syncthreads(); // Note: This only works within a single block!
        if (i >= stride) {
            g_idata[i] += val;
        }
        __syncthreads();
    }
    g_odata[i] = g_idata[i];
}

// --- KERNEL 2: Optimized Shared Memory Scan (Blelloch) ---
// Uses Coalesced Global Loads/Stores and avoids bank conflicts 
// by using sequential addressing in shared memory.
__global__ void scanOptimized(float* g_idata, float* g_odata, int n) {
    __shared__ float temp[BLOCK_SIZE * 2]; // Twice the block size for Blelloch
    int tid = threadIdx.x;
    int offset = 1;

    // Coalesced load from global memory to shared memory
    int ai = tid;
    int bi = tid + (n / 2);
    temp[ai] = g_idata[ai];
    temp[bi] = g_idata[bi];

    // Up-Sweep (Reduction Phase)
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int i = offset * (2 * tid + 1) - 1;
            int j = offset * (2 * tid + 2) - 1;
            temp[j] += temp[i];
        }
        offset *= 2;
    }

    // Clear the last element for exclusive scan
    if (tid == 0) { temp[n - 1] = 0; }

    // Down-Sweep Phase
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int i = offset * (2 * tid + 1) - 1;
            int j = offset * (2 * tid + 2) - 1;
            float t = temp[i];
            temp[i] = temp[j];
            temp[j] += t;
        }
    }
    __syncthreads();

    // Coalesced store back to global memory
    g_odata[ai] = temp[ai];
    g_odata[bi] = temp[bi];
}

int main() {
    const int N = BLOCK_SIZE; // Simplified for single-block profiling
    size_t size = N * sizeof(float);

    float h_in[N];
    for (int i = 0; i < N; i++) h_in[i] = 1.0f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    // Launch Naive
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    scanGlobalNaive<<<1, N>>>(d_in, d_out, N);
    
    // Launch Optimized
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    scanOptimized<<<1, N / 2>>>(d_in, d_out, N);

    cudaDeviceSynchronize();
    std::cout << "Scan profiling kernels complete." << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}