#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// --- KERNEL 1: Shared Memory with BANK CONFLICTS ---
__global__ void reduceBankConflicts(float* g_idata, float* g_odata, unsigned int n) {
    __shared__ uint8_t sdata_raw[BLOCK_SIZE * sizeof(float)]; // Raw bytes for alignment
    float* sdata = reinterpret_cast<float*>(sdata_raw);

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();

    // Strategy: 2-way interleaved addressing (Causes 2-way, 4-way, etc. bank conflicts)
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;
        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// --- KERNEL 2: Shared Memory (Optimized, No Bank Conflicts) ---
__global__ void reduceOptimizedShared(float* g_idata, float* g_odata, unsigned int n) {
    __shared__ float sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? g_idata[i] : 0;
    __syncthreads();

    // Strategy: Sequential addressing (No bank conflicts)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// --- KERNEL 3: Warp Shuffle (Bypassing Shared Memory) ---
__global__ void reduceWarpShuffle(float* g_idata, float* g_odata, unsigned int n) {
    float val = 0;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) val = g_idata[i];

    // Intra-warp reduction using shuffle
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    // Shared memory only used once per block to collect warp results
    __shared__ float warpSums[32]; 
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    if (lane == 0) warpSums[wid] = val;
    __syncthreads();

    // Final reduction of warp sums in the first warp
    val = (threadIdx.x < blockDim.x / warpSize) ? warpSums[lane] : 0;
    if (wid == 0) {
        for (int offset = (blockDim.x / warpSize) / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
    }

    if (tid == 0) g_odata[blockIdx.x] = val;
}

int main() {
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);

    float *h_in = new float[N];
    for (int i = 0; i < N; i++) h_in[i] = 1.0f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    dim3 threads(BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // --- Launch Kernels ---
    std::cout << "Launching kernels for profiling..." << std::endl;
    
    reduceBankConflicts<<<blocks, threads>>>(d_in, d_out, N);
    reduceOptimizedShared<<<blocks, threads>>>(d_in, d_out, N);
    reduceWarpShuffle<<<blocks, threads>>>(d_in, d_out, N);

    cudaDeviceSynchronize();
    std::cout << "Done." << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    return 0;
}