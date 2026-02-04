#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define WARP_SIZE 32

// --- KERNEL 1: Scalar (Naive) ---
// Each thread processes one row. Poor coalescing.
__global__ void spmv_scalar(const int* row_ptr, const int* col_idx, const float* values, 
                            const float* x, float* y, int num_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float sum = 0;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        for (int i = row_start; i < row_end; i++) {
            sum += values[i] * x[col_idx[i]];
        }
        y[row] = sum;
    }
}

// --- KERNEL 2: Vector (Shared Memory) ---
// One warp processes one row. Coalesced reads, but uses Shared Mem.
__global__ void spmv_vector_smem(const int* row_ptr, const int* col_idx, const float* values, 
                                 const float* x, float* y, int num_rows) {
    __shared__ float sdata[128]; // Enough for 4 warps per block (128 threads)
    
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = thread_id / WARP_SIZE;
    int lane = thread_id % WARP_SIZE;
    int row = warp_id;

    if (row < num_rows) {
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        float sum = 0;

        for (int i = row_start + lane; i < row_end; i += WARP_SIZE) {
            sum += values[i] * x[col_idx[i]];
        }

        sdata[threadIdx.x] = sum;
        __syncwarp();

        // Manual reduction in shared memory (prone to bank conflicts if not careful)
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            if (lane < offset) sdata[threadIdx.x] += sdata[threadIdx.x + offset];
            __syncwarp();
        }

        if (lane == 0) y[row] = sdata[threadIdx.x];
    }
}

// --- KERNEL 3: Warp Shuffle (Optimized) ---
// One warp per row, bypassing shared memory for reduction.
__global__ void spmv_warp_shuffle(const int* row_ptr, const int* col_idx, const float* values, 
                                  const float* x, float* y, int num_rows) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = thread_id / WARP_SIZE;
    int lane = thread_id % WARP_SIZE;
    int row = warp_id;

    if (row < num_rows) {
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        float sum = 0;

        for (int i = row_start + lane; i < row_end; i += WARP_SIZE) {
            sum += values[i] * x[col_idx[i]];
        }

        // Warp Shuffle Reduction
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }

        if (lane == 0) y[row] = sum;
    }
}

int main() {
    const int num_rows = 10000;
    const int nnz_per_row = 32;
    const int nnz = num_rows * nnz_per_row;

    // Host Data
    std::vector<int> h_row_ptr(num_rows + 1);
    std::vector<int> h_col_idx(nnz);
    std::vector<float> h_values(nnz);
    std::vector<float> h_x(num_rows, 1.0f);

    for (int i = 0; i < num_rows; i++) {
        h_row_ptr[i] = i * nnz_per_row;
        for (int j = 0; j < nnz_per_row; j++) {
            h_col_idx[i * nnz_per_row + j] = (i + j) % num_rows;
            h_values[i * nnz_per_row + j] = 1.0f;
        }
    }
    h_row_ptr[num_rows] = nnz;

    // Device Data
    int *d_row_ptr, *d_col_idx;
    float *d_values, *d_x, *d_y;
    cudaMalloc(&d_row_ptr, (num_rows + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, nnz * sizeof(int));
    cudaMalloc(&d_values, nnz * sizeof(float));
    cudaMalloc(&d_x, num_rows * sizeof(float));
    cudaMalloc(&d_y, num_rows * sizeof(float));

    cudaMemcpy(d_row_ptr, h_row_ptr.data(), (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, h_col_idx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, h_values.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), num_rows * sizeof(float), cudaMemcpyHostToDevice);

    // Launch
    dim3 block(128);
    dim3 grid_scalar((num_rows + 127) / 128);
    dim3 grid_vector((num_rows * WARP_SIZE + 127) / 128);

    spmv_scalar<<<grid_scalar, block>>>(d_row_ptr, d_col_idx, d_values, d_x, d_y, num_rows);
    spmv_vector_smem<<<grid_vector, block>>>(d_row_ptr, d_col_idx, d_values, d_x, d_y, num_rows);
    spmv_warp_shuffle<<<grid_vector, block>>>(d_row_ptr, d_col_idx, d_values, d_x, d_y, num_rows);

    cudaDeviceSynchronize();
    return 0;
}