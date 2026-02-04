#include <iostream>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define KERNEL_RADIUS 2
#define KERNEL_WIDTH (2 * KERNEL_RADIUS + 1)

// --- KERNEL 1: Naive 2D Conv (High Branch Divergence) ---
__global__ void conv2d_naive(float* input, float* output, float* kernel, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < height && col < width) {
        float sum = 0.0f;
        for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++) {
            for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {
                int curRow = row + i;
                int curCol = col + j;

                // BRANCH DIVERGENCE: Edge threads take the 'if', inner threads don't.
                // In a warp, if one thread hits the boundary, the whole warp executes both paths.
                if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
                    sum += input[curRow * width + curCol] * kernel[(i + KERNEL_RADIUS) * KERNEL_WIDTH + (j + KERNEL_RADIUS)];
                }
            }
        }
        output[row * width + col] = sum;
    }
}

// --- KERNEL 2: Tiled Shared Memory (Optimized) ---
__global__ void conv2d_tiled(float* input, float* output, float* kernel, int width, int height) {
    // Shared memory needs to be larger than the tile to account for the "halo"
    __shared__ float s_data[TILE_SIZE + 2 * KERNEL_RADIUS][TILE_SIZE + 2 * KERNEL_RADIUS];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + tx;
    int row = blockIdx.y * TILE_SIZE + ty;

    // Load main tile + halo into shared memory
    for (int i = ty; i < TILE_SIZE + 2 * KERNEL_RADIUS; i += TILE_SIZE) {
        for (int j = tx; j < TILE_SIZE + 2 * KERNEL_RADIUS; j += TILE_SIZE) {
            int curRow = blockIdx.y * TILE_SIZE + i - KERNEL_RADIUS;
            int curCol = blockIdx.x * TILE_SIZE + j - KERNEL_RADIUS;

            if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width)
                s_data[i][j] = input[curRow * width + curCol];
            else
                s_data[i][j] = 0.0f;
        }
    }
    __syncthreads();

    if (row < height && col < width) {
        float sum = 0.0f;
        for (int i = 0; i < KERNEL_WIDTH; i++) {
            for (int j = 0; j < KERNEL_WIDTH; j++) {
                sum += s_data[ty + i][tx + j] * kernel[i * KERNEL_WIDTH + j];
            }
        }
        output[row * width + col] = sum;
    }
}

int main() {
    const int W = 2048;
    const int H = 2048;
    size_t img_size = W * H * sizeof(float);
    size_t kernel_size = KERNEL_WIDTH * KERNEL_WIDTH * sizeof(float);

    float *h_in = new float[W * H];
    float *h_kern = new float[KERNEL_WIDTH * KERNEL_WIDTH];
    for (int i = 0; i < W * H; i++) h_in[i] = 1.0f;
    for (int i = 0; i < KERNEL_WIDTH * KERNEL_WIDTH; i++) h_kern[i] = 0.1f;

    float *d_in, *d_out, *d_kern;
    cudaMalloc(&d_in, img_size);
    cudaMalloc(&d_out, img_size);
    cudaMalloc(&d_kern, kernel_size);

    cudaMemcpy(d_in, h_in, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kern, h_kern, kernel_size, cudaMemcpyHostToDevice);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((W + TILE_SIZE - 1) / TILE_SIZE, (H + TILE_SIZE - 1) / TILE_SIZE);

    std::cout << "Profiling 2D Convolution..." << std::endl;
    conv2d_naive<<<blocks, threads>>>(d_in, d_out, d_kern, W, H);
    conv2d_tiled<<<blocks, threads>>>(d_in, d_out, d_kern, W, H);

    cudaDeviceSynchronize();
    std::cout << "Done." << std::endl;

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_kern);
    delete[] h_in; delete[] h_kern;
    return 0;
}