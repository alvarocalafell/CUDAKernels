// Include necessary libraries
#include <stdio.h>      // Standard I/O (e.g., printf)
#include <stdlib.h>     // Standard library (e.g., malloc, rand)
#include <cuda_runtime.h> // CUDA runtime functions (e.g., cudaMalloc)

// CUDA kernel to add two vectors (runs on the GPU)
// The __global__ keyword marks this as a GPU kernel
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    // Calculate the global thread index:
    // blockIdx.x = index of the current block in the grid
    // blockDim.x = number of threads per block
    // threadIdx.x = index of the current thread within the block
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if the thread index is within the array bounds
    if (i < n) {
        // Each thread computes one element of the output vector
        c[i] = a[i] + b[i];
    }
}

// Helper function to check for CUDA errors
void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        // Print error message and exit if something went wrong
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

int main() {
    // Define problem size: 1 million elements
    int n = 1 << 20; // 1 << 20 = 2^20 = 1,048,576
    size_t size = n * sizeof(float); // Total bytes needed

    // Allocate memory on the CPU (host)
    float *h_a = (float*)malloc(size); // Host vector A
    float *h_b = (float*)malloc(size); // Host vector B
    float *h_c = (float*)malloc(size); // Host result vector C

    // Initialize host vectors with random values
    for (int i = 0; i < n; i++) {
        h_a[i] = rand() / (float)RAND_MAX; // Random float between 0 and 1
        h_b[i] = rand() / (float)RAND_MAX;
    }

    // Allocate memory on the GPU (device)
    float *d_a, *d_b, *d_c;
    checkCudaError(cudaMalloc(&d_a, size), "cudaMalloc d_a failed"); // Allocate GPU memory for A
    checkCudaError(cudaMalloc(&d_b, size), "cudaMalloc d_b failed"); // Allocate GPU memory for B
    checkCudaError(cudaMalloc(&d_c, size), "cudaMalloc d_c failed"); // Allocate GPU memory for C

    // Copy data from CPU to GPU
    checkCudaError(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice), "Copy to d_a failed");
    checkCudaError(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice), "Copy to d_b failed");

    // Configure kernel launch parameters
    int blockSize = 256; // Number of threads per block (common choice: 256 or 512)
    int numBlocks = (n + blockSize - 1) / blockSize; // Calculate number of blocks needed

    // Launch the kernel on the GPU
    // Syntax: <<<numBlocks, blockSize>>> specifies grid/block dimensions
    vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);

    // Copy result back from GPU to CPU
    checkCudaError(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost), "Copy to h_c failed");

    // Verify correctness by comparing with CPU result
    for (int i = 0; i < n; i++) {
        float expected = h_a[i] + h_b[i];
        // Check if GPU and CPU results match (allow small floating-point error)
        if (fabs(h_c[i] - expected) > 1e-5) {
            printf("Error at index %d: %f vs %f\n", i, h_c[i], expected);
            break;
        }
    }

    // Free CPU memory
    free(h_a);
    free(h_b);
    free(h_c);

    // Free GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    printf("Vector addition completed successfully!\n");
    return 0;
}