// ==========================================================
// softmax.cpp — Main benchmark + shmoo performance graph
// ==========================================================

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include <vector>
#include <cmath>
#include "softmax.h"

#define MAX_BLOCK_DIM_SIZE 65535
#ifndef MIN
#define MIN(x,y) ((x)<(y)?(x):(y))
#endif

// Compute next power of 2 (used to tune thread count)
unsigned int nextPow2(unsigned int x) {
    --x; x |= x >> 1; x |= x >> 2; x |= x >> 4;
    x |= x >> 8; x |= x >> 16; return ++x;
}

// ---------------------------------------------------------------
// Utility: compute blocks/threads configuration (like reduction sample)
// ---------------------------------------------------------------
void getNumBlocksAndThreads(int kernel, int N, int maxBlocks,
                            int maxThreads, int &blocks, int &threads) {
    cudaDeviceProp prop; int dev;
    checkCudaErrors(cudaGetDevice(&dev));
    checkCudaErrors(cudaGetDeviceProperties(&prop, dev));

    threads = (N < maxThreads) ? nextPow2(N) : maxThreads;
    blocks  = (N + threads - 1) / threads;

    if (blocks > prop.maxGridSize[0]) {
        blocks = prop.maxGridSize[0];
    }
}

// ---------------------------------------------------------------
// Benchmark one variant of softmax for given N
// ---------------------------------------------------------------
float benchmarkSoftmax(int N, int threads, int blocks, int whichKernel,
                       float* d_input, float* d_output,
                       float* d_partial, float* h_partial) {

    StopWatchInterface *timer = nullptr;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    // --- Run kernel ---
    sdkStartTimer(&timer);
    softmax_launch<float>(N, threads, blocks, whichKernel,
                          d_input, d_output, d_partial);
    cudaDeviceSynchronize();

    // --- Reduce partial sums on host ---
    checkCudaErrors(cudaMemcpy(h_partial, d_partial,
                               blocks * sizeof(float),
                               cudaMemcpyDeviceToHost));

    float sum_total = 0.f;
    for (int i = 0; i < blocks; ++i)
        sum_total += h_partial[i];

    // Normalize on device
    normalize_kernel<<<blocks, threads>>>(d_output, sum_total, N);
    cudaDeviceSynchronize();

    sdkStopTimer(&timer);
    float time_ms = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);

    return time_ms;
}

// ---------------------------------------------------------------
// Generate Shmoo table across kernel variants and input sizes
// ---------------------------------------------------------------
int main(int argc, char **argv) {
    printf("=== CUDA Softmax Benchmark (no max(x) normalization) ===\n");
    int Nmax = 1 << 24;
    int maxThreads = 256, maxBlocks = 64;
    int numVariants = 3;

    size_t bytes = Nmax * sizeof(float);
    std::vector<float> h_input(Nmax);
    for (int i = 0; i < Nmax; ++i)
        h_input[i] = sinf(i * 0.001f) + (i % 10) * 0.1f;

    // Allocate device memory
    float *d_input, *d_output, *d_partial;
    checkCudaErrors(cudaMalloc(&d_input, bytes));
    checkCudaErrors(cudaMalloc(&d_output, bytes));
    checkCudaErrors(cudaMalloc(&d_partial, maxBlocks * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_input, h_input.data(), bytes,
                               cudaMemcpyHostToDevice));
    std::vector<float> h_partial(maxBlocks);

    // CSV header
    printf("Variant");
    for (int n = 1<<10; n <= 1<<24; n <<= 1)
        printf(", %d", n);
    printf("\n");

    // Benchmark each variant
    for (int k = 0; k < numVariants; ++k) {
        printf("%d", k);
        for (int n = 1<<10; n <= 1<<24; n <<= 1) {
            int threads = 0, blocks = 0;
            getNumBlocksAndThreads(k, n, maxBlocks, maxThreads,
                                   blocks, threads);
            float time_ms = benchmarkSoftmax(n, threads, blocks, k,
                                             d_input, d_output,
                                             d_partial, h_partial.data());
            printf(", %.5f", time_ms);
        }
        printf("\n");
    }

    // ---- Validate correctness for final variant ----
    int Ntest = 1<<20;
    std::vector<float> h_out(Ntest), h_ref(Ntest);
    checkCudaErrors(cudaMemcpy(h_out.data(), d_output,
                               Ntest*sizeof(float), cudaMemcpyDeviceToHost));
    softmax_cpu<float>(h_input.data(), h_ref.data(), Ntest);

    float max_err = 0.f;
    for (int i=0;i<Ntest;i++)
        max_err = fmaxf(max_err, fabsf(h_out[i] - h_ref[i]));
    printf("\nVerification max abs error: %.6e\n", max_err);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_partial);
    printf("✅ Benchmark completed.\n");
    return 0;
}
