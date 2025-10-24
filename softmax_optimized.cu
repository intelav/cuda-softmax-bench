// ==========================================================
// softmax_kernel.cu — Multiple CUDA kernel variants
// ==========================================================

#include <cuda_runtime.h>
#include <cstdio>
#include <math.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <float.h>
#include <chrono>

#define DEBUG_MODE  0  // <-- change to 1 to re-enable debug prints

#if DEBUG_MODE
    #define DBG_PRINT(...)   printf(__VA_ARGS__)
#else
    #define DBG_PRINT(...)
#endif


#define MAX_BLOCK_DIM_SIZE 65535
#ifndef MIN
#define MIN(x,y) ((x)<(y)?(x):(y))
#endif

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

// ------------------------------------------------------------------
// Utility: warp-level reduction for fast sum using shuffle
// ------------------------------------------------------------------
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__inline__ __device__ float warpReduceMax(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset));
    return val;
}


// ================================================================
// Variant 0 — Naive softmax (global memory only)
// Each thread computes exp(x[i]) and accumulates partial sums
// ================================================================
__global__ void softmax_naive_kernel(const float* x, float* y,
                                     int N, float* partialSum) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float local_sum = 0.f;

    for (int i = idx; i < N; i += gridDim.x * blockDim.x) {
        float val = expf(x[i]);  // No numerical stabilization
        y[i] = val;
        local_sum += val;
    }

    // Store partial sum into shared memory
    __shared__ float sdata[256];
    sdata[tid] = local_sum;
    __syncthreads();

    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
        if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();

    if (tid == 0)
        partialSum[blockIdx.x] = sdata[0];
}

// ================================================================
// Variant 1 — Shared-memory reduction with better coalescing
// ================================================================
__global__ void softmax_shared_kernel(const float* x, float* y,
                                      int N, float* partialSum) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    float val = (i < N) ? expf(x[i]) : 0.f;
    y[i] = val;
    sdata[tid] = val;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
        if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();

    if (tid == 0)
        partialSum[blockIdx.x] = sdata[0];
}

// ================================================================
// Variant 2 — Warp-shuffle reduction (fastest)
// ================================================================
__global__ void softmax_warp_kernel(const float* x, float* y,
                                    int N, float* partialSum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float local_sum = 0.f;

    for (int i = idx; i < N; i += gridDim.x * blockDim.x) {
        float val = expf(x[i]);
        y[i] = val;
        local_sum += val;
    }

    float sum = warpReduceSum(local_sum);
    __shared__ float warpSum[32];
    if ((threadIdx.x & 31) == 0) warpSum[threadIdx.x / 32] = sum;
    __syncthreads();

    float block_sum = 0.f;
    if (threadIdx.x < 32)
        block_sum = warpReduceSum((threadIdx.x < blockDim.x/32) ? warpSum[threadIdx.x] : 0.f);

    if (threadIdx.x == 0)
        partialSum[blockIdx.x] = block_sum;
}

__global__ void softmax_warp_shared_kernel(const float* __restrict__ x,
                                         float* __restrict__ y,
                                         int N,
                                         float* __restrict__ partialSum) {
    // Shared memory for one float per warp
    __shared__ float warpSums[32];  // enough for up to 1024 threads

    int tid  = threadIdx.x;
    int gid  = blockIdx.x * blockDim.x + tid;

    // --- 1. Coalesced load: one element per thread
    float val = 0.f;
    if (gid < N) {
        val = expf(x[gid]);  // expf is heavy, dominates runtime
        y[gid] = val;
    }

    // --- 2. Warp-level reduction of val across threads in the same warp
    float local_sum = warpReduceSum(val);

    // --- 3. Lane 0 of each warp writes its warp sum to shared memory
    int warpId = tid / warpSize;
    if ((tid & 31) == 0)
        warpSums[warpId] = local_sum;
    __syncthreads();

    // --- 4. Warp 0 reduces all warp sums
    float block_sum = 0.f;
    if (warpId == 0) {
        float warp_val = (tid < blockDim.x / warpSize) ? warpSums[tid] : 0.f;
        block_sum = warpReduceSum(warp_val);
    }

    // --- 5. Store per-block sum
    if (tid == 0)
        partialSum[blockIdx.x] = block_sum;
}

__global__ void softmax_warp_shared_double_kernel(const float* x, float* y, int N, float* partialSum) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + tid;

    float mySum = 0.f;
    if (i < N) mySum = expf(x[i]);
    if (i + blockDim.x < N) mySum += expf(x[i + blockDim.x]);
    y[i] = mySum;
    sdata[tid] = mySum;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Warp-level reduction (unrolled)
    if (tid < 32) {
        float val = sdata[tid];
        for (int offset = 16; offset > 0; offset /= 2)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (tid == 0)
            partialSum[blockIdx.x] = val;
    }
}

__global__ void softmax_warp_vectorized_kernel(const float *__restrict__ x,
                                                     float *__restrict__ y,
                                                     int N,
                                                     float *__restrict__ partialSum) {
    extern __shared__ float warpBuf[];
    int tid   = threadIdx.x;
    int lane  = tid % WARP_SIZE;
    int warpId = tid / WARP_SIZE;

    int vecIndex = blockIdx.x * blockDim.x + tid;
    int baseIdx  = vecIndex * 4;
    const float4 *x4 = reinterpret_cast<const float4 *>(x);

    float4 val4 = {0.f, 0.f, 0.f, 0.f};
    float local_sum = 0.f;

    int numVec = N / 4;

    // vectorized load (or tail-safe scalar)
    if (vecIndex < numVec) {
        val4 = x4[vecIndex];
    } else if (baseIdx < N) {
        val4.x = (baseIdx + 0 < N) ? x[baseIdx + 0] : -FLT_MAX;
        val4.y = (baseIdx + 1 < N) ? x[baseIdx + 1] : -FLT_MAX;
        val4.z = (baseIdx + 2 < N) ? x[baseIdx + 2] : -FLT_MAX;
        val4.w = (baseIdx + 3 < N) ? x[baseIdx + 3] : -FLT_MAX;
    }

    // compute exp(x)
    float4 exp4;
    exp4.x = expf(val4.x);
    exp4.y = expf(val4.y);
    exp4.z = expf(val4.z);
    exp4.w = expf(val4.w);
    local_sum = exp4.x + exp4.y + exp4.z + exp4.w;

    // store exp(x) directly
    if (baseIdx < N)
        reinterpret_cast<float4 *>(y)[vecIndex] = exp4;

    // reduce for sum
    float warp_sum = warpReduceSum(local_sum);
    if (lane == 0) warpBuf[warpId] = warp_sum;
    __syncthreads();

    float block_sum = 0.f;
    if (warpId == 0) {
        float v = (tid < blockDim.x / WARP_SIZE) ? warpBuf[lane] : 0.f;
        float tmp = warpReduceSum(v);
        if (lane == 0) warpBuf[0] = tmp;
    }
    __syncthreads();
    block_sum = warpBuf[0];

    if (tid == 0)
        partialSum[blockIdx.x] = block_sum;
}


// ================================================================
// Normalization kernel (same for all variants)
// ================================================================
__global__ void normalize_kernel(float* y, const float *totalSum, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < N; i += gridDim.x * blockDim.x)
        y[i] /= *totalSum;
}

// ================================================================
// Kernel dispatcher
// ================================================================
template <class T>
void softmax_launch(int N, int threads, int blocks,
                    int whichKernel, T *d_input, T *d_output,
                    T *d_partial) {
    switch (whichKernel) {
        default:
        case 0:
            softmax_naive_kernel<<<blocks, threads, threads * sizeof(T)>>>(
                d_input, d_output, N, d_partial);
            break;
        case 1:
            softmax_shared_kernel<<<blocks, threads, threads * sizeof(T)>>>(
                d_input, d_output, N, d_partial);
            break;
        case 2:
            softmax_warp_kernel<<<blocks, threads, 0>>>(
                d_input, d_output, N, d_partial);
            break;
        case 3: 
            softmax_warp_shared_kernel<<<blocks, threads, threads * sizeof(T)>>>(
                d_input, d_output, N, d_partial);    
            break;  
        case 4: 
            softmax_warp_shared_double_kernel<<<blocks, threads, threads * sizeof(T)>>>(
                d_input, d_output, N, d_partial);    
            break;  
        case 5: 
            int vecN = (N + 3) / 4;  // number of float4s
            int threads_vec = threads;
            int blocks_vec  = (vecN + threads_vec - 1) / threads_vec;
            softmax_warp_vectorized_kernel<<<blocks_vec, threads_vec, threads_vec * sizeof(T)>>>(
                d_input, d_output, N, d_partial);   

            break;    
    }
}

// // Explicit instantiation
// template void softmax_launch<float>(int, int, int, int, float*, float*, float*);

// ================================================================
// CPU reference softmax (for correctness verification)
// ================================================================
template <class T>
void softmax_cpu(const T* x, T* y, int N) {
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        y[i] = exp(x[i]);
        sum += y[i];
    }
    for (int i = 0; i < N; ++i)
        y[i] /= sum;
}

template void softmax_cpu<float>(const float*, float*, int);

// Compute next power of 2 (used to tune thread count)
unsigned int nextPow2(unsigned int x) {
    --x; x |= x >> 1; x |= x >> 2; x |= x >> 4;
    x |= x >> 8; x |= x >> 16; return ++x;
}

// ---------------------------------------------------------------
// Utility: compute blocks/threads configuration (like reduction sample)
// ---------------------------------------------------------------
void  getNumBlocksAndThreads(int kernel, int N, int maxBlocks,
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


// ================================================================
// Device reduction kernel: reduce array of block partial sums
// ================================================================
__global__ void reduce_partial_sum_kernel(const float* __restrict__ d_partial,
                                          float* __restrict__ d_total,
                                          int numBlocks)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load data into shared memory
    float val = (idx < numBlocks) ? d_partial[idx] : 0.f;
    sdata[tid] = val;
    __syncthreads();

    // Parallel reduction within the block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    // Write one partial sum per block
    if (tid == 0)
        d_total[blockIdx.x] = sdata[0];
}

// ======================================================================
//  Optimized main() for softmax benchmark
//  ---------------------------------------------------------------
//  Key changes:
//   1. Allocate GPU buffers once and reuse across all kernel variants.
//   2. Copy input to GPU once at start, copy output back once at end.
//   3. Use CUDA events to time only kernel execution.
//   4. Chain kernels on device without unnecessary host syncs.
// ======================================================================

int main(int argc, char **argv)
{
    printf("=== CUDA Softmax Benchmark (Optimized, GPU-resident) ===\n");
    auto t_start = std::chrono::high_resolution_clock::now();
    // -----------------------------------------------------------
    // 1. Configuration parameters
    // -----------------------------------------------------------
    uint64_t Nmax = (argc > 1) ? nextPow2(atoll(argv[1])) : (1ULL << 24);  // e.g., 16M elements
    const int maxThreads = 256;    // per block
    const int maxBlocks  = 64;     // limited for benchmark
    const int numVariants = 6;     // total kernel variants to test

    // Host buffer allocation (input + output + partials)
    size_t bytes = Nmax * sizeof(float);
    std::vector<float> h_input(Nmax);
    std::vector<float> h_output(Nmax);
    
    int maxPossibleBlocks = (Nmax + maxThreads - 1) / maxThreads;
    std::vector<float> h_partial(maxPossibleBlocks);
    // Initialize input with deterministic pattern
    for (size_t i = 0; i < Nmax; ++i)
        h_input[i] = sinf(i * 0.001f) + (i % 10) * 0.1f;

    // -----------------------------------------------------------
    // 2. Allocate GPU memory ONCE and reuse
    // -----------------------------------------------------------
    float *d_input = nullptr, *d_output = nullptr, *d_partial = nullptr;
    checkCudaErrors(cudaMalloc(&d_input,  bytes));
    checkCudaErrors(cudaMalloc(&d_output, bytes));
    checkCudaErrors(cudaMalloc(&d_partial, maxPossibleBlocks * sizeof(float)));

    // Copy input data once to device
    checkCudaErrors(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));

    // -----------------------------------------------------------
    // 3. Create CUDA events for precise timing of GPU kernels only
    // -----------------------------------------------------------
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    // CSV header for timing results
    printf("Variant");
    for (int n = (1 << 10); n <= Nmax; n <<= 1)
        printf(", %d", n);
    printf("\n");

    // -----------------------------------------------------------
    // 4. Benchmark loop for all variants, keeping data resident
    // -----------------------------------------------------------
    for (int variant = 0; variant < numVariants; ++variant) {
        printf("%d", variant);

        for (int n = (1 << 10); n <= Nmax; n <<= 1) {
            int threads = 0, blocks = 0;
            getNumBlocksAndThreads(variant, n, maxBlocks, maxThreads, blocks, threads);

            // // Reset partial sums (optional; not necessary unless kernel assumes zero)
            // checkCudaErrors(cudaMemset(d_partial, 0, blocks * sizeof(float)));

            // ---------------------------------------------------
            // 5. Start timing just before kernel launch
            // ---------------------------------------------------
            cudaEventRecord(startEvent);

            // Launch the selected softmax kernel variant (pure GPU compute)
            softmax_launch<float>(n, threads, blocks, variant,
                                  d_input, d_output, d_partial);

            // // Synchronize device once per variant to ensure kernel finished
            // checkCudaErrors(cudaDeviceSynchronize());

            // --- 3️⃣ Reduce per-block partial sums to one total sum (on GPU)
            int threads_reduce = 256;
            int blocks_reduce  = (blocks + threads_reduce - 1) / threads_reduce;
            reduce_partial_sum_kernel<<<blocks_reduce, threads_reduce,
                                        threads_reduce * sizeof(float)>>>(d_partial, d_partial, blocks);
            // checkCudaErrors(cudaDeviceSynchronize());
            
            if (blocks_reduce > 1) {
            reduce_partial_sum_kernel<<<1, threads_reduce,
                                        threads_reduce * sizeof(float)>>>(d_partial, d_partial, blocks_reduce);
            // checkCudaErrors(cudaDeviceSynchronize());
            }

            // --- 5️⃣ Now d_partial[0] holds total sum; use directly in normalize_kernel
            normalize_kernel<<<blocks, threads>>>(d_output, d_partial, n);
            checkCudaErrors(cudaDeviceSynchronize());

            // ---------------------------------------------------
            // 6. Stop timing after normalization
            // ---------------------------------------------------
            cudaEventRecord(stopEvent);
            cudaEventSynchronize(stopEvent);

            float ms = 0.f;
            cudaEventElapsedTime(&ms, startEvent, stopEvent);

            // Print per-variant timing result
            printf(", %.5f", ms);
        }
        printf("\n");
    }

    // // -----------------------------------------------------------
    // // 7. Copy final output back once (for verification)
    // // -----------------------------------------------------------
    // checkCudaErrors(cudaMemcpy(h_output.data(), d_output, bytes, cudaMemcpyDeviceToHost));

    // // -----------------------------------------------------------
    // // 8. Validate correctness vs CPU reference
    // // -----------------------------------------------------------
    // const int Ntest = std::min<uint64_t>(Nmax, 1ULL << 20);
    // std::vector<float> h_ref(Ntest);
    // softmax_cpu<float>(h_input.data(), h_ref.data(), Ntest);

    // float max_err = 0.f;
    // for (int i = 0; i < Ntest; i++)
    //     max_err = fmaxf(max_err, fabsf(h_output[i] - h_ref[i]));

    // printf("\nVerification max abs error: %.6e\n", max_err);

    // -----------------------------------------------------------
    // 9. Cleanup (device + events)
    // -----------------------------------------------------------
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_partial);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    printf("✅ GPU-resident benchmark completed successfully.\n");
    
    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    printf("⏱️  Total runtime (end-to-end): %.3f ms\n", total_ms);
    return 0;
}