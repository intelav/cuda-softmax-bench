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

#define DEBUG_MODE  1  // <-- change to 1 to re-enable debug prints
#define SINGLE_TEST_MODE  1
#define SINGLE_KERNEL_ID_DEFAULT  3   // which kernel variant to test
#define SINGLE_INPUT_SIZE_DEFAULT (1 << 24)  // 16M elements

#if DEBUG_MODE
    #define DBG_PRINT(...)   printf(__VA_ARGS__)
#else
    #define DBG_PRINT(...)
#endif


#define MAX_BLOCK 64
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
/*!SECTION
two-pass softmax implementation without numerical stabilization (no max(x) subtraction).
Memory-access pattern
Within a warp, thread addresses are now 16 384 floats apart (≈ 64 kB stride).
This destroys coalescing: each warp access causes 32 separate 4-byte transactions instead of one combined 128-byte transaction.
Hardware still caches, but L2/L1 caching gives limited help because each line is used once before skipping far ahead.
Compute overhead
Each thread executes 1024 iterations (N/stride) for 16M, with branch/loop control and dependency on the same registers (local_sum).
Increased instruction count per byte moved → lower effective throughput.
*/
__global__ void softmax_naive_kernel(const float* x, float* y,
                                     int N, float* partialSum) {
    int tid = threadIdx.x;
    //global index
    int idx = blockIdx.x * blockDim.x + tid;
    double local_sum = 0.f;
    
    //Each thread handles multiple elements, spaced out by the full grid size (via grid-stride loop).
    for (int i = idx; i < N; i += gridDim.x * blockDim.x) {
        float val = expf(x[i]); 
        y[i] = val;
        local_sum += val;
    }

    // Store partial sum into dynamic  shared memory
    extern __shared__ float sdata[];
    sdata[tid] = local_sum;
    __syncthreads();

    // Reduce within block, stride based reduction or simple tree-reduction loop
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

    double val = (i < N) ? expf(x[i]) : 0.f;
    //Each thread handles one element only.
    y[i] = val;
    sdata[tid] = val;
    __syncthreads();
     /*!SECTION
     Memory access pattern: contiguous, perfectly coalesced 32-thread warps.
     Each warp issues full-width 128-byte memory transactions → the hardware bus is kept busy.
    */                                   
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
     
    //Each thread handles multiple elements, spaced out by the full grid size (via grid-stride loop).
    for (int i = idx; i < N; i += gridDim.x * blockDim.x) {
        float val = expf(x[i]);
        y[i] = val;
        local_sum += val;
    }

    float sum = warpReduceSum(local_sum);
    __shared__ float warpSum[32];
    //Fast bitwise way to compute threadIdx.x % 32 (warp lane ID)
    //threadIdx.x / 32 is the warp ID within the block
    if ((threadIdx.x & 31) == 0) warpSum[threadIdx.x / 32] = sum;
    __syncthreads();

    
    /*!SECTION
    At most 32 warps per block (because 32×32=1024 threads, the max block size on CUDA).
    combines all the warp-level partial sums into one final block sum.
    blockDim.x / 32 = number of warps in this block.
    */
    float block_sum = 0.f;
    if (threadIdx.x < 32)
        block_sum = warpReduceSum((threadIdx.x < blockDim.x/32) ? warpSum[threadIdx.x] : 0.f);

    if (threadIdx.x == 0)
        partialSum[blockIdx.x] = block_sum;
}
/*if blocks launched are less than N/threads, each thread processes multiple elements in a grid-stride loop.  prvoided grid
stride is implemented . but in this kenrel each thread processes only one element. 
so this kernel is not fully utilizing the grid-stride loop concept.Hence its important to launch enough blocks to cover all elements.
*/

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
//worst numerical stability among all variants due to rounding off error 
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

//best kernel for numerical stability as wel as speed
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

    // store exp(x) directly, tail elements (partial vector near the end), load scalars safely
    //cudamalloc overprovisioned so that out-of-bounds vectorized accesses within a few bytes past the 
    //logical end** (say, 4–16 bytes) don’t fault, hence if N is 10 , baseIdx is 0,4,8 , 
    //all 3 vectors are written even though last one writes 2 extra floats beyond N.

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

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

template <typename T, unsigned int BLOCK_SIZE>
__global__ void softmax_coopgrid_kernel(const T * __restrict__ d_input,
                                        T * __restrict__ d_output,
                                        T * __restrict__ g_max,
                                        T * __restrict__ g_sum,
                                        int N)
{
    // cooperative grid handle
    cg::grid_group grid = cg::this_grid();

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * BLOCK_SIZE + tid;

    extern __shared__ T sdata[];
    T *sdata_max = sdata;
    T *sdata_sum = sdata + BLOCK_SIZE;

    // ------------------------------------------------------------------
    // 1. Local max
    // ------------------------------------------------------------------
    T x = (idx < N) ? d_input[idx] : -INFINITY;
    sdata_max[tid] = x;
    __syncthreads();

    // Block reduction (max)
    #pragma unroll
    for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1)
        if (tid < offset)
            sdata_max[tid] = max(sdata_max[tid], sdata_max[tid + offset]);

    
    //broadcast block max to all threads in the block
    T block_max = sdata_max[0];

    // ------------------------------------------------------------------
    // 2. Global max (atomic reduction)
    // ------------------------------------------------------------------
    if (tid == 0)
        atomicMax((int*)g_max, __float_as_int(block_max));  // assuming float
    grid.sync();

    T global_max = *g_max; // broadcast to all threads
    __syncthreads();

    // ------------------------------------------------------------------
    // 3. Compute exp(x - global_max)
    // ------------------------------------------------------------------
    T exp_val = (idx < N) ? exp(x - global_max) : T(0);
    sdata_sum[tid] = exp_val;
    __syncthreads();


    // Block reduction (sum)
    #pragma unroll
    for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1)
        if (tid < offset)
            sdata_sum[tid] += sdata_sum[tid + offset];

    T block_sum = sdata_sum[0];


    // ------------------------------------------------------------------
    // 4. Global sum (atomic add)
    // ------------------------------------------------------------------
    if (tid == 0)
        atomicAdd(g_sum, block_sum);
    grid.sync();

    T global_sum = *g_sum;
    __syncthreads();

    // ------------------------------------------------------------------
    // 5. Normalize and write
    // ------------------------------------------------------------------
    if (idx < N)
        d_output[idx] = exp_val / global_sum;
}

// ================================================================
// Normalization kernel (same for all variants)
// ================================================================
__global__ void normalize_kernel(float* y, float *totalSum, int N) {
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
                    T *d_partial)
{
    switch (whichKernel)
    {
        // ----------------------------------------------------------
        case 0: {
            DBG_PRINT("  >> Using softmax_naive_kernel\n");
            softmax_naive_kernel<<<blocks, threads, threads * sizeof(T)>>>(
                d_input, d_output, N, d_partial);
            checkCudaErrors(cudaDeviceSynchronize());
            break;
        }

        // ----------------------------------------------------------
        case 1: {
            DBG_PRINT("  >> Using softmax_shared_kernel\n");
            softmax_shared_kernel<<<blocks, threads, threads * sizeof(T)>>>(
                d_input, d_output, N, d_partial);
            checkCudaErrors(cudaDeviceSynchronize());
            break;
        }

        // ----------------------------------------------------------
        case 2: {
            DBG_PRINT("  >> Using softmax_reduce_kernel\n");
            softmax_warp_kernel<<<blocks, threads, 0>>>(
                d_input, d_output, N, d_partial);
            checkCudaErrors(cudaDeviceSynchronize());
            break;
        }

        // ----------------------------------------------------------
        case 3: {
            DBG_PRINT("  >> Using softmax_warp_shared_kernel\n");
            softmax_warp_shared_kernel<<<blocks, threads, threads * sizeof(T)>>>(
                d_input, d_output, N, d_partial);
            checkCudaErrors(cudaDeviceSynchronize());
            break;
        }

        // ----------------------------------------------------------
        case 4: {
            DBG_PRINT("  >> Using softmax_warp_shared_double_kernel\n");
            softmax_warp_shared_double_kernel<<<blocks, threads, threads * sizeof(T)>>>(
                d_input, d_output, N, d_partial);
            checkCudaErrors(cudaDeviceSynchronize());
            break;
        }

        // ----------------------------------------------------------
        case 5: {
            DBG_PRINT("  >> Using softmax_warp_vectorized_kernel\n");
            int vecN = (N + 3) / 4;              // number of float4s
            int threads_vec = threads;
            int blocks_vec  = (vecN + threads_vec - 1) / threads_vec;

            softmax_warp_vectorized_kernel<<<blocks_vec, threads_vec,
                                             threads_vec * sizeof(T)>>>(
                d_input, d_output, N, d_partial);

            checkCudaErrors(cudaDeviceSynchronize());
            break;
        }

        // ----------------------------------------------------------
        case 6: {
                DBG_PRINT("  >> Using softmax_coopgrid_kernel (Cooperative Groups)\n");

                // 1. Use const block size since it's a template parameter
                const int BLOCK_SIZE = 256;
                
                // 2. Create temporary device memory for max and sum
                float *d_max = nullptr;
                float *d_sum = nullptr;
                checkCudaErrors(cudaMalloc(&d_max, sizeof(float)));
                checkCudaErrors(cudaMalloc(&d_sum, sizeof(float)));
                
                // 3. Initialize d_max and d_sum
                float init_max = -INFINITY;
                float init_sum = 0.0f;
                checkCudaErrors(cudaMemcpy(d_max, &init_max, sizeof(float), cudaMemcpyHostToDevice));
                checkCudaErrors(cudaMemcpy(d_sum, &init_sum, sizeof(float), cudaMemcpyHostToDevice));

                // 4. Calculate grid size based on N
                int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
                
                // 5. Check cooperative launch capability
                int dev = 0;
                int supportsCoopLaunch = 0;
                checkCudaErrors(cudaDeviceGetAttribute(&supportsCoopLaunch, 
                    cudaDevAttrCooperativeLaunch, dev));
                
                if (!supportsCoopLaunch) {
                    printf("Error: Device does not support Cooperative Launch\n");
                    break;
                }

                // 6. Launch kernel with proper configuration
                void *args[] = {(void *)&d_input, 
                                (void *)&d_output,
                                (void *)&d_max,
                                (void *)&d_sum,
                                (void *)&N};

                dim3 gridDim(numBlocks);
                dim3 blockDim(BLOCK_SIZE);

                checkCudaErrors(cudaLaunchCooperativeKernel(
                    (void*)softmax_coopgrid_kernel<float, BLOCK_SIZE>,
                    gridDim,
                    blockDim,
                    args,
                    BLOCK_SIZE * sizeof(float),  // shared memory size
                    nullptr));                   // stream

                // 7. Cleanup temporary memory
                checkCudaErrors(cudaFree(d_max));
                checkCudaErrors(cudaFree(d_sum));
                
                checkCudaErrors(cudaDeviceSynchronize());
                break;
            }

        // ----------------------------------------------------------
        default: {
            DBG_PRINT("  >> Invalid kernel ID (%d)\n", whichKernel);
            break;
        }
    } // end switch
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

// ---------------------------------------------------------------
// Benchmark one variant of softmax for given N
// ---------------------------------------------------------------
float benchmarkSoftmax(int N, int threads, int blocks, int whichKernel,
                       float* d_input, float* d_output, float* d_partial) {
    // --- CUDA event-based timing for GPU-only measurement ---
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    // --- Start GPU timing ---
    cudaEventRecord(startEvent);

    // 1️⃣ Launch softmax kernel (computes exp(x[i]) and writes to d_output)
    softmax_launch<float>(N, threads, blocks, whichKernel, d_input, d_output, d_partial);
    checkCudaErrors(cudaDeviceSynchronize());

    // 2️⃣ Reduce per-block partial sums (entirely on GPU)
    int threads_reduce = 256;
    int blocks_reduce = (blocks + threads_reduce - 1) / threads_reduce;
     
    /*!SECTION
    First pass: reduce per-block
    partial sums into d_partial (overwriting the input). 
    Each block processes up to threads_reduce elements from d_partial, writing one output per block.
    */
    reduce_partial_sum_kernel<<<blocks_reduce, threads_reduce,
                                 threads_reduce * sizeof(float)>>>(d_partial, d_partial, blocks);
    checkCudaErrors(cudaDeviceSynchronize());

    /*!SECTION
    Second pass: reduce across blocks if needed (blocks_reduce > 1).Note here that gridsize is 1, 
    so we launch a single block that processes all remaining partial sums in d_partial,
    producing the final total sum in d_partial[0].
    */
    if (blocks_reduce > 1) {
        reduce_partial_sum_kernel<<<1, threads_reduce,
                                     threads_reduce * sizeof(float)>>>(d_partial, d_partial, blocks_reduce);
        checkCudaErrors(cudaDeviceSynchronize());
    }

    // 3️⃣ Normalize output using the final d_partial[0] on GPU
    normalize_kernel<<<blocks, threads>>>(d_output, d_partial, N);
    checkCudaErrors(cudaDeviceSynchronize());

    // --- Stop GPU timing ---
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, startEvent, stopEvent);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    return elapsed_ms;
}


// ---------------------------------------------------------------
// Generate Shmoo table across kernel variants and input sizes
// ---------------------------------------------------------------
int main(int argc, char **argv) {
    printf("=== CUDA Softmax Benchmark (no max(x) normalization) ===\n");
    auto t_start = std::chrono::high_resolution_clock::now();
    uint64_t Nmax;
    //int Nmax = 1 << 24;
    int maxThreads = 256;
    int numVariants = 6;
    int maxBlocks = 64;
  
    if (argc > 1) {
        Nmax = atoll(argv[1]);   // user-specified value
        //printf("[INFO] Using custom Nmax = %llu elements\n", (unsigned long long)Nmax);
        Nmax = nextPow2(Nmax) ;
    } else {
        Nmax = 1ULL << 24;       // default (16 million)
       // printf("[INFO] Using default Nmax = %llu elements\n", (unsigned long long)Nmax);
    }

    //int maxBlocks  = std::min(65535, int((Nmax + maxThreads - 1) / maxThreads));
    size_t bytes = Nmax * sizeof(float);
    std::vector<float> h_input(Nmax);
    
    for (int i = 0; i < Nmax; ++i)
        h_input[i] = sinf(i * 0.001f) + (i % 10) * 0.1f;

    // Allocate device memory
    float *d_input, *d_output, *d_partial;
    checkCudaErrors(cudaMalloc(&d_input, bytes));
    checkCudaErrors(cudaMalloc(&d_output, bytes));
    cudaMalloc(&d_partial, (Nmax / maxThreads + 1) * sizeof(float));
    checkCudaErrors(cudaMemcpy(d_input, h_input.data(), bytes,
                               cudaMemcpyHostToDevice));
    std::vector<float> h_partial(Nmax / maxThreads + 1);
    
#if SINGLE_TEST_MODE
    // ===========================================================
    // 🧪 SINGLE TEST MODE
       //./basesoftmax 16777216 3
    // ===========================================================
    int k = SINGLE_KERNEL_ID_DEFAULT;;
    int n = SINGLE_INPUT_SIZE_DEFAULT;
    
 

    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) k = atoi(argv[2]);
    
    // Round N up to next power of 2 (optional)
    n = nextPow2(n);
    if (k > 6) {
        printf("Invalid kernel variant %d, must be in [0-%d]\n",
               k, numVariants - 1);
        return -1;
    }
    
    int threads = 0, blocks = 0;
    getNumBlocksAndThreads(k, n, maxBlocks, maxThreads, blocks, threads);
    // blocks = MIN(blocks, maxBlocks);
    DBG_PRINT("\n=== SINGLE TEST MODE ENABLED ===\n");
    DBG_PRINT("Variant (kernel): %d\n", k);
    DBG_PRINT("Input size (N):  %d\n", n);
    DBG_PRINT("Threads/block:   %d\n", threads);
    DBG_PRINT("Blocks:          %d\n", blocks);
    DBG_PRINT("=================================\n");

    float time_ms = benchmarkSoftmax(n, threads, blocks, k,
                                     d_input, d_output, d_partial);
    printf("\nSingle test result:\n");
    printf("  Kernel variant: %d\n", k);
    printf("  N = %d, Threads = %d, Blocks = %d, Time = %.5f ms\n",
           n, threads, blocks, time_ms);
    
    // memory accessed for bytes read + written (approx) + memory accessed for partial sums
    double bytes_moved = 2.0 * n * sizeof(float) + blocks * sizeof(float);
    double gbps = (bytes_moved / (time_ms / 1000.0)) / 1e9;

    printf("\nSingle test result:\n");
    printf("  Kernel variant: %d\n", k);
    printf("  N = %d, Threads = %d, Blocks = %d\n", n, threads, blocks);
    printf("  Time = %.5f ms\n", time_ms);
    printf("  Approx. memory throughput = %.2f GB/s\n", gbps);   
    
    std::vector<float> h_out(n);
    cudaMemcpy(h_out.data(), d_output, n*sizeof(float), cudaMemcpyDeviceToHost);

    float sum=0;
    for (int i=0;i<n;i++) sum+=h_out[i];
    printf("Output sum: %.6f\n", sum);

    float denom;
    cudaMemcpy(&denom, d_partial, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Final denominator (sumExp) = %.6f\n", denom);

    // 2. Recompute CPU sum of exp(x) and compare:
    double ref_sum = 0;
    for (int i = 0; i < n; ++i)
        ref_sum += exp(h_input[i]);
    printf("Reference CPU sumExp = %.6f\n", ref_sum);
    printf("Relative error = %.6f%%\n",
        fabs(ref_sum - denom) / ref_sum * 100.0);
#else
    // ===========================================================
    // 🧮 FULL BENCHMARK MODE
    // ===========================================================
    printf("Variant");
    for (int n = 1<<10; n <= Nmax; n <<= 1)
        printf(", %d", n);
    printf("\n");

    for (int k = 0; k < numVariants; ++k) {
        printf("%d", k);
        for (int n = 1<<10; n <= Nmax; n <<= 1) {
            int threads = 0, blocks = 0;
            getNumBlocksAndThreads(k, n, maxBlocks, maxThreads,
                                   blocks, threads);
            float time_ms = benchmarkSoftmax(n, threads, blocks, k,
                                             d_input, d_output, d_partial);
            printf(", %.5f", time_ms);
        }
        printf("\n");
    }
#endif

    // // ---- Validate correctness for final variant ----
    // int Ntest = std::min<uint64_t>(Nmax, 1ULL << 20);
    // std::vector<float> h_out(Ntest), h_ref(Ntest);
    // checkCudaErrors(cudaMemcpy(h_out.data(), d_output,
    //                            Ntest*sizeof(float), cudaMemcpyDeviceToHost));
    // softmax_cpu<float>(h_input.data(), h_ref.data(), Ntest);

    // float max_err = 0.f;
    // for (int i=0;i<Ntest;i++)
    //     max_err = fmaxf(max_err, fabsf(h_out[i] - h_ref[i]));
    // printf("\nVerification max abs error: %.6e\n", max_err);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_partial);
    printf("✅ Benchmark completed.\n");
    
    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    printf("⏱️  Total runtime (end-to-end): %.3f ms\n", total_ms);
    return 0;
}
