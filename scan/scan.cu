#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include <algorithm>

#include "CycleTimer.h"

extern float toBW(int bytes, float sec);


/* Helper function to round up to a power of 2. 
 */
static inline int nextPow2(int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

///correct only when N <= 512(threadsPerBlock)
__global__ void exclusive_scan_kernel(int* start, int N, int* output, int P2) {
    // compute overall index from position of thread in current block,
    // and given the block we are in
    unsigned int block_index = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    unsigned int thread_index = block_index * blockDim.x * blockDim.y * blockDim.z + \
        threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    unsigned int index = thread_index;//blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= P2) return ;
    // upsweep phase
    for (int twod = 1; twod < P2; twod*=2) {
        int stride = twod*2;
        if ((index+1-stride) % stride == 0)
            output[index] += output[index-stride+twod];
        __syncthreads();
    }
    if (index == P2 - 1) output[index] = 0;
    __syncthreads();
    // downsweep phase
    for (int twod = P2/2; twod >= 1; twod /= 2) {
        int stride = twod*2;
        if ((index+1-stride) % stride == 0) {
            int tmp = output[index-stride+twod];
            output[index-stride+twod] = output[index];
            output[index] += tmp;
        }
        __syncthreads();
    }
}

__global__ void exclusive_scan_upsweep(int* start, int N, int* output) {
    // compute overall index from position of thread in current block, and given the block we are in
    unsigned int block_index = blockIdx.x;
    unsigned int thread_index = block_index * blockDim.x + threadIdx.x;
    unsigned int index = thread_index;
    unsigned int P2 = 512;///blockDim.x: #threads per block
    if (index >= N) return ;
    ///synchronize all threads in block between every step 
    for (int twod = 1; twod < P2; twod*=2) {
        int stride = twod*2;
        if ((index+1-stride - block_index*P2) % stride == 0)
            output[index] += output[index-stride+twod];
        __syncthreads();
    }
    if (index == N-1) output[index] = 0;
    __syncthreads();
}

__global__ void exclusive_scan_merge(int* start, int N, int* output) {
    unsigned int P2 = 512;///blockDim.x: #threads per block
    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int index = (thread_index+1)*P2 - 1;
    if (index >= N) return ;
    ///merge last elements from all blocks
    for (int twod = P2; twod < N; twod*=2) {
        int stride = twod*2;
        if ((index+1-stride) % stride == 0)
            output[index] += output[index-stride+twod];
        __syncthreads();
    }
    if (index == N-1) output[index] = 0;
    __syncthreads();
    ///exchange last elements from all blocks
    for (int twod = N/2; twod >= P2; twod /= 2) {
        int stride = twod*2;
        if ((index+1-stride) % stride == 0) {
            int tmp = output[index-stride+twod];
            output[index-stride+twod] = output[index];
            output[index] += tmp;
        }
        __syncthreads();
    }
}

__global__ void exclusive_scan_downsweep(int* start, int N, int* output) {
    unsigned int block_index = blockIdx.x;
    unsigned int thread_index = block_index * blockDim.x + threadIdx.x;
    unsigned int index = thread_index;
    unsigned int P2 = 512;///blockDim.x: #threads per block
    if (index >= N) return ;
    ///downsweep within every block
    for (int twod = P2/2; twod >= 1; twod /= 2) {
        int stride = twod*2;
        if ((index+1-stride - block_index*P2) % stride == 0) {
            int tmp = output[index-stride+twod];
            output[index-stride+twod] = output[index];
            output[index] += tmp;
        }
        __syncthreads();
    }
}

void exclusive_scan(int* device_start, int length, int* device_result)
{
    /* Fill in this function with your exclusive scan implementation.
     * You are passed the locations of the input and output in device memory,
     * but this is host code -- you will need to declare one or more CUDA 
     * kernels (with the __global__ decorator) in order to actually run code
     * in parallel on the GPU.
     * Note you are given the real length of the array, but may assume that
     * both the input and the output arrays are sized to accommodate the next
     * power of 2 larger than the input.
     */
    int threadsPerBlock = 512;
    int nextP2 = nextPow2(length);
    int blocksPerGrid = (nextP2 + threadsPerBlock - 1) / threadsPerBlock;
    if (length <= threadsPerBlock) {
        exclusive_scan_kernel<<<blocksPerGrid, threadsPerBlock>>>(device_start, length, device_result, nextP2);
    } else {
        ///upsweep phase : no problem to parallelize all blocks in a grid
        exclusive_scan_upsweep<<<blocksPerGrid, threadsPerBlock>>>(device_start, nextP2, device_result);
        ///merge phase
        assert(blocksPerGrid <= threadsPerBlock);///512*512 = 110080
        exclusive_scan_merge<<<1, blocksPerGrid>>>(device_start, nextP2, device_result);
        ///downsweep phase
        exclusive_scan_downsweep<<<blocksPerGrid, threadsPerBlock>>>(device_start, nextP2, device_result);
    }
}

/* This function is a wrapper around the code you will write - it copies the
 * input to the GPU and times the invocation of the exclusive_scan() function
 * above. You should not modify it.
 */
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input; 
    // We round the array sizes up to a power of 2, but elements after
    // the end of the original input are left uninitialized and not checked
    // for correctness. 
    // You may have an easier time in your implementation if you assume the 
    // array's length is a power of 2, but this will result in extra work on
    // non-power-of-2 inputs.
    int rounded_length = nextPow2(end - inarray);
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), 
               cudaMemcpyHostToDevice);

    // For convenience, both the input and output vectors on the device are
    // initialized to the input values. This means that you are free to simply
    // implement an in-place scan on the result vector if you wish.
    // If you do this, you will need to keep that fact in mind when calling
    // exclusive_scan from find_repeats.
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), 
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, end - inarray, device_result);

    // Wait for any work left over to be completed.
    cudaThreadSynchronize();
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;
    
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int),
               cudaMemcpyDeviceToHost);
// for (int i = 0; i < end - inarray; ++i) printf("%d ", resultarray[i]);
// printf("\n");
    return overallDuration;
}

/* Wrapper around the Thrust library's exclusive scan function
 * As above, copies the input onto the GPU and times only the execution
 * of the scan itself.
 * You are not expected to produce competitive performance to the
 * Thrust version.
 */
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), 
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaThreadSynchronize();
    double endTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int),
               cudaMemcpyDeviceToHost);
    thrust::device_free(d_input);
    thrust::device_free(d_output);
    double overallDuration = endTime - startTime;
    return overallDuration;
}

__global__ void find_repeats_kernel(int* start, int N, int* output, int* counter) {
    // compute overall index from position of thread in current block,
    // and given the block we are in
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N - 1) return ;

    if (start[index] == start[index+1]) {
        int tc = atomicAdd(counter, 1);
        output[tc] = index;
    }
}

int find_repeats(int *device_input, int length, int *device_output) {
    /* Finds all pairs of adjacent repeated elements in the list, storing the
     * indices of the first element of each pair (in order) into device_result.
     * Returns the number of pairs found.
     * Your task is to implement this function. You will probably want to
     * make use of one or more calls to exclusive_scan(), as well as
     * additional CUDA kernel launches.
     * Note: As in the scan code, we ensure that allocated arrays are a power
     * of 2 in size, so you can use your exclusive_scan function with them if 
     * it requires that. However, you must ensure that the results of
     * find_repeats are correct given the original length.
     */
    int threadsPerBlock = 512;
    int blocks = (length + threadsPerBlock - 1) / threadsPerBlock;
    // ///compute exclusive_scan
    // exclusive_scan_kernel<<<blocks, threadsPerBlock>>>(device_start, length, device_result);
    int* counter;
    cudaMalloc((void**)&counter, sizeof(int));
    ///find_repeat
    find_repeats_kernel<<<blocks, threadsPerBlock>>>(device_input, length, device_output, counter);
    int c;
    cudaMemcpy(&c, counter, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(counter);
    return c;
}

/* Timing wrapper around find_repeats. You should not modify this function.
 */
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {
    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), 
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    cudaThreadSynchronize();
    double endTime = CycleTimer::currentSeconds();

    *output_length = result;

    cudaMemcpy(output, device_output, length * sizeof(int),
               cudaMemcpyDeviceToHost);
std::sort(output, output + result);
    cudaFree(device_input);
    cudaFree(device_output);

    return endTime - startTime;
}

void printCudaInfo()
{
    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}
