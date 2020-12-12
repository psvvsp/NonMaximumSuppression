#include "NMS_gpu.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

#ifndef max
#define max(x, y) (((x) > (y))?(x):(y))
#endif

#ifndef min
#define min(x, y) (((x) < (y))?(x):(y))
#endif

__device__
inline integer area(const Box& b)
{
    return (b.right - b.left + 1) * (b.bottom - b.top + 1);
}

__device__
inline real iou_device(const Box& b1, const Box& b2)
{
    Box intersection;

    intersection.left = max(b1.left, b2.left);
    intersection.right = min(b1.right, b2.right);

    if (intersection.left > intersection.right)
        return real(0.0);

    intersection.top = max(b1.top, b2.top);
    intersection.bottom = min(b1.bottom, b2.bottom);

    if (intersection.top > intersection.bottom)
        return real(0.0);

    integer intersection_area = area(intersection);
    integer union_area = area(b1) + area(b2) - intersection_area;

    return real(intersection_area) / real(union_area);
}

//__global__ void NmsKernel(const Box *boxes, real threshold, char *result)
//{
//    extern __shared__ Box boxesShared[];
//    char *resultShared = (char *)(boxesShared + blockDim.x);
//
//    unsigned j = threadIdx.x;
//    boxesShared[j] = boxes[j];
//    resultShared[j] = 0;
//
//    __syncthreads();
//
//    for (unsigned i = 0; i < blockDim.x; i++) {
//        if (resultShared[i]) continue;
//        if (j > i && resultShared[j] == 0 && iou_device(boxesShared[i], boxesShared[j]) >= threshold) {
//            resultShared[j] = 1;
//        }
//        __syncthreads();
//    }
//
//    result[j] = resultShared[j];
//}

__global__ void NmsTriangle(const Box *boxes, real threshold, char *result, unsigned i0, unsigned size)
{
    unsigned j = i0 + threadIdx.x;
    if (j >= size) return;
 
    extern __shared__ Box boxesShared[];
    char *resultShared = (char *)(boxesShared + blockDim.x);

    boxesShared[threadIdx.x] = boxes[j];
    resultShared[threadIdx.x] = result[j];

    __syncthreads();

    for (unsigned i = 0; i < min(blockDim.x, size); i++) {
        if (resultShared[i]) continue;
        if (threadIdx.x > i && resultShared[threadIdx.x] == 0 && iou_device(boxesShared[i], boxesShared[threadIdx.x]) >= threshold) {
            resultShared[threadIdx.x] = 1;
        }
        __syncthreads();
    }

    result[j] = resultShared[threadIdx.x];
}

__global__ void NmsRectangle(const Box* boxes, real threshold, char* result, unsigned i0, unsigned size)
{
    unsigned i = i0 + threadIdx.x;
    if (i >= size) return;
    
    Box box = boxes[i];
    bool res = false;

    unsigned j0 = blockIdx.x * blockDim.x;

    for (unsigned j = j0; j < min(j0 + blockDim.x, size); j++) {
        if (result[j]) continue;
        if (iou_device(box, boxes[j]) >= threshold)
            res = true;
    }

    if (res)
        result[i] = 1;
}

//int main()
//{
//    const int arraySize = 5;
//    const int a[arraySize] = { 1, 2, 3, 4, 5 };
//    const int b[arraySize] = { 10, 20, 30, 40, 50 };
//    int c[arraySize] = { 0 };
//
//    // Add vectors in parallel.
//    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addWithCuda failed!");
//        return 1;
//    }
//
//    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
//        c[0], c[1], c[2], c[3], c[4]);
//
//    // cudaDeviceReset must be called before exiting in order for profiling and
//    // tracing tools such as Nsight and Visual Profiler to show complete traces.
//    cudaStatus = cudaDeviceReset();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceReset failed!");
//        return 1;
//    }
//
//    return 0;
//}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

bool nms_gpu(
    const std::vector<Box>& boxesIn,
    const std::vector<real>& scoresIn,
    real threshold,
    std::vector<Box>& boxesOut,
    std::vector<real>& scoresOut)
{
    Box* boxesInBuffer = nullptr;
    real* scoresInBuffer = nullptr;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&boxesInBuffer, boxesIn.size() * sizeof(Box));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&scoresInBuffer, scoresIn.size() * sizeof(real));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(boxesInBuffer, boxesIn.data(), boxesIn.size() * sizeof(Box), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(scoresInBuffer, scoresIn.data(), scoresIn.size() * sizeof(real), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    //// Launch a kernel on the GPU with one thread for each element.
    //addKernel << <1, size >> > (dev_c, dev_a, dev_b);

    //// Check for any errors launching the kernel
    //cudaStatus = cudaGetLastError();
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    //    goto Error;
    //}

    //// cudaDeviceSynchronize waits for the kernel to finish, and returns
    //// any errors encountered during the launch.
    //cudaStatus = cudaDeviceSynchronize();
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
    //    goto Error;
    //}

    // Copy output vector from GPU buffer to host memory.
    //cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaMemcpy failed!");
    //    goto Error;
    //}

Error:
    cudaFree(boxesInBuffer);
    cudaFree(scoresInBuffer);

    return cudaStatus;
}

NMS_gpu::NMS_gpu(size_t boxesCountMax)
    : m_boxesInCPU(nullptr)
    , m_boxesInGPU(nullptr)
    , m_scoresInGPU(nullptr)
    , m_resultGPU(nullptr)
    , m_resultCPU(nullptr)
    , m_boxesCountMax(boxesCountMax)
{
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }

    cudaStatus = cudaMallocHost(
        (void**)&m_boxesInCPU, m_boxesCountMax * sizeof(Box));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocHost failed!");
    }

    cudaStatus = cudaMalloc(
        (void**)&m_boxesInGPU, m_boxesCountMax * sizeof(Box));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    cudaStatus = cudaMalloc(
        (void**)&m_scoresInGPU, m_boxesCountMax * sizeof(real));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    cudaStatus = cudaMalloc(
        (void**)&m_resultGPU, m_boxesCountMax * sizeof(char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }

    cudaStatus = cudaMallocHost(
        (void**)&m_resultCPU, m_boxesCountMax * sizeof(char));

}

NMS_gpu::~NMS_gpu()
{
    cudaFreeHost(m_resultCPU);
    cudaFreeHost(m_boxesInCPU);
    cudaFree(m_boxesInGPU);
    cudaFree(m_scoresInGPU);
    cudaFree(m_resultGPU);
}

namespace {

    struct Record
    {
        Box box;
        real score;
    };
}

bool NMS_gpu::doIt(
    const std::vector<Box>& boxesIn,
    const std::vector<real>& scoresIn,
    real threshold,
    std::vector<Box>& boxesOut,
    std::vector<real>& scoresOut)
{
    assert(boxesIn.size() == scoresIn.size());
    const size_t size = boxesIn.size();

    if (size > m_boxesCountMax)
        return false;

    std::vector<Record> records;
    records.reserve(size);
    for (size_t i = 0; i < size; i++)
        records.push_back({ boxesIn[i], scoresIn[i] });

    using namespace std::chrono;

    // cpu
    steady_clock::time_point t1 = steady_clock::now();

    std::sort(records.begin(), records.end(),
        [](const Record& l, const Record& r) { return l.score > r.score; });

    steady_clock::time_point t2 = steady_clock::now();

    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

    std::cout << "Sorting took " << time_span.count() << " seconds" << std::endl;

    for (size_t i = 0; i < size; i++)
        m_boxesInCPU[i] = records[i].box;

    cudaError_t cudaStatus = cudaMemcpy(
        m_boxesInGPU, m_boxesInCPU,
        size * sizeof(Box), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return false;
    }

    t1 = steady_clock::now();

    const size_t BLOCK_SIZE = 1024;
    const size_t numLaunches = (size / BLOCK_SIZE + ((size % BLOCK_SIZE) ? 1 : 0));
    for (size_t i = 0; i < numLaunches; i++) {
        
        if (i > 0)
            NmsRectangle << <i, BLOCK_SIZE >> > (
                m_boxesInGPU, threshold, m_resultGPU, static_cast<unsigned>(i * BLOCK_SIZE), static_cast<unsigned>(size));
        
        NmsTriangle<<<1, BLOCK_SIZE, (sizeof(Box) + 1)* BLOCK_SIZE >> > (
            m_boxesInGPU, threshold, m_resultGPU, static_cast<unsigned>(i * BLOCK_SIZE), static_cast<unsigned>(size));
    }

    //NmsKernel << <1, static_cast<unsigned>(size), (sizeof(Box) + 1)*size >> > (m_boxesInGPU, threshold, m_resultGPU);

    ////// Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return false;
    }

    ////// cudaDeviceSynchronize waits for the kernel to finish, and returns
    ////// any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        fprintf(stderr, "error string: %s\n", cudaGetErrorString(cudaStatus));
        return false;
    }

    t2 = steady_clock::now();

    time_span = duration_cast<duration<double>>(t2 - t1);

    std::cout << "Kernel took " << time_span.count() << " seconds" << std::endl;

    cudaStatus = cudaMemcpy(
        m_resultCPU, m_resultGPU,
        size * sizeof(char), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return false;
    }

    boxesOut.clear();
    scoresOut.clear();

    for (size_t i = 0; i < size; i++) {
        if (!m_resultCPU[i]) {
            boxesOut.push_back(records[i].box);
            scoresOut.push_back(records[i].score);
        }
    }

    return true;
}
