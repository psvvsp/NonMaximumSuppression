#include "NMS_gpu.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>

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

NMS_gpu::NMS_gpu()
    : m_boxesCountMax(0)
    , m_maxBlockSize(0)
    , m_boxesInCPU(nullptr)
    , m_boxesInGPU(nullptr)
    , m_resultCPU(nullptr)
    , m_resultGPU(nullptr)
{
}

bool NMS_gpu::init(size_t boxesCountMax)
{
    assert(m_boxesCountMax == 0);
    m_boxesCountMax = boxesCountMax;

    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) return false;

    cudaDeviceProp* props = new cudaDeviceProp();
    cudaStatus = cudaGetDeviceProperties(props, 0);
    m_maxBlockSize = props->maxThreadsPerBlock;
    delete props;
    if (cudaStatus != cudaSuccess) return false;

    cudaStatus = cudaMallocHost(
        (void**)&m_boxesInCPU, m_boxesCountMax * sizeof(Box));
    if (cudaStatus != cudaSuccess) return false;

    cudaStatus = cudaMalloc(
        (void**)&m_boxesInGPU, m_boxesCountMax * sizeof(Box));
    if (cudaStatus != cudaSuccess) return false;

    cudaStatus = cudaMallocHost(
        (void**)&m_resultCPU, m_boxesCountMax * sizeof(char));
    if (cudaStatus != cudaSuccess) return false;

    cudaStatus = cudaMalloc(
        (void**)&m_resultGPU, m_boxesCountMax * sizeof(char));
    if (cudaStatus != cudaSuccess) return false;

    return true;
}

NMS_gpu::~NMS_gpu()
{
    cudaFreeHost(m_resultCPU);
    cudaFreeHost(m_boxesInCPU);
    cudaFree(m_boxesInGPU);
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

    //using namespace std::chrono;

    //steady_clock::time_point t1 = steady_clock::now();

    std::sort(records.begin(), records.end(),
        [](const Record& l, const Record& r) { return l.score > r.score; });

    //steady_clock::time_point t2 = steady_clock::now();

    //duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

    //std::cout << "Sorting took " << time_span.count() << " seconds" << std::endl;

    for (size_t i = 0; i < size; i++)
        m_boxesInCPU[i] = records[i].box;

    cudaError_t cudaStatus = cudaMemcpy(
        m_boxesInGPU, m_boxesInCPU,
        size * sizeof(Box), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        return false;
    }

    cudaStatus = cudaMemset(m_resultGPU, 0, size);
    if (cudaStatus != cudaSuccess) {
        return false;
    }

    //t1 = steady_clock::now();

    const unsigned maxBlockSize = m_maxBlockSize;
    const unsigned numLaunches = static_cast<unsigned>(
        size / maxBlockSize + ((size % maxBlockSize) ? 1 : 0));
    
    for (unsigned i = 0; i < numLaunches; i++) {
        
        if (i > 0)
            NmsRectangle << <i, maxBlockSize >> > (
                m_boxesInGPU, threshold, m_resultGPU,
                i * maxBlockSize, static_cast<unsigned>(size));
        
        NmsTriangle<<<1, maxBlockSize, (sizeof(Box) + 1)* maxBlockSize >> > (
            m_boxesInGPU, threshold, m_resultGPU,
            i * maxBlockSize, static_cast<unsigned>(size));
    }

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        return false;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        return false;
    }

    //t2 = steady_clock::now();

    //time_span = duration_cast<duration<double>>(t2 - t1);

    //std::cout << "Kernel took " << time_span.count() << " seconds" << std::endl;

    cudaStatus = cudaMemcpy(
        m_resultCPU, m_resultGPU,
        size * sizeof(char), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
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
