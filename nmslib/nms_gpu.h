#pragma once

#include "Box.h"

#include <vector>

class NMS_gpu
{
public:
    NMS_gpu(size_t boxesCountMax);
    ~NMS_gpu();

    NMS_gpu(const NMS_gpu& other) = delete;
    NMS_gpu& operator =(const NMS_gpu& other) = delete;

    bool doIt(
        const std::vector<Box>& boxesIn,
        const std::vector<real>& scoresIn,
        real threshold,
        std::vector<Box>& boxesOut,
        std::vector<real>& scoresOut
    );

private:
    Box* m_boxesInCPU;
    Box* m_boxesInGPU;
    real* m_scoresInGPU;
    char* m_resultGPU;
    char* m_resultCPU;
    size_t m_boxesCountMax;
};
