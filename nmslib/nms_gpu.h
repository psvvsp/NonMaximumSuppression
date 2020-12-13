#pragma once

#include "nms_interface.h"

#include <vector>

class NMS_gpu : public NMS_interface
{
public:
    NMS_gpu();
    virtual ~NMS_gpu();

    NMS_gpu(const NMS_gpu& other) = delete;
    NMS_gpu& operator =(const NMS_gpu& other) = delete;

    bool doIt(
        const std::vector<Box>& boxesIn,
        const std::vector<real>& scoresIn,
        real threshold,
        std::vector<Box>& boxesOut,
        std::vector<real>& scoresOut
    ) override;

    bool init(size_t boxesCountMax) override;

private:
    size_t m_boxesCountMax;
    int m_maxBlockSize;
    Box* m_boxesInCPU;
    Box* m_boxesInGPU;
    char* m_resultGPU;
    char* m_resultCPU;
};
