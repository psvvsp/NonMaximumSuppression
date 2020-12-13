#pragma once

#include "nms_gpu.h"
#include "nms_single_thread.h"
#include "nms_multiple_threads.h"

#include <vector>

class NMS_impl : public NMS_interface
{
public:
    NMS_impl() : m_boxesCountMax(0) {}
    virtual ~NMS_impl() {}

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

    NMS_gpu m_nms_gpu;
    NMS_single_thread m_nms_single_thread;
    NMS_multiple_threads m_nms_multiple_threads;

    std::vector<std::pair<size_t, NMS_interface*>> m_implementations;
};