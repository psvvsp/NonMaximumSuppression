#pragma once

#include "nms_interface.h"

#include <vector>

class NMS_multiple_threads : public NMS_interface
{
public:
    virtual ~NMS_multiple_threads() {}

    bool doIt(
        const std::vector<Box>& boxesIn,
        const std::vector<real>& scoresIn,
        real threshold,
        std::vector<Box>& boxesOut,
        std::vector<real>& scoresOut
    ) override;

    bool init(size_t boxesCountMax) override { return true; }
};
