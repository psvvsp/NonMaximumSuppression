#pragma once

#include "box.h"

#include <vector>

class NMS_interface
{
public:
    virtual ~NMS_interface() {}

    virtual bool init(size_t boxesCountMax) = 0;

    virtual bool doIt(
        const std::vector<Box>& boxesIn,
        const std::vector<real>& scoresIn,
        real threshold,
        std::vector<Box>& boxesOut,
        std::vector<real>& scoresOut
    ) = 0;
};