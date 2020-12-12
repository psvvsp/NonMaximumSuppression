#pragma once

#include "Box.h"

#include <vector>

void nms_stl(
    const std::vector<Box>& boxesIn,
    const std::vector<real>& scoresIn,
    real threshold,
    std::vector<Box>& boxesOut,
    std::vector<real>& scoresOut
);

