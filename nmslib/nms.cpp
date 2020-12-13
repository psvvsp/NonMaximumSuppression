#include "nms.h"
#include "nms_impl.h"

NMS::NMS()
{
    m_impl = new NMS_impl();
}

NMS::~NMS()
{
    delete m_impl;
}

bool NMS::doIt(
    const std::vector<Box>& boxesIn,
    const std::vector<real>& scoresIn,
    real threshold,
    std::vector<Box>& boxesOut,
    std::vector<real>& scoresOut)
{
    return m_impl->doIt(boxesIn, scoresIn, threshold, boxesOut, scoresOut);
}

bool NMS::init(size_t boxesCountMax)
{
    return m_impl->init(boxesCountMax);
}
