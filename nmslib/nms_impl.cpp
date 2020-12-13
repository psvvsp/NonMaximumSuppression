#include "nms_impl.h"
#include "box_generator.h"

#include <cassert>
#include <chrono>

static double runSMP(
    const std::vector<Box>& boxesIn,
    const std::vector<real>& scoresIn,
    real threshold,
    std::vector<Box>& boxesOut,
    std::vector<real>& scoresOut,
    NMS_interface& nms_interface
)
{
    using namespace std::chrono;

    steady_clock::time_point t1 = steady_clock::now();

    nms_interface.doIt(boxesIn, scoresIn, threshold, boxesOut, scoresOut);

    steady_clock::time_point t2 = steady_clock::now();

    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

    return time_span.count();
}

bool NMS_impl::init(size_t boxesCountMax)
{
    if (boxesCountMax < 100) {
        assert(false);
        return false;
    }

    m_boxesCountMax = boxesCountMax;

    if (!m_nms_gpu.init(boxesCountMax))
        return false;

    if (!m_nms_single_thread.init(boxesCountMax))
        return false;

    if (!m_nms_multiple_threads.init(boxesCountMax))
        return false;

    for (size_t boxesCount = boxesCountMax / 10;
        boxesCount < boxesCountMax; boxesCount++) {

        std::vector<Box> boxesIn;
        std::vector<real> scoresIn;

        const real threshold = real(0.7);
        generateBoxes(boxesCount, boxesIn, scoresIn);

        std::vector<Box> boxesOut;
        std::vector<real> scoresOut;

        double durationGPU = runSMP(boxesIn, scoresIn, threshold,
            boxesOut, scoresOut, m_nms_gpu);

        double durationSingleThread = runSMP(boxesIn, scoresIn, threshold,
            boxesOut, scoresOut, m_nms_single_thread);

        double durationMultipleThreads = runSMP(boxesIn, scoresIn, threshold,
            boxesOut, scoresOut, m_nms_multiple_threads);

        std::pair<size_t, NMS_interface*> pair;
        pair.first = boxesCount;

        if (durationGPU < durationSingleThread && 
            durationGPU < durationMultipleThreads) {

            pair.second = &m_nms_gpu;

        }
        else if (durationSingleThread < durationGPU &&
            durationSingleThread < durationMultipleThreads) {

            pair.second = &m_nms_single_thread;
        }
        else
            pair.second = &m_nms_multiple_threads;

        m_implementations.push_back(pair);
    }

    return false;
}

bool NMS_impl::doIt(
    const std::vector<Box>& boxesIn,
    const std::vector<real>& scoresIn,
    real threshold,
    std::vector<Box>& boxesOut,
    std::vector<real>& scoresOut)
{
    if (boxesIn.size() != scoresIn.size() ||
        m_boxesCountMax == 0 ||
        boxesIn.size() > m_boxesCountMax) {
        
        assert(false);
        return false;
    }

    const size_t size = boxesIn.size();
    if (size == 0)
        return true;

    NMS_interface* impl = nullptr;

    if (size < m_implementations.front().first)
        impl = m_implementations.front().second;

    for (size_t i = 0; i < m_implementations.size() - 1; i++) {
        if (m_implementations[i].first <= size &&
            size < m_implementations[i + 1].first)
            impl = m_implementations[i].second;
    }
    
    if (m_implementations.back().first <= size)
        impl = m_implementations.back().second;

    return impl->doIt(boxesIn, scoresIn, threshold, boxesOut, scoresOut);
}
