#include "box_generator.h"
#include "nms_single_thread.h"
#include "nms_multiple_threads.h"
#include "nms_gpu.h"
#include "nms_impl.h"

#include <cassert>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

static bool compareResults(
    std::vector<Box>& boxes1, std::vector<real>& scores1,
    std::vector<Box>& boxes2, std::vector<real>& scores2)
{
    assert(boxes1.size() == scores1.size());
    assert(boxes2.size() == scores2.size());

    if (boxes1.size() != boxes2.size())
        return false;

    for (size_t i = 0; i < boxes1.size(); i++) {
        if (!(boxes1[i] == boxes2[i]))
            return false;
        if (scores1[i] != scores2[i])
            return false;
    }

    return true;
}

static void runNMS(
    const std::vector<Box>& boxesIn,
    const std::vector<real>& scoresIn,
    real threshold,
    std::vector<Box>& boxesOut,
    std::vector<real>& scoresOut,
    NMS_interface& nms_interface,
    const std::string name
)
{
    using namespace std::chrono;

    if (nms_interface.init(boxesIn.size())) {

        steady_clock::time_point t1 = steady_clock::now();

        nms_interface.doIt(boxesIn, scoresIn, threshold, boxesOut, scoresOut);

        steady_clock::time_point t2 = steady_clock::now();

        duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

        std::cout << "Output boxes count (" << name << "): " << boxesOut.size() << std::endl;
        std::cout << "It took me " << time_span.count() << " seconds.";
        std::cout << std::endl << std::endl;
    }
    else {
        std::cout << "Failed to initialize " << name << "!" << std::endl << std::endl;
    }
}

int main(int argc, char* argv[])
{
    int boxCount = 3000;
    if (argc > 1) {
        boxCount = std::stoi(argv[1]);
    }

    std::vector<Box> boxes;
    std::vector<real> scores;
    real threshold = real(0.7);

    generateBoxes(size_t(boxCount), boxes, scores);

    std::vector<Box> boxesOut;
    std::vector<real> scoresOut;

    // single thread
    NMS_single_thread nms_single_thread;
    runNMS(boxes, scores, threshold, boxesOut,
        scoresOut, nms_single_thread, "single thread");

    // gpu
    NMS_gpu nms_gpu;

    std::vector<Box> boxesOutGPU;
    std::vector<real> scoresOutGPU;

    runNMS(boxes, scores, threshold, boxesOutGPU,
        scoresOutGPU, nms_gpu, "gpu");

    if (!compareResults(boxesOut, scoresOut, boxesOutGPU, scoresOutGPU))
        std::cout << "Results are different!" << std::endl << std::endl;

    // multiple threads
    NMS_multiple_threads nms_multiple_threads;
    
    std::vector<Box> boxesOutThreads;
    std::vector<real> scoresOutThreads;

    runNMS(boxes, scores, threshold, boxesOutThreads,
        scoresOutThreads, nms_multiple_threads, "multiple threads");

    if (!compareResults(boxesOut, scoresOut, boxesOutThreads, scoresOutThreads))
        std::cout << "Results are different!" << std::endl;

    // optimal implementation
    NMS_impl nms_impl;

    std::vector<Box> boxesOutImpl;
    std::vector<real> scoresOutImpl;

    runNMS(boxes, scores, threshold, boxesOutImpl,
        scoresOutImpl, nms_impl, "optimal implementation");

    if (!compareResults(boxesOut, scoresOut, boxesOutImpl, scoresOutImpl))
        std::cout << "Results are different!" << std::endl;

    return 0;
}