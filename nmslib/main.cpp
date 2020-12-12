#include "box_generator.h"
#include "nms_single_thread.h"
#include "nms_multiple_threads.h"
#include "nms_gpu.h"

#include <cassert>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

static void generateBoxes(int count,
    std::vector<Box>& boxes, std::vector<real>& scores)
{
    boxes.clear();
    scores.clear();

    boxes.reserve(count);
    scores.reserve(count);

    unsigned seed = static_cast<unsigned>(
        std::chrono::system_clock::now().time_since_epoch().count());
    std::default_random_engine generator(seed);

    BoxGenerator boxGenerator(4096, 2160);
    std::uniform_real_distribution<real> distr;

    for (int i = 0; i < count; i++) {
        boxes.push_back(boxGenerator.generate(generator));
        scores.push_back(distr(generator));
    }
}

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

int main(int argc, char* argv[])
{
    int boxCount = 3000;
    if (argc > 1) {
        boxCount = std::stoi(argv[1]);
    }

    std::vector<Box> boxes;
    std::vector<real> scores;
    real threshold = real(0.7);

    generateBoxes(boxCount, boxes, scores);

    std::vector<Box> boxesOut;
    std::vector<real> scoresOut;

    using namespace std::chrono;

    // single thread
    steady_clock::time_point t1 = steady_clock::now();

    nms_single_thread(boxes, scores, threshold, boxesOut, scoresOut);

    steady_clock::time_point t2 = steady_clock::now();

    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

    std::cout << "Output boxes count (single thread): " << boxesOut.size() << std::endl;
    std::cout << "It took me " << time_span.count() << " seconds.";
    std::cout << std::endl << std::endl;

    // gpu
    NMS_gpu nms_gpu;
    if (nms_gpu.init(boxCount)) {

        std::vector<Box> boxesOutGPU;
        std::vector<real> scoresOutGPU;

        t1 = steady_clock::now();

        nms_gpu.doIt(boxes, scores, threshold, boxesOutGPU, scoresOutGPU);

        t2 = steady_clock::now();

        time_span = duration_cast<duration<double>>(t2 - t1);

        std::cout << "Output boxes count (gpu): " << boxesOutGPU.size() << std::endl;
        std::cout << "It took me " << time_span.count() << " seconds.";
        std::cout << std::endl << std::endl;

        if (!compareResults(boxesOut, scoresOut, boxesOutGPU, scoresOutGPU))
            std::cout << "Results are different!" << std::endl << std::endl;
    }
    else {
        std::cout << "Failed to initialize nms_gpu!" << std::endl << std::endl;
    }

    // multiple threads
    std::vector<Box> boxesOutStl;
    std::vector<real> scoresOutStl;

    t1 = steady_clock::now();

    nms_multiple_threads(boxes, scores, threshold, boxesOutStl, scoresOutStl);

    t2 = steady_clock::now();

    time_span = duration_cast<duration<double>>(t2 - t1);

    std::cout << "Output boxes count (multiple threads): " << boxesOutStl.size() << std::endl;
    std::cout << "It took me " << time_span.count() << " seconds.";
    std::cout << std::endl << std::endl;

    if (!compareResults(boxesOut, scoresOut, boxesOutStl, scoresOutStl))
        std::cout << "Results are different!" << std::endl;

    return 0;
}