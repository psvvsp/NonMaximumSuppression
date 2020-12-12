#include "nms_multiple_threads.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>
#include <thread>

namespace {
    struct Record
    {
        Record(const Box& b, real s, bool r)
            : box(b), score(s), removed(r) {}

        Box box;
        real score;
        std::atomic<bool> removed;
    };
}

static void threadFunction(
    std::vector<std::unique_ptr<Record>>& records,
    std::vector<std::atomic<size_t>> &positions,
    size_t threadIndex, real threshold)
{
    const size_t increment = positions.size();
    while(positions[threadIndex] < records.size()) {
        
        const size_t i = positions[threadIndex];
        
        while (true) {
            bool ready = true;
            for (const auto& position : positions)
                ready = ready && (position >= i);
            if (ready) break;
        }

        const Record& record = *records[i];

        if (!record.removed) {
            for (size_t j = i + 1; j < records.size(); j++) {
                if (j == i + increment)
                    positions[threadIndex] += increment;
                if (records[j]->removed) continue;
                if (iou(record.box, records[j]->box) >= threshold)
                    records[j]->removed = true;
            }
        }
        
        if (i == positions[threadIndex])
            positions[threadIndex] += increment;
    }
}

void nms_multiple_threads(
    const std::vector<Box>& boxesIn,
    const std::vector<real>& scoresIn,
    real threshold,
    std::vector<Box>& boxesOut,
    std::vector<real>& scoresOut)
{
    assert(boxesIn.size() == scoresIn.size());
    const size_t size = boxesIn.size();

    std::vector<std::unique_ptr<Record>> records(size);
    for (size_t i = 0; i < size; i++)
        records[i].reset(new Record(boxesIn[i], scoresIn[i], false));

    //using namespace std::chrono;

    //steady_clock::time_point t1 = steady_clock::now();

    std::sort(records.begin(), records.end(),
        [](const std::unique_ptr<Record>& l, const std::unique_ptr<Record>& r) { return l->score > r->score; });

    //steady_clock::time_point t2 = steady_clock::now();

    //duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

    //std::cout << "Sorting took " << time_span.count() << " seconds" << std::endl;

    size_t threadsNum = std::thread::hardware_concurrency();
    std::vector<std::unique_ptr<std::thread>> threads(threadsNum);
    
    std::vector<std::atomic<size_t>> positions(threadsNum);
    for (size_t i = 0; i < threadsNum; i++) {
        positions[i] = i;
    }

    for (size_t i = 0; i < threadsNum; i++) {
        threads[i].reset(new std::thread(threadFunction, std::ref(records), std::ref(positions), i, threshold));
    }

    for (const auto& thread : threads) {
        thread->join();
    }

    boxesOut.clear();
    scoresOut.clear();

    for (const auto& record : records) {
        if (!record->removed) {
            boxesOut.push_back(record->box);
            scoresOut.push_back(record->score);
        }
    }
}
