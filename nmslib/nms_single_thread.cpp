#include "nms_single_thread.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <set>

namespace {

    struct Record
    {
        Box box;
        real score;
        bool removed;
    };

}

bool NMS_single_thread::doIt(
    const std::vector<Box>& boxesIn,
    const std::vector<real>& scoresIn,
    real threshold,
    std::vector<Box>& boxesOut,
    std::vector<real>& scoresOut)
{
    assert(boxesIn.size() == scoresIn.size());
    const size_t size = boxesIn.size();

    std::vector<Record> records;
    records.reserve(size);
    for (size_t i = 0; i < size; i++)
        records.push_back({ boxesIn[i], scoresIn[i], false });

    //using namespace std::chrono;

    //steady_clock::time_point t1 = steady_clock::now();

    std::sort(records.begin(), records.end(),
        [](const Record& l, const Record& r) { return l.score > r.score; });

    //steady_clock::time_point t2 = steady_clock::now();

    //duration<double> time_span = duration_cast<duration<double>>(t2 - t1);

    //std::cout << "Sorting took " << time_span.count() << " seconds" << std::endl;

    boxesOut.clear();
    scoresOut.clear();

    for (size_t i = 0; i < size; ) {

        Record& record = records[i];

        boxesOut.push_back(record.box);
        scoresOut.push_back(record.score);
        
        record.removed = true;

        for (size_t j = i + 1; j < size; j++) {
            
            if (records[j].removed)
                continue;
            
            if (iou(record.box, records[j].box) >= threshold) {
                records[j].removed = true;
            }
        }

        while (i < size && records[i].removed) i++;
    }

    return true;
}
