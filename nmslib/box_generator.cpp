#include "box_generator.h"
#include <chrono>

void generateBoxes(size_t count,
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

    for (size_t i = 0; i < count; i++) {
        boxes.push_back(boxGenerator.generate(generator));
        scores.push_back(distr(generator));
    }
}
