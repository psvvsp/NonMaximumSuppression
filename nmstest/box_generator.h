#pragma once

#include "Box.h"

#include <random>

class BoxGenerator
{
public:
    BoxGenerator(integer screenWidth, integer screenHeight)
        : m_distrWidth(0, screenWidth - 1)
        , m_distrHeight(0, screenHeight - 1)
    {
    }

    template<class GeneratorType>
    Box generate(GeneratorType& generator)
    {
        Box box;

        integer x1 = m_distrWidth(generator);
        integer x2 = m_distrWidth(generator);

        box.left = std::min(x1, x2);
        box.right = std::max(x1, x2);

        integer y1 = m_distrHeight(generator);
        integer y2 = m_distrHeight(generator);

        box.top = std::min(y1, y2);
        box.bottom = std::max(y1, y2);

        return box;
    }
    
private:
    std::uniform_int_distribution<integer> m_distrWidth;
    std::uniform_int_distribution<integer> m_distrHeight;
};

