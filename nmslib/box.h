#pragma once

#include "Common.h"

#include <algorithm>

struct Box
{
    integer area() const { return (right - left + 1) * (bottom - top + 1); }

    integer left;
    integer right;
    integer top;
    integer bottom;
};

inline real iou(const Box& b1, const Box& b2)
{
    Box intersection;
    
    intersection.left = std::max(b1.left, b2.left);
    intersection.right = std::min(b1.right, b2.right);

    if (intersection.left > intersection.right)
        return real(0.0);

    intersection.top = std::max(b1.top, b2.top);
    intersection.bottom = std::min(b1.bottom, b2.bottom);

    if (intersection.top > intersection.bottom)
        return real(0.0);

    integer intersection_area = intersection.area();
    integer union_area = b1.area() + b2.area() - intersection_area;

    return real(intersection_area) / real(union_area);
}

inline bool operator ==(const Box& b1, const Box& b2)
{
    return b1.left == b2.left && b1.right == b2.right
        && b1.top == b2.top && b1.bottom == b2.bottom;
}