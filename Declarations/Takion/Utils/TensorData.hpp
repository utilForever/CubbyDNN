// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_UTIL_TENSORDATA_HPP
#define TAKION_UTIL_TENSORDATA_HPP

#include <Takion/Utils/Shape.hpp>
#include <vector>

namespace Takion::Util
{
template <typename T>
struct TensorData
{
    TensorData(std::vector<T> data, Shape shape, std::size_t batchSize)
        : Data(std::move(data)),
          TensorShape(shape),
          BatchSize(batchSize)
    {
    }

    std::vector<T> Data;
    Shape TensorShape;
    std::size_t BatchSize;
};
} // namespace Takion::Util

#endif
