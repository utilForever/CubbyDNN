// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.


#ifndef TAKION_UTIL_LOADER_HPP
#define TAKION_UTIL_LOADER_HPP

#include <vector>
#include <Takion/Utils/Shape.hpp>

namespace Takion::Util
{
template <typename T>
class Loader
{
public:
    Loader(Shape shape, std::size_t batchSize)
        : m_data(shape.Size() * batchSize)
    {
    }

    void SetData(std::vector<T> vector)
    {
        m_data = std::move(vector);
    }

    virtual std::vector<T> operator()()
    {
        return m_data;
    }

protected:
    std::vector<T> m_data;
};
}

#endif
