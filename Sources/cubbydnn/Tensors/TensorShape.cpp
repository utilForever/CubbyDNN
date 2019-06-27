// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cstdio>
#include <cubbydnn/Tensors/TensorShape.hpp>

namespace CubbyDNN
{
TensorShape::TensorShape(long row, long column, long depth)
    : m_dimension({ row, column, depth }), m_totalSize(1)
{
    for (auto mul : m_dimension)
    {
        m_totalSize *= mul;
    }
}

bool TensorShape::operator==(const TensorShape& shape) const
{
    return m_dimension == shape.m_dimension;
}

bool TensorShape::operator!=(const TensorShape& shape) const
{
    return !(*this == shape);
}

size_t TensorShape::Size() const noexcept
{
    return m_totalSize;
}

bool TensorShape::IsEmpty() const noexcept
{
    return m_dimension.empty();
}

long TensorShape::Row() const
{
    return m_dimension.at(0);
}

long TensorShape::Col() const
{
    return m_dimension.at(1);
}

long TensorShape::Depth() const
{
    return m_dimension.at(2);
}
}  // namespace CubbyDNN