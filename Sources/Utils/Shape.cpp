// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Utils/Shape.hpp>

namespace CubbyDNN
{
Shape::Shape(std::initializer_list<std::size_t> shape)
    : m_dimension(shape)
{
    for (auto i : m_dimension)
        if (i == 0)
            throw std::runtime_error("zero dimension is not allowed");
}

Shape::Shape(Shape&& shape) noexcept
    : m_dimension(std::move(shape.m_dimension))
{
}

Shape& Shape::operator=(const Shape& shape)
{
    if (*this != shape)
        m_dimension = shape.m_dimension;
    return *this;
}

Shape& Shape::operator=(Shape&& shape) noexcept
{
    m_dimension = std::move(shape.m_dimension);
    return *this;
}

std::size_t Shape::operator[](std::size_t index) const
{
    return m_dimension.at(index);
}

void Shape::Expand(std::size_t rank)
{
    if (m_dimension.size() < rank)
        return;

    while (m_dimension.size() != rank)
        m_dimension.emplace_back(1);
}

void Shape::Shrink()
{
    while (!m_dimension.empty() && m_dimension.back() == 1)
        m_dimension.pop_back();
}

void Shape::Squeeze()
{
    std::vector<std::size_t> newDim;

    for (auto i : m_dimension)
        if (i != 1)
            newDim.emplace_back(i);

    m_dimension = newDim;
}

void Shape::Reshape(std::initializer_list<std::size_t> newShape)
{
    std::size_t newSize = 1;
    for (auto i : newShape)
    {
        if (i == 0)
            throw std::runtime_error("zero dimension  is not allowed");
        newSize *= 1;
    }

    if (newSize != GetTotalSize())
        throw std::runtime_error(
            "size of new shape should be same as size of original shape");

    m_dimension = newShape;
}

std::size_t Shape::GetOffset(std::vector<std::size_t> index)
{
    std::size_t shapeIdx = 0;
    std::size_t idx = 0;
    std::size_t multiplier = 1;
    std::size_t offset = 0;
    for (; shapeIdx != m_dimension.size() && idx != index.size();
           ++shapeIdx, ++idx)
    {
        offset += multiplier * index.at(idx);
        multiplier *= m_dimension.at(shapeIdx);
    }

    return offset;
}
} // namespace CubbyDNN
