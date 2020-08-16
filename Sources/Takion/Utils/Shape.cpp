// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include<Takion/Utils/Shape.hpp>

namespace Takion
{
Shape::Shape(std::initializer_list<std::size_t> shape)
    : m_shapeVector(shape)
{
    for (auto i : m_shapeVector)
        if (i == 0)
            throw std::runtime_error("zero dimension is not allowed");
}

Shape::Shape(std::vector<std::size_t> shape)
    : m_shapeVector(std::move(shape))
{
    for (auto i : m_shapeVector)
        if (i == 0)
            throw std::runtime_error("zero dimension is not allowed");
}

Shape::Shape(Shape&& shape) noexcept
    : m_shapeVector(std::move(shape.m_shapeVector))
{
}

Shape& Shape::operator=(const Shape& shape)
{
    if (this == &shape)
        return *this;
    m_shapeVector = shape.m_shapeVector;
    return *this;
}

Shape& Shape::operator=(Shape&& shape) noexcept
{
    m_shapeVector = std::move(shape.m_shapeVector);
    return *this;
}

std::size_t& Shape::operator[](std::size_t index)
{
    return m_shapeVector.at(index);
}

bool Shape::operator==(const Shape& shape) const
{
    return m_shapeVector == shape.m_shapeVector;
}

bool Shape::operator!=(const Shape& shape) const
{
    return !(*this == shape);
}

std::size_t Shape::At(std::size_t index) const
{
    if (index >= m_shapeVector.size())
        throw std::invalid_argument("Requested index exceeds shape dimension");
    return m_shapeVector.at(index);
}


Shape Shape::operator*(const Shape& shape) const
{
    if (this->Dim() != shape.Dim())
        throw std::runtime_error("Dimension mismatch");
    if (this->NumCol() != shape.NumRow())
        throw std::runtime_error("Multiply-shape mismatch");;

    std::vector<std::size_t> shapeVector;
    shapeVector.reserve(shape.Dim());

    for (std::size_t idx = 2; idx < shape.Dim(); ++idx)
    {
        if (m_shapeVector.at(idx) == shape.m_shapeVector.at(idx))
            shapeVector.at(idx) = shape.m_shapeVector.at(idx);
        else
            throw std::runtime_error("Batch Shape mismatch");
    }

    Shape derivedShape;
    derivedShape.m_shapeVector = shapeVector;
    return derivedShape;
}

void Shape::Expand(std::size_t rank)
{
    if (m_shapeVector.size() < rank)
        return;

    while (m_shapeVector.size() != rank)
    {
        m_shapeVector.emplace_back(1);
    }
}


void Shape::Shrink()
{
    while (!m_shapeVector.empty() && m_shapeVector.back() == 1)
    {
        m_shapeVector.pop_back();
    }
}

void Shape::Squeeze()
{
    std::vector<std::size_t> newDim;

    for (auto i : m_shapeVector)
        if (i != 1)
            newDim.emplace_back(i);

    m_shapeVector = newDim;
}

std::size_t Shape::Size() const noexcept
{
    std::size_t size = 1;
    for (auto i : m_shapeVector)
        size *= i;

    return size;
}

std::size_t Shape::Dim() const
{
    return m_shapeVector.size();
}

Shape& Shape::Reshape(std::initializer_list<std::size_t> newShape)
{
    std::size_t newSize = 1;
    for (auto i : newShape)
    {
        if (i == 0)
            throw std::runtime_error("zero dimension is not allowed");
        newSize *= 1;
    }

    if (newSize != Size())
        throw std::runtime_error(
            "size of new shape should be same as size of original shape");

    m_shapeVector = newShape;
    return *this;
}

void Shape::Transpose()
{
    const auto temp = m_shapeVector.at(0);
    m_shapeVector.at(0) = m_shapeVector.at(1);
    m_shapeVector.at(1) = temp;
}

Shape Shape::GetTransposedShape() const
{
    Shape shape(m_shapeVector);
    shape.SetNumRows(NumCol());
    shape.SetNumCols(NumRow());
    return shape;
}
} // namespace Takion
