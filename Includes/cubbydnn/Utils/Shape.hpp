// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_SHAPE_HPP
#define CUBBYDNN_SHAPE_HPP

#include <stdexcept>
#include <vector>

namespace CubbyDNN
{
class Shape
{
public:
    Shape();
    ~Shape() = default;

    Shape(std::initializer_list<std::size_t> shape);

    Shape(const Shape& shape) = default;
    Shape(Shape&& shape) noexcept;

    Shape& operator=(const Shape& shape);
    Shape& operator=(Shape&& shape) noexcept;

    std::size_t& operator[](std::size_t index);

    Shape operator*(const Shape& shape) const;

    [[nodiscard]] std::size_t At(std::size_t index) const
    {
        return m_shapeVector.at(index);
    }

    void Expand(std::size_t rank);

    void Shrink();

    void Squeeze();

    [[nodiscard]] std::size_t Dim() const;

    friend bool operator==(const Shape& lhs, const Shape& rhs)
    {
        return lhs.m_shapeVector == rhs.m_shapeVector;
    }

    friend bool operator!=(const Shape& lhs, const Shape& rhs)
    {
        return !(lhs == rhs);
    }

    [[nodiscard]] std::size_t Size() const noexcept;

    [[nodiscard]] std::size_t Offset(std::vector<std::size_t> index) const
    noexcept;

    [[nodiscard]] std::size_t BatchSize() const;

    [[nodiscard]] std::size_t NumRows() const
    {
        return m_shapeVector.at(1);
    }

    [[nodiscard]] std::size_t NumCols() const
    {
        return m_shapeVector.at(0);
    }

    void SetNumRows(std::size_t row)
    {
        m_shapeVector.at(1) = row;
    }

    void SetNumCols(std::size_t col)
    {
        m_shapeVector.at(0) = col;
    }

    Shape& ChangeDimension(std::size_t axis, std::size_t value)
    {
        m_shapeVector.at(axis) = value;
        return *this;
    }

    Shape& Reshape(std::initializer_list<std::size_t> newShape);

    [[nodiscard]] Shape& Transpose();

private:
    std::vector<std::size_t> m_shapeVector;
};
} // namespace CubbyDNN

#endif
