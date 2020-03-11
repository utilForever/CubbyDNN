// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_SHAPER_HPP
#define CUBBYDNN_SHAPER_HPP

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

    std::size_t operator[](std::size_t index) const;

    Shape operator*(const Shape& shape) const;

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

    [[nodiscard]] std::size_t TotalSize() const noexcept;

    [[nodiscard]] std::size_t Offset(std::vector<std::size_t> index) const
    noexcept;

    [[nodiscard]] std::size_t BatchSize() const;

    [[nodiscard]] std::size_t MatrixSize() const;

    [[nodiscard]] std::size_t PadSize() const
    {
        return m_padding;
    }

    [[nodiscard]] std::size_t PaddedMatrixSize() const;

    [[nodiscard]] bool IsAligned() const;

    [[nodiscard]] std::size_t Row() const
    {
        return m_shapeVector.at(1);
    }

    [[nodiscard]] std::size_t Col() const
    {
        return m_shapeVector.at(0);
    }

    void Reshape(std::initializer_list<std::size_t> newShape);

private:
    std::vector<std::size_t> m_shapeVector;
    std::size_t m_padding = 0;
};
} // namespace CubbyDNN

#endif
