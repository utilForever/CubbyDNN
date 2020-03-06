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
    Shape() = default;
   ~Shape() = default;

    Shape(std::initializer_list<std::size_t> shape);

    Shape(const Shape& shape) = default;
    Shape(Shape&& shape) noexcept;

    Shape& operator=(const Shape& shape);
    Shape& operator=(Shape&& shape) noexcept;

    std::size_t
    operator[](std::size_t index) const;

    void Expand(std::size_t rank);
    void Shrink();
    void Squeeze();

    friend bool
    operator==(const Shape& lhs, const Shape& rhs)
    {
        return lhs.m_dimension == rhs.m_dimension;
    }

    friend bool operator!=(const Shape& lhs, const Shape& rhs)
    {
        return !(lhs == rhs);
    }

    [[nodiscard]] std::size_t GetTotalSize() const
    {
        std::size_t size = 1;
        for (auto i : m_dimension)
            size *= i;

        return size;
    }

    [[nodiscard]] std::size_t GetOffset(std::vector<std::size_t> index);

    void Reshape(std::initializer_list<std::size_t> newShape);

private:
    std::vector<std::size_t> m_dimension;
};
} // namespace CubbyDNN

#endif
