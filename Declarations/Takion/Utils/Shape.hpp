// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_SHAPE_HPP
#define TAKION_SHAPE_HPP

#include <stdexcept>
#include <string>
#include <vector>

namespace Takion
{
class Shape
{
public:
    Shape() = default;
    ~Shape() = default;

    Shape(std::initializer_list<std::size_t> shape);
    Shape(std::vector<std::size_t> shape);

    Shape(const Shape& shape) = default;
    Shape(Shape&& shape) noexcept;

    Shape& operator=(const Shape& shape);
    Shape& operator=(Shape&& shape) noexcept;
    std::size_t& operator[](std::size_t index);

    bool operator==(const Shape& shape) const;
    bool operator!=(const Shape& shape) const;

    Shape operator*(const Shape& shape) const;

    [[nodiscard]] std::string ToString() const
    {
        std::string msg;
        msg += "Dim : " + std::to_string(Dim()) + " ";
        msg += " [";

        for (auto dim : m_shapeVector)
            msg += (std::to_string(dim) + " ");

        msg += " ] ";
        return msg;
    }

    [[nodiscard]] std::size_t At(std::size_t index) const;

    void Expand(std::size_t rank);

    void Shrink();

    void Squeeze();

    [[nodiscard]] std::size_t Dim() const;

    [[nodiscard]] std::size_t Size() const noexcept;

    [[nodiscard]] std::size_t NumRow() const
    {
        if (m_shapeVector.empty())
            return 0;
        if (m_shapeVector.size() == 1)
            return 1;
        return m_shapeVector.at(m_shapeVector.size() - 2);
    }

    [[nodiscard]] std::size_t NumCol() const
    {
        if (m_shapeVector.empty())
            return 0;
        return m_shapeVector.at(m_shapeVector.size() - 1);
    }

    void SetNumRows(std::size_t row)
    {
        if (m_shapeVector.size() == 1)
        {
            const auto col = NumCol();
            m_shapeVector = { row, col };
        }

        if (m_shapeVector.size() < 2)
            throw std::invalid_argument(
                "SetNumRows requires dimension larger than 1");
        m_shapeVector.at(m_shapeVector.size() - 2) = row;
    }

    void SetNumCols(std::size_t col)
    {
        if (m_shapeVector.empty())
            throw std::invalid_argument(
                "SetNumRows requires dimension larger than 0");
        m_shapeVector.at(m_shapeVector.size() - 1) = col;
    }

    Shape& ChangeDimension(std::size_t axis, std::size_t value)
    {
        if (axis >= m_shapeVector.size())
            throw std::invalid_argument(
                "Requested to change dimension " + std::to_string(axis) +
                " But this tensor has only " + std::to_string(
                    m_shapeVector.size()) + " dimensions");
        m_shapeVector.at(axis) = value;
        return *this;
    }

    Shape& Reshape(std::initializer_list<std::size_t> newShape);

    void Transpose();

    [[nodiscard]] Shape GetTransposedShape() const;

private:
    std::vector<std::size_t> m_shapeVector;
};
} // namespace Takion

#endif
