#include <CubbyDNN/Core/Shape.hpp>

namespace CubbyDNN
{
Shape::Shape()
{
    m_dimension.reserve(4);
}

Shape::Shape(std::initializer_list<std::size_t> dimension)
    : Shape(dimension.begin(), dimension.end())
{
    // Do nothing
}

Shape& Shape::operator=(std::initializer_list<std::size_t> dimension)
{
    Assign(dimension.begin(), dimension.end());

    return *this;
}

bool Shape::operator==(std::initializer_list<std::size_t> dimension) const
{
    return Equals(dimension.begin(), dimension.end());
}

bool Shape::operator==(const Shape& rhs) const
{
    return Equals(rhs.m_dimension.cbegin(), rhs.m_dimension.cend());
}

bool Shape::operator!=(std::initializer_list<std::size_t> dimension) const
{
    return !this->operator==(dimension);
}

bool Shape::operator!=(const Shape& rhs) const
{
    return !this->operator==(rhs);
}

std::size_t& Shape::operator[](std::size_t numAxis)
{
    return m_dimension[numAxis];
}

std::size_t Shape::operator[](std::size_t numAxis) const
{
    return m_dimension[numAxis];
}
}  // namespace CubbyDNN