#include <CubbyDNN/Core/Shape.hpp>

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace CubbyDNN::Core
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

std::size_t Shape::Rank() const noexcept
{
    return m_dimension.size();
}

std::size_t Shape::Size() const noexcept
{
    return std::accumulate(m_dimension.cbegin(), m_dimension.cend(),
                           static_cast<std::size_t>(1),
                           std::multiplies<std::size_t>{});
}

Shape Shape::Expand() const
{
    Shape result{ *this };

    result.m_dimension.emplace_back(1);

    return result;
}

Shape Shape::Expand(std::size_t numRank) const
{
    Shape result{ *this };

    if (numRank < Rank())
    {
        return result;
    }

    for (const std::size_t rank{ Rank() }; rank < numRank; ++numRank)
    {
        result.m_dimension.emplace_back(1);
    }

    return result;
}

Shape Shape::Shrink() const
{
    Shape result{ *this };

    while (!result.m_dimension.empty() && result.m_dimension.back() == 1)
    {
        result.m_dimension.pop_back();
    }

    return result;
}

Shape Shape::Squeeze() const
{
    Shape result;

    for (auto numSize : m_dimension)
    {
        if (numSize != 1)
        {
            result.m_dimension.emplace_back(numSize);
        }
    }

    return result;
}

Shape Shape::Broadcast(const Shape& left, const Shape& right)
{
    if (!left.Rank() && !right.Rank())
    {
        return Shape{};
    }

    Shape result;
    std::size_t numAxis{ 0 };

    for (const auto numMinAxis{ std::min(left.Rank(), right.Rank()) };
         numAxis < numMinAxis; ++numAxis)
    {
        if (left[numAxis] != right[numAxis] && left[numAxis] != 1 &&
            right[numAxis] != 1)
        {
            throw std::runtime_error("Unable to broadcast");
        }

        result.m_dimension.emplace_back(
            std::max(left[numAxis], right[numAxis]));
    }

    for (const auto numMaxAxis{ std::max(left.Rank(), right.Rank()) };
         numAxis < numMaxAxis; ++numAxis)
    {
        result.m_dimension.emplace_back(
            (numAxis < left.Rank() ? left : right)[numAxis]);
    }

    return result;
}
}  // namespace CubbyDNN