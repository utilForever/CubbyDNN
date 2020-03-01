#ifndef CUBBYDNN_SHAPE_HPP
#define CUBBYDNN_SHAPE_HPP

#include <vector>

namespace CubbyDNN
{
class Shape
{
 public:
    Shape();
    Shape(std::initializer_list<std::size_t> dimension);
    ~Shape() noexcept = default;
    Shape(const Shape& rhs) = default;
    Shape(Shape&& rhs) noexcept = default;

    template <typename T>
    Shape(T begin, T end) : m_dimension(begin, end)
    {
        // Do nothing
    }

    Shape& operator=(const Shape& rhs) = default;
    Shape& operator=(Shape&& rhs) noexcept = default;
    Shape& operator=(std::initializer_list<std::size_t> dimension);

    bool operator==(const Shape& rhs) const;
    bool operator==(std::initializer_list<std::size_t> dimension) const;
    bool operator!=(const Shape& rhs) const;
    bool operator!=(std::initializer_list<std::size_t> dimension) const;

    std::size_t& operator[](std::size_t numAxis);
    std::size_t operator[](std::size_t numAxis) const;

    std::size_t Rank() const noexcept;
    std::size_t Size() const noexcept;
    Shape Expand() const;
    Shape Expand(std::size_t numRank) const;
    Shape Shrink() const;
    Shape Squeeze() const;
    static Shape Broadcast(const Shape& left, const Shape& right);

    template <typename T>
    void Assign(T begin, T end)
    {
        m_dimension.assign(begin, end);
    }

    template <typename T>
    bool Equals(T begin, T end) const
    {
        auto dimBegin{ this->m_dimension.cbegin() };
        const auto dimEnd{ this->m_dimension.cend() };

        for (; begin != end && dimBegin != dimEnd; ++begin, ++dimBegin)
        {
            if (*begin != *dimBegin)
            {
                return false;
            }
        }

        for (; begin != end; ++begin)
        {
            if (*begin != 1)
            {
                return false;
            }
        }

        for (; dimBegin != dimEnd; ++dimBegin)
        {
            if (*dimBegin != 1)
            {
                return false;
            }
        }

        return true;
    }

    friend void Swap(Shape& left, Shape& right) noexcept
    {
        std::swap(left.m_dimension, right.m_dimension);
    }

 private:
    std::vector<std::size_t> m_dimension;
};
}  // namespace CubbyDNN

#endif