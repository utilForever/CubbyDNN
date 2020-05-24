#ifndef CUBBYDNN_SPAN_HPP
#define CUBBYDNN_SPAN_HPP

#include <cstddef>

namespace CubbyDNN::Core
{
template <typename T>
class Span
{
 public:
    Span() noexcept;
    Span(T* base, std::size_t length) noexcept;
    template <class Iter>
    Span(Iter begin, Iter end) noexcept;

	T& operator[](std::size_t index);
    const T& operator[](std::size_t index) const;

    T* begin() noexcept;
    const T* begin() const noexcept;
    const T* cbegin() const noexcept;
    T* end() noexcept;
    const T* end() const noexcept;
    const T* cend() const noexcept;
    std::size_t Length() const noexcept;

    T* Min();
    const T* Min() const;
    T* Max();
    const T* Max() const;

    Span SubSpan(std::size_t offset) const noexcept;
    Span SubSpan(std::size_t offset, std::size_t length) const noexcept;

    void FillZero();
    void FillOne();

    void CopyFrom(const Span& span);

    void AccumulateFrom(const Span& span);
    void AccumulateFrom(T factor, const Span& span);

 private:
    T* m_base;
    std::size_t m_length;
};
}  // namespace CubbyDNN::Core

#include <CubbyDNN/Core/Span-Impl.hpp>

#endif