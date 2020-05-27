#ifndef CUBBYDNN_SPAN_IMPL_HPP
#define CUBBYDNN_SPAN_IMPL_HPP

#include <algorithm>
#include <cmath>

namespace CubbyDNN::Core
{
template <typename T>
Span<T>::Span() noexcept : m_base(nullptr), m_length(0)
{
    // Do nothing
}

template <typename T>
Span<T>::Span(T* base, std::size_t length) noexcept
    : m_base(base), m_length(length)
{
    // Do nothing
}

template <typename T>
template <typename Iter>
Span<T>::Span(Iter begin, Iter end) noexcept
    : m_base(&*begin),
      m_length(static_cast<std::size_t>(std::distance(begin, end)))
{
    // Do nothing
}

template <typename T>
T& Span<T>::operator[](std::size_t index)
{
    return m_base[index];
}

template <typename T>
const T& Span<T>::operator[](std::size_t index) const
{
    return m_base[index];
}

template <typename T>
T* Span<T>::begin() noexcept
{
    return m_base;
}

template <typename T>
const T* Span<T>::begin() const noexcept
{
    return m_base;
}

template <typename T>
const T* Span<T>::cbegin() const noexcept
{
    return m_base;
}

template <typename T>
T* Span<T>::end() noexcept
{
    return m_base + m_length;
}

template <typename T>
const T* Span<T>::end() const noexcept
{
    return m_base + m_length;
}

template <typename T>
const T* Span<T>::cend() const noexcept
{
    return m_base + m_length;
}

template <typename T>
std::size_t Span<T>::Length() const noexcept
{
    return m_length;
}

template <typename T>
T* Span<T>::Min()
{
    return std::min_element(begin(), end());
}

template <typename T>
const T* Span<T>::Min() const
{
    return std::min_element(begin(), end());
}

template <typename T>
T* Span<T>::Max()
{
    return std::max_element(begin(), end());
}

template <typename T>
const T* Span<T>::Max() const
{
    return std::max_element(begin(), end());
}

template <typename T>
Span<T> Span<T>::SubSpan(std::size_t offset) const noexcept
{
    return Span(m_base + offset, m_base + m_length);
}

template <typename T>
Span<T> Span<T>::SubSpan(std::size_t offset, std::size_t length) const noexcept
{
    return Span(m_base + offset, m_base + std::min(m_length, length));
}

template <typename T>
void Span<T>::FillZero()
{
    std::fill(m_base, m_base + m_length, static_cast<T>(0));
}

template <typename T>
void Span<T>::FillOne()
{
    std::fill(m_base, m_base + m_length, static_cast<T>(1));
}

template <typename T>
void Span<T>::FillScalar(T scalar)
{
    std::fill(m_base, m_base + m_length, scalar);
}

template <typename T>
void Span<T>::CopyFrom(const Span& span)
{
    std::copy(span.m_base, span.m_base + std::min(span.m_length, m_length),
              m_base);
}

template <typename T>
void Span<T>::AccumulateFrom(const Span& span)
{
    std::transform(span.m_base, span.m_base + std::min(span.m_length, m_length),
                   m_base, m_base, std::plus<T>());
}

template <typename T>
void Span<T>::AccumulateFrom(T factor, const Span& span)
{
    std::transform(span.m_base, span.m_base + std::min(span.m_length, m_length),
                   m_base, m_base, [=](auto left, auto right) {
                       return std::fma(left, factor, right);
                   });
}
}  // namespace CubbyDNN::Core

#endif