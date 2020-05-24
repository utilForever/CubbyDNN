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

template <class T>
T* Span<T>::begin() noexcept
{
    return m_base;
}

template <class T>
const T* Span<T>::begin() const noexcept
{
    return m_base;
}

template <class T>
const T* Span<T>::cbegin() const noexcept
{
    return m_base;
}

template <class T>
T* Span<T>::end() noexcept
{
    return m_base + m_length;
}

template <class T>
const T* Span<T>::end() const noexcept
{
    return m_base + m_length;
}

template <class T>
const T* Span<T>::cend() const noexcept
{
    return m_base + m_length;
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