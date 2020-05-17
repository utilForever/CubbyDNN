#ifndef CUBBYDNN_MEMORY_IMPL_HPP
#define CUBBYDNN_MEMORY_IMPL_HPP

namespace CubbyDNN::Core
{
template <typename T>
Memory<T>::Memory() : m_size(0), m_capacity(0), m_pointer()
{
    // Do nothing
}

template <typename T>
Memory<T>::Memory(std::size_t size)
    : m_size(size), m_capacity(size), m_pointer(std::make_unique<T[]>(size))
{
    // Do nothing
}

template <typename T>
Memory<T>::Memory(const Memory& rhs)
    : m_size(rhs.m_size),
      m_capacity(rhs.m_capacity),
      m_pointer(std::make_unique<T[]>(rhs.m_capacity))
{
    std::memcpy(m_pointer.get(), rhs.m_pointer.get(), sizeof(T) * rhs.m_size);
}

template <typename T>
Memory<T>::Memory(Memory&& rhs) noexcept
    : m_size(rhs.m_size),
      m_capacity(rhs.m_capacity),
      m_pointer(std::move(rhs.m_pointer))
{
    // Do nothing
}

template <typename T>
Memory<T>& Memory<T>::operator=(const Memory& rhs)
{
    Swap(*this, rhs);

    return *this;
}

template <typename T>
Memory<T>& Memory<T>::operator=(Memory&& rhs) noexcept
{
    Swap(*this, rhs);

    return *this;
}

template <typename T>
std::size_t Memory<T>::Size() const noexcept
{
    return m_size;
}

template <typename T>
std::size_t Memory<T>::Capacity() const noexcept
{
    return m_capacity;
}

template <typename T>
Span<T> Memory<T>::GetSpan() const noexcept
{
    return Span(m_pointer.get(), m_size);
}

template <typename T>
void Memory<T>::Resize(std::size_t size)
{
    m_size = size;

    if (m_capacity >= m_size)
    {
        return;
    }

    m_capacity = size;
    m_pointer = std::make_unique<T[]>(m_capacity);
}

template <typename T>
void Memory<T>::Reserve(std::size_t capacity)
{
    if (m_capacity >= capacity)
    {
        return;
    }

    m_capacity = capacity;
    m_pointer = std::make_unique<T[]>(m_capacity);
}

template <typename T>
void Swap(Memory<T>& left, Memory<T>& right) noexcept
{
    using std::swap;

    swap(left.m_size, right.m_size);
    swap(left.m_capacity, right.m_capacity);
    swap(left.m_pointer, right.m_pointer);
}
}  // namespace CubbyDNN::Core

#endif