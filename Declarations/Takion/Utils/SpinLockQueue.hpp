// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_SPINLOCKQUEUE_HPP
#define CUBBYDNN_SPINLOCKQUEUE_HPP

#include <Takion/Utils/SpinLockQueue-Decl.hpp>
#include <type_traits>


namespace Takion
{
template <typename T>
SpinLockQueue<T>::SpinLockQueue(std::size_t maxCapacity)
    : m_maxCapacity(maxCapacity), m_container(maxCapacity)
{
}

template <typename T>
SpinLockQueue<T>::SpinLockQueue(SpinLockQueue<T> &&spinLockQueue) noexcept
    : m_maxCapacity(spinLockQueue.m_maxCapacity),
      m_spinLock(std::move(spinLockQueue.m_spinLock)),
      m_container(spinLockQueue.m_container),
      m_frontIndex(spinLockQueue.m_frontIndex),
      m_backIndex(spinLockQueue.m_backIndex),
      m_empty(spinLockQueue.m_empty)
{
}

template <typename T>
template <typename U>
void SpinLockQueue<T>::Enqueue(U &&object)
{
   // static_assert(std::is_same<std::decay_t<T>, std::decay_t<U>>::value);
    while (!TryEnqueue(std::forward<U>(object)))
        std::this_thread::yield();
        ;
}

template <typename T>
template <typename U>
bool SpinLockQueue<T>::TryEnqueue(U &&object)
{
    /// Check if type of class and argument is identical
   // static_assert(std::is_same<std::decay_t<T>, std::decay_t<U>>::value);

    bool isAvailable = false;
    m_spinLock.ExclusiveLock();
    if (this->size() < m_maxCapacity)
    {
        if (m_empty)
            m_empty = false;
        else
            incrementBack();

        m_container.at(m_backIndex) = std::forward<U>(object);
        isAvailable = true;
    }
    else
        isAvailable = false;

    m_spinLock.ExclusiveRelease();
    return isAvailable;
}

template <typename T>
T SpinLockQueue<T>::Dequeue()
{
    while (m_empty)
        std::this_thread::yield();
        
    auto rtn = TryDequeue();
    while (!std::get<1>(rtn))
    {
        rtn = TryDequeue();
    }
    return std::get<0>(rtn);
}

template <typename T>
std::tuple<T, bool> SpinLockQueue<T>::TryDequeue()
{
    bool isSuccessful = false;
    m_spinLock.ExclusiveLock();
    T temp;
    // assert(!m_empty);
    if (!m_empty)
    {
        temp = m_container.at(m_frontIndex);
        if (m_frontIndex == m_backIndex)
            m_empty = true;
        else
            incrementFront();
        isSuccessful = true;
    }
    m_spinLock.ExclusiveRelease();
    return std::make_tuple(temp, isSuccessful);
}

template <typename T>
T &SpinLockQueue<T>::At(std::size_t index)
{
    // assert(index > size());
    m_spinLock.SharedLock();
    std::size_t vectorIndex = m_frontIndex + index;
    if (vectorIndex >= m_maxCapacity)
    {
        vectorIndex -= m_maxCapacity;
    }
    T returnVal = m_container.at(vectorIndex);
    m_spinLock.SharedRelease();
    return std::move(returnVal);
}

template <typename T>
std::size_t SpinLockQueue<T>::Size()
{
    m_spinLock.SharedLock();
    const std::size_t currentSize = size();
    m_spinLock.SharedRelease();
    return currentSize;
}

template <typename T>
void SpinLockQueue<T>::incrementBack()
{
    if (m_backIndex == m_maxCapacity - 1)
    {
        m_backIndex = 0;
    }
    else
        m_backIndex += 1;
}

template <typename T>
void SpinLockQueue<T>::incrementFront()
{
    if (m_frontIndex == m_maxCapacity - 1)
    {
        m_frontIndex = 0;
    }
    else
        m_frontIndex += 1;
}

template <typename T>
std::size_t SpinLockQueue<T>::size()
{
    if (m_empty)
        return 0;

    if (m_backIndex < m_frontIndex)
    {
        return (m_maxCapacity - m_frontIndex) + m_backIndex + 1;
    }
    else
    {
        return m_backIndex - m_frontIndex + 1;
    }
}
}  // namespace Captain

#endif  // CAPTAIN_SPINLOCKQUEUE_HPP