// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#pragma once
#include <Takion/Utils/WeakPtr-Decl.hpp>

namespace Takion
{

template <typename T>
template <typename U>
WeakPtr<T>::WeakPtr(const WeakPtr<U>& weakPtr)
    : m_objectPtr(weakPtr.m_objectPtr),
      m_sharedObjectInfoPtr(weakPtr.m_sharedObjectInfoPtr)
{
    static_assert(std::is_same<std::decay_t<T>, std::decay_t<U>>::value ||
                  std::is_base_of<std::decay_t<T>, std::decay_t<U>>::value);
}

template <typename T>
WeakPtr<T>::WeakPtr(const WeakPtr<T>& weakPtr)
    : m_objectPtr(weakPtr.m_objectPtr),
      m_sharedObjectInfoPtr(weakPtr.m_sharedObjectInfoPtr)
{
}

template <typename T>
template <typename U>
WeakPtr<T>::WeakPtr(const SharedPtr<U>& sharedPtr)
{
    static_assert(std::is_same<std::decay_t<T>, std::decay_t<U>>::value ||
                  std::is_base_of<std::decay_t<T>, std::decay_t<U>>::value);
    m_objectPtr = sharedPtr.m_objectPtr;
    m_sharedObjectInfoPtr = sharedPtr.m_sharedObjectInfoPtr;
}

template <typename T>
template <typename U>
WeakPtr<T>::WeakPtr(WeakPtr<U>&& weakPtr) noexcept
    : m_objectPtr(weakPtr.m_objectPtr),
      m_sharedObjectInfoPtr(weakPtr.m_sharedObjectInfoPtr)
{
    static_assert(std::is_same<std::decay_t<T>, std::decay_t<U>>::value ||
                  std::is_base_of<std::decay_t<T>, std::decay_t<U>>::value);
    weakPtr.m_objectPtr = nullptr;
    weakPtr.m_sharedObjectInfoPtr = nullptr;
}

template <typename T>
WeakPtr<T>::WeakPtr(WeakPtr<T>&& weakPtr) noexcept
    : m_objectPtr(weakPtr.m_objectPtr),
      m_sharedObjectInfoPtr(weakPtr.m_sharedObjectInfoPtr)
{
    weakPtr.m_objectPtr = nullptr;
    weakPtr.m_sharedObjectInfoPtr = nullptr;
}

template <typename T>
SharedPtr<T> WeakPtr<T>::Lock() const
{
    int oldRefCount =
        m_sharedObjectInfoPtr->RefCount.load(std::memory_order_relaxed);
    while (!m_sharedObjectInfoPtr->RefCount.compare_exchange_weak(
        oldRefCount, oldRefCount + 1, std::memory_order_release,
        std::memory_order_relaxed))
        ;
    return SharedPtr<T>(m_objectPtr, m_sharedObjectInfoPtr);
}

template <typename T>
template <typename U>
WeakPtr<T>& WeakPtr<T>::operator=(const WeakPtr<U>& weakPtr)
{
    static_assert(std::is_same<std::decay_t<T>, std::decay_t<U>>::value ||
                  std::is_base_of<std::decay_t<T>, std::decay_t<U>>::value);

    m_objectPtr = weakPtr.m_objectPtr;
    m_sharedObjectInfoPtr = weakPtr.m_sharedObjectInfoPtr;
    return *this;
}

template <typename T>
WeakPtr<T>& WeakPtr<T>::operator=(const WeakPtr<T>& weakPtr)
{
    m_objectPtr = weakPtr.m_objectPtr;
    m_sharedObjectInfoPtr = weakPtr.m_sharedObjectInfoPtr;
    return *this;
}

template <typename T>
template <typename U>
WeakPtr<T>& WeakPtr<T>::operator=(WeakPtr<U>&& weakPtr) noexcept
{
    static_assert(std::is_same<std::decay_t<T>, std::decay_t<U>>::value ||
                  std::is_base_of<std::decay_t<T>, std::decay_t<U>>::value);

    if (weakPtr == *this)
        return *this;

    m_objectPtr = weakPtr.m_objectPtr;
    m_sharedObjectInfoPtr = weakPtr.m_sharedObjectInfoPtr;

    weakPtr.m_objectPtr = nullptr;
    weakPtr.m_sharedObjectInfoPtr = nullptr;

    return *this;
}

template <typename T>
WeakPtr<T>& WeakPtr<T>::operator=(WeakPtr<T>&& weakPtr) noexcept
{
    if (weakPtr == *this) // Check for self assignment
        return *this;
    m_objectPtr = weakPtr.m_objectPtr;
    m_sharedObjectInfoPtr = weakPtr.m_sharedObjectInfoPtr;

    weakPtr.m_objectPtr = nullptr;
    weakPtr.m_sharedObjectInfoPtr = nullptr;

    return *this;
}

template <typename T>
T* WeakPtr<T>::operator->() const
{
    return m_objectPtr;
}
}  // namespace Takion
