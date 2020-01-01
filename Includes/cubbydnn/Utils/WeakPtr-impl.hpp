// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#pragma once

#include <cubbydnn/Utils/WeakPtr.hpp>

namespace CubbyDNN
{
template <typename T>
constexpr WeakPtr<T>::WeakPtr()
    : m_objectPtr(nullptr), m_sharedObjectInfoPtr(nullptr)
{
}

template <typename T>
template <typename U>
WeakPtr<T>::WeakPtr(const SharedPtr<U>& sharedPtr)
{
    static_assert(std::is_same<T, U>::value || std::is_base_of<T, U>::value);

    m_objectPtr = sharedPtr.m_objectPtr;
    m_sharedObjectInfoPtr = sharedPtr.m_sharedObjectInfoPtr;
}

template <typename T>
template <typename U>
WeakPtr<T>::WeakPtr(WeakPtr<U>&& weakPtr)
    : m_objectPtr(weakPtr.m_objectPtr),
      m_sharedObjectInfoPtr(weakPtr.m_sharedObjectInfoPtr)
{
    static_assert(std::is_same<T, U>::value || std::is_base_of<T, U>::value);
    weakPtr.m_objectPtr = nullptr;
    weakPtr.m_sharedObjectInfoPtr = nullptr;
}

template <typename T>
template <typename U>
WeakPtr<T>::WeakPtr(const WeakPtr<U>& weakPtr)
    : m_objectPtr(weakPtr.m_weakPtr),
      m_sharedObjectInfoPtr(weakPtr.m_sharedObjectInfoPtr)
{
    static_assert(std::is_same<T, U>::value || std::is_base_of<T, U>::value);
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
T* WeakPtr<T>::operator->() const
{
    return m_objectPtr;
}



}  // namespace CubbyDNN
