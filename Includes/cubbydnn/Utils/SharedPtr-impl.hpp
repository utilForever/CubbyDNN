// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_SHAREDPTR_IMPL_HPP
#define CUBBYDNN_SHAREDPTR_IMPL_HPP

#include <cubbydnn/Utils/SharedPtr.hpp>

namespace CubbyDNN
{
template <typename T>
SharedPtr<T>::SharedPtr(T* objectPtr, SharedObjectInfo* informationPtr)
    : m_objectPtr(objectPtr), m_sharedObjectPtr(informationPtr)
{
}

template <typename T>
SharedPtr<T>::SharedPtr(const SharedPtr<T>& sharedPtr)
{
    int oldRefCount =
        sharedPtr.m_sharedObjectPtr->RefCount.load(std::memory_order_relaxed);
    while (!sharedPtr.m_sharedObjectPtr->RefCount.compare_exchange_weak(
        oldRefCount, oldRefCount + 1, std::memory_order_release,
        std::memory_order_relaxed))
        ;
    m_objectPtr = sharedPtr.m_objectPtr;
    m_sharedObjectPtr = sharedPtr.m_sharedObjectPtr;
}

template <typename T>
SharedPtr<T>::SharedPtr(SharedPtr<T>&& sharedPtr) noexcept
    : m_objectPtr(sharedPtr.m_objectPtr),
      m_sharedObjectPtr(std::move(sharedPtr.m_sharedObjectPtr))
{
    sharedPtr.m_objectPtr = nullptr;
    sharedPtr.m_sharedObjectPtr = nullptr;
}

template <typename T>
SharedPtr<T>& SharedPtr<T>::operator=(const SharedPtr<T>& sharedPtr)
{
    int oldRefCount =
        sharedPtr.m_sharedObjectPtr->RefCount.load(std::memory_order_relaxed);
    while (!sharedPtr.m_sharedObjectPtr->RefCount.compare_exchange_weak(
        oldRefCount, oldRefCount + 1, std::memory_order_release,
        std::memory_order_relaxed))
        ;

    m_objectPtr = sharedPtr.m_objectPtr;
    m_sharedObjectPtr = sharedPtr.m_sharedObjectPtr;
    return *this;
}

template <typename T>
SharedPtr<T>& SharedPtr<T>::operator=(SharedPtr<T>&& sharedPtr) noexcept
{
    m_objectPtr = sharedPtr.m_objectPtr;
    m_sharedObjectPtr = sharedPtr.m_sharedObjectPtr;

    sharedPtr.m_objectPtr = nullptr;
    sharedPtr.m_sharedObjectPtr = nullptr;
    return *this;
}

template <typename T>
SharedPtr<T>::~SharedPtr()
{
    if (m_sharedObjectPtr)
    {
        if (m_sharedObjectPtr->RefCount == 1)
        {
            delete m_sharedObjectPtr;
            delete m_objectPtr;
        }
        else
        {
            m_sharedObjectPtr->RefCount.fetch_sub(1, std::memory_order_release);
        }
    }
}

template <typename T>
SharedPtr<T> SharedPtr<T>::Make(T* objectPtr)
{
    auto* infoPtr = new SharedObjectInfo();
    return SharedPtr<T>(objectPtr, infoPtr);
}

template <typename T>
SharedPtr<T> SharedPtr<T>::Make()
{
    T* objectPtr = new T();
    auto* infoPtr = new SharedObjectInfo();
    return SharedPtr<T>(objectPtr, infoPtr);
}

template <typename T>
template <typename... Ts>
SharedPtr<T> SharedPtr<T>::Make(Ts&&... args)
{
    T* objectPtr = new T(args...);
    auto* infoPtr = new SharedObjectInfo();
    return SharedPtr<T>(objectPtr, infoPtr);
}

template <typename T>
T* SharedPtr<T>::operator->()
{
    return m_objectPtr;
}

template <typename T>
const T* SharedPtr<T>::operator->() const
{
    return m_objectPtr;
}
}  // namespace CubbyDNN

#endif  // CUBBYDNN_SHAREDPTR_IMPL_HPP
