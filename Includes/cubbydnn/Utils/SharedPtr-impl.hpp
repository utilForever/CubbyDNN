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
template <typename U>
SharedPtr<T>::SharedPtr(U* objectPtr, SharedObjectInfo* informationPtr)
    : m_objectPtr(objectPtr), m_sharedObjectInfoPtr(informationPtr)
{
    static_assert(std::is_same<T, U>::value || std::is_base_of<T, U>::value);
}

template <typename T>
template <typename U>
SharedPtr<T>::SharedPtr(const SharedPtr<U>& sharedPtr)
{
    static_assert(std::is_same<T, U>::value || std::is_base_of<T, U>::value);

    int oldRefCount = sharedPtr.m_sharedObjectInfoPtr->RefCount.load(
        std::memory_order_relaxed);
    while (!sharedPtr.m_sharedObjectInfoPtr->RefCount.compare_exchange_weak(
        oldRefCount, oldRefCount + 1, std::memory_order_release,
        std::memory_order_relaxed))
        ;
    m_objectPtr = sharedPtr.m_objectPtr;
    m_sharedObjectInfoPtr = sharedPtr.m_sharedObjectInfoPtr;
}

template <typename T>
template <typename U>
SharedPtr<T>::SharedPtr(SharedPtr<U>&& sharedPtr) noexcept
{
    static_assert(std::is_same<T, U>::value || std::is_base_of<T, U>::value);

    m_objectPtr = sharedPtr.m_objectPtr;
    m_sharedObjectInfoPtr = sharedPtr.m_sharedObjectInfoPtr;
    sharedPtr.m_objectPtr = nullptr;
    sharedPtr.m_sharedObjectInfoPtr = nullptr;
}

template <typename T>
template <typename U>
SharedPtr<T>& SharedPtr<T>::operator=(const SharedPtr<U>& sharedPtr)
{
    static_assert(std::is_same<T, U>::value || std::is_base_of<T, U>::value);

    int oldRefCount =
        sharedPtr.m_sharedObjectInfoPtr->RefCount.load(std::memory_order_relaxed);
    while (!sharedPtr.m_sharedObjectInfoPtr->RefCount.compare_exchange_weak(
        oldRefCount, oldRefCount + 1, std::memory_order_release,
        std::memory_order_relaxed))
        ;

    m_objectPtr = sharedPtr.m_objectPtr;
    m_sharedObjectInfoPtr = sharedPtr.m_sharedObjectInfoPtr;
    return *this;
}

template <typename T>
template <typename U>
SharedPtr<T>& SharedPtr<T>::operator=(SharedPtr<U>&& sharedPtr) noexcept
{
    static_assert(std::is_same<T, U>::value || std::is_base_of<T, U>::value);

    m_objectPtr = sharedPtr.m_objectPtr;
    m_sharedObjectInfoPtr = sharedPtr.m_sharedObjectInfoPtr;

    sharedPtr.m_objectPtr = nullptr;
    sharedPtr.m_sharedObjectInfoPtr = nullptr;
    return *this;
}

template <typename T>
SharedPtr<T>::~SharedPtr()
{
    if (m_sharedObjectInfoPtr)
    {
        if (m_sharedObjectInfoPtr->RefCount == 1)
        {
            delete m_sharedObjectInfoPtr;
            delete m_objectPtr;
        }
        else
        {
            m_sharedObjectInfoPtr->RefCount.fetch_sub(
                1, std::memory_order_release);
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
    T* objectPtr = new T(std::forward<Ts>(args)...);
    auto* infoPtr = new SharedObjectInfo();
    return SharedPtr<T>(objectPtr, infoPtr);
}

template <typename T>
T* SharedPtr<T>::operator->() const
{
    return m_objectPtr;
}

}  // namespace CubbyDNN

#endif  // CUBBYDNN_SHAREDPTR_IMPL_HPP
