//
// Created by jwkim98 on 6/21/19.
//

#ifndef CUBBYDNN_SHAREDPTR_IMPL_HPP
#define CUBBYDNN_SHAREDPTR_IMPL_HPP

#include <cubbydnn/Utils/SharedPtr.hpp>

namespace CubbyDNN
{
template <typename T>
SharedPtr<T>::SharedPtr(SharedObject* objectPtr, PtrState state)
    : m_sharedObjectPtr(objectPtr), m_ptrState(state){};

template <typename T>
SharedPtr<T>::SharedPtr()
    : m_sharedObjectPtr(nullptr), m_ptrState(PtrState::invalid)
{
}

template <typename T>
SharedPtr<T>::~SharedPtr()
{
    if (m_sharedObjectPtr)
    {
        if (m_sharedObjectPtr->RefCount == 1)
        {
            delete m_sharedObjectPtr;
        }
        else
        {
            m_sharedObjectPtr->RefCount--;
        }
    }
}

template <typename T>
SharedPtr<T> SharedPtr<T>::tryMakeCopy()
{
    int oldRefCount = m_sharedObjectPtr->RefCount;
    if (oldRefCount < m_sharedObjectPtr->MaxRefCount)
    {
        if (m_sharedObjectPtr->RefCount.compare_exchange_strong(
                oldRefCount, (oldRefCount + 1), std::memory_order_relaxed))
            return SharedPtr<T>(m_sharedObjectPtr, PtrState::valid);
        else
            return SharedPtr<T>(nullptr, PtrState::dirty);
    }
    else
        return SharedPtr<T>(nullptr, PtrState::invalid);
}

template <typename T>
SharedPtr<T> SharedPtr<T>::Make(int maxReferenceCount)
{
    auto* ptr = new SharedObject(T(), maxReferenceCount);
    return SharedPtr<T>(ptr, PtrState::valid);
}

template <typename T>
template <typename... Ts>
SharedPtr<T> SharedPtr<T>::Make(int maxReferenceCount, Ts&&... args)
{
    auto* ptr = new SharedObject(T(args...), maxReferenceCount);
    return SharedPtr<T>(ptr, PtrState::valid);
}

template <typename T>
SharedPtr<T>::SharedPtr(SharedPtr<T>&& sharedPtr) noexcept
    : m_sharedObjectPtr(std::move(sharedPtr.m_sharedObjectPtr)),
      m_ptrState(sharedPtr.m_ptrState)
{
    sharedPtr.m_sharedObjectPtr = nullptr;
    sharedPtr.m_ptrState = PtrState::invalid;
}

template <typename T>
SharedPtr<T>& SharedPtr<T>::operator=(SharedPtr<T>&& sharedPtr) noexcept
{
    m_sharedObjectPtr = sharedPtr.m_sharedObjectPtr;
    m_ptrState = sharedPtr.m_ptrState;

    sharedPtr.m_sharedObjectPtr = nullptr;
    sharedPtr.m_ptrState = PtrState::invalid;
    return *this;
}

template <typename T>
T* SharedPtr<T>::operator->()
{
    return &m_sharedObjectPtr->Object;
}

template <typename T>
SharedPtr<T> SharedPtr<T>::MakeCopy()
{
    if (m_ptrState == PtrState::valid)
    {
        auto sharedPtr = tryMakeCopy();
        while (sharedPtr.GetState() == PtrState::dirty)
            sharedPtr = tryMakeCopy();
        return sharedPtr;
    }
    return SharedPtr<T>(nullptr, PtrState::invalid);
}

}  // namespace CubbyDNN

#endif  // CUBBYDNN_SHAREDPTR_IMPL_HPP
