//
// Created by jwkim98 on 6/21/19.
//

#ifndef CUBBYDNN_PTRWRAPPER_IMPL_HPP
#define CUBBYDNN_PTRWRAPPER_IMPL_HPP

#include <cubbydnn/Utils/PtrWrapper.hpp>

namespace CubbyDNN
{
template <typename T>
SharedPtr<T>::SharedPtr(SharedObject* objectPtr, SharedPtrState state)
    : m_sharedObjectPtr(objectPtr), m_ptrState(state){};

template <typename T>
SharedPtr<T> SharedPtr<T>::tryMakeCopy()
{
    const int oldRefCount = m_sharedObjectPtr->RefCount;
    if (oldRefCount < m_sharedObjectPtr->MaxRefCount)
    {
        if (m_sharedObjectPtr->RefCount.compare_exchange_strong(
                oldRefCount, oldRefCount + 1))
            return SharedPtr<T>(m_sharedObjectPtr, SharedPtrState::valid);
        else
            return SharedPtr<T>(nullptr, SharedPtrState::dirty);
    }
    else
        return SharedPtr<T>(nullptr, SharedPtrState::invalid);
}

template <typename T>
SharedPtr<T> SharedPtr<T>::Make()
{
    T* ptr = new T();
    return SharedPtr<T>(ptr, SharedPtrState::valid);
}

template <typename T>
template <typename... Ts>
SharedPtr<T> SharedPtr<T>::Make(int maxReferenceCount, Ts&... args)
{
    T* ptr = new T(args...);
    return std::move(SharedPtr<T>(ptr, SharedPtrState::valid));
}

template <typename T>
SharedPtr<T>::SharedPtr(SharedPtr<T>&& sharedPtr) noexcept
    : m_sharedObjectPtr(std::move(sharedPtr.m_sharedObjectPtr)),
      m_ptrState(m_ptrState)
{
    sharedPtr.m_sharedObjectPtr = nullptr;
    m_ptrState = SharedPtrState::invalid;
}

template <typename T>
SharedPtr<T>& SharedPtr<T>::operator=(SharedPtr<T>&& sharedPtr) noexcept
{
    m_sharedObjectPtr = sharedPtr.m_sharedObjectPtr;
    m_ptrState = sharedPtr.m_ptrState;

    sharedPtr.m_sharedObjectPtr = nullptr;
    m_ptrState = SharedPtrState::invalid;
    return *this;
}

template <typename T>
SharedPtr<T> SharedPtr<T>::MakeCopy()
{
    if(m_ptrState == SharedPtrState::valid) {
        auto sharedPtr = tryMakeCopy();
        while (sharedPtr.GetState()==SharedPtrState::dirty)
            sharedPtr = tryMakeCopy();
        return sharedPtr;
    }
    return SharedPtr<T>(nullptr, SharedPtrState::invalid);
}

}  // namespace CubbyDNN

#endif  // CUBBYDNN_PTRWRAPPER_IMPL_HPP
