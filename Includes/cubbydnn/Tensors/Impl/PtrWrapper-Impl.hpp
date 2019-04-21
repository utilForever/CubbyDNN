//
// Created by jwkim98 on 4/21/19.
//

#ifndef CUBBYDNN_PTRWRAPPER_IMPL_HPP
#define CUBBYDNN_PTRWRAPPER_IMPL_HPP

#include <cubbydnn/Tensors/Decl/PtrWrapper.hpp>

namespace CubbyDNN
{

template <typename T>
Ptr<TensorObject<T>>::Ptr(Ptr<TensorObject<T>>&& tensorObjectPtr) noexcept
    : m_tensorObjectPtr(std::move(tensorObjectPtr))
{
}

template <typename T>
Ptr<TensorObject<T>>& Ptr<TensorObject<T>>::operator=(
    Ptr<TensorObject<T>>&& ptrWrapper) noexcept
{
    m_tensorObjectPtr = std::move(ptrWrapper.m_tensorObjectPtr);
}

template <typename T>
template <typename ...Ts>
Ptr<TensorObject<T>> Ptr<TensorObject<T>>::Make(Ts... args)
{
    auto ptr = Ptr<TensorObject<T>>();
    ptr.m_tensorObjectPtr = std::make_unique<TensorObject<T>>(args...);
    return std::move(ptr);

}

template <typename T>
Ptr<TensorSocket<T>>::Ptr(Ptr<TensorSocket<T>>&& ptrWrapper) noexcept

    : m_tensorSocketPtr(ptrWrapper.m_tensorSocketPtr),
      m_reference_count(ptrWrapper.m_reference_count)
{
    ptrWrapper.m_tensorSocketPtr = nullptr;
}

template<typename T>
Ptr<TensorSocket<T>>::Ptr(const Ptr<TensorSocket<T>>& ptrWrapper)
    : m_tensorSocketPtr(ptrWrapper.m_tensorSocketPtr),
    m_reference_count(ptrWrapper.m_reference_count + 1)
{

}

template <typename T>
template <typename... Ts>
Ptr<TensorSocket<T>> Ptr<TensorSocket<T>>::Make(Ts... args)
{
    auto ptrWrapper = Ptr<TensorSocket<T>>();
    ptrWrapper.m_tensorSocketPtr = new TensorSocket<T>(args...);
    ptrWrapper.m_reference_count = 0;
    return std::move(ptrWrapper);
}
}  // namespace CubbyDNN

#endif  // CUBBYDNN_PTRWRAPPER_IMPL_HPP
