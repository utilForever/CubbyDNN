//
// Created by jwkim98 on 4/21/19.
//

#ifndef CUBBYDNN_PTRWRAPPER_HPP
#define CUBBYDNN_PTRWRAPPER_HPP

#include <cubbydnn/Tensors/Decl/TensorObject.hpp>
#include <cubbydnn/Tensors/Decl/TensorSocket.hpp>

#include <atomic>
#include <memory>

namespace CubbyDNN
{


template <typename T, typename = void>
class PtrWrapper
{
};

template <typename T>
class PtrWrapper<TensorObject<T>>
{
 public:
    explicit PtrWrapper(std::unique_ptr<TensorObject<T>> tensorObjectPtr);

    PtrWrapper(PtrWrapper<TensorObject<T>>&& tensorObjectPtr) noexcept
        : m_tensorObjectPtr(std::move(tensorObjectPtr))
    {
    }

    PtrWrapper<TensorObject<T>>& operator=(
        PtrWrapper<TensorObject<T>>&& ptrWrapper) noexcept
    {
        m_tensorObjectPtr = std::move(ptrWrapper.m_tensorObjectPtr);
    }

    template <typename... Ts>
    explicit PtrWrapper(Ts... args)
        : m_tensorObjectPtr(std::make_unique<TensorObject<T>>(args...))
    {
    }

 private:
    std::unique_ptr<TensorObject<T>> m_tensorObjectPtr;
};

template <typename T>
class PtrWrapper<TensorSocket<T>>
{
 public:
    PtrWrapper() = default;

    PtrWrapper(PtrWrapper<TensorSocket<T>>&& ptrWrapper) noexcept;

    PtrWrapper(const PtrWrapper<TensorSocket<T>>& ptrWrapper)
        : m_tensorSocketPtr(ptrWrapper.m_tensorSocketPtr),
          m_reference_count(ptrWrapper.m_reference_count + 1)
    {
    }

    template <typename... Ts>
    static PtrWrapper<TensorSocket<T>> make(Ts... args);

    PtrWrapper<TensorSocket<T>>& operator=(
        PtrWrapper<TensorSocket<T>>&& ptrWrapper) noexcept
    {
        m_tensorSocketPtr = ptrWrapper.m_tensorSocketPtr;
        m_reference_count = ptrWrapper.m_reference_count;
        ptrWrapper.m_tensorSocketPtr = nullptr;
    }

    PtrWrapper<TensorSocket<T>>& operator=(
        const PtrWrapper<TensorSocket<T>>& ptrWrapper)
    {
        m_tensorSocketPtr = ptrWrapper.m_tensorSocketPtr;
        m_reference_count = ptrWrapper.m_reference_count + 1;
    }

    TensorSocket<T>& operator->()
    {
        return *m_tensorSocketPtr;
    }

 private:
    TensorSocket<T>* m_tensorSocketPtr = nullptr;
    std::atomic_int m_reference_count = 0;
};



}  // namespace CubbyDNN

#endif  // CUBBYDNN_PTRWRAPPER_HPP
