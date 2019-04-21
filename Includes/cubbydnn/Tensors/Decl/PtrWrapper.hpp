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
class Ptr
{
};

template <typename T>
class Ptr<TensorObject<T>>
{
 public:
    Ptr() = default;

    Ptr(Ptr<TensorObject<T>>&& tensorObjectPtr) noexcept;

    Ptr<TensorObject<T>>& operator=(Ptr<TensorObject<T>>&& ptrWrapper) noexcept;

    template <typename... Ts>
    Ptr<TensorObject<T>> Make(Ts... args);

 private:
    std::unique_ptr<TensorObject<T>> m_tensorObjectPtr = nullptr;
};

template <typename T>
class Ptr<TensorSocket<T>>
{
 public:
    Ptr() = default;

    Ptr(Ptr<TensorSocket<T>>&& ptrWrapper) noexcept;

    Ptr(const Ptr<TensorSocket<T>>& ptrWrapper);

    template <typename... Ts>
    static Ptr<TensorSocket<T>> Make(Ts... args);

    Ptr<TensorSocket<T>>& operator=(Ptr<TensorSocket<T>>&& ptrWrapper) noexcept
    {
        m_tensorSocketPtr = ptrWrapper.m_tensorSocketPtr;
        m_reference_count = ptrWrapper.m_reference_count;
        ptrWrapper.m_tensorSocketPtr = nullptr;
    }

    Ptr<TensorSocket<T>>& operator=(const Ptr<TensorSocket<T>>& ptrWrapper)
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
