//
// Created by jwkim98 on 4/21/19.
//

#ifndef CUBBYDNN_PTRWRAPPER_HPP
#define CUBBYDNN_PTRWRAPPER_HPP

#include <cubbydnn/Tensors/Decl/TensorPlug.hpp>
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
class Ptr<TensorPlug<T>>
{
 public:
    Ptr() = default;

    Ptr(Ptr<TensorPlug<T>>&& tensorObjectPtr) noexcept
    {
    }

    Ptr<TensorPlug<T>>& operator=(Ptr<TensorPlug<T>>&& ptrWrapper) noexcept
    {
    }

    template <typename... Ts>
    Ptr<TensorPlug<T>> Make(Ts... args)
    {
        auto ptr = Ptr<TensorPlug<T>>();
        ptr.m_tensorObjectPtr = std::make_unique<TensorPlug<T>>(args...);
        return std::move(ptr);
    }

 private:
    std::unique_ptr<TensorPlug<T>> m_tensorObjectPtr = nullptr;
};

template <typename T>
class Ptr<TensorSocket<T>>
{
 public:
    Ptr() = default;

    Ptr(Ptr<TensorSocket<T>>&& ptrWrapper) noexcept
    {
        ptrWrapper.m_tensorSocketPtr = nullptr;
    }

    Ptr(const Ptr<TensorSocket<T>>& ptrWrapper)
        : m_tensorSocketPtr(ptrWrapper.m_tensorSocketPtr),
          m_reference_count(ptrWrapper.m_reference_count + 1)
    {
    }

    template <typename... Ts>
    static Ptr<TensorSocket<T>> Make(Ts... args)
    {
        auto ptrWrapper = Ptr<TensorSocket<T>>();
        ptrWrapper.m_tensorSocketPtr = new TensorSocket<T>(args...);
        ptrWrapper.m_reference_count = 0;
        return std::move(ptrWrapper);
    }

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
