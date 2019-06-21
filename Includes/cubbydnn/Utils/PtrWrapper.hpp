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

enum class SharedPtrState
{
    valid,
    dirty,
    invalid,
};

template <typename T>
class SharedPtr
{
 private:
    struct SharedObject
    {
        SharedObject(T&& object, const int maxRefCount)
            : Object(std::move(object)),
              RefCount(1),
              MaxRefCount(maxRefCount){};

        T Object;
        std::atomic<int> RefCount;
        const int MaxRefCount;
    };

    SharedObject* m_sharedObjectPtr;

    SharedPtrState m_ptrState;

    explicit SharedPtr<T>(SharedObject* objectPtr, SharedPtrState state)
        : m_sharedObjectPtr(objectPtr), m_ptrState(state){};

    SharedPtr<T> tryMakeCopy()
    {
        const int oldRefCount = m_sharedObjectPtr->RefCount;
        if (oldRefCount < m_sharedObjectPtr->MaxRefCount)
        {
            if (m_sharedObjectPtr->RefCount.compare_exchange_strong(
                    oldRefCount, oldRefCount + 1))
                return SharedPtr(m_sharedObjectPtr, SharedPtrState::valid);
            else
                return SharedPtr(nullptr, SharedPtrState::dirty);
        }
        else
            return SharedPtr(nullptr, SharedPtrState::invalid);
    }

 public:
    static SharedPtr<T> Make()
    {
        T* ptr = new T();
        return SharedPtr<T>(ptr, SharedPtrState::valid);
    }

    template <typename... Ts>
    static SharedPtr<T> Make(int maxReferenceCount, Ts&... args)
    {
        T* ptr = new T(args...);
        return std::move(SharedPtr<T>(ptr, SharedPtrState::valid));
    }

    SharedPtr(const SharedPtr& sharedPtr) = delete;

    SharedPtr& operator=(const SharedPtr& sharedPtr) = delete;

    SharedPtr(SharedPtr&& sharedPtr) noexcept
        : m_sharedObjectPtr(std::move(sharedPtr.m_sharedObjectPtr)),
          m_ptrState(m_ptrState)
    {
        sharedPtr.m_sharedObjectPtr = nullptr;
        m_ptrState = SharedPtrState::invalid;
    }

    SharedPtr& operator=(SharedPtr&& sharedPtr) noexcept
    {
        m_sharedObjectPtr = sharedPtr.m_sharedObjectPtr;
        m_ptrState = sharedPtr.m_ptrState;

        sharedPtr.m_sharedObjectPtr = nullptr;
        m_ptrState = SharedPtrState::invalid;
    }

    SharedPtr<T> MakeCopy()
    {
        auto sharedPtr = tryMakeCopy();
        while (sharedPtr.GetState() == SharedPtrState::dirty)
            sharedPtr = tryMakeCopy();
        return sharedPtr;
    }


    SharedPtrState GetState()
    {
        return m_ptrState;
    }
};

}  // namespace CubbyDNN

#endif  // CUBBYDNN_PTRWRAPPER_HPP
