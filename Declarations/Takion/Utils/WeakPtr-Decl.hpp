// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.


#pragma once
#include <Takion/Utils/SharedPtr.hpp>

namespace Takion
{
//! \brief : This class declares custom version of atomic std::weak_ptr
template <typename T>
class WeakPtr
{
    T* m_objectPtr = nullptr;

    SharedObjectInfo* m_sharedObjectInfoPtr = nullptr;

    template <typename U>
    friend class WeakPtr;

 public:
    constexpr WeakPtr() = default;

    ~WeakPtr() = default;

    //! Copy constructor
    //! \tparam U : template type of weakPtr to copy from
    //! U must be same type as T or subclass of T or assertion will fail
    //! \param weakPtr : weakPtr to copy from
    template <typename U>
    WeakPtr(const WeakPtr<U>& weakPtr);

    //! Copy constructor
    //! \param weakPtr : weakPtr to copy from
    WeakPtr(const WeakPtr<T>& weakPtr);

    //! Constructs WeakPtr from SharedPtr<U>
    //! \tparam U : template type of sharedPtr to copy from
    template <typename U>
    WeakPtr(const SharedPtr<U>& sharedPtr);

    //! Move constructor
    template <typename U>
    WeakPtr(WeakPtr<U>&& weakPtr) noexcept;

    WeakPtr(WeakPtr<T>&& weakPtr) noexcept;

    template <typename U>
    WeakPtr<T>& operator=(const WeakPtr<U>& weakPtr);

    WeakPtr<T>& operator=(const WeakPtr<T>& weakPtr);

    template <typename U>
    WeakPtr<T>& operator=(WeakPtr<U>&& weakPtr) noexcept;

    WeakPtr<T>& operator=(WeakPtr<T>&& weakPtr) noexcept;

    template <typename U>
    bool operator==(const WeakPtr<U>& weakPtr) const
    {
        static_assert(std::is_same<std::decay<T>(), std::decay<U>()>::value ||
                      std::is_base_of<std::decay<T>(), std::decay<U>()>::value);

        return m_objectPtr == weakPtr.m_objectPtr &&
               m_sharedObjectInfoPtr == weakPtr.m_sharedObjectInfoPtr;
    }

    bool operator==(const WeakPtr<T>& weakPtr) const
    {
        return m_objectPtr == weakPtr.m_objectPtr &&
               m_sharedObjectInfoPtr == weakPtr.m_sharedObjectInfoPtr;
    }

    //! Creates new SharedPtr<T> that shares ownership of the managed object
    //! Increments reference count
    //! If ownership is already expired, returns empty shared_ptr
    [[nodiscard]] SharedPtr<T> Lock() const;

    T* operator->() const;
};

}  // namespace Takion
