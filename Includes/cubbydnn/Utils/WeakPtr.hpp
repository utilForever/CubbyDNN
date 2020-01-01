// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#pragma once

#include <cubbydnn/Utils/SharedPtr-impl.hpp>

namespace CubbyDNN
{
template <typename T>
class WeakPtr
{
    T* m_objectPtr;

    SharedObjectInfo* m_sharedObjectInfoPtr;

    template<typename U>
    friend class WeakPtr;

 public:
    constexpr WeakPtr();

    template <typename U>
    WeakPtr(const WeakPtr<U>& weakPtr);

    template <typename U>
    WeakPtr(const SharedPtr<U>& sharedPtr);

    //! Move constructor
    template <typename U>
    WeakPtr(WeakPtr<U>&& weakPtr);

    //! Creates new SharedPtr<T> that shares ownership of the managed object
    //! Increments reference count
    //! If ownership is already expired, returns empty shared_ptr
    SharedPtr<T> Lock() const;

    T* operator->() const;
};

}  // namespace CubbyDNN
