// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_SHAREDPTR_HPP
#define TAKION_SHAREDPTR_HPP

#include <atomic>

namespace Takion
{
//! Shared m_objectPtr stores the actual m_objectPtr with atomic reference
//! counter
struct SharedObjectInfo
{
    SharedObjectInfo()
        : RefCount(1)
    {
    }

    /// T m_objectPtr might not be initializable
    std::atomic<int> RefCount;
};

template <typename T>
class WeakPtr;

//! \brief : This class declares custom version of atomic std::shared_ptr
template <typename T>
class SharedPtr
{
    /// Unique id of this sharedPtr object
    int m_id = -1;

    /// Ptr to the object
    T* m_objectPtr = nullptr;

    /// Ptr to the reference counter
    SharedObjectInfo* m_sharedObjectInfoPtr = nullptr;

    /// Make all other classes with different template types friends
    template <typename U>
    friend class SharedPtr;

    /// WeakPtr<T> is allowed to access m_objectPtr and m_sharedObjectInfoPtr of
    /// SharedPtr<T>
    friend class WeakPtr<T>;

    //! Deletes object if this shared_ptr is last shared_ptr owning object
    //! or decrements reference count
    void m_delete() const;

    //! private constructor for constructing the m_objectPtr for the first time
    //! \param objectPtr : ptr for the object
    //! \param informationPtr : informationPtr that has been created
    template <typename U>
    explicit SharedPtr(U* objectPtr, SharedObjectInfo* informationPtr);

public:
    //! Default constructor that will construct empty SharedPtr object will NULL
    //! object pointer
    SharedPtr() = default;

    //! Copy constructor
    //! \tparam U : template type of sharedPtr to copy from
    //! U must be same type as T or subclass of T or assertion will fail
    //! \param sharedPtr : sharedPtr to copy from
    template <typename U>
    SharedPtr(const SharedPtr<U>& sharedPtr);

    //! Copy constructor
    //! \param sharedPtr : sharedPtr to copy from
    SharedPtr(const SharedPtr<T>& sharedPtr);

    //! Move constructor
    //! This will make given parameter (sharedPtr) invalid
    //! \param sharedPtr : SharedPtr<T> to move from
    template <typename U>
    SharedPtr(SharedPtr<U>&& sharedPtr) noexcept;

    //! Move constructor
    //! This will make given parameter (sharedPtr) invalid
    SharedPtr(SharedPtr<T>&& sharedPtr) noexcept;

    //! Copy assign operator
    //! Deletes object if reference counter of original object was 1
    //! decrements reference counter otherwise
    //! \tparam U : template type of sharedPtr to copy from
    //! U must be same type as T or subclass of T or assertion will fail
    //! \param sharedPtr : SharedPtr<T> to copy from
    //! \return reference to current object
    template <typename U>
    SharedPtr<T>& operator=(const SharedPtr<U>& sharedPtr);

    //! Copy assign operator
    //! Deletes object if reference counter of original object was 1
    //! decrements reference counter otherwise
    //! \param sharedPtr : SharedPtr<T> to copy from
    SharedPtr<T>& operator=(const SharedPtr<T>& sharedPtr);

    //! Move assign operator
    //! This will make given parameter (sharedPtr) invalid
    //! Deletes object if reference counter of original object was 1
    //! decrements reference counter otherwise
    //! \tparam U : template type of sharedPtr to copy from
    //! U must be same type as T or subclass of T or assertion will fail
    //! \param sharedPtr : SharedPtr<T> to move from
    //! \return : SharedPtr<T>
    template <typename U>
    SharedPtr<T>& operator=(SharedPtr<U>&& sharedPtr) noexcept;

    //! Move assign operator
    //! This will make given parameter (sharedPtr) invalid
    //! Deletes object if reference counter of original object was 1
    //! decrements reference counter otherwise
    //! \param sharedPtr : SharedPtr<T> to move from
    //! \return : SharedPtr<T>
    SharedPtr<T>& operator=(SharedPtr<T>&& sharedPtr) noexcept;

    template <typename U>
    bool operator==(const SharedPtr<U>& sharedPtr);

    bool operator==(const SharedPtr<T>& sharedPtr);

    template <typename U>
    bool operator!=(const SharedPtr<U>& sharedPtr);

    bool operator!=(const SharedPtr<T>& sharedPtr);

    //! Destructor will automatically decrease the reference counter if this
    //! ptr has valid pointer
    ~SharedPtr();

    auto& operator[](std::ptrdiff_t idx);

    [[nodiscard]] T* get() const noexcept
    {
        return m_objectPtr;
    }

    //! Builds SharedPtr m_objectPtr using raw pointer
    //! \param objectPtr : pointer to the object
    static SharedPtr<T> Make(T* objectPtr);

    //! Builds new SharedPtr m_objectPtr with no parameters
    //! \return : default empty SharedPtr<T>
    static SharedPtr<T> Make();

    //! Builds new SharedPtr m_objectPtr with parameters
    //! \tparam Ts : template parameter package
    //! \param args : arguments to build new m_objectPtr
    //! \return : SharedPtr<T> generated using given type and parameters
    template <typename... Ts>
    static SharedPtr<T> Make(Ts&&... args);

    //! Access operator to object ptr
    //! \return : ptr to m_objectPtr
    T* operator->() const;

    //! Gets internal object pointer
    T* Get() const
    {
        return m_objectPtr;
    }


    //! Gets current reference count
    //! \return :  current reference count
    [[nodiscard]] int GetCurrentRefCount() const
    {
        return m_sharedObjectInfoPtr->RefCount.load(std::memory_order_acquire);
    }
};
} // namespace Takion

#endif  // CUBBYDNN_SHAREDPTR_HPP
