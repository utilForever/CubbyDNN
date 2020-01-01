// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_SHAREDPTR_HPP
#define CUBBYDNN_SHAREDPTR_HPP

#include <atomic>

namespace CubbyDNN
{
//! Shared m_objectPtr stores the actual m_objectPtr with atomic reference
//! counter
struct SharedObjectInfo
{
    SharedObjectInfo() : RefCount(1)
    {
    }

    /// T m_objectPtr might not be initializable
    std::atomic<int> RefCount;
};

template <typename T>
class WeakPtr;

template <typename T>
class SharedPtr
{
    /// Ptr to the object
    T* m_objectPtr = nullptr;

    /// Ptr to the reference counter
    SharedObjectInfo* m_sharedObjectInfoPtr = nullptr;

    template<typename U>
    friend class SharedPtr;

    friend class WeakPtr<T>;

    //! private constructor for constructing the m_objectPtr for the first time
    //! \param objectPtr : ptr for the object
    //! \param informationPtr : informationPtr that has been created
    template <typename U>
    explicit SharedPtr(U* objectPtr, SharedObjectInfo* informationPtr);

 public:
    SharedPtr() = default;

    //! Copy constructor
    //! \param sharedPtr
    template <typename U>
    SharedPtr(const SharedPtr<U>& sharedPtr);

    //! Move constructor
    //! This will make given parameter (sharedPtr) invalid
    //! \param sharedPtr : SharedPtr<T> to move from
    template<typename U>
    SharedPtr(SharedPtr<U>&& sharedPtr) noexcept;

    //! Compute assign operator is explicitly deleted
    //! \param sharedPtr
    //! \return reference to current object
    template<typename U>
    SharedPtr<T>& operator=(const SharedPtr<U>& sharedPtr);

    //! Move assign operator
    //! This will make given parameter (sharedPtr) invalid
    //! \param sharedPtr : SharedPtr<T> to move from
    //! \return : SharedPtr<T>
    template<typename U>
    SharedPtr<T>& operator=(SharedPtr<U>&& sharedPtr) noexcept;

    //! Destructor will automatically decrease the reference counter if this ptr
    //! has valid pointer
    ~SharedPtr();

    //! Builds SharedPtr m_objectPtr using raw pointer
    //! \param objectPtr : pointer to the object
    static SharedPtr<T> Make(T* objectPtr);

    //! Builds new SharedPtr m_objectPtr with no parameters
    //! \return : SharedPtr
    static SharedPtr<T> Make();

    //! Builds new SharedPtr m_objectPtr with parameters
    //! \tparam Ts : template parameter pack
    //! \param args : arguments to build new m_objectPtr
    //! \return : SharedPtr
    template <typename... Ts>
    static SharedPtr<T> Make(Ts&&... args);

    //! Access operator to m_objectPtr this possesses
    //! \return : ptr to m_objectPtr
    T* operator->() const;

    //! Gets current reference count
    //! \return :  current reference count
    [[nodiscard]] int GetCurrentRefCount() const
    {
        return m_sharedObjectInfoPtr->RefCount.load(std::memory_order_acquire);
    }
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_SHAREDPTR_HPP
