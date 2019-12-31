// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_SHAREDPTR_HPP
#define CUBBYDNN_SHAREDPTR_HPP

#include <atomic>

namespace CubbyDNN
{
template <typename T>
class SharedPtr
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

    /// Ptr to the object
    T* m_objectPtr = nullptr;

    /// Ptr to the reference counter
    SharedObjectInfo* m_sharedObjectPtr = nullptr;

    //! private constructor for constructing the m_objectPtr for the first time
    //! \param objectPtr : ptr for the object
    //! \param informationPtr : informationPtr that has been created
    explicit SharedPtr(T* objectPtr, SharedObjectInfo* informationPtr);

 public:
    SharedPtr() = default;

    //! Copy constructor
    //! \param sharedPtr
    SharedPtr(const SharedPtr<T>& sharedPtr);

    //! Move constructor
    //! This will make given parameter (sharedPtr) invalid
    //! \param sharedPtr : SharedPtr<T> to move from
    SharedPtr(SharedPtr<T>&& sharedPtr) noexcept;

    //! Compute assign operator is explicitly deleted
    //! \param sharedPtr
    //! \return
    SharedPtr<T>& operator=(const SharedPtr<T>& sharedPtr);

    //! Move assign operator
    //! This will make given parameter (sharedPtr) invalid
    //! \param sharedPtr : SharedPtr<T> to move from
    //! \return : SharedPtr<T>
    SharedPtr<T>& operator=(SharedPtr<T>&& sharedPtr) noexcept;

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
    T* operator->();

    //! Const access operator
    //! \return : Const ptr to m_objectPtr
    const T* operator->() const;

    //! Gets current reference count
    //! \return :  current reference count
    [[nodiscard]] int GetCurrentRefCount() const
    {
        return m_sharedObjectPtr->RefCount.load(std::memory_order_acquire);
    }
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_SHAREDPTR_HPP
