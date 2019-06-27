//
// Created by jwkim98 on 4/21/19.
//

#ifndef CUBBYDNN_SHAREDPTR_HPP
#define CUBBYDNN_SHAREDPTR_HPP

#include <atomic>
#include <memory>

namespace CubbyDNN
{
enum class PtrState
{
    valid,
    dirty,
    invalid,
};

template <typename T>
class SharedPtr
{
 private:
    /**
     * Shared object stores the actual object with atomic reference counter
     */
    struct SharedObject
    {
        SharedObject(T&& object, const int maxRefCount)
            : Object(std::move(object)),
              RefCount(1),
              MaxRefCount(maxRefCount){};

        T Object;
        std::atomic<int> RefCount;
        /// Maximum reference count that it can reach
        const int MaxRefCount;
    };

    SharedObject* m_sharedObjectPtr;

    PtrState m_ptrState;

    /**
     * private constructor for constructing the object for the first time
     * @param objectPtr : objectPtr that has been created
     * @param state : state of the sharedObject
     */
    explicit SharedPtr(SharedObject* objectPtr, PtrState state);

    /**
     * Makes copy of the sharedPtr
     * @return
     */
    SharedPtr<T> tryMakeCopy();

 public:
    /**
     * Destructor will automatically decrease the reference counter if this ptr
     * has valid pointer
     */
    ~SharedPtr();

    /**
     * Builds new SharedPtr object with no parameters
     * @return : SharedPtr
     */
    static SharedPtr<T> Make(int maxReferenceCount);

    /**
     * Builds new SharedPtr object with parameters
     * @tparam Ts : template parameter pack
     * @param maxReferenceCount : maximum reference count of this object
     * @param args : arguments to build new object
     * @return : SharedPtr
     */
    template <typename... Ts>
    static SharedPtr<T> Make(int maxReferenceCount, Ts&&... args);

    /**
     * Copy constructor is explicitly deleted
     * @param sharedPtr
     */
    SharedPtr(const SharedPtr<T>& sharedPtr) = delete;

    /**
     * Copy assign operator is explicitly deleted
     * @param sharedPtr
     * @return
     */
    SharedPtr<T>& operator=(const SharedPtr<T>& sharedPtr) = delete;

    /**
     * Move constructor
     * This will make given parameter (sharedPtr) invalid
     * @param sharedPtr : SharedPtr<T> to move from
     */
    SharedPtr(SharedPtr<T>&& sharedPtr) noexcept;

    /**
     * Move assign operator
     * This will make given parameter (sharedPtr) invalid
     * @param sharedPtr : SharedPtr<T> to move from
     * @return : SharedPtr<T>
     */
    SharedPtr<T>& operator=(SharedPtr<T>&& sharedPtr) noexcept;

    /**
     * Access operator to object this possesses
     * @return : Reference of the object
     */
    T& operator->();

    /**
     * Makes copy of this SharedPtr
     * Increments reference count of the object
     * @return : Copied SharedPtr
     */
    SharedPtr<T> MakeCopy();

    /**
     * Returns state of this SharedPtr
     * This is used to determine if SharePtr is in valid state
     * @return : state of this SharedPtr
     */
    PtrState GetState() const
    {
        return m_ptrState;
    }

    int GetCurrentRefCount() const
    {
        return m_sharedObjectPtr->RefCount;
    }

    int GetMaximumRefCount() const
    {
        return m_sharedObjectPtr->MaxRefCount;
    }
};

}  // namespace CubbyDNN

#endif  // CUBBYDNN_SHAREDPTR_HPP
