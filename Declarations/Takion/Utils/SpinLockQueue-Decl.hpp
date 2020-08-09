// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_SPINLOCKQUEUE_DECL_HPP
#define CUBBYDNN_SPINLOCKQUEUE_DECL_HPP

#include <Takion/Utils/SpinLock.hpp>

#include <vector>

namespace Takion
{
//! Thread safe spinLockQueue which enqueue and dequeue, accesses element using
//! spin lock
//! \tparam T : type of m_objectPtr to store
template <typename T>
class SpinLockQueue
{
public:

    //! Constructor
    //! Initializes queue with given capacity (type must support default
    //! constructor
    //! \param maxCapacity : size of the queue
    explicit SpinLockQueue(size_t maxCapacity = 20);

    ~SpinLockQueue() = default;

    //! Move constructor
    //! \param spinLockQueue  : spinLockQueue object to move from
    SpinLockQueue(SpinLockQueue&& spinLockQueue) noexcept;

    //!  Enqueue elements into the queue
    //!  Tries to enqueue object until it's successful
    //! \tparam U : template type to move. This type must match queue's template type T
    //! \param object : object to enqueue
    template <typename U>
    void Enqueue(U&& object);

    //! Enqueue m_objectPtr to the queue
    //! \tparam U : type of m_objectPtr to store this type must match
    //! SpinLockQueue's template type T
    //! \param object : element to enqueue
    //! \return : whether enqueue has been succeeded. This can fail if queue goes
    //! over the capacity
    template <typename U>
    bool TryEnqueue(U&& object);


    //! Dequeue m_objectPtr from the queue
    //! This waits until there are some elements ready to be dequeued. Otherwise,
    //! it busy waits
    //! \return : dequeued m_objectPtr from the queue
    T Dequeue();

    //! Tries to dequeue element from the queue
    //! \return : tuple which contains first element as dequeued m_objectPtr(if
    //! successful) and second as whether dequeue was successful
    std::tuple<T, bool> TryDequeue();

    //! Access method to the index
    //! internally locks the spin lock in shared state
    //!  \param index : to access
    //! \return
    T& At(size_t index);

    //! Gets size of the queue after locking shared spin lock
    //! \return : current size (number of used elements) of the queue
    size_t Size();

private:
    /// Called when some element has been enqueued (front the back)
    /// this method does not check the condition (condition must be checked before
    /// calling this method)
    void incrementBack();

    /// Called when some element has been dequeued (front the front)
    /// this method does not check the condition (condition must be checked before
    /// calling this method)
    void incrementFront();

    /// brings size without locking anything (Should be called after locking
    /// externally)
    size_t size();
    /// Stores capacity of this SpinLockQueue
    const size_t m_maxCapacity;

    SpinLock<T> m_spinLock;
    std::vector<T> m_container;
    /// Index for indicating front of the queue (the index one after last element
    /// is stored)
    size_t m_frontIndex = 0;
    /// Index for indicating back of the queue
    size_t m_backIndex = 0;
    /// True if queue is empty false otherwise
    bool m_empty = true;
};
} // namespace Captain

#endif // CAPTAIN_SPINLOCKQUEUE_DECL_HPP
