//
// Created by jwkim98 on 6/30/19.
//

#ifndef CAPTAIN_SPINLOCKQUEUE_DECL_HPP
#define CAPTAIN_SPINLOCKQUEUE_DECL_HPP

#include <captain/Engine/SpinLock.hpp>

#include <type_traits>
#include <vector>

namespace Captain {

/**
 * Thread safe spinLockQueue which enqueues and dequeues, accesses element using
 * spinlocks
 * @tparam T : type of object to store
 */
template <typename T> class SpinLockQueue {
public:
  /**
   * Constructor
   * Initializes queue with given capacity (type must support default
   * constructor
   * @param maxCapacity : size of the queue
   */
  explicit SpinLockQueue(size_t maxCapacity);

  SpinLockQueue(SpinLockQueue &&spinLockQueue) noexcept;

  template <typename U> void Enqueue(U &&elem);

  /**
   * Enqueues object to the queue
   * @tparam U : type of object to store this type must be same as
   * SpinLockQueue's template type 'T'
   * @param elem : element to enqueue
   * @return : whether enqueue has been succeeded. This can fail if queue goes
   * over the capacity
   */
  template <typename U> bool TryEnqueue(U &&elem);

  /**
   * Dequeues object from the queue
   * This waits until there are some elements ready to be dequeued. Otherwise,
   * it busy waits
   * @return : dequeued object from the queue
   */

  T Dequeue();

  /**
   * Tries to dequeue element from the queue
   * @return : tuple which contains first element as dequeued object(if
   * successful) and second as whether dequeue was successful
   */
  std::tuple<T, bool> TryDequeue();

  /**
   * Access method to the index
   * internally locks the spinlock in shared state
   * @param index : to access
   * @return
   */
  T &At(size_t index);

  /**
   * Brings back size of the queue after locking shared spinlock
   * @return : current size (number of used elements) of the queue
   */
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