// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_SPINLOCK_DECL_HPP
#define TAKION_SPINLOCK_DECL_HPP

#include <atomic>

namespace Takion
{
//! Shared spinlock class
//! thread will busy wait for accessing some data with atomic variable
//!  \tparam T :
template <typename T>
class SpinLock
{
public:
    SpinLock()
        : m_exclusiveAccess(false),
          m_referenceCount(0)
    {
    }

    //! Exclusively acquires resource
    void ExclusiveLock();

    //!  Once thread was locked exclusively, it must call this method to release
    //! exclusive access to data
    void ExclusiveRelease();

    //! If some operation can be shared while exclusive operation is not being
    //! done, many threads can access the data using Shared lock. It internally
    //! waits while exclusive resource is free, and acquires shared lock
    void SharedLock();

    //! If some thread acquired SharedLock, it must free the lock using this method
    void SharedRelease();

private:
    /// Exclusive Access flag
    std::atomic<bool> m_exclusiveAccess;
    /// Reference counter for shared accesses
    std::atomic<int> m_referenceCount;
};

} // namespace Captain

#endif //TAKION_SPINLOCK_DECL_HPP
