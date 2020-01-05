// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_SYNC_HPP
#define CUBBYDNN_SYNC_HPP

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <atomic>

namespace CubbyDNN
{

/**
 * Mtx and Cond_var for controlling synchronization from Operation to Linker
 */
struct Sync
{
 public:
    explicit Sync(int waitFor) : m_resetVal(waitFor), m_counter(waitFor)
    {
    }

    /**
     * WaitUntilAllFinish
     * Waits until every operation finishes by checking counter is 0
     */
    void WaitUntilAllFinish()
    {
        std::unique_lock<std::mutex> lock(m_mtx);
        auto checkCompleted = [this]() {
            return (m_counter == 0) || m_forceFinish;
        };
        m_condVar.wait(lock, checkCompleted);
    }

    /**
     * ResetCounter
     * Resets counter to initial value
     */
    void ResetCounter()
    {
        std::unique_lock<std::mutex> lock(m_mtx);
        m_counter = m_resetVal;
    }

    /**
     * NotifyFinish
     * Decrements counter by 1 Which means one operation is finished
     */
    void NotifyFinish()
    {
        std::unique_lock<std::mutex> lock(m_mtx);
        if (m_counter > 0)
            m_counter--;
        m_condVar.notify_all();
    }

    /**
     * ForceFinish
     * Forces synchronization process to Finish
     */
    void ForceFinish()
    {
        std::unique_lock<std::mutex> lock(m_mtx);
        m_forceFinish = true;
        m_condVar.notify_all();
    }

 private:
    int m_resetVal;
    int m_counter;
    std::mutex m_mtx;
    std::condition_variable m_condVar;
    bool m_forceFinish = false;
};

struct LockFreeSync
{ public:
    explicit LockFreeSync(int maxConnections);

    void NotifyFinish();

    bool IsReady();

    void waitUntilReady();

 private:
    std::atomic<size_t> m_count;
    std::atomic<bool> m_isOccupied;
    int m_maxConnections;
};

using SyncPtr = Sync*;
}  // namespace CubbyDNN
#endif  // CUBBYDNN_SYNC_HPP
