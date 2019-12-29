// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include "SharedPtrTests.hpp"
#include <deque>
#include <thread>
#include "gtest/gtest.h"

namespace CubbyDNN
{
void CopyandDestruct(const SharedPtr<int>& sharedPtr, bool* stop)
{
    volatile auto ptr = sharedPtr;
    while (!*stop)
    {
        std::this_thread::yield();
    }
    /// MaximumRefCount should be always be greater or equal than reference
    /// counter
}

void ConcurrentCopy(int spawnNum)
{
    auto* ptr = new int(1);
    SharedPtr<int> shared = SharedPtr<int>::Make(ptr);
    std::deque<std::thread> threadPool;
    bool stop = std::atomic_bool(false);

    for (int i = 0; i < spawnNum; ++i)
    {
        threadPool.emplace_back(
            std::thread(std::move(CopyandDestruct), shared, &stop));
        volatile const auto count = shared.GetCurrentRefCount();
        std::cout << count << std::endl;
    }

    while (shared.GetCurrentRefCount() < spawnNum + 1)
    {
        volatile const auto count = shared.GetCurrentRefCount();
        std::cout << count << std::endl;
        count;
    }

    EXPECT_EQ(shared.GetCurrentRefCount(), spawnNum + 1);
    stop = true;

    for (auto& thread : threadPool)
    {
        if (thread.joinable())
            thread.join();
    }

    /// Only 1 refCount should be left at the end
    EXPECT_EQ(shared.GetCurrentRefCount(), 1);
}

TEST(ConcurrentCopy_basic, ConcurrentCopy)
{
    /**
     * Spawn 10 threads with enough maxRefCount
     * Check if reference counter successfully reterns 1 at the end
     */
    ConcurrentCopy(10);
}

TEST(ConcurrentCopy_RefLimit, ConcurrentCopy)
{
    /**
     * Spawn 50 threads and copy SharedPtr
     * Checks if reference counter does not exceed maximumRefCount
     */
    ConcurrentCopy(50);
}
}  // namespace CubbyDNN