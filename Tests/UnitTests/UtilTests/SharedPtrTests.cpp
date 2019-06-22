//
// Created by jwkim98 on 6/21/19.
//

#include "SharedPtrTests.hpp"

namespace CubbyDNN
{
void CopyandDestruct(SharedPtr<int>&& ptr)
{
    SharedPtr<int> copy = ptr.MakeCopy();
    /// MaximumRefCount should be always be greater or equal than reference
    /// counter
    EXPECT_GE(ptr.GetMaximumRefCount(), ptr.GetCurrentRefCount());
}

void ConcurrentCopy(int spawnNum, int maxRefCount)
{
    SharedPtr<int> ptr = SharedPtr<int>::Make(maxRefCount, 1);
    std::deque<std::thread> threadPool;

    for (int i = 0; i < spawnNum; ++i)
    {
        threadPool.emplace_back(
            std::thread(std::move(CopyandDestruct), ptr.MakeCopy()));
    }

    for (auto&& thread : threadPool)
    {
        if (thread.joinable())
            thread.join();
    }

    /// Only 1 refCount should be left at the end
    EXPECT_EQ(ptr.GetCurrentRefCount(), 1);
}

TEST(ConcurrentCopy_basic, ConcurrentCopy)
{
    /**
     * Spawn 10 threads with enough maxRefCount
     * Check if reference counter successfully reterns 1 at the end
     */
    ConcurrentCopy(10, 20);
}

TEST(ConcurrentCopy_RefLimit, ConcurrentCopy)
{
    /**
     * Spawn 50 threads and copy SharedPtr
     * Checks if reference counter does not exceed maximumRefCount
     */
    ConcurrentCopy(50, 10);
}
}  // namespace CubbyDNN