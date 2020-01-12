// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include "SharedPtrTests.hpp"
#include "gtest/gtest.h"

#include <deque>
#include <thread>

namespace CubbyDNN
{
void CopyAndDestruct(const SharedPtr<int>& sharedPtr, int numCopy, bool* stop)
{
    std::vector<SharedPtr<int>> intVector;
    intVector.reserve(numCopy);
    for (auto i = 0; i < numCopy; ++i)
    {
        intVector.emplace_back(sharedPtr);
    }

    while (!*stop)
    {
        std::this_thread::yield();
    }
    /// MaximumRefCount should be always be greater or equal than reference
    /// counter
}

void ConcurrentCopy(int spawnNum, int numCopy)
{
    auto* ptr = new int(1);
    SharedPtr<int> shared = SharedPtr<int>::Make(ptr);
    std::deque<std::thread> threadPool;
    bool stop = std::atomic_bool(false);

    for (int i = 0; i < spawnNum; ++i)
    {
        threadPool.emplace_back(
            std::thread(std::move(CopyAndDestruct), shared, numCopy, &stop));

    }

    while (shared.GetCurrentRefCount() < spawnNum * (numCopy + 1) + 1)
        ;

    EXPECT_EQ(shared.GetCurrentRefCount(), spawnNum * (numCopy + 1) + 1);
    stop = true;

    for (auto& thread : threadPool)
    {
        if (thread.joinable())
            thread.join();
    }

    /// Only 1 refCount should be left at the end
    EXPECT_EQ(shared.GetCurrentRefCount(), 1);
}

// TEST(ConcurrentCopy_basic, ConcurrentCopy)
// {
//     //! Spawn 10 threads and copy SharedPtr
//     //! Check if reference counter successfully returns 1 at the end
//     ConcurrentCopy(10, 100);
// }
//
// TEST(ConcurrentCopy_RefLimit, ConcurrentCopy)
// {
//     //! Spawn 100 threads and copy SharedPtr
//     //!  Check if reference counter successfully returns 1 at the end
//     ConcurrentCopy(100, 100);
// }
}  // namespace CubbyDNN
