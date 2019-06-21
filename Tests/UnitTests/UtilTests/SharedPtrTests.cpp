//
// Created by jwkim98 on 6/21/19.
//

#include <cubbydnn/Utils/SharedPtr-impl.hpp>

#include <googletest/googletest/include/gtest/gtest.h>
#include <deque>
#include <thread>

namespace CubbyDNN
{
void CopyandDestruct(SharedPtr<int>&& ptr)
{
    SharedPtr<int> copy = ptr.MakeCopy();
    /// MaximumRefCount should be always be greater or equal than reference
    /// counter
    EXPECT_GE(ptr.GetMaximumRefCount(), ptr.GetCurrentRefCount());
}

void ConcurrentCopy()
{
    SharedPtr<int> ptr = SharedPtr<int>::Make(20, 1);
    std::deque<std::thread> threadPool;

    for (int i = 0; i < 10; ++i)
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
    ConcurrentCopy();
}
}  // namespace CubbyDNN