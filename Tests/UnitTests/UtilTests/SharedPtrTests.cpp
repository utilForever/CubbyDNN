// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.
#include "SharedPtrTests.hpp"
#include <doctest.h>
#include <deque>
#include <thread>
#include <vector>

namespace CubbyDNN::Test
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

    while (shared.GetCurrentRefCount() < spawnNum * (numCopy + 1) + 1);

    CHECK(shared.GetCurrentRefCount() == spawnNum * (numCopy + 1) + 1);
    stop = true;

    for (auto& thread : threadPool)
    {
        if (thread.joinable())
            thread.join();
    }

    /// Only 1 refCount should be left at the end
    CHECK(shared.GetCurrentRefCount() == 1);
}


} // namespace CubbyDNN
