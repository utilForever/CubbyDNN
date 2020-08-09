// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include "TaskQueueTest.hpp"
#include <atomic>
#include <Takion/Utils/SpinLockQueue.hpp>
#include <functional>
#include <thread>
#include <vector>
#include <random>
#include <doctest.h>

namespace Takion::Tests
{
void StressTask()
{
    std::random_device rn;
    std::mt19937 gen(rn());
    
   std::uniform_int_distribution<> rand(0, 10000);
    volatile auto num = 10;
    const auto size = static_cast<std::size_t>(rand(gen));
    for (std::size_t i = 0; i < size; ++i)
        if (i % 2 == 0)
            num *= 2;
        else
            num /= 2;
}

void TaskQueueTest(int workers)
{
    const auto desired = 100;
    std::atomic_int count = 0;

    std::vector<std::thread> threadVector;
    threadVector.reserve(workers);

    SpinLockQueue<std::function<bool(void)>> taskQueue(desired);

    for (auto i = 0; i < workers; ++i)
    {
        auto run = [&taskQueue]()
        {
            auto func = taskQueue.Dequeue();
            while (func())
            {
                func = taskQueue.Dequeue();
            }
        };
        threadVector.emplace_back(std::thread(std::move(run)));
    }

    for (auto i = 0; i < desired; ++i)
    {
        std::function<bool(void)> run = [&count]()
        {
            StressTask();
            count.fetch_add(1);
            return true;
        };
        taskQueue.Enqueue(run);
    }

    while (count != desired)
        std::this_thread::yield();

    CHECK(count == desired);

    for (std::size_t i = 0; i < threadVector.size(); ++i)
    {
        auto stop = []() { return false; };
        taskQueue.Enqueue(stop);
    }

    for (auto& thread : threadVector)
    {
        thread.join();
    }
}

} // namespace Takion
