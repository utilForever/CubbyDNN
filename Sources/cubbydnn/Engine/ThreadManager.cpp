//
// Created by jwkim98 on 6/26/19.
//
#include <cubbydnn/Engine/ThreadManager.hpp>
#include <iostream>

namespace CubbyDNN
{
std::vector<std::thread> ThreadManager::m_threadPool =
    std::vector<std::thread>();

SpinLockQueue<TaskWrapper> ThreadManager::m_taskQueue =
    SpinLockQueue<TaskWrapper>(10000);

void ThreadManager::InitializeThreadPool(size_t threadNum)
{
    auto hardwareThreads = std::thread::hardware_concurrency();
    auto activeThreadNum =
        (threadNum < hardwareThreads) ? threadNum : hardwareThreads;
    std::cout << "Creating " << activeThreadNum << " threads" << std::endl;
    m_threadPool.reserve(activeThreadNum);
    for (unsigned count = 0; count < hardwareThreads; ++count)
    {
        m_threadPool.emplace_back(std::thread(run));
    }
}

// TODO : turn this into spinlock based pending function
void ThreadManager::run()
{
    TaskWrapper taskWrapper = m_taskQueue.Dequeue();
    while (taskWrapper.Type != TaskType::Join)
    {
        auto task = taskWrapper.GetTask();
        task();
        std::atomic_exchange_explicit(&m_dirty, true,
                                      std::memory_order_release);
        taskWrapper = m_taskQueue.Dequeue();
    }
}

void ThreadManager::EnqueueTask(TaskWrapper&& task)
{
    m_taskQueue.Enqueue(std::move(task));
}

TaskWrapper ThreadManager::DequeueTask()
{
    return m_taskQueue.Dequeue();
}

void ThreadManager::Scan()
{
    while (m_active)
    {
        if (!m_dirty)
            std::this_thread::yield();

        for (auto& copyUnit : m_copyUnitVector)
        {
            if (copyUnit.IsReady())
            {
                auto computeFunc = [&copyUnit]() { copyUnit.Compute(); };
                auto updateState = [&copyUnit]() { copyUnit.ReleaseUnit(); };
                TaskWrapper taskWrapper(TaskType::Copy, computeFunc,
                                        updateState);
                EnqueueTask(std::move(taskWrapper));
            }
        }
    }
}

}  // namespace CubbyDNN