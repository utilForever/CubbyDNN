//
// Created by jwkim98 on 6/26/19.
//
#include <cubbydnn/Engine/ThreadManager.hpp>
#include <iostream>

namespace CubbyDNN
{
std::vector<std::thread> ThreadManager::m_mainThreadPool =
    std::vector<std::thread>();

SpinLockQueue<TaskWrapper> ThreadManager::m_mainTaskQueue =
    SpinLockQueue<TaskWrapper>(10000);

void ThreadManager::InitializeThreadPool(size_t mainThreadNum, size_t copyThreadNum)
{
    assert(mainThreadNum + copyThreadNum < std::thread::hardware_concurrency());

    auto hardwareThreads = std::thread::hardware_concurrency() - 2;

    std::cout << "Creating " << mainThreadNum << "main threads" << std::endl;
    m_mainThreadPool.reserve(mainThreadNum);
    for (unsigned count = 0; count < mainThreadNum; ++count)
    {
        m_mainThreadPool.emplace_back(std::thread(m_runMain));
    }

    for(unsigned count = 0; count < copyThreadNum; count++)
    {
        m_copyThreadPool.emplace_back(std::thread(m_runMain));
    }
}

// TODO : turn this into spinlock based pending function
void ThreadManager::m_runMain()
{
    TaskWrapper taskWrapper = m_mainTaskQueue.Dequeue();
    while (taskWrapper.Type != TaskType::Join)
    {
        auto task = taskWrapper.GetTask();
        task();
        std::atomic_exchange_explicit(&m_dirty, true,
                                      std::memory_order_release);
        taskWrapper = m_mainTaskQueue.Dequeue();
    }
}

void ThreadManager::m_runCopy()
{
    TaskWrapper taskWrapper = m_copyTaskQueue.Dequeue();
    while (taskWrapper.Type != TaskType::Join)
    {
        auto task = taskWrapper.GetTask();
        task();
        std::atomic_exchange_explicit(&m_dirty, true,
                std::memory_order_release);
        taskWrapper = m_copyTaskQueue.Dequeue();
    }
}

void ThreadManager::EnqueueTask(TaskWrapper&& task)
{
    m_mainTaskQueue.Enqueue(std::move(task));
}

TaskWrapper ThreadManager::DequeueTask()
{
    return m_mainTaskQueue.Dequeue();
}

void ThreadManager::ScanUnitTasks()
{
    bool isFinished = false;
    while (m_active && !isFinished)
    {
        if (m_dirty)
        {
            isFinished = true;
            for (auto& sourceUnit : m_sourceUnitVector)
            {
                if (sourceUnit.IsReady() &&
                    sourceUnit.GetStateNum() < m_maxEpochs)
                {
                    auto computeFunc = [&sourceUnit]() {
                        sourceUnit.Compute();
                    };
                    auto updateState = [&sourceUnit]() {
                        sourceUnit.ReleaseUnit();
                    };
                    sourceUnit.AcquireUnit();
                    TaskWrapper taskWrapper(TaskType::ComputeSource,
                                            computeFunc, updateState);
                    m_mainTaskQueue.Enqueue(std::move(taskWrapper));
                }
                else if (sourceUnit.GetStateNum() != m_maxEpochs)
                {
                    isFinished = false;
                }
            }

            for (auto& intermediateUnit : m_intermediateUnitVector)
            {
                if (intermediateUnit.IsReady() &&
                    intermediateUnit.GetStateNum() < m_maxEpochs)
                {
                    auto computeFunc = [&intermediateUnit]() {
                        intermediateUnit.Compute();
                    };
                    auto updateState = [&intermediateUnit]() {
                        intermediateUnit.ReleaseUnit();
                    };
                    intermediateUnit.AcquireUnit();
                    TaskWrapper taskWrapper(TaskType::ComputeIntermediate,
                                            computeFunc, updateState);
                    m_mainTaskQueue.Enqueue(std::move(taskWrapper));
                }
                else if (intermediateUnit.GetStateNum() != m_maxEpochs)
                {
                    isFinished = false;
                }
            }

            for (auto& sinkUnit : m_sinkUnitVector)
            {
                if (sinkUnit.IsReady() && sinkUnit.GetStateNum() < m_maxEpochs)
                {
                    auto computeFunc = [&sinkUnit]() { sinkUnit.Compute(); };
                    auto updateState = [&sinkUnit]() {
                        sinkUnit.ReleaseUnit();
                    };
                    sinkUnit.AcquireUnit();
                    TaskWrapper taskWrapper(TaskType::ComputeSink, computeFunc,
                                            updateState);
                    m_mainTaskQueue.Enqueue(std::move(taskWrapper));
                }
                else if (sinkUnit.GetStateNum() != m_maxEpochs)
                {
                    isFinished = false;
                }
            }
        }
        else
        {
            std::this_thread::yield();
        }
    }
}

void ThreadManager::ScanCopyTasks()
{
    while (m_active)
    {
        if (m_dirty)
        {
            for (auto& copyUnit : m_copyUnitVector)
            {
                if (copyUnit.IsReady() && copyUnit.GetStateNum() < m_maxEpochs)
                {
                    auto computeFunc = [&copyUnit]() { copyUnit.Compute(); };
                    auto updateState = [&copyUnit]() {
                        copyUnit.ReleaseUnit();
                    };
                    copyUnit.AcquireUnit();
                    TaskWrapper taskWrapper(TaskType::Copy, computeFunc,
                                            updateState);
                    m_copyTaskQueue.Enqueue(std::move(taskWrapper));
                }
            }
        }
        else
        {
            std::this_thread::yield();
        }
    }
}

}  // namespace CubbyDNN