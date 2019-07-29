//
// Created by jwkim98 on 6/26/19.
//
#include <cubbydnn/Engine/Engine.hpp>
#include <iostream>

namespace CubbyDNN
{
std::vector<std::thread> Engine::m_mainThreadPool = std::vector<std::thread>();

SpinLockQueue<TaskWrapper> Engine::m_mainTaskQueue =
    SpinLockQueue<TaskWrapper>(10000);

void Engine::InitializeThreadPool(size_t mainThreadNum, size_t copyThreadNum)
{
    if (mainThreadNum + copyThreadNum > std::thread::hardware_concurrency())
    {
        auto hardwareConcurrency = std::thread::hardware_concurrency();
        std::cout << "This computer has only " << hardwareConcurrency
                  << " threads available, but total "
                  << (mainThreadNum + copyThreadNum)
                  << " threads were requested" << std::endl;
    }

    std::cout << "Creating " << mainThreadNum << "main threads" << std::endl;
    m_mainThreadPool.reserve(mainThreadNum);
    for (size_t count = 0; count < mainThreadNum; ++count)
    {
        m_mainThreadPool.emplace_back(std::thread(m_runMain));
    }

    for (size_t count = 0; count < copyThreadNum; count++)
    {
        m_copyThreadPool.emplace_back(std::thread(m_runMain));
    }
}

size_t Engine::AddSourceUnit(SourceUnit&& sourceUnit)
{
    auto id = m_sourceUnitVector.size();
    m_sourceUnitVector.emplace_back(std::move(sourceUnit));
    return id;
}

size_t Engine::AddIntermediateUnit(IntermediateUnit&& intermediateUnit)
{
    auto id = m_intermediateUnitVector.size();
    m_intermediateUnitVector.emplace_back(std::move(intermediateUnit));
    return id;
}

size_t Engine::AddSinkUnit(SinkUnit&& sinkUnit)
{
    auto id = m_sinkUnitVector.size();
    m_sinkUnitVector.emplace_back(std::move(sinkUnit));
    return id;
}

void Engine::ConnectSourceToIntermediate(size_t originID, size_t destID,
                                         size_t destInputIndex)
{
    assert(originID < m_sourceUnitVector.size());
    assert(destID < m_intermediateUnitVector.size());
    auto& sourceUnit = m_sourceUnitVector.at(originID);
    auto& intermediateUnit = m_intermediateUnitVector.at(destID);
    auto& copyUnit = m_copyUnitVector.emplace_back(CopyUnit());
    copyUnit.SetOriginPtr(&sourceUnit);
    copyUnit.SetDestinationPtr(&intermediateUnit);
    sourceUnit.AddOutputPtr(&copyUnit);
    intermediateUnit.AddInputPtr(&copyUnit, destInputIndex);
}

void Engine::ConnectIntermediateToIntermediate(size_t originID, size_t destID, size_t destInputIndex)
{
    assert(originID < m_intermediateUnitVector.size());
    assert(destID < m_intermediateUnitVector.size());
    auto& originIntermediateUnit = m_intermediateUnitVector.at(originID);
    auto& destIntermediateUnit = m_intermediateUnitVector.at(destID);
    auto& copyUnit = m_copyUnitVector.emplace_back(CopyUnit());
    copyUnit.SetOriginPtr(&originIntermediateUnit);
    copyUnit.SetDestinationPtr(&destIntermediateUnit);
    originIntermediateUnit.AddOutputPtr(&copyUnit);
    destIntermediateUnit.AddInputPtr(&copyUnit, destInputIndex);
}

void Engine::ConnectIntermediateToSink(size_t originID, size_t destID, size_t destInputIndex){
    assert(originID < m_intermediateUnitVector.size());
    assert(destID < m_sinkUnitVector.size());
    auto& intermediateUnit = m_intermediateUnitVector.at(originID);
    auto& sinkUnit = m_sinkUnitVector.at(destID);
    auto& copyUnit = m_copyUnitVector.emplace_back(CopyUnit());
    copyUnit.SetOriginPtr(&intermediateUnit);
    copyUnit.SetDestinationPtr(&sinkUnit);
    intermediateUnit.AddOutputPtr(&copyUnit);
    sinkUnit.AddInputPtr(&copyUnit, destInputIndex);
}

// TODO : turn this into spinlock based pending function
void Engine::m_runMain()
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

void Engine::m_runCopy()
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

void Engine::EnqueueTask(TaskWrapper&& task)
{
    m_mainTaskQueue.Enqueue(std::move(task));
}

TaskWrapper Engine::DequeueTask()
{
    return m_mainTaskQueue.Dequeue();
}

void Engine::ScanUnitTasks()
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

void Engine::ScanCopyTasks()
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