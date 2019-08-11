//
// Created by jwkim98 on 6/26/19.
//
#include <cubbydnn/Engine/Engine.hpp>
#include <iostream>

namespace CubbyDNN
{
std::thread Engine::m_scanMainThread;

std::thread Engine::m_scanCopyThread;

std::vector<std::thread> Engine::m_mainThreadPool = std::vector<std::thread>();

std::vector<std::thread> Engine::m_copyThreadPool;

SpinLockQueue<TaskWrapper> Engine::m_mainTaskQueue(10000);

SpinLockQueue<TaskWrapper> Engine::m_copyTaskQueue(10000);

std::atomic<bool> Engine::m_dirty(true);

bool Engine::m_active = true;

std::vector<SourceUnit*> Engine::m_sourceUnitVector;

std::vector<SinkUnit*> Engine::m_sinkUnitVector;

std::vector<HiddenUnit*> Engine::m_intermediateUnitVector;

std::vector<CopyUnit*> Engine::m_copyUnitVector;

size_t Engine::m_maxEpochs = 0;

void Engine::StartExecution(size_t mainThreadNum, size_t copyThreadNum,
                            size_t epochs)
{
    if (mainThreadNum + copyThreadNum > std::thread::hardware_concurrency())
    {
        auto hardwareConcurrency = std::thread::hardware_concurrency();
        std::cout << "This computer has only " << hardwareConcurrency
                  << " threads available, but total "
                  << (mainThreadNum + copyThreadNum)
                  << " threads were requested" << std::endl;
    }
    m_maxEpochs = epochs;

    std::cout << "Creating " << mainThreadNum << " main threads" << std::endl;
    m_mainThreadPool.reserve(mainThreadNum);
    for (size_t count = 0; count < mainThreadNum; ++count)
    {
        m_mainThreadPool.emplace_back(std::thread(m_runMain));
    }
    std::cout << "Creating " << copyThreadNum << " copy threads" << std::endl;
    for (size_t count = 0; count < copyThreadNum; count++)
    {
        m_copyThreadPool.emplace_back(std::thread(m_runCopy));
    }

    m_scanMainThread = std::thread(ScanUnitTasks);
    m_scanCopyThread = std::thread(ScanCopyTasks);
}

size_t Engine::AddSourceUnit(SourceUnit* sourceUnit)
{
    auto id = m_sourceUnitVector.size();
    m_sourceUnitVector.emplace_back(std::move(sourceUnit));
    return id;
}

size_t Engine::AddHiddenUnit(HiddenUnit* hiddenUnit)
{
    auto id = m_intermediateUnitVector.size();
    m_intermediateUnitVector.emplace_back(std::move(hiddenUnit));
    return id;
}

size_t Engine::AddSinkUnit(SinkUnit* sinkUnit)
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
    auto* sourceUnit = m_sourceUnitVector.at(originID);
    auto* intermediateUnit = m_intermediateUnitVector.at(destID);
    m_copyUnitVector.emplace_back(new CopyUnit());
    auto* copyUnit = m_copyUnitVector.at(m_copyUnitVector.size() - 1);
    copyUnit->SetInputPtr(sourceUnit);
    copyUnit->SetOutputPtr(intermediateUnit);
    sourceUnit->AddOutputPtr(copyUnit);
    intermediateUnit->AddInputPtr(copyUnit, destInputIndex);
}

void Engine::ConnectIntermediateToIntermediate(size_t originID, size_t destID,
                                               size_t destInputIndex)
{
    assert(originID < m_intermediateUnitVector.size());
    assert(destID < m_intermediateUnitVector.size());
    auto* originIntermediateUnit = m_intermediateUnitVector.at(originID);
    auto* destIntermediateUnit = m_intermediateUnitVector.at(destID);
    m_copyUnitVector.emplace_back(new CopyUnit());
    auto* copyUnit = m_copyUnitVector.at(m_copyUnitVector.size() - 1);
    copyUnit->SetInputPtr(originIntermediateUnit);
    copyUnit->SetOutputPtr(destIntermediateUnit);
    originIntermediateUnit->AddOutputPtr(copyUnit);
    destIntermediateUnit->AddInputPtr(copyUnit, destInputIndex);
}

void Engine::ConnectIntermediateToSink(size_t originID, size_t destID,
                                       size_t destInputIndex)
{
    assert(originID < m_intermediateUnitVector.size());
    assert(destID < m_sinkUnitVector.size());
    auto* intermediateUnit = m_intermediateUnitVector.at(originID);
    auto* sinkUnit = m_sinkUnitVector.at(destID);
    m_copyUnitVector.emplace_back(new CopyUnit());
    auto* copyUnit = m_copyUnitVector.at(m_copyUnitVector.size() - 1);
    copyUnit->SetInputPtr(intermediateUnit);
    copyUnit->SetOutputPtr(sinkUnit);
    intermediateUnit->AddOutputPtr(copyUnit);
    sinkUnit->AddInputPtr(copyUnit, destInputIndex);
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

void Engine::JoinThreads()
{
    for (auto& thread : m_mainThreadPool)
    {
        if (thread.joinable())
            thread.join();
    }

    for (auto& thread : m_copyThreadPool)
    {
        if (thread.joinable())
            thread.join();
    }
    if (m_scanMainThread.joinable())
        m_scanMainThread.join();
    if (m_scanCopyThread.joinable())
        m_scanCopyThread.join();
}

void Engine::Abort()
{
    for (size_t count = 0; count < m_mainThreadPool.size(); ++count)
        m_mainTaskQueue.Enqueue(TaskWrapper(TaskType::Join, []() {}, []() {}));

    for (size_t count = 0; count < m_copyThreadPool.size(); ++count)
        m_mainTaskQueue.Enqueue(TaskWrapper(TaskType::Join, []() {}, []() {}));

    for (auto& thread : m_mainThreadPool)
    {
        if (thread.joinable())
            thread.join();
    }
    m_active = false;
    if (m_scanMainThread.joinable())
        m_scanMainThread.join();
    if (m_scanCopyThread.joinable())
        m_scanCopyThread.join();
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
                if (sourceUnit->IsReady() &&
                    sourceUnit->GetStateNum() < m_maxEpochs)
                {
                    auto computeFunc = [&sourceUnit]() {
                        sourceUnit->Compute();
                    };
                    auto updateState = [&sourceUnit]() {
                        sourceUnit->ReleaseUnit();
                    };
                    sourceUnit->AcquireUnit();
                    TaskWrapper taskWrapper(TaskType::ComputeSource,
                                            computeFunc, updateState);
                    m_mainTaskQueue.Enqueue(std::move(taskWrapper));
                }
                else if (sourceUnit->GetStateNum() != m_maxEpochs)
                {
                    isFinished = false;
                }
            }

            for (auto& hiddenUnit : m_intermediateUnitVector)
            {
                if (hiddenUnit->IsReady() &&
                    hiddenUnit->GetStateNum() < m_maxEpochs)
                {
                    auto computeFunc = [&hiddenUnit]() {
                        hiddenUnit->Compute();
                    };
                    auto updateState = [&hiddenUnit]() {
                        hiddenUnit->ReleaseUnit();
                    };
                    hiddenUnit->AcquireUnit();
                    TaskWrapper taskWrapper(TaskType::ComputeIntermediate,
                                            computeFunc, updateState);
                    m_mainTaskQueue.Enqueue(std::move(taskWrapper));
                }
                else if (hiddenUnit->GetStateNum() != m_maxEpochs)
                {
                    isFinished = false;
                }
            }

            for (auto& sinkUnit : m_sinkUnitVector)
            {
                if (sinkUnit->IsReady() &&
                    sinkUnit->GetStateNum() < m_maxEpochs)
                {
                    auto computeFunc = [&sinkUnit]() { sinkUnit->Compute(); };
                    auto updateState = [&sinkUnit]() {
                        sinkUnit->ReleaseUnit();
                    };
                    sinkUnit->AcquireUnit();
                    TaskWrapper taskWrapper(TaskType::ComputeSink, computeFunc,
                                            updateState);
                    m_mainTaskQueue.Enqueue(std::move(taskWrapper));
                }
                else if (sinkUnit->GetStateNum() != m_maxEpochs)
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

    for (size_t count = 0; count < m_mainThreadPool.size(); ++count)
    {
        auto dummyFunc = []() {};
        TaskWrapper taskWrapper(TaskType::Join, dummyFunc, dummyFunc);
        m_mainTaskQueue.Enqueue(std::move(taskWrapper));
    }
}

void Engine::ReleaseResources()
{
    for (auto* sourcePtr : m_sourceUnitVector)
    {
        delete sourcePtr;
    }

    for (auto* hiddenPtr : m_intermediateUnitVector)
    {
        delete hiddenPtr;
    }

    for(auto* sinkPtr : m_sinkUnitVector){
        delete sinkPtr;
    }

    for (auto* copyPtr : m_copyUnitVector)
    {
        delete copyPtr;
    }
}

void Engine::ScanCopyTasks()
{
    bool isFinished = false;
    while (m_active && !isFinished)
    {
        if (m_dirty)
        {
            isFinished = true;
            for (auto& copyUnit : m_copyUnitVector)
            {
                if (copyUnit->IsReady() &&
                    copyUnit->GetStateNum() < m_maxEpochs)
                {
                    auto computeFunc = [&copyUnit]() { copyUnit->Compute(); };
                    auto updateState = [&copyUnit]() {
                        copyUnit->ReleaseUnit();
                    };
                    copyUnit->AcquireUnit();
                    TaskWrapper taskWrapper(TaskType::Copy, computeFunc,
                                            updateState);
                    m_copyTaskQueue.Enqueue(std::move(taskWrapper));
                }
                if (copyUnit->GetStateNum() < m_maxEpochs)
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

    for (size_t count = 0; count < m_copyThreadPool.size(); ++count)
    {
        auto dummyFunc = []() {};
        TaskWrapper taskWrapper(TaskType::Join, dummyFunc, dummyFunc);
        m_copyTaskQueue.Enqueue(std::move(taskWrapper));
    }
}

}  // namespace CubbyDNN