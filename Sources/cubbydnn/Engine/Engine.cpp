// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Engine/Engine.hpp>

namespace CubbyDNN
{
std::thread Engine::m_scanMainThread;

std::thread Engine::m_scanCopyThread;

std::vector<std::thread> Engine::m_mainThreadPool = std::vector<std::thread>();

std::vector<std::thread> Engine::m_copyThreadPool;

SpinLockQueue<TaskWrapper> Engine::m_mainTaskQueue(1000);

SpinLockQueue<TaskWrapper> Engine::m_copyTaskQueue(1000);

std::atomic<bool> Engine::m_dirty(true);

bool Engine::m_active = true;

std::vector<SharedPtr<SourceUnit>> Engine::m_sourceUnitVector;

std::vector<SharedPtr<SinkUnit>> Engine::m_sinkUnitVector;

std::vector<SharedPtr<HiddenUnit>> Engine::m_hiddenUnitVector;

std::vector<SharedPtr<CopyUnit>> Engine::m_copyUnitVector;

size_t Engine::m_maxEpochs = 0;

void Engine::StartExecution(size_t epochs)
{
    bool isFinished = false;
    m_maxEpochs = epochs;

    while (m_active && !isFinished)
    {
        isFinished = true;
        for (auto& sourceUnit : m_sourceUnitVector)
        {
            if (sourceUnit->IsReady() &&
                sourceUnit->GetStateNum() < m_maxEpochs)
            {
                sourceUnit->Compute();
                sourceUnit->ReleaseUnit();
                isFinished = false;
            }
            else if (sourceUnit->GetStateNum() < m_maxEpochs)
            {
                isFinished = false;
            }
        }

        for (auto& hiddenUnit : m_hiddenUnitVector)
        {
            if (hiddenUnit->IsReady() &&
                hiddenUnit->GetStateNum() < m_maxEpochs)
            {
                hiddenUnit->Compute();
                hiddenUnit->ReleaseUnit();
                isFinished = false;
            }
            else if (hiddenUnit->GetStateNum() < m_maxEpochs)
            {
                isFinished = false;
            }
        }

        for (auto& sinkUnit : m_sinkUnitVector)
        {
            if (sinkUnit->IsReady() && sinkUnit->GetStateNum() < m_maxEpochs)
            {
                sinkUnit->Compute();
                sinkUnit->ReleaseUnit();
                isFinished = false;
            }
            else if (sinkUnit->GetStateNum() < m_maxEpochs)
            {
                isFinished = false;
            }
        }

        for (auto& copyUnit : m_copyUnitVector)
        {
            if (copyUnit->IsReady() && copyUnit->GetStateNum() < m_maxEpochs)
            {
                copyUnit->Compute();
                copyUnit->ReleaseUnit();
                isFinished = false;
            }
            else if (copyUnit->GetStateNum() < m_maxEpochs)
            {
                isFinished = false;
            }
        }
    }
}

void Engine::StartExecution(size_t mainThreadSize, size_t copyThreadSize,
                            size_t epochs)
{
    if (mainThreadSize + copyThreadSize > std::thread::hardware_concurrency())
    {
        const auto hardwareConcurrency = std::thread::hardware_concurrency();
        std::cout << "This computer has only " << hardwareConcurrency
            << " threads available, but total "
            << mainThreadSize + copyThreadSize
            << " threads were requested" << std::endl;
    }
    m_maxEpochs = epochs;

    std::cout << "Creating " << mainThreadSize << " main threads" << std::endl;
    m_mainThreadPool.reserve(mainThreadSize);
    for (size_t count = 0; count < mainThreadSize; ++count)
    {
        m_mainThreadPool.emplace_back(std::thread(m_runMain));
    }
    std::cout << "Creating " << copyThreadSize << " copy threads" << std::endl;
    for (size_t count = 0; count < copyThreadSize; count++)
    {
        m_copyThreadPool.emplace_back(std::thread(m_runCopy));
    }

    m_scanMainThread = std::thread(ScanUnitTasks);
}

size_t Engine::AddSourceUnit(
    const std::vector<TensorInfo>& outputTensorInfoVector)
{
    const auto unitId = m_sourceUnitVector.size();
    m_sourceUnitVector.emplace_back(
        SharedPtr<SourceUnit>::Make(outputTensorInfoVector));
    return unitId;
}

size_t Engine::Constant(const TensorInfo& output, void* dataPtr,
                        int numberOfOutputs)
{
    const auto unitId = m_sourceUnitVector.size();
    m_sourceUnitVector.emplace_back(
        SharedPtr<ConstantUnit>::Make(output, numberOfOutputs, dataPtr));
    return unitId;
}

size_t Engine::AddHiddenUnit(
    const std::vector<TensorInfo>& inputTensorInfoVector,
    const std::vector<TensorInfo>& outputTensorInfoVector)
{
    const auto unitId = m_hiddenUnitVector.size();
    m_hiddenUnitVector.emplace_back(SharedPtr<HiddenUnit>::Make(
        inputTensorInfoVector, outputTensorInfoVector));
    return unitId;
}

size_t Engine::Multiply(const TensorInfo& inputA, const TensorInfo& inputB,
                        const TensorInfo& output)
{
    const auto unitId = m_hiddenUnitVector.size();
    m_hiddenUnitVector.emplace_back(
        SharedPtr<MatMul>::Make(inputA, inputB, output));
    return unitId;
}

size_t Engine::AddSinkUnit(const std::vector<TensorInfo>& inputTensorInfoVector)
{
    const auto id = m_sinkUnitVector.size();
    m_sinkUnitVector.emplace_back(
        SharedPtr<SinkUnit>::Make(inputTensorInfoVector));
    return id;
}

size_t Engine::AddSinkUnitWithTest(
    const std::vector<TensorInfo>& inputTensorInfoVector,
    const std::function<void(const Tensor& tensor)>& testFunction)
{
    const auto id = m_sinkUnitVector.size();
    auto sinkUnitPtr =
        SharedPtr<SinkTestUnit>::Make(inputTensorInfoVector, testFunction);
    m_sinkUnitVector.emplace_back(sinkUnitPtr);
    return id;
}

void Engine::ConnectSourceToHidden(size_t originID, size_t destID,
                                   size_t destInputIndex)
{
    assert(originID < m_sourceUnitVector.size());
    assert(destID < m_hiddenUnitVector.size());
    auto sourceUnit = m_sourceUnitVector.at(originID);
    auto intermediateUnit = m_hiddenUnitVector.at(destID);
    m_copyUnitVector.emplace_back(SharedPtr<CopyUnit>::Make());
    auto copyUnit = m_copyUnitVector.at(m_copyUnitVector.size() - 1);
    copyUnit->SetInputPtr(sourceUnit);
    copyUnit->SetOutputPtr(intermediateUnit);
    const auto inputIndex = sourceUnit->AddOutputPtr(copyUnit);
    intermediateUnit->AddInputPtr(copyUnit, destInputIndex);
    copyUnit->SetInputTensorIndex(inputIndex);
    copyUnit->SetOutputTensorIndex(destInputIndex);
}

void Engine::ConnectHiddenToHidden(size_t originID, size_t destID,
                                   size_t destInputIndex)
{
    assert(originID < m_hiddenUnitVector.size());
    assert(destID < m_hiddenUnitVector.size());
    auto originIntermediateUnit = m_hiddenUnitVector.at(originID);
    auto destIntermediateUnit = m_hiddenUnitVector.at(destID);
    m_copyUnitVector.emplace_back(SharedPtr<CopyUnit>::Make());
    auto copyUnit = m_copyUnitVector.at(m_copyUnitVector.size() - 1);
    copyUnit->SetInputPtr(originIntermediateUnit);
    copyUnit->SetOutputPtr(destIntermediateUnit);
    const auto inputIndex = originIntermediateUnit->AddOutputPtr(copyUnit);
    destIntermediateUnit->AddInputPtr(copyUnit, destInputIndex);
    copyUnit->SetInputTensorIndex(inputIndex);
    copyUnit->SetOutputTensorIndex(destInputIndex);
}

void Engine::ConnectHiddenToSink(size_t originID, size_t destID,
                                 size_t destInputIndex)
{
    assert(originID < m_hiddenUnitVector.size());
    assert(destID < m_sinkUnitVector.size());
    auto intermediateUnit = m_hiddenUnitVector.at(originID);
    auto sinkUnit = m_sinkUnitVector.at(destID);
    m_copyUnitVector.emplace_back(SharedPtr<CopyUnit>::Make());
    auto copyUnit = m_copyUnitVector.at(m_copyUnitVector.size() - 1);
    copyUnit->SetInputPtr(intermediateUnit);
    copyUnit->SetOutputPtr(sinkUnit);
    const auto inputIndex = intermediateUnit->AddOutputPtr(copyUnit);
    sinkUnit->AddInputPtr(copyUnit, destInputIndex);
    copyUnit->SetInputTensorIndex(inputIndex);
    copyUnit->SetOutputTensorIndex(destInputIndex);
}

void Engine::m_runMain()
{
    TaskWrapper taskWrapper = m_mainTaskQueue.Dequeue();
    while (taskWrapper.Type != TaskType::Join)
    {
        //std::cout << "main execute" << std::endl;
        auto task = taskWrapper.GetTask();
        task();
        std::atomic_exchange_explicit(&m_dirty, true,
                                      std::memory_order_seq_cst);
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
                                      std::memory_order_seq_cst);
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
        {
            thread.join();
            std::cout << "Joined main" << std::endl;
        }
    }

    for (auto& thread : m_copyThreadPool)
    {
        if (thread.joinable())
        {
            std::cout << "Joined Copy" << std::endl;
            thread.join();
        }
    }
    if (m_scanMainThread.joinable())
        m_scanMainThread.join();
    if (m_scanCopyThread.joinable())
        m_scanCopyThread.join();
}

void Engine::Abort()
{
    for (size_t count = 0; count < m_mainThreadPool.size(); ++count)
        m_mainTaskQueue.Enqueue(TaskWrapper(
            TaskType::Join, []()
            {
            }, []()
            {
            }));

    for (size_t count = 0; count < m_copyThreadPool.size(); ++count)
        m_copyTaskQueue.Enqueue(TaskWrapper(
            TaskType::Join, []()
            {
            }, []()
            {
            }));

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
    static size_t copyEnqueueCount = 0;
    static size_t SourceEnqueueCount = 0;
    static size_t HiddenEnqueueCount = 0;
    static size_t SinkEnqueueCount = 0;
    while (m_active && !isFinished)
    {
        isFinished = true;
        for (auto& sourceUnit : m_sourceUnitVector)
        {
            if (sourceUnit->IsReady() &&
                sourceUnit->GetStateNum() < m_maxEpochs)
            {
                const auto computeFunc = [&sourceUnit]()
                {
                    sourceUnit->Compute();
                };
                const auto updateState = [&sourceUnit]()
                {
                    sourceUnit->ReleaseUnit();
                };
                sourceUnit->AcquireUnit();
                TaskWrapper taskWrapper(TaskType::ComputeSource,
                                        computeFunc, updateState);
                m_mainTaskQueue.Enqueue(std::move(taskWrapper));
                ++SourceEnqueueCount;
            }
            else if (sourceUnit->GetStateNum() < m_maxEpochs)
            {
                isFinished = false;
            }
        }

        for (auto& hiddenUnit : m_hiddenUnitVector)
        {
            if (hiddenUnit->IsReady() &&
                hiddenUnit->GetStateNum() < m_maxEpochs)
            {
                //std::cout << "hidden ready" << std::endl;
                const auto computeFunc = [&hiddenUnit]()
                {
                    hiddenUnit->Compute();
                };
                const auto updateState = [&hiddenUnit]()
                {
                    hiddenUnit->ReleaseUnit();
                };
                hiddenUnit->AcquireUnit();
                TaskWrapper taskWrapper(TaskType::ComputeIntermediate,
                                        computeFunc, updateState);
                m_mainTaskQueue.Enqueue(std::move(taskWrapper));
                ++HiddenEnqueueCount;
            }
            else if (hiddenUnit->GetStateNum() < m_maxEpochs)
            {
                isFinished = false;
            }
        }

        for (auto& sinkUnit : m_sinkUnitVector)
        {
            if (sinkUnit->IsReady() &&
                sinkUnit->GetStateNum() < m_maxEpochs)
            {
                //std::cout << "sink ready" << std::endl;
                const auto computeFunc = [&sinkUnit]()
                {
                    sinkUnit->Compute();
                };
                const auto updateState = [&sinkUnit]()
                {
                    sinkUnit->ReleaseUnit();
                };
                sinkUnit->AcquireUnit();
                TaskWrapper taskWrapper(TaskType::ComputeSink, computeFunc,
                                        updateState);
                m_mainTaskQueue.Enqueue(std::move(taskWrapper));
                ++SinkEnqueueCount;
            }
            else if (sinkUnit->GetStateNum() < m_maxEpochs)
            {
                isFinished = false;
            }
        }

        for (auto& copyUnit : m_copyUnitVector)
        {
            if (copyUnit->IsReady() &&
                copyUnit->GetStateNum() < m_maxEpochs)
            {
                const auto computeFunc = [&copyUnit]()
                {
                    copyUnit->Compute();
                };
                const auto updateState = [&copyUnit]()
                {
                    copyUnit->ReleaseUnit();
                };
                copyUnit->AcquireUnit();
                TaskWrapper taskWrapper(TaskType::Copy, computeFunc,
                                        updateState);

                // TODO : Do not put same task into the queue again
                m_copyTaskQueue.Enqueue(std::move(taskWrapper));
                copyEnqueueCount++;
            }
            if (copyUnit->GetStateNum() < m_maxEpochs)
            {
                isFinished = false;
            }
        }
    }

    for (size_t count = 0; count < m_mainThreadPool.size(); ++count)
    {
        const auto dummyFunc = []()
        {
        };
        TaskWrapper taskWrapper(TaskType::Join, dummyFunc, dummyFunc);
        m_mainTaskQueue.Enqueue(std::move(taskWrapper));
    }

    for (size_t count = 0; count < m_copyThreadPool.size(); ++count)
        m_copyTaskQueue.Enqueue(TaskWrapper(
            TaskType::Join, []()
            {
            }, []()
            {
            }));
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
                    const auto computeFunc = [&copyUnit]()
                    {
                        copyUnit->Compute();
                    };
                    const auto updateState = [&copyUnit]()
                    {
                        copyUnit->ReleaseUnit();
                    };
                    copyUnit->AcquireUnit();
                    TaskWrapper taskWrapper(TaskType::Copy, computeFunc,
                                            updateState);

                    // TODO : Do not put same task into the queue again
                    m_copyTaskQueue.Enqueue(std::move(taskWrapper));
                }
                if (copyUnit->GetStateNum() < m_maxEpochs)
                {
                    isFinished = false;
                }
            }
            m_dirty.exchange(false, std::memory_order_seq_cst);
        }
        else
        {
            std::this_thread::yield();
        }
    }
}
} // namespace CubbyDNN
