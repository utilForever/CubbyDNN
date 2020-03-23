// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cassert>
#include <cubbydnn/Engine/Engine.hpp>
#include "cubbyDnn/Units/HiddenComputableUnits/Dense.hpp"

namespace CubbyDNN
{
void Graph::ExecuteForward(std::size_t epochs)
{
    bool isFinished = false;
    m_maxEpochs = epochs;

    while (m_active && !isFinished)
    {
        isFinished = true;
        for (auto& sourceUnit : m_sourceUnitVector)
        {
            if (sourceUnit->IsReady() && sourceUnit->GetStateNum() < epochs)
            {
                sourceUnit->Forward();
                sourceUnit->ReleaseUnit();
                isFinished = false;
            }
        }

        for (auto& hiddenUnit : m_hiddenUnitVector)
        {
            if (hiddenUnit->IsReady() && hiddenUnit->GetStateNum() < epochs)
            {
                hiddenUnit->Forward();
                hiddenUnit->ReleaseUnit();
                isFinished = false;
            }
        }

        if (m_sinkUnit->IsReady() && m_sinkUnit->GetStateNum() < epochs)
        {
            m_sinkUnit->Forward();
            m_sinkUnit->ReleaseUnit();
            isFinished = false;
        }

        for (auto& copyUnit : m_copyUnitVector)
        {
            if (copyUnit->IsReady() && copyUnit->GetStateNum() < epochs)
            {
                copyUnit->Forward();
                copyUnit->ReleaseUnit();
                isFinished = false;
            }
        }
    }
}

void Graph::ExecuteBackward(size_t epochs)
{
    bool isFinished = false;
    m_maxEpochs = epochs;

    while (m_active && !isFinished)
    {
        isFinished = true;
        for (auto& sourceUnit : m_sourceUnitVector)
        {
            if (sourceUnit->IsReady() && sourceUnit->GetStateNum() < epochs)
            {
                sourceUnit->Backward();
                sourceUnit->ReleaseUnit();
                isFinished = false;
            }
        }

        for (auto& hiddenUnit : m_hiddenUnitVector)
        {
            if (hiddenUnit->IsReady() && hiddenUnit->GetStateNum() < epochs)
            {
                hiddenUnit->Backward();
                hiddenUnit->ReleaseUnit();
                isFinished = false;
            }
        }

        for (auto& sinkUnit : m_sinkUnitVector)
        {
            if (sinkUnit->IsReady() && sinkUnit->GetStateNum() < epochs)
            {
                sinkUnit->Backward();
                sinkUnit->ReleaseUnit();
                isFinished = false;
            }
        }

        for (auto& copyUnit : m_copyUnitVector)
        {
            if (copyUnit->IsReady() && copyUnit->GetStateNum() < epochs)
            {
                copyUnit->Backward();
                copyUnit->ReleaseUnit();
                isFinished = false;
            }
        }
    }
}

void Graph::ExecuteForwardParallel(std::size_t workers, std::size_t epochs)
{
    const auto hardwareConcurrency = std::thread::hardware_concurrency();
    if (workers + 1 > hardwareConcurrency)
    {
        std::cout << "This computer has only " << hardwareConcurrency
            << " threads available, but total " << workers
            << " threads were requested" << std::endl;
    }

    if (hardwareConcurrency < workers)
        workers = hardwareConcurrency;

    std::cout << "Creating " << workers << " workers" << std::endl;

    for (std::size_t count = 0; count < workers; ++count)
        m_mainThreadPool.emplace_back(std::thread(m_run));

    while (!m_IsComplete(epochs))
    {
        m_executeForwardUnits();
        m_executeCopyUnits();
    }

    for (std::size_t count = 0; count < m_mainThreadPool.size(); ++count)
    {
        TaskWrapper taskWrapper(TaskType::Join);
        m_taskQueue.Enqueue(std::move(taskWrapper));
    }
}

UnitIdentifier Graph::PlaceHolder(const Shape& shape)
{
    SharedPtr<PlaceHolderUnit> ptr =
        SharedPtr<PlaceHolderUnit>::Make(TensorInfo(shape, m_numberSystem));

    const auto id = m_sourceUnitVector.size();
    m_sourceUnitVector.emplace_back(std::move(ptr));
    return { UnitType::Source, id };
}

UnitIdentifier Graph::Hidden(
    const std::vector<UnitIdentifier>& previousUnitVector,
    TensorInfo outputTensorInfo, std::size_t numberOfOutputs)
{
    std::vector<TensorInfo> inputTensorInfoVector;
    inputTensorInfoVector.reserve(previousUnitVector.size());
    for (const auto& unitIdentifier : previousUnitVector)
    {
        if (unitIdentifier.Type == UnitType::Hidden)
            inputTensorInfoVector.emplace_back(
                m_hiddenUnitVector.at(unitIdentifier.ID)
                                  ->GetOutputTensorInfo());
        if (unitIdentifier.Type == UnitType::Source)
            inputTensorInfoVector.emplace_back(
                m_sourceUnitVector.at(unitIdentifier.ID)
                                  ->GetOutputTensorInfo());
    }

    const auto unitId = m_hiddenUnitVector.size();
    m_hiddenUnitVector.emplace_back(SharedPtr<HiddenUnit>::Make(
        inputTensorInfoVector, outputTensorInfo, numberOfOutputs));
    const UnitIdentifier unitIdentifier = { UnitType::Hidden, unitId };
    m_connectWithPreviousUnit(previousUnitVector, unitIdentifier);
    return unitIdentifier;
}

// TODO : put activation, initializing methods, etc.
UnitIdentifier Graph::Dense(const UnitIdentifier& input, std::size_t units)
{
    TensorInfo inputTensorInfo;
    TensorInfo weightTensorInfo;
    TensorInfo biasTensorInfo;
    if (input.Type == UnitType::Hidden)
    {
        inputTensorInfo =
            m_hiddenUnitVector.at(input.ID)->GetOutputTensorInfo();
    }
    else if (input.Type == UnitType::Source)
    {
        inputTensorInfo =
            m_hiddenUnitVector.at(input.ID)->GetOutputTensorInfo();
    }
    else
        throw std::runtime_error("input unit must be source or hidden");

    const Shape weightShape = { inputTensorInfo.GetShape().Row(), units };
    const Shape biasShape = { 1, inputTensorInfo.GetShape().Row() };

    // TODO : specify number system other than float
    TensorInfo weightInfo(weightShape);
    TensorInfo biasInfo(biasShape);

    SharedPtr<ConstantUnit> weightPtr = SharedPtr<ConstantUnit>::Make(
        weightInfo, AllocateData<float>(weightShape));
    SharedPtr<ConstantUnit> biasPtr = SharedPtr<ConstantUnit>::Make(
        weightInfo, AllocateData<float>(biasShape));

    const auto weightUnitID = m_sourceUnitVector.size();
    m_sourceUnitVector.emplace_back(std::move(weightPtr));
    const auto biasUnitID = m_sourceUnitVector.size();
    m_sourceUnitVector.emplace_back(std::move(biasPtr));

    const UnitIdentifier weightUnit = { UnitType::Source, weightUnitID };
    const UnitIdentifier biasUnit = { UnitType::Source, biasUnitID };

    const auto denseUnitID = m_hiddenUnitVector.size();
    SharedPtr<DenseUnit> densePtr = SharedPtr<DenseUnit>::Make(
        inputTensorInfo, weightTensorInfo, biasTensorInfo);

    densePtr->SetInputUnitVector({ input, weightUnit, biasUnit });

    m_hiddenUnitVector.emplace_back(SharedPtr<DenseUnit>::Make(
        inputTensorInfo, weightTensorInfo, biasTensorInfo));

    const UnitIdentifier unitIdentifier = { UnitType::Hidden, denseUnitID };
    return unitIdentifier;
}

//! TODO : Do dfs search to seek and connect units together
void Graph::Compile()
{
    auto identifier = m_sinkUnit.GetIdentifier();
    auto inputUnitVector = m_sinkUnit.GetInputUnitVector();
    m_executionOrder.emplace_front(identifier);
    for (auto unit : inputUnitVector)
    {
        m_getExecutionOrder(unit, m_executionOrder);
    }
}

void Graph::m_connectSourceToHidden(std::size_t originID, std::size_t destID,
                                    std::size_t destInputIndex)
{
    assert(originID < m_sourceUnitVector.size());
    assert(destID < m_hiddenUnitVector.size());
    auto sourceUnit = m_sourceUnitVector.at(originID);
    auto intermediateUnit = m_hiddenUnitVector.at(destID);
    m_copyUnitVector.emplace_back(SharedPtr<CopyUnit>::Make());
    auto copyUnit = m_copyUnitVector.at(m_copyUnitVector.size() - 1);
    copyUnit->SetInputPtr(sourceUnit);
    copyUnit->AddOutputPtr(intermediateUnit);
    const auto inputIndex = sourceUnit->AddOutputPtr(copyUnit);
    intermediateUnit->AddInputPtr(copyUnit, destInputIndex);
    copyUnit->SetInputTensorIndex(inputIndex);
    copyUnit->SetOutputTensorIndex(destInputIndex);
}

void Graph::m_connectHiddenToHidden(std::size_t originID, std::size_t destID,
                                    std::size_t destInputIndex)
{
    assert(originID < m_hiddenUnitVector.size());
    assert(destID < m_hiddenUnitVector.size());
    auto originIntermediateUnit = m_hiddenUnitVector.at(originID);
    auto destIntermediateUnit = m_hiddenUnitVector.at(destID);
    m_copyUnitVector.emplace_back(SharedPtr<CopyUnit>::Make());
    auto copyUnit = m_copyUnitVector.at(m_copyUnitVector.size() - 1);
    copyUnit->SetInputPtr(originIntermediateUnit);
    copyUnit->AddOutputPtr(destIntermediateUnit);
    const auto inputIndex = originIntermediateUnit->AddOutputPtr(copyUnit);
    destIntermediateUnit->AddInputPtr(copyUnit, destInputIndex);
    copyUnit->SetInputTensorIndex(inputIndex);
    copyUnit->SetOutputTensorIndex(destInputIndex);
}

void Graph::m_connectHiddenToSink(std::size_t originID, std::size_t destID,
                                  std::size_t destInputIndex)
{
    assert(originID < m_hiddenUnitVector.size());
    assert(destID < m_sinkUnitVector.size());
    auto hiddenUnit = m_hiddenUnitVector.at(originID);
    auto sinkUnit = m_sinkUnitVector.at(destID);
    m_copyUnitVector.emplace_back(SharedPtr<CopyUnit>::Make());
    auto copyUnit = m_copyUnitVector.at(m_copyUnitVector.size() - 1);
    copyUnit->SetInputPtr(hiddenUnit);
    copyUnit->AddOutputPtr(sinkUnit);
    const auto inputIndex = hiddenUnit->AddOutputPtr(copyUnit);
    sinkUnit->AddInputPtr(copyUnit, destInputIndex);
    copyUnit->SetInputTensorIndex(inputIndex);
    copyUnit->SetOutputTensorIndex(destInputIndex);
}

void Graph::m_connectWithPreviousUnit(
    const std::vector<UnitIdentifier>& previousUnitVector,
    UnitIdentifier subjectUnitIdentifier)
{
    std::size_t inputIdx = 0;
    if (subjectUnitIdentifier.Type == UnitType::Hidden)
        for (const auto& unit : previousUnitVector)
        {
            if (unit.Type == UnitType::Source)
            {
                m_connectSourceToHidden(unit.ID, subjectUnitIdentifier.ID,
                                        inputIdx++);
            }
            else if (unit.Type == UnitType::Hidden)
            {
                m_connectHiddenToHidden(unit.ID, subjectUnitIdentifier.ID,
                                        inputIdx++);
            }
            else
                assert("Unsupported type of unit");
        }

    else if (subjectUnitIdentifier.Type == UnitType::Sink)
        for (const auto& unit : previousUnitVector)
        {
            if (unit.Type == UnitType::Hidden)
            {
                m_connectHiddenToSink(unit.ID, subjectUnitIdentifier.ID,
                                      inputIdx++);
            }
            else
                assert("Unsupported type of unit");
        }
    else
        assert("Unsupported type of unit");
}

void Graph::m_run()
{
    TaskWrapper taskWrapper = m_taskQueue.Dequeue();
    while (taskWrapper.Type != TaskType::Join)
    {
        auto task = taskWrapper.GetTask();
        task();
        taskWrapper = m_taskQueue.Dequeue();
    }
}

void Graph::EnqueueTask(TaskWrapper&& task)
{
    m_taskQueue.Enqueue(std::move(task));
}

TaskWrapper Graph::DequeueTask()
{
    return m_taskQueue.Dequeue();
}

void Graph::JoinThreads()
{
    for (auto& thread : m_mainThreadPool)
    {
        if (thread.joinable())
        {
            thread.join();
            std::cout << "Joined main" << std::endl;
        }
    }

    if (m_scanMainThread.joinable())
        m_scanMainThread.join();
    if (m_scanCopyThread.joinable())
        m_scanCopyThread.join();
}

void Graph::Abort()
{
    for (std::size_t count = 0; count < m_mainThreadPool.size(); ++count)
        m_taskQueue.Enqueue(TaskWrapper(
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

void Graph::m_executeForwardUnits()
{
    int desired = 0;
    std::atomic_int count = 0;

    for (auto& sourceUnit : m_sourceUnitVector)
    {
        if (sourceUnit->IsReady())
        {
            const auto computeFunc = [&sourceUnit]() { sourceUnit->Forward(); };
            const auto updateState = [&sourceUnit, &count]()
            {
                count.fetch_add(1);
                sourceUnit->ReleaseUnit();
            };
            TaskWrapper taskWrapper(TaskType::ComputeSource, computeFunc,
                                    updateState);
            m_taskQueue.Enqueue(std::move(taskWrapper));
            ++desired;
        }
    }
    for (auto& hiddenUnit : m_hiddenUnitVector)
    {
        if (hiddenUnit->IsReady())
        {
            const auto computeFunc = [&hiddenUnit]() { hiddenUnit->Forward(); };
            const auto updateState = [&hiddenUnit, &count]()
            {
                count.fetch_add(1);
                hiddenUnit->ReleaseUnit();
            };
            TaskWrapper taskWrapper(TaskType::ComputeHidden, computeFunc,
                                    updateState);
            m_taskQueue.Enqueue(std::move(taskWrapper));
            ++desired;
        }
    }
    for (auto& sinkUnit : m_sinkUnitVector)
    {
        if (sinkUnit->IsReady())
        {
            const auto computeFunc = [&sinkUnit]() { sinkUnit->Forward(); };
            const auto updateState = [&sinkUnit, &count]()
            {
                count.fetch_add(1);
                sinkUnit->ReleaseUnit();
            };
            TaskWrapper taskWrapper(TaskType::ComputeSink, computeFunc,
                                    updateState);
            m_taskQueue.Enqueue(std::move(taskWrapper));
            ++desired;
        }
    }

    m_ready.exchange(true);
    while (count.load(std::memory_order_acquire) != desired)
        std::this_thread::yield();
    m_ready.exchange(false);
}

void Graph::m_executeCopyUnits()
{
    int desired = 0;
    std::atomic_int count = 0;
    for (auto& copyUnit : m_copyUnitVector)
    {
        if (copyUnit->IsReady())
        {
            const auto computeFunc = [&copyUnit]() { copyUnit->Forward(); };
            const auto updateState = [&copyUnit, &count]()
            {
                count.fetch_add(1);
                copyUnit->ReleaseUnit();
            };

            TaskWrapper taskWrapper(TaskType::Copy, computeFunc, updateState);
            // std::cout << "Enqueue Copy" << std::endl;
            m_taskQueue.Enqueue(std::move(taskWrapper));
            ++desired;
        }
    }

    m_ready.exchange(true);
    while (count.load(std::memory_order_acquire) != desired)
        std::this_thread::yield();
    m_ready.exchange(false);
}

bool Graph::m_IsComplete(std::size_t epochs)
{
    bool isComplete = true;
    for (auto& sourceUnit : m_sourceUnitVector)
        if (sourceUnit->GetStateNum() < epochs)
            isComplete = false;
    for (auto& hiddenUnit : m_hiddenUnitVector)
        if (hiddenUnit->GetStateNum() < epochs)
            isComplete = false;
    for (auto& sinkUnit : m_sinkUnitVector)
        if (sinkUnit->GetStateNum() < epochs)
            isComplete = false;
    for (auto& copyUnit : m_copyUnitVector)
        if (copyUnit->GetStateNum() < epochs)
            isComplete = false;

    return isComplete;
}
} // namespace CubbyDNN
