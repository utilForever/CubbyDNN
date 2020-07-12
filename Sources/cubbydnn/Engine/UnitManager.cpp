// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Engine/UnitManager.hpp>
#include <cubbydnn/Units/HiddenComputableUnits/Dense.hpp>
#include <cubbydnn/Units/SinkComputableUnits/LossUnit.hpp>
#include <cubbydnn/Units/SourceComputableUnits/ConstantUnit.hpp>


namespace CubbyDNN::Graph
{
UnitManager::UnitManager(UnitManager&& unitManager) noexcept
    : m_unitMetaDataMap(std::move(unitManager.m_unitMetaDataMap)),
      m_unitMap(std::move(unitManager.m_unitMap))
{
}

UnitManager& UnitManager::operator=(UnitManager&& unitManager) noexcept
{
    m_unitMetaDataMap = std::move(unitManager.m_unitMetaDataMap);
    m_unitMap = std::move(unitManager.m_unitMap);
    return *this;
}

void UnitManager::AppendUnit(UnitMetaData&& unitMetaData)
{
    const auto unitId = unitMetaData.Id();

    m_unitMetaDataMap[unitId] = std::make_unique<UnitMetaData>(
        std::move(unitMetaData));
}

void UnitManager::Compile(const std::string& optimizerName,
                          const Parameter& optimizerParameters)
{
    m_connectUnits();

    for (auto& [key, metaDataPtr] : m_unitMetaDataMap)
    {
        const auto unitId = metaDataPtr->Id();
        auto type = unitId.Type;

        if (type.Name() == "DataLoader")
        {
            throw std::runtime_error("Not implemented");
        }
        if (type.Name() == "Dense")
        {
            auto unit = DenseUnit::CreateUnit(
                *metaDataPtr,
                m_makeOptimizer(optimizerName, optimizerParameters));
            m_unitMap[metaDataPtr->Id()] =
                std::make_unique<DenseUnit>(std::move(unit));
            continue;
        }
        if (type.Name() == "Dropout")
        {
            throw std::runtime_error("Not implemented");
        }
        if (type.Name() == "Activation")
        {
            auto unit = ActivationUnit::CreateUnit(*metaDataPtr);
            m_unitMap[metaDataPtr->Id()] =
                std::make_unique<ActivationUnit>(std::move(unit));
            continue;
        }
        if (type.Name() == "Reshape")
        {
            throw std::runtime_error("Not implemented");
        }
        if (type.Name() == "Loss")
        {
            auto unit = LossUnit::CreateUnit(*metaDataPtr);
            m_unitMap[metaDataPtr->Id()] =
                std::make_unique<LossUnit>(std::move(unit));
            continue;
        }
        if (type.Name() == "Constant")
        {
            auto unit = ConstantUnit::CreateUnit(*metaDataPtr);
            m_unitMap[metaDataPtr->Id()] =
                std::make_unique<ConstantUnit>(std::move(unit));
            continue;
        }
        if (type.Name() == "Multiply")
        {
            throw std::runtime_error("Not implemented");
        }
        if (type.Name() == "Add")
        {
            throw std::runtime_error("Not implemented");
        }

        throw std::runtime_error("No matching unit type");
    }
}

void UnitManager::Forward(std::size_t cycle)
{
    for (const auto& [key, unitPtr] : m_unitMap)
    {
        if (unitPtr->IsForwardReady(cycle))
            unitPtr->Forward();
        m_forwardCopy(key);
    }
}

void UnitManager::Backward(std::size_t cycle)
{
    for (const auto& [key, unitPtr] : m_unitMap)
    {
        if (unitPtr->IsBackwardReady(cycle))
            unitPtr->Backward();
        m_backwardCopy(key);
    }
}

void UnitManager::AsyncForward(std::size_t cycle)
{
    std::unordered_map<UnitId, std::future<bool>> futureVector;
    futureVector.reserve(10);

    for (const auto& [key, unitPtr] : m_unitMap)
    {
        if (unitPtr->IsForwardReady(cycle))
        {
            std::promise<bool> promise;
            futureVector[key] = promise.get_future();
            unitPtr->AsyncForward(std::move(promise));
        }
    }

    for (auto& [key, future] : futureVector)
    {
        future.wait();
        m_forwardCopy(key);
    }
}

void UnitManager::AsyncBackward(std::size_t cycle)
{
    std::unordered_map<UnitId, std::future<bool>> futureVector;
    futureVector.reserve(10);

    for (const auto& [key, unitPtr] : m_unitMap)
    {
        if (unitPtr->IsBackwardReady(cycle))
        {
            std::promise<bool> promise;
            futureVector[key] = promise.get_future();
            unitPtr->AsyncBackward(std::move(promise));
        }
    }

    for (auto& [key, future] : futureVector)
    {
        future.wait();
        m_backwardCopy(key);
    }
}

void UnitManager::m_forwardCopy(const UnitId& subjectUnitId)
{
    const auto& sourceMetaData = m_unitMetaDataMap[subjectUnitId];
    auto& subjectOutputTensor = m_unitMap[subjectUnitId]->ForwardOutput;

    for (const auto& outputUnitId : sourceMetaData->OutputUnitVector())
    {
        auto& nextInputTensorVector = m_unitMap[outputUnitId]->
            ForwardInputVector;
        for (auto& destTensor : nextInputTensorVector)
        {
            Tensor::ForwardTensor(subjectOutputTensor, destTensor);
            subjectOutputTensor.ForwardState.fetch_add(1);
            destTensor.ForwardState.fetch_add(1);
        }
    }
}

void UnitManager::m_backwardCopy(const UnitId& subjectUnitId)
{
    const auto& sourceMetaData = m_unitMetaDataMap[subjectUnitId];
    int index = 0;
    for (const auto& [key, subjectInputUnitId] : sourceMetaData->InputUnitMap())
    {
        auto& outputTensor = m_unitMap[subjectUnitId]->BackwardOutputVector[
            index];
        auto& nextBackwardInputTensorVector = m_unitMap[subjectInputUnitId]->
            BackwardInputVector;
        auto nextBackwardInputUnitVector =
            m_unitMetaDataMap[subjectInputUnitId]->OutputUnitVector();

        for (std::size_t i = 0; i < nextBackwardInputTensorVector.size(); ++i)
        {
            auto targetUnitId = nextBackwardInputUnitVector.at(i);
            if (targetUnitId == sourceMetaData->Id())
            {
                auto& destTensor =
                    nextBackwardInputTensorVector.at(i);
                Tensor::ForwardTensor(outputTensor, destTensor);
                outputTensor.BackwardState.fetch_add(1);
                destTensor.BackwardState.fetch_add(1);
            }
        }
        index += 1;
    }
}

void UnitManager::m_connectUnits()
{
    for (auto& [subjectKey, metaDataPtr] : m_unitMetaDataMap)
    {
        const auto unitId = metaDataPtr->Id();
        //! Analyzes dependency between units
        for (const auto& [inputKey, inputUnitId] : metaDataPtr->InputUnitMap())
        {
            m_unitMetaDataMap[inputUnitId]->AppendOutputUnitId(unitId);
        }
    }
}


std::unique_ptr<Compute::Optimizer> UnitManager::m_makeOptimizer(
    const std::string& optimizerName, const Parameter& parameters) const
{
    if (optimizerName == "SGD")
    {
        auto optimizer = std::make_unique<Compute::SGD>(
            parameters.GetFloatingPointParam("epsilon"));
        return std::move(optimizer);
    }
    throw std::runtime_error("Unsupported optimizer type");
}
}
