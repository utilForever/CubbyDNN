// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.


#include <cubbydnn/Engine/UnitManager.hpp>
#include <cubbydnn/Units/HiddenComputableUnits/Dense.hpp>


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

    m_unitMetaDataMap[unitId.Id] = std::make_unique<UnitMetaData>(
        std::move(unitMetaData));
}

void UnitManager::Compile(std::string optimizerName,
                          const ParameterPack& optimizerParameters)
{
    m_connectUnits();

    for (auto& [key, metaDataPtr] : m_unitMetaDataMap)
    {
        const auto unitId = metaDataPtr->Id();
        auto type = unitId.Type;

        if (type.Name() == "PlaceHolder")
        {
        }
        else if (type.Name() == "Dense")
        {
            DenseUnit::CreateUnit(
                *metaDataPtr,
                m_makeOptimizer(optimizerName, optimizerParameters));
        }
        else if (type.Name() == "Activation")
        {
            ActivationUnit::CreateUnit(*metaDataPtr);
        }
        else
            throw std::runtime_error("UnImplemented unit type");
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
    std::unordered_map<std::size_t, std::future<bool>> futureVector;
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
    std::unordered_map<std::size_t, std::future<bool>> futureVector;
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

void UnitManager::m_forwardCopy(std::size_t subjectUnitKey)
{
    const auto& sourceMetaData = m_unitMetaDataMap[subjectUnitKey];
    for (const auto& unitId : sourceMetaData->OutputUnitVector())
    {
        auto& outputTensor = m_unitMap[subjectUnitKey]->ForwardOutput;
        auto& nextInputTensorVector = m_unitMap[unitId.Id]->ForwardInputVector;
        for (auto& destTensor : nextInputTensorVector)
        {
            Tensor::ForwardTensor(outputTensor, destTensor);
            outputTensor.ForwardState.fetch_add(1);
            destTensor.ForwardState.fetch_add(1);
        }
    }
}

void UnitManager::m_backwardCopy(std::size_t subjectUnitKey)
{
    const auto& sourceMetaData = m_unitMetaDataMap[subjectUnitKey];
    int index = 0;
    for (const auto& unitId : sourceMetaData->InputUnitVector())
    {
        auto& outputTensor = m_unitMap[subjectUnitKey]->BackwardOutputVector[
            index];
        auto& nextBackwardInputTensorVector = m_unitMap[unitId.Id]->
            BackwardInputVector;
        auto nextBackwardInputUnitVector =
            m_unitMetaDataMap[unitId.Id]->OutputUnitVector();

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
    for (auto& [key, metaDataPtr] : m_unitMetaDataMap)
    {
        auto unitId = metaDataPtr->Id();
        const auto& inputPtrVector = metaDataPtr->InputUnitVector();
        //! Analyzes dependency between units
        for (const auto& inputUnitId : inputPtrVector)
        {
            metaDataPtr->AppendOutputUnitId(inputUnitId);
        }
    }
}


std::unique_ptr<Compute::Optimizer> UnitManager::m_makeOptimizer(
    const std::string& optimizerName, const ParameterPack& parameters) const
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
