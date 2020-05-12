// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.


#include <cubbydnn/Engine/UnitManager.hpp>

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
}


void UnitManager::AddUnit(UnitMetaData unitMetaData)
{
    std::vector<Tensor> forwardInputVector;
    std::vector<Tensor> backwardInputVector;
    std::vector<Tensor> backwardOutputVector;

    Tensor forwardOutput =
        CreateTensor(unitMetaData.OutputShape(), NumberSystem::Float);

    for (const auto& shape : unitMetaData.InputShapeVector())
    {
        forwardInputVector.emplace_back(CreateTensor(
            shape, unitMetaData.NumericType, unitMetaData.PadSize));
        backwardOutputVector.emplace_back(CreateTensor(
            shape, unitMetaData.NumericType, unitMetaData.PadSize));
    }

    for (std::size_t i = 0; i < unitMetaData.OutputUnitVector().size(); ++i)
        backwardInputVector.emplace_back(
            CreateTensor(unitMetaData.OutputShape(), unitMetaData.NumericType,
                         unitMetaData.PadSize));

    m_unitMetaDataMap[unitMetaData.Id().Id] = std::move(unitMetaData);
    // TODO : Create appropriate Unit by examining UnitID
}


void UnitManager::Forward(std::size_t cycle)
{
    for (const auto& [key, unitPtr] : m_unitMap)
    {
        if (unitPtr->IsForwardReady(cycle))
            unitPtr->Forward(cycle);
        m_forwardCopy(key);
    }
}

void UnitManager::Backward(std::size_t cycle)
{
    for (const auto& [key, unitPtr] : m_unitMap)
    {
        if (unitPtr->IsBackwardReady(cycle))
            unitPtr->Backward(cycle);
        m_backwardCopy(key);
    }
}

void UnitManager::AsyncForward(std::size_t cycle)
{
    std::unordered_map<int, std::future<bool>> futureVector;
    futureVector.reserve(10);

    for (const auto& [key, unitPtr] : m_unitMap)
    {
        if (unitPtr->IsForwardReady(cycle))
        {
            std::promise<bool> promise;
            futureVector[key] = promise.get_future();
            unitPtr->AsyncForward(cycle, std::move(promise));
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
    std::unordered_map<int, std::future<bool>> futureVector;
    futureVector.reserve(10);

    for (const auto& [key, unitPtr] : m_unitMap)
    {
        if (unitPtr->IsBackwardReady(cycle))
        {
            std::promise<bool> promise;
            futureVector[key] = promise.get_future();
            unitPtr->AsyncBackward(cycle, std::move(promise));
        }
    }

    for (auto& [key, future] : futureVector)
    {
        future.wait();
        m_backwardCopy(key);
    }
}

void UnitManager::m_forwardCopy(int sourceKey)
{
    const auto& sourceMetaData = m_unitMetaDataMap[sourceKey];
    for (const auto& unitId : sourceMetaData.OutputUnitVector())
    {
        auto& outputTensor = m_unitMap[unitId.Id]->ForwardOutput;
        auto& nextInputTensorVector = m_unitMap[unitId.Id]->ForwardInputVector;
        for (auto& destTensor : nextInputTensorVector)
        {
            Tensor::CopyTensor(outputTensor, destTensor);
            outputTensor.ForwardStateNum += 1;
            destTensor.ForwardStateNum += 1;
        }
    }
}

void UnitManager::m_backwardCopy(int sourceKey)
{
    const auto& sourceMetaData = m_unitMetaDataMap[sourceKey];
    for (const auto& unitId : sourceMetaData.OutputUnitVector())
    {
        auto& outputTensor = m_unitMap[unitId.Id]->BackwardOutput;
        auto& nextInputTensorVector = m_unitMap[unitId.Id]->BackwardInputVector;
        for (auto& destTensor : nextInputTensorVector)
        {
            Tensor::CopyTensor(outputTensor, destTensor);
            outputTensor.BackwardStateNum += 1;
            destTensor.BackwardStateNum += 1;
        }
    }
}
}
