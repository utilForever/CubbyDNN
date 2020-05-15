// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.


#include <cubbydnn/Engine/UnitManager.hpp>
#include <cubbydnn/Units/HiddenComputableUnits/Dense.hpp>
#include <cubbydnn/Units/HiddenComputableUnits/ActivationUnit.hpp>

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

template <typename... Ts>
void UnitManager::AppendUnit(const UnitMetaData& unitMetaData, Ts ... type)
{
    const auto unitId = unitMetaData.Id();
    if (unitId.Type.Name() == "Dense")
    {
        m_unitMap[unitId.Id] =
            std::make_unique<DenseUnit>(DenseUnit::CreateUnit(unitMetaData, type...));
    }
    if (unitId.Type.Name() == "Activation")
    {
        m_unitMap[unitId.Id] = std::make_unique<ActivationUnit>(
            ActivationUnit::CreateUnit(unitMetaData, type...));
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
    std::unordered_map<int, std::future<bool>> futureVector;
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
    std::unordered_map<int, std::future<bool>> futureVector;
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

void UnitManager::m_forwardCopy(int subjectUnitKey)
{
    const auto& sourceMetaData = m_unitMetaDataMap[subjectUnitKey];
    for (const auto& unitId : sourceMetaData.OutputUnitVector())
    {
        auto& outputTensor = m_unitMap[subjectUnitKey]->ForwardOutput;
        auto& nextInputTensorVector = m_unitMap[unitId.Id]->ForwardInputVector;
        for (auto& destTensor : nextInputTensorVector)
        {
            Tensor::CopyTensor(outputTensor, destTensor);
            outputTensor.ForwardStateNum.fetch_add(1);
            destTensor.ForwardStateNum.fetch_add(1);
        }
    }
}

void UnitManager::m_backwardCopy(int subjectUnitKey)
{
    const auto& sourceMetaData = m_unitMetaDataMap[subjectUnitKey];
    int index = 0;
    for (const auto& unitId : sourceMetaData.InputUnitVector())
    {
        auto& outputTensor = m_unitMap[subjectUnitKey]->BackwardOutputVector[
            index];
        auto& nextBackwardInputTensorVector = m_unitMap[unitId.Id]->
            BackwardInputVector;
        auto nextBackwardInputUnitVector =
            m_unitMetaDataMap[unitId.Id].OutputUnitVector();

        for (std::size_t i = 0; i < nextBackwardInputTensorVector.size(); ++i)
        {
            auto targetUnitId = nextBackwardInputUnitVector.at(i);
            if (targetUnitId == sourceMetaData.Id())
            {
                auto& destTensor =
                    nextBackwardInputTensorVector.at(i);
                Tensor::CopyTensor(outputTensor, destTensor);
                outputTensor.BackwardStateNum.fetch_add(1);
                destTensor.BackwardStateNum.fetch_add(1);
            }
        }
        index += 1;
    }
}
}
