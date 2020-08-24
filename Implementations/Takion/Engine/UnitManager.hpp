// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_GRAPH_UNITMANAGER_HPP
#define TAKION_GRAPH_UNITMANAGER_HPP

#include <Takion/Engine/UnitManagerDecl.hpp>
#include <Takion/Units/HiddenUnits/Dense.hpp>
#include <Takion/Units/SourceUnits/ConstantUnit.hpp>
#include <Takion/Units/SourceUnits/PlaceHolder.hpp>
#include <Takion/Units/HiddenUnits/Activations/ReLU.hpp>
#include <Takion/Units/HiddenUnits/Activations/Sigmoid.hpp>
#include <Takion/Units/HiddenUnits/Activations/SoftMax.hpp>
#include <Takion/Units/SinkUnits/MSE.hpp>
#include <Takion/Units/SinkUnits/CrossEntropy.hpp>


namespace Takion::Engine
{
template <typename T>
UnitManager<T>::UnitManager(UnitManager<T>&& unitManager) noexcept
    : m_unitMetaDataMap(std::move(unitManager.m_unitMetaDataMap)),
      m_unitMap(std::move(unitManager.m_unitMap)),
      m_batchSize(unitManager.m_batchSize)
{
}

template <typename T>
UnitManager<T>& UnitManager<T>::operator=(UnitManager<T>&& unitManager) noexcept
{
    m_unitMetaDataMap = std::move(unitManager.m_unitMetaDataMap);
    m_unitMap = std::move(unitManager.m_unitMap);
    return *this;
}

template <typename T>
FrontEnd::UnitMetaData<T>& UnitManager<T>::GetUnitMetaData(const UnitId& unitId)
{
    return m_unitMetaDataMap[unitId];
}

template <typename T>
void UnitManager<T>::AppendUnit(FrontEnd::UnitMetaData<T>&& unitMetaData)
{
    const auto unitId = unitMetaData.Id();
    m_unitMetaDataMap[unitId] = std::move(unitMetaData);
}

template <typename T>
void UnitManager<T>::SetLoader(const UnitId& unitId,
                               std::unique_ptr<Util::Loader<T>> loader)
{
    m_loaderMap[unitId] = std::move(loader);
}


template <typename T>
Shape UnitManager<T>::GetUnitOutputShape(const UnitId& unitId)
{
    return m_unitMetaDataMap[unitId].GetOutputShape();
}

template <typename T>
void UnitManager<T>::Compile(const std::string& optimizerName,
                             const Parameter& parameter)
{
    for (const auto& [key, unitMetaData] : m_unitMetaDataMap)
    {
        if (m_appendSource(unitMetaData))
            continue;
        if (m_appendHidden(unitMetaData, optimizerName, parameter))
            continue;
        if (m_appendLoss(unitMetaData))
            continue;
        throw std::runtime_error("No matching unit type");
    }
}

template <typename T>
void UnitManager<T>::Forward()
{
    for (const auto& [key, unitPtr] : m_unitMap)
        if (key.Type.BaseType == UnitBaseType::Fetcher ||
            key.Type.BaseType == UnitBaseType::Constant)
        {
            for (auto& [unitId, tensor] : unitPtr->ForwardInputMap)
                tensor.State.fetch_add(1);
        }

    bool done = false;
    while (!done)
    {
        done = true;
        for (const auto& [key, unitPtr] : m_unitMap)
        {
            if (unitPtr->IsForwardReady(0))
            {
                unitPtr->Forward();
                unitPtr->UpdateForwardState();
                done = false;
            }
            if (m_isForwardCopyReady(key))
            {
                m_forwardCopy(key);
                done = false;
            }
        }
    }
}

template <typename T>
void UnitManager<T>::Backward()
{
    for (const auto& [key, unitPtr] : m_unitMap)
        if (key.Type.BaseType == UnitBaseType::Loss)
            for (auto& [unitId, tensor] : unitPtr->BackwardInputMap)
                tensor.State.fetch_add(1);

    bool done = false;
    while (!done)
    {
        done = true;
        for (const auto& [key, unitPtr] : m_unitMap)
        {
            if (unitPtr->IsBackwardReady(0))
            {
                unitPtr->Backward();
                unitPtr->UpdateBackwardState();
                done = false;
            }
            if (m_isBackwardCopyReady(key))
            {
                m_backwardCopy(key);
                done = false;
            }
        }
    }
}

template <typename T>
void UnitManager<T>::AsyncForward(std::size_t cycle)
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

template <typename T>
void UnitManager<T>::AsyncBackward(std::size_t cycle)
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

template <typename T>
void UnitManager<T>::ResetState()
{
    for (const auto& [key, unitPtr] : m_unitMap)
        unitPtr->ResetState();
}

template <typename T>
void UnitManager<T>::ChangeBatchSize(std::size_t batchSize)
{
    for (const auto& [key, unitPtr] : m_unitMap)
        unitPtr->ChangeBatchSize(batchSize);

    m_batchSize = batchSize;
}


template <typename T>
const Tensor<T>& UnitManager<T>::GetOutput(UnitId unitId) const
{
    return m_unitMap.at(unitId)->ForwardOutput;
}

template <typename T>
std::unique_ptr<Graph::ComputableUnit<T>>& UnitManager<T>::GetUnit(
    const UnitId& unitId)
{
    return m_unitMap[unitId];
}

template <typename T>
bool UnitManager<T>::m_isForwardCopyReady(const UnitId& subjectUnitId) const
{
    const auto& sourceMetaData = m_unitMetaDataMap.at(subjectUnitId);
    if (sourceMetaData.Id().Type.BaseType == UnitBaseType::Loss)
        return false;

    const auto& subjectOutputTensor =
        m_unitMap.at(subjectUnitId)->ForwardOutput;

    for (const auto& outputUnitId : sourceMetaData.OutputUnitVector())
    {
        const auto& nextInputTensorMap =
            m_unitMap.at(outputUnitId)->ForwardInputMap;
        for (const auto& [targetUnitId, destTensor] : nextInputTensorMap)
        {
            if (targetUnitId == subjectUnitId)
            {
                if (subjectOutputTensor.State != destTensor.State + 1)
                    return false;
            }
        }
    }
    return true;
}

template <typename T>
bool UnitManager<T>::m_isBackwardCopyReady(const UnitId& subjectUnitId) const
{
    const auto& sourceMetaData = m_unitMetaDataMap.at(subjectUnitId);
    if (sourceMetaData.Id().Type.BaseType == UnitBaseType::Fetcher)
        return false;

    bool hasValidBackwardUnit = false;
    for (const auto& [unitId, outputTensor] :
         m_unitMap.at(subjectUnitId)->BackwardOutputMap)
    {
        const auto& nextBackwardInputTensorMap =
            m_unitMap.at(unitId)->BackwardInputMap;

        for (const auto& [targetUnitId, destTensor] :
             nextBackwardInputTensorMap)
        {
            if (targetUnitId == subjectUnitId)
            {
                hasValidBackwardUnit = true;
                if (outputTensor.State != destTensor.State + 1)
                    return false;
            }
        }
    }
    return hasValidBackwardUnit;
}

template <typename T>
void UnitManager<T>::m_forwardCopy(const UnitId& subjectUnitId)
{
    const auto& sourceMetaData = m_unitMetaDataMap[subjectUnitId];
    auto& subjectOutputTensor = m_unitMap[subjectUnitId]->ForwardOutput;

    for (const auto& outputUnitId : sourceMetaData.OutputUnitVector())
    {
        auto& nextInputTensorMap = m_unitMap[outputUnitId]->ForwardInputMap;
        for (auto& [targetUnitId, destTensor] : nextInputTensorMap)
        {
            if (targetUnitId == subjectUnitId)
            {
                Tensor<T>::CopyTensorData(subjectOutputTensor, destTensor);
                destTensor.State.fetch_add(1);
            }
        }
    }
}

template <typename T>
void UnitManager<T>::m_backwardCopy(const UnitId& subjectUnitId)
{
    for (const auto& [unitId, outputTensor] :
         m_unitMap.at(subjectUnitId)->BackwardOutputMap)
    {
        auto& nextBackwardInputTensorMap =
            m_unitMap.at(unitId)->BackwardInputMap;

        for (auto& [targetUnitId, destTensor] : nextBackwardInputTensorMap)
        {
            if (targetUnitId == subjectUnitId)
            {
                Tensor<T>::CopyTensorData(outputTensor, destTensor);
                destTensor.State.fetch_add(1);
            }
        }
    }
}

template <typename T>
std::unique_ptr<Compute::Optimizer<T>> UnitManager<T>::m_makeOptimizer(
    const std::string& optimizerName, const Parameter& parameter) const
{
    if (optimizerName == "SGD")
    {
        auto optimizer = std::make_unique<Compute::SGD<T>>(
            parameter.GetFloatingPointParam("LearningRate"));
        return std::move(optimizer);
    }
    throw std::runtime_error("Unsupported optimizer type");
}

template <typename T>
bool UnitManager<T>::m_appendSource(
    const FrontEnd::UnitMetaData<T>& unitMetaData)
{
    const auto unitId = unitMetaData.Id();
    auto type = unitId.Type;

    if (type.Name() == "Fetcher")
    {
        auto unit = Graph::PlaceHolder<T>::CreateUnit(unitMetaData,
                                                      std::move(
                                                          m_loaderMap[unitId]));
        m_unitMap[unitId] =
            std::make_unique<Graph::PlaceHolder<T>>(std::move(unit));
        return true;
    }
    if (type.Name() == "Constant")
    {
        auto unit = Graph::ConstantUnit<T>::CreateUnit(unitMetaData);
        m_unitMap[unitId] =
            std::make_unique<Graph::ConstantUnit<T>>(std::move(unit));
        return true;
    }
    return false;
}

template <typename T>
bool UnitManager<T>::m_appendHidden(
    const FrontEnd::UnitMetaData<T>& unitMetaData,
    const std::string& optimizerName, const Parameter& parameter)
{
    const auto unitId = unitMetaData.Id();
    auto type = unitId.Type;

    if (type.Name() == "Dense")
    {
        auto unit = Graph::DenseUnit<T>::CreateUnit(
            unitMetaData, m_makeOptimizer(optimizerName, parameter));

        m_unitMap[unitId] =
            std::make_unique<Graph::DenseUnit<T>>(std::move(unit));
        return true;
    }
    if (type.Name() == "ReLU")
    {
        auto unit = Graph::ReLU<T>::CreateUnit(unitMetaData);
        m_unitMap[unitId] = std::make_unique<Graph::ReLU<T>>(std::move(unit));
        return true;
    }
    if (type.Name() == "Sigmoid")
    {
        auto unit = Graph::Sigmoid<T>::CreateUnit(unitMetaData);
        m_unitMap[unitId] =
            std::make_unique<Graph::Sigmoid<T>>(std::move(unit));
        return true;
    }
    if (type.Name() == "SoftMax")
    {
        auto unit = Graph::SoftMax<T>::CreateUnit(unitMetaData);
        m_unitMap[unitId] =
            std::make_unique<Graph::SoftMax<T>>(std::move(unit));
        return true;
    }
    if (type.Name() == "Dropout")
    {
        throw std::runtime_error("Not implemented");
    }
    if (type.Name() == "Reshape")
    {
        throw std::runtime_error("Not implemented");
    }
    if (type.Name() == "Multiply")
    {
        throw std::runtime_error("Not implemented");
    }
    if (type.Name() == "Add")
    {
        throw std::runtime_error("Not implemented");
    }
    return false;
}

template <typename T>
bool UnitManager<T>::m_appendLoss(const FrontEnd::UnitMetaData<T>& unitMetaData)
{
    const auto unitId = unitMetaData.Id();
    auto type = unitId.Type;

    if (type.Name() == "CrossEntropy")
    {
        auto unit = Graph::CrossEntropy<T>::CreateUnit(unitMetaData);
        m_unitMap[unitId] =
            std::make_unique<Graph::CrossEntropy<T>>(std::move(unit));
        return true;
    }
    if (type.Name() == "MSE")
    {
        auto unit = Graph::MSELoss<T>::CreateUnit(unitMetaData);
        m_unitMap[unitId] =
            std::make_unique<Graph::MSELoss<T>>(std::move(unit));
        return true;
    }
    return false;
}
} // namespace Takion::Graph

#endif
