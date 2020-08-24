// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_GRAPH_COMPUTABLEUNIT_HPP
#define TAKION_GRAPH_COMPUTABLEUNIT_HPP

#include <Takion/Units/ComputableUnitDecl.hpp>

namespace Takion::Graph
{
template <typename T>
ComputableUnit<T>::ComputableUnit(
    UnitId subjectUnitId,
    std::unordered_map<UnitId, Tensor<T>> forwardInputMap,
    std::unordered_map<UnitId, Tensor<T>> backwardInputMap,
    Tensor<T> forwardOutput,
    std::unordered_map<UnitId, Tensor<T>> backwardOutputMap,
    std::unordered_map<std::string, Tensor<T>> internalTensorMap,
    std::size_t batchSize)
    : ForwardInputMap(std::move(forwardInputMap)),
      BackwardInputMap(std::move(backwardInputMap)),
      ForwardOutput(forwardOutput),
      BackwardOutputMap(std::move(backwardOutputMap)),
      InternalTensorMap(std::move(internalTensorMap)),
      BatchSize(batchSize),
      m_unitId(std::move(subjectUnitId))
{
}

template <typename T>
ComputableUnit<T>::ComputableUnit(ComputableUnit<T>&& computableUnit) noexcept
    : ForwardInputMap(std::move(computableUnit.ForwardInputMap)),
      BackwardInputMap(std::move(computableUnit.BackwardInputMap)),
      ForwardOutput(computableUnit.ForwardOutput),
      BackwardOutputMap(std::move(computableUnit.BackwardOutputMap)),
      InternalTensorMap(std::move(computableUnit.InternalTensorMap)),
      BatchSize(computableUnit.BatchSize),
      m_unitId(std::move(computableUnit.m_unitId))
{
}

template <typename T>
ComputableUnit<T>& ComputableUnit<T>::operator=(
    ComputableUnit<T>&& computableUnit)
noexcept
{
    ForwardInputMap = std::move(computableUnit.ForwardInputMap);
    BackwardInputMap = std::move(computableUnit.BackwardInputMap);
    ForwardOutput = std::move(computableUnit.ForwardOutput);
    BackwardOutputMap = std::move(computableUnit.BackwardOutputMap);
    InternalTensorMap = std::move(computableUnit.InternalTensorMap);
    BatchSize = computableUnit.BatchSize;
    m_unitId = std::move(computableUnit.m_unitId);
    return *this;
}

template <typename T>
bool ComputableUnit<T>::IsForwardReady(std::size_t cycle) const
{
    for (const auto& [unitId, tensor] : ForwardInputMap)
    {
        if (tensor.State != cycle + 1)
            return false;
    }

    if (ForwardOutput.State != cycle)
        return false;
    return true;
}

template <typename T>
bool ComputableUnit<T>::IsBackwardReady(std::size_t cycle) const
{
    if (BackwardOutputMap.empty())
        return false;

    for (const auto& [unitId, tensor] : BackwardInputMap)
    {
        if (tensor.State != cycle + 1)
            return false;
    }

    for (const auto& [unitId, tensor] : BackwardOutputMap)
    {
        if (tensor.State != cycle)
            return false;
    }
    return true;
}

template <typename T>
void ComputableUnit<T>::UpdateForwardState()
{
    m_unitState.ForwardStateCount.fetch_add(1);
    ForwardOutput.State.fetch_add(1);
}

template <typename T>
void ComputableUnit<T>::UpdateBackwardState()
{
    m_unitState.BackwardStateCount.fetch_add(1);
    for (auto& [unitId, tensor] : BackwardOutputMap)
        tensor.State.fetch_add(1);
}

template <typename T>
void ComputableUnit<T>::ResetState()
{
    for (auto& [unitId, tensor] : ForwardInputMap)
        tensor.State = 0;
    for (auto& [unitId, tensor] : BackwardInputMap)
        tensor.State = 0;
    ForwardOutput.State = 0;
    for (auto& [unitId, tensor] : BackwardOutputMap)
        tensor.State = 0;

    m_unitState.BackwardStateCount = 0;
    m_unitState.ForwardStateCount = 0;
}

template <typename T>
void ComputableUnit<T>::ChangeBatchSize(std::size_t batchSize)
{
    for (auto& [unitId, tensor] : ForwardInputMap)
        tensor.ChangeBatchSize(batchSize);
    for (auto& [unitId, tensor] : BackwardInputMap)
        tensor.ChangeBatchSize(batchSize);
    ForwardOutput.ChangeBatchSize(batchSize);
    for (auto& [unitId, tensor] : BackwardOutputMap)
        tensor.ChangeBatchSize(batchSize);
    BatchSize = batchSize;
}
} // namespace Takion::Graph

#endif
