// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Units/ComputableUnit.hpp>

namespace CubbyDNN::Graph
{
ComputableUnit::ComputableUnit(
    UnitId subjectUnitId, NumberSystem numberSystem,
    std::unordered_map<UnitId, Tensor> forwardInputMap,
    std::unordered_map<UnitId, Tensor> backwardInputMap, Tensor forwardOutput,
    std::unordered_map<UnitId, Tensor> backwardOutputMap)
    : ForwardInputMap(std::move(forwardInputMap)),
      BackwardInputMap(std::move(backwardInputMap)),
      ForwardOutput(std::move(forwardOutput)),
      BackwardOutputMap(std::move(backwardOutputMap)),
      m_unitId(std::move(subjectUnitId)),
      m_numericType(numberSystem)
{
    for (const auto& [unitId, tensor] : ForwardInputMap)
    {
        if (tensor.NumericType != m_numericType)
            throw std::invalid_argument("Number system mismatch");
    }

    for (const auto& [unitId, tensor] : BackwardOutputMap)
    {
        if (tensor.NumericType != m_numericType)
            throw std::invalid_argument("Number system mismatch");
    }

    for (const auto& [unitId, tensor] : BackwardInputMap)
    {
        if (tensor.NumericType != m_numericType)
            throw std::invalid_argument("Number system mismatch");
    }

    if (ForwardOutput.NumericType != m_numericType)
        throw std::invalid_argument("Number system mismatch");
}

ComputableUnit::ComputableUnit(ComputableUnit&& computableUnit) noexcept
    : ForwardInputMap(std::move(computableUnit.ForwardInputMap)),
      BackwardInputMap(std::move(computableUnit.BackwardInputMap)),
      ForwardOutput(std::move(computableUnit.ForwardOutput)),
      BackwardOutputMap(std::move(computableUnit.BackwardOutputMap)),
      m_unitId(std::move(computableUnit.m_unitId)),
      m_numericType(computableUnit.m_numericType)
{
}

ComputableUnit& ComputableUnit::operator=(
    ComputableUnit&& computableUnit) noexcept
{
    ForwardInputMap = std::move(computableUnit.ForwardInputMap);
    BackwardInputMap = std::move(computableUnit.BackwardInputMap);
    ForwardOutput = std::move(computableUnit.ForwardOutput);
    BackwardOutputMap = std::move(computableUnit.BackwardOutputMap);
    m_unitId = std::move(computableUnit.m_unitId);
    m_numericType = computableUnit.m_numericType;
    return *this;
}

bool ComputableUnit::IsForwardReady(std::size_t cycle) const
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

bool ComputableUnit::IsBackwardReady(std::size_t cycle) const
{
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


void ComputableUnit::UpdateForwardState()
{
    m_unitState.ForwardStateCount.fetch_add(1);
    ForwardOutput.State.fetch_add(1);
}

void ComputableUnit::UpdateBackwardState()
{
    m_unitState.BackwardStateCount.fetch_add(1);
    for (auto& [unitId, tensor] : BackwardOutputMap)
        tensor.State.fetch_add(1);
}
} // namespace CubbyDNN
