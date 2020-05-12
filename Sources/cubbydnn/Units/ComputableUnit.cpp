// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Units/ComputableUnit.hpp>

namespace CubbyDNN::Graph
{
ComputableUnit::ComputableUnit(UnitId unitId, NumberSystem numberSystem,
                               std::vector<Tensor> forwardInputVector,
                               std::vector<Tensor> backwardInputVector,
                               Tensor forwardOutput, Tensor backwardOutput)
    : ForwardInputVector(std::move(forwardInputVector)),
      BackwardInputVector(std::move(backwardInputVector)),
      ForwardOutput(std::move(forwardOutput)),
      BackwardOutput(std::move(backwardOutput)),
      m_unitId(std::move(unitId)),
      m_numberSystem(numberSystem)
{
}

ComputableUnit::ComputableUnit(ComputableUnit&& computableUnit) noexcept
    : ForwardInputVector(std::move(computableUnit.ForwardInputVector)),
      BackwardInputVector(std::move(computableUnit.BackwardInputVector)),
      ForwardOutput(std::move(computableUnit.ForwardOutput)),
      BackwardOutput(std::move(computableUnit.BackwardOutput)),
      m_unitId(std::move(computableUnit.m_unitId)),
      m_numberSystem(computableUnit.m_numberSystem)
{
}

ComputableUnit& ComputableUnit::operator=(ComputableUnit&& computableUnit)
{
    ForwardInputVector = std::move(computableUnit.ForwardInputVector);
    BackwardInputVector = std::move(computableUnit.BackwardInputVector);
    ForwardOutput = std::move(computableUnit.ForwardOutput);
    BackwardOutput = std::move(computableUnit.BackwardOutput);
    m_unitId = std::move(computableUnit.m_unitId);
    m_numberSystem = computableUnit.m_numberSystem;
    return *this;
}

bool ComputableUnit::IsForwardReady(std::size_t cycle) const
{
    for (const auto& tensor : ForwardInputVector)
    {
        if (tensor.ForwardStateNum != cycle)
            return false;
    }

    if (ForwardOutput.ForwardStateNum != cycle)
        return false;
    return true;
}

bool ComputableUnit::IsBackwardReady(std::size_t cycle) const
{
    for (const auto& tensor : BackwardInputVector)
    {
        if (tensor.BackwardStateNum != cycle)
            return false;
    }

    if (BackwardOutput.BackwardStateNum != cycle)
        return false;
    return true;
}


void ComputableUnit::m_updateForwardState()
{
    m_unitState.ForwardStateCount.fetch_add(1);
}

void ComputableUnit::m_updateBackwardState()
{
    m_unitState.BackwardStateCount.fetch_add(1);
}
} // namespace CubbyDNN
