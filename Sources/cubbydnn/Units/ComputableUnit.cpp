// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Units/ComputableUnit.hpp>

namespace CubbyDNN
{
UnitState::UnitState() : StateNum(0), IsBusy(false)
{
}

ComputableUnit::ComputableUnit(size_t inputSize, size_t outputSize)
    : m_inputPtrVector(inputSize), m_outputPtrVector(outputSize)
{
    m_inputPtrVector.reserve(inputSize);
    m_outputPtrVector.reserve(outputSize);
}

ComputableUnit::ComputableUnit(ComputableUnit&& computableUnit) noexcept
    : m_inputPtrVector(std::move(computableUnit.m_inputPtrVector)),
      m_outputPtrVector(std::move(computableUnit.m_outputPtrVector)),
      m_logVector(std::move(computableUnit.m_logVector))
{
}

std::atomic<std::size_t>& ComputableUnit::GetStateNum()
{
    return m_unitState.StateNum;
}
}  // namespace CubbyDNN