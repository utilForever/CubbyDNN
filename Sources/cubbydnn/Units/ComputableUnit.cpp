// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Units/ComputableUnit.hpp>

namespace CubbyDNN
{
UnitState::UnitState() = default;

ComputableUnit::ComputableUnit(size_t inputSize, size_t outputSize,
                               UnitType unitType)
    : Type(unitType), m_inputPtrVector(inputSize), m_outputPtrVector(outputSize)
{
    m_inputPtrVector.reserve(inputSize);
    m_outputPtrVector.reserve(outputSize);
}

ComputableUnit::ComputableUnit(ComputableUnit&& computableUnit) noexcept
    : Type(computableUnit.Type),
      m_inputPtrVector(std::move(computableUnit.m_inputPtrVector)),
      m_outputPtrVector(std::move(computableUnit.m_outputPtrVector)),
      m_logVector(std::move(computableUnit.m_logVector)),
      m_outputVectorIndex(computableUnit.m_outputVectorIndex)
{
}

size_t ComputableUnit::AddOutputPtr(const SharedPtr<ComputableUnit>& computableUnitPtr)
{
    assert(m_outputVectorIndex < m_outputPtrVector.size() && "Number of outputs exceeds number given from decleration");

    m_outputPtrVector.at(m_outputVectorIndex) = computableUnitPtr;
    return m_outputVectorIndex++;
}

void ComputableUnit::AddInputPtr(const SharedPtr<ComputableUnit>& computableUnitPtr, size_t index)
{
    assert(index< m_inputPtrVector.size() &&
           "Number of inputs exceeds number given from decleration");
    m_inputPtrVector.at(index) = computableUnitPtr;
}


}  // namespace CubbyDNN
