// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Units/ComputableUnit.hpp>

namespace CubbyDNN
{
UnitState::UnitState() = default;

ComputableUnit::ComputableUnit(std::vector<TensorInfo> inputTensorInfoVector,
                               std::vector<TensorInfo> outputTensorInfoVector,
                               UnitType unitType)
    : Type(unitType),
      m_inputTensorInfoVector(std::move(inputTensorInfoVector)),
      m_outputTensorInfoVector(std::move(outputTensorInfoVector))
{
}

size_t ComputableUnit::AddOutputPtr(
    const SharedPtr<ComputableUnit>& computableUnitPtr)
{
    assert(m_outputVectorIndex < m_outputPtrVector.size() &&
           "Number of outputs exceeds number given from decleration");

    m_outputPtrVector.at(m_outputVectorIndex) = computableUnitPtr;
    return m_outputVectorIndex++;
}

void ComputableUnit::AddInputPtr(
    const SharedPtr<ComputableUnit>& computableUnitPtr, size_t index)
{
    assert(index < m_inputPtrVector.size() &&
           "Number of inputs exceeds number given from decleration");
    m_inputPtrVector.at(index) = computableUnitPtr;
}
}  // namespace CubbyDNN
