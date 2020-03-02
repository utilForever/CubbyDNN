// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Units/ComputableUnit.hpp>

namespace CubbyDNN
{
UnitState::UnitState() = default;

ComputableUnit::ComputableUnit(UnitType unitType)
    : Type(unitType)
{
}

ComputableUnit::ComputableUnit(ComputableUnit&& other) noexcept
    : Type(other.Type),
      m_inputPtrVector(std::move(other.m_inputPtrVector)),
      m_outputPtrVector(std::move(other.m_outputPtrVector)),
      m_logVector(std::move(other.m_logVector)),
      m_inputTensorInfoVector(std::move(other.m_inputTensorInfoVector)),
      m_outputTensorInfo(other.m_outputTensorInfo),
      m_inputTensorVector(std::move(other.m_inputTensorVector)),
      m_outputTensorVector(std::move(other.m_outputTensorVector)),
      m_tensor(std::move(other.m_tensor)),
      m_outputVectorIndex(other.m_outputVectorIndex)
{
}

ComputableUnit& ComputableUnit::operator=(ComputableUnit&& other) noexcept
{
    if (this == &other)
        return *this;
    Type = other.Type;
    m_inputPtrVector = std::move(other.m_inputPtrVector);
    m_outputPtrVector = std::move(other.m_outputPtrVector);
    m_logVector = std::move(other.m_logVector);
    m_inputTensorInfoVector = std::move(other.m_inputTensorInfoVector);
    m_outputTensorInfo = other.m_outputTensorInfo;
    m_inputTensorVector = std::move(other.m_inputTensorVector);
    m_outputTensorVector = std::move(other.m_outputTensorVector);
    m_tensor = std::move(other.m_tensor);
    m_outputVectorIndex = other.m_outputVectorIndex;
    return *this;
}

ComputableUnit::ComputableUnit(std::vector<TensorInfo> inputTensorInfoVector,
                               TensorInfo outputTensorInfo,
                               UnitType unitType)
    : Type(unitType),
      m_inputTensorInfoVector(std::move(inputTensorInfoVector)),
      m_outputTensorInfo(outputTensorInfo)
{
}

std::size_t ComputableUnit::AddOutputPtr(
    const SharedPtr<ComputableUnit>& computableUnitPtr)
{
    assert(m_outputVectorIndex < m_outputPtrVector.size() &&
        "Number of outputs exceeds number given from decleration");

    m_outputPtrVector.at(m_outputVectorIndex) = computableUnitPtr;
    return m_outputVectorIndex++;
}

void ComputableUnit::AddInputPtr(
    const SharedPtr<ComputableUnit>& computableUnitPtr, std::size_t index)
{
    assert(index < m_inputPtrVector.size() &&
        "Number of inputs exceeds number given from decleration");
    m_inputPtrVector.at(index) = computableUnitPtr;
}

void ComputableUnit::ReleaseUnit()
{
    m_incrementStateNum();
}
} // namespace CubbyDNN
