// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Units/ComputableUnit.hpp>

namespace CubbyDNN
{
UnitState::UnitState() = default;

ComputableUnit::ComputableUnit(UnitId unitId,
                               std::vector<Shape> inputShapeVector,
                               Shape outputShape, NumberSystem numberSystem)
    : m_id(unitId),
      m_inputShapeVector(std::move(inputShapeVector)),
      m_outputTensorShape(std::move(outputShape)),
      m_numberSystem(numberSystem)
{
    m_inputForwardTensorVector.reserve(m_inputShapeVector.size());
    m_outputBackwardTensorVector.reserve(m_inputShapeVector.size());

    for (const auto& shape : m_inputShapeVector)
    {
        m_inputForwardTensorVector.emplace_back(
            CreateTensor(shape, numberSystem));
        m_outputBackwardTensorVector.emplace_back(
            CreateTensor(shape, numberSystem));
    }
    m_outputForwardTensor = CreateTensor(outputShape, numberSystem);
    m_inputBackwardTensor = CreateTensor(outputShape, numberSystem);
}

ComputableUnit::ComputableUnit(ComputableUnit&& computableUnit) noexcept
    : m_id(computableUnit.m_id),
      m_inputShapeVector(std::move(computableUnit.m_inputShapeVector)),
      m_outputTensorShape(std::move(computableUnit.m_outputTensorShape)),
      m_numberSystem(computableUnit.m_numberSystem),
      m_inputUnitIdVector(computableUnit.m_inputUnitIdVector),
      m_outputUnitIdVector(std::move(computableUnit.m_outputUnitIdVector)),
      m_outputForwardTensor(std::move(computableUnit.m_outputForwardTensor)),
      m_inputBackwardTensor(std::move(m_inputBackwardTensor)),
      m_inputForwardTensorVector(std::move(m_inputForwardTensorVector)),
      m_outputBackwardTensorVector(std::move(m_outputBackwardTensorVector))
{
}

ComputableUnit& ComputableUnit::operator=(
    ComputableUnit&& computableUnit) noexcept
{
    if (this == &computableUnit)
        return *this;
    m_id = computableUnit.m_id;
    m_inputUnitIdVector = std::move(computableUnit.m_inputUnitIdVector);
    m_outputUnitIdVector = std::move(computableUnit.m_outputUnitIdVector);
    m_inputShapeVector = std::move(computableUnit.m_inputShapeVector);
    m_outputTensorShape = std::move(computableUnit.m_outputTensorShape);
    m_outputForwardTensor = std::move(computableUnit.m_outputForwardTensor);
    m_inputBackwardTensor = std::move(computableUnit.m_inputBackwardTensor);
    m_inputForwardTensorVector =
        std::move(computableUnit.m_inputForwardTensorVector);
    m_outputBackwardTensorVector = std::move(m_outputBackwardTensorVector);
    m_numberSystem = std::move(computableUnit.m_numberSystem);
    return *this;
}

void ComputableUnit::ReleaseUnit()
{
    m_unitState.StateNum.fetch_add(1, std::memory_order_release);
}
} // namespace CubbyDNN
