// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Units/UnitMetadata.hpp>

namespace CubbyDNN::Graph
{
UnitMetaData::UnitMetaData(UnitId unitId,
                           std::unordered_map<std::string, Shape>
                           internalVariableShapeMap,
                           std::unordered_map<
                               std::string, std::unique_ptr<Initializer>>
                           initializerVector,
                           std::vector<Shape> inputShapeVector,
                           Shape outputShape,
                           std::vector<UnitId> inputUnitIdVector,
                           NumberSystem numericType,
                           Compute::Device device)
    : NumericType(numericType),
      Device(std::move(device)),
      m_unitId(std::move(unitId)),
      m_internalVariableShapeMap(std::move(internalVariableShapeMap)),
      m_initializerVector(std::move(initializerVector)),
      m_inputShapeVector(std::move(inputShapeVector)),
      m_outputShape(std::move(outputShape)),
      m_inputUnitIdVector(std::move(inputUnitIdVector))
{
}


UnitMetaData::UnitMetaData(
    UnitId unitId,
    std::unordered_map<std::string, Shape> internalVariableShapeMap,
    std::unordered_map<std::string, std::unique_ptr<Initializer>>
    initializerVector,
    std::vector<Shape> inputShapeVector, Shape outputShape,
    std::vector<UnitId> inputUnitIdVector, ParameterPack params,
    NumberSystem numericType, Compute::Device device)
    : Parameters(std::move(params)),
      NumericType(numericType),
      Device(std::move(device)),
      m_unitId(std::move(unitId)),
      m_internalVariableShapeMap(std::move(internalVariableShapeMap)),
      m_initializerVector(std::move(initializerVector)),
      m_inputShapeVector(std::move(inputShapeVector)),
      m_outputShape(std::move(outputShape)),
      m_inputUnitIdVector(std::move(inputUnitIdVector))
{
}

void UnitMetaData::AppendOutputUnitId(UnitId unitId)
{
    m_outputUnitIdVector.emplace_back(unitId);
}


void UnitMetaData::SetOutputUnitIdVector(std::vector<UnitId> unitIdVector)
{
    m_outputUnitIdVector = std::move(unitIdVector);
}

UnitId UnitMetaData::Id() const
{
    return m_unitId;
}

std::vector<Shape> UnitMetaData::InputShapeVector() const
{
    return m_inputShapeVector;
}

Shape UnitMetaData::OutputShape() const
{
    return m_outputShape;
}

std::vector<UnitId> UnitMetaData::InputUnitVector() const
{
    return m_inputUnitIdVector;
}

std::vector<UnitId> UnitMetaData::OutputUnitVector() const
{
    return m_outputUnitIdVector;
}

const std::unique_ptr<Initializer>& UnitMetaData::
GetInitializer(const std::string& name) const
{
    return m_initializerVector.at(name);
}

Shape UnitMetaData::GetShape(const std::string& name) const
{
    return m_internalVariableShapeMap.at(name);
}
}
