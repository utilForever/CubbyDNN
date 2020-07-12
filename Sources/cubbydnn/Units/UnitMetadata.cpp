// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Units/UnitMetadata.hpp>

namespace CubbyDNN::Graph
{
ParameterPack::ParameterPack(std::unordered_map<std::string, int> integerParams,
                             std::unordered_map<std::string, float>
                             floatingPointParams,
                             std::unordered_map<std::string, std::string>
                             stringParams)
    : m_integerParameters(std::move(integerParams)),
      m_floatingPointParameters(std::move(floatingPointParams)),
      m_stringParameters(std::move(stringParams))
{
}


int ParameterPack::GetIntegerParam(const std::string& name) const
{
    return m_integerParameters.at(name);
}

float ParameterPack::GetFloatingPointParam(const std::string& name) const
{
    return m_floatingPointParameters.at(name);
}

std::string ParameterPack::GetStringParam(const std::string& name) const
{
    return m_stringParameters.at(name);
}


UnitMetaData::UnitMetaData(UnitId unitId,
                           std::unordered_map<std::string, Shape>
                           internalVariableShapeMap,
                           std::unordered_map<
                               std::string, std::unique_ptr<Initializer>>
                           initializerMap,
                           std::vector<Shape> inputShapeVector,
                           Shape outputShape,
                           std::vector<UnitId> inputUnitIdVector,
                           NumberSystem numericType,
                           Compute::Device device)
    : NumericType(numericType),
      Device(std::move(device)),
      m_unitId(std::move(unitId)),
      m_internalVariableShapeMap(std::move(internalVariableShapeMap)),
      m_initializerMap(std::move(initializerMap)),
      m_inputShapeVector(std::move(inputShapeVector)),
      m_outputShape(std::move(outputShape)),
      m_inputUnitIdVector(std::move(inputUnitIdVector))
{
}


UnitMetaData::UnitMetaData(
    UnitId unitId,
    std::unordered_map<std::string, Shape> internalVariableShapeMap,
    std::unordered_map<std::string, std::unique_ptr<Initializer>>
    initializerMap,
    std::vector<Shape> inputShapeVector, Shape outputShape,
    std::vector<UnitId> inputUnitIdVector, ParameterPack params,
    NumberSystem numericType, Compute::Device device)
    : Parameters(std::move(params)),
      NumericType(numericType),
      Device(std::move(device)),
      m_unitId(std::move(unitId)),
      m_internalVariableShapeMap(std::move(internalVariableShapeMap)),
      m_initializerMap(std::move(initializerMap)),
      m_inputShapeVector(std::move(inputShapeVector)),
      m_outputShape(std::move(outputShape)),
      m_inputUnitIdVector(std::move(inputUnitIdVector))
{
}

UnitMetaData::UnitMetaData(UnitMetaData&& unitMetaData) noexcept
    : Parameters(std::move(unitMetaData.Parameters)),
      NumericType(unitMetaData.NumericType),
      Device(std::move(unitMetaData.Device)),
      m_unitId(std::move(unitMetaData.m_unitId)),
      m_internalVariableShapeMap(
          std::move(unitMetaData.m_internalVariableShapeMap)),
      m_initializerMap(std::move(unitMetaData.m_initializerMap)),
      m_inputShapeVector(std::move(unitMetaData.m_inputShapeVector)),
      m_outputShape(std::move(unitMetaData.m_outputShape)),
      m_inputUnitIdVector(std::move(unitMetaData.m_inputUnitIdVector)),
      m_outputUnitIdVector(std::move(unitMetaData.m_outputUnitIdVector))
{
}

UnitMetaData& UnitMetaData::operator=(UnitMetaData&& unitMetaData) noexcept
{
    Parameters = std::move(unitMetaData.Parameters);
    NumericType = unitMetaData.NumericType;
    Device = std::move(unitMetaData.Device);
    m_unitId = std::move(unitMetaData.m_unitId);
    m_internalVariableShapeMap =
        std::move(unitMetaData.m_internalVariableShapeMap);
    m_initializerMap = std::move(unitMetaData.m_initializerMap);
    m_inputShapeVector = std::move(unitMetaData.m_inputShapeVector);
    m_outputShape = std::move(unitMetaData.m_outputShape);
    m_inputUnitIdVector = std::move(unitMetaData.m_inputUnitIdVector);
    m_outputUnitIdVector = std::move(unitMetaData.m_outputUnitIdVector);
    return *this;
}


void UnitMetaData::AppendOutputUnitId(UnitId unitId)
{
    m_outputUnitIdVector.emplace_back(unitId);
}


void UnitMetaData::SetOutputUnitIdVector(std::vector<UnitId> unitIdVector)
{
    m_outputUnitIdVector = std::move(unitIdVector);
}

void UnitMetaData::AddInternalTensor(const std::string& key, Tensor tensor)
{
    m_internalTensorMap[key] = std::move(tensor);
}

const Tensor& UnitMetaData::GetInternalTensor(const std::string& key) const
{
    return m_internalTensorMap.at(key);
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
    return m_initializerMap.at(name);
}

Shape UnitMetaData::GetShape(const std::string& name) const
{
    return m_internalVariableShapeMap.at(name);
}
}
