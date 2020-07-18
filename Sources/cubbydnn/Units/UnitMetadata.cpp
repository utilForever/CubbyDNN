// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Units/UnitMetadata.hpp>

namespace CubbyDNN::Graph
{
Parameter::Parameter(std::unordered_map<std::string, int> integerParams,
                     std::unordered_map<std::string, float>
                     floatingPointParams,
                     std::unordered_map<std::string, std::string>
                     stringParams)
    : m_integerParameters(std::move(integerParams)),
      m_floatingPointParameters(std::move(floatingPointParams)),
      m_stringParameters(std::move(stringParams))
{
}

Parameter::Parameter(std::unordered_map<std::string, int> integerParams)
    : m_integerParameters(std::move(integerParams))
{
}

Parameter::Parameter(std::unordered_map<std::string, float> floatingPointParams)
    : m_floatingPointParameters(std::move(floatingPointParams))
{
}

Parameter::Parameter(std::unordered_map<std::string, std::string> stringParams)
    : m_stringParameters(std::move(stringParams))
{
}

int Parameter::GetIntegerParam(const std::string& name) const
{
    return m_integerParameters.at(name);
}

float Parameter::GetFloatingPointParam(const std::string& name) const
{
    return m_floatingPointParameters.at(name);
}

std::string Parameter::GetStringParam(const std::string& name) const
{
    return m_stringParameters.at(name);
}


UnitMetaData::UnitMetaData(
    UnitId unitId,
    std::unordered_map<std::string, Shape> internalVariableShapeMap,
    std::unordered_map<std::string, std::unique_ptr<Initializer>>
    initializerMap,
    std::unordered_map<std::string, Shape> inputShapeMap, Shape outputShape,
    std::unordered_map<std::string, UnitId> inputUnitIdMap,
    NumberSystem numericType, Compute::Device device, Parameter params)
    : m_unitId(std::move(unitId)),
      m_internalVariableShapeMap(std::move(internalVariableShapeMap)),
      m_initializerMap(std::move(initializerMap)),
      m_inputShapeMap(std::move(inputShapeMap)),
      m_outputShape(std::move(outputShape)),
      m_inputUnitMap(std::move(inputUnitIdMap)),
      NumericType(numericType),
      Device(std::move(device)),
      Params(std::move(params))
{
}

UnitMetaData::UnitMetaData(UnitMetaData&& unitMetaData) noexcept
    : m_unitId(std::move(unitMetaData.m_unitId)),
      m_internalVariableShapeMap(
          std::move(unitMetaData.m_internalVariableShapeMap)),
      m_initializerMap(std::move(unitMetaData.m_initializerMap)),
      m_internalTensorMap(std::move(unitMetaData.m_internalTensorMap)),
      m_inputShapeMap(std::move(unitMetaData.m_inputShapeMap)),
      m_outputShape(std::move(unitMetaData.m_outputShape)),
      m_inputUnitMap(std::move(unitMetaData.m_inputUnitMap)),
      m_outputUnitIdVector(std::move(unitMetaData.m_outputUnitIdVector)),
      NumericType(unitMetaData.NumericType),
      Device(std::move(unitMetaData.Device)),
      Params(std::move(unitMetaData.Params))
{
}

UnitMetaData& UnitMetaData::operator=(UnitMetaData&& unitMetaData) noexcept
{
    Params = std::move(unitMetaData.Params);
    NumericType = unitMetaData.NumericType;
    Device = std::move(unitMetaData.Device);
    m_unitId = std::move(unitMetaData.m_unitId);
    m_internalVariableShapeMap =
        std::move(unitMetaData.m_internalVariableShapeMap);
    m_initializerMap = std::move(unitMetaData.m_initializerMap);
    m_internalTensorMap = std::move(unitMetaData.m_internalTensorMap);
    m_inputShapeMap = std::move(unitMetaData.m_inputShapeMap);
    m_outputShape = std::move(unitMetaData.m_outputShape);
    m_inputUnitMap = std::move(unitMetaData.m_inputUnitMap);
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

Shape UnitMetaData::GetInputShape(const std::string& key) const
{
    return m_inputShapeMap.at(key);
}

UnitId UnitMetaData::GetInputUnitId(const std::string& key) const
{
    return m_inputUnitMap.at(key);
}


Shape UnitMetaData::OutputShape() const
{
    return m_outputShape;
}

std::unordered_map<std::string, UnitId> UnitMetaData::InputUnitMap() const
{
    return m_inputUnitMap;
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

Shape UnitMetaData::GetInternalVariableShape(const std::string& name) const
{
    return m_internalVariableShapeMap.at(name);
}
}
