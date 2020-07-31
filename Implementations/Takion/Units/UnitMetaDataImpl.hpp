// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_UNITMETADATA_IMPL_HPP
#define TAKION_UNITMETADATA_IMPL_HPP

#include <Takion/Units/UnitMetadata.hpp>

namespace Takion::Graph
{
template <typename T>
UnitMetaData<T>::UnitMetaData(
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
      Device(std::move(device)),
      Params(std::move(params))
{
}

template <typename T>
UnitMetaData<T>::UnitMetaData(UnitMetaData&& unitMetaData)
noexcept
    : m_unitId(std::move
          (unitMetaData.m_unitId)),
      m_internalVariableShapeMap(
          std::move(unitMetaData.m_internalVariableShapeMap)),
      m_initializerMap(std::move(unitMetaData.m_initializerMap)),
      m_internalTensorMap(std::move(unitMetaData.m_internalTensorMap)),
      m_inputShapeMap(std::move(unitMetaData.m_inputShapeMap)),
      m_outputShape(std::move(unitMetaData.m_outputShape)),
      m_inputUnitMap(std::move(unitMetaData.m_inputUnitMap)),
      m_outputUnitIdVector(std::move(unitMetaData.m_outputUnitIdVector)),
      Device(std::move(unitMetaData.Device)),
      Params(std::move(unitMetaData.Params))
{
}

template <typename T>
UnitMetaData<T>& UnitMetaData<T>::operator=(UnitMetaData<T>&& unitMetaData)
noexcept
{
    Params = std::move(unitMetaData.Params);
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

template <typename T>
void UnitMetaData<T>::AppendOutputUnitId(UnitId unitId)
{
    m_outputUnitIdVector.emplace_back(unitId);
}

template <typename T>
void UnitMetaData<T>::SetOutputUnitIdVector(std::vector<UnitId> unitIdVector)
{
    m_outputUnitIdVector = std::move(unitIdVector);
}

template <typename T>
void UnitMetaData<T>::AddInternalTensor(const std::string& key, Tensor tensor)
{
    m_internalTensorMap[key] = std::move(tensor);
}

template <typename T>
const Tensor<T>& UnitMetaData<T>::GetInternalTensor(const std::string& key) const
{
    return m_internalTensorMap.at(key);
}

template <typename T>
UnitId UnitMetaData<T>::Id() const
{
    return m_unitId;
}

template <typename T>
Shape UnitMetaData<T>::GetInputShape(const std::string& key) const
{
    return m_inputShapeMap.at(key);
}

template <typename T>
UnitId UnitMetaData<T>::GetInputUnitId(const std::string& key) const
{
    return m_inputUnitMap.at(key);
}

template <typename T>
Shape UnitMetaData<T>::OutputShape() const
{
    return m_outputShape;
}

template <typename T>
std::unordered_map<std::string, UnitId> UnitMetaData<T>::InputUnitMap() const
{
    return m_inputUnitMap;
}

template <typename T>
std::vector<UnitId> UnitMetaData<T>::OutputUnitVector() const
{
    return m_outputUnitIdVector;
}

template <typename T>
const std::unique_ptr<Initializer>& UnitMetaData<T>::GetInitializer(
    const std::string& name) const
{
    return m_initializerMap.at(name);
}

template <typename T>
Shape UnitMetaData<T>::GetInternalVariableShape(const std::string& name) const
{
    return m_internalVariableShapeMap.at(name);
}
} // namespace Takion::Graph

#endif
