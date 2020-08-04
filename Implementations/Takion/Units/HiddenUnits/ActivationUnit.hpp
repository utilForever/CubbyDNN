// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_GRAPH_ACTIVATIONUNIT_HPP
#define TAKION_GRAPH_ACTIVATIONUNIT_HPP

#include <Takion/Computations/Activations/ActivationWrapper.hpp>
#include <Takion/Computations/TensorOperations/Computations.hpp>
#include <Takion/Units/HiddenComputableUnits/ActivationUnitDecl.hpp>

namespace Takion::Graph
{
using namespace Compute;

template <typename T>
ActivationUnit<T>::ActivationUnit(
    const UnitId& unitId, const UnitId& sourceUnitId, Tensor<T> forwardInput,
    std::unordered_map<UnitId, Tensor<T>> backwardInputVector,
    Tensor<T> forwardOutput, Tensor<T> backwardOutput,
    std::unordered_map<std::string, Tensor<T>> trainableUnit,
    std::string activationType)
    : ComputableUnit(unitId,
                     { { sourceUnitId, std::move(forwardInput) } },
                     std::move(backwardInputVector), std::move(forwardOutput),
                     { { sourceUnitId, std::move(backwardOutput) } }),
      TrainableUnit(std::move(trainableUnit)),
      m_activationType(std::move(activationType)),
      m_sourceUnitId(sourceUnitId)
{
}

template <typename T>
ActivationUnit<T>::ActivationUnit(ActivationUnit<T>&& activationUnit) noexcept
    : ComputableUnit(std::move(activationUnit)),
      TrainableUnit(std::move(activationUnit)),
      m_activationType(std::move(activationUnit.m_activationType)),
      m_sourceUnitId(std::move(activationUnit.m_sourceUnitId))
{
}

template <typename T>
ActivationUnit<T>& ActivationUnit<T>::operator=(
    ActivationUnit<T>&& activationUnit) noexcept
{
    m_activationType = std::move(activationUnit.m_activationType);
    ComputableUnit<T>::operator=(std::move(activationUnit));
    return *this;
}

template <typename T>
ActivationUnit ActivationUnit<T>::CreateUnit(
    const UnitMetaData<T>& unitMetaData)
{
    std::string activationName =
        unitMetaData.Params.GetStringParam("activationName");

    auto sourceUnitId = unitMetaData.GetInputUnitId("input");

    Tensor forwardInputTensor(unitMetaData.GetInputShape("input"),
                              unitMetaData.Device, unitMetaData.NumericType);

    std::unordered_map<UnitId, Tensor<T>> backwardInputMap;
    for (const auto& backwardInputUnitId : unitMetaData.OutputUnitVector())
    {
        Tensor tensor(unitMetaData.OutputShape(), unitMetaData.Device,
                      unitMetaData.NumericType);
        backwardInputMap[backwardInputUnitId] = std::move(tensor);
    }

    Tensor forwardOutputTensor(unitMetaData.OutputShape(), unitMetaData.Device,
                               unitMetaData.NumericType);

    Tensor backwardOutputTensor(unitMetaData.GetInputShape("input"),
                                unitMetaData.Device, unitMetaData.NumericType);

    Tensor backwardTempTensor(unitMetaData.OutputShape(), unitMetaData.Device,
                              unitMetaData.NumericType);

    auto activationUnit = ActivationUnit<T>(
        unitMetaData.Id(), sourceUnitId, std::move(forwardInputTensor),
        std::move(backwardInputMap), std::move(forwardOutputTensor),
        std::move(backwardOutputTensor),
        { { "backwardTemp", backwardTempTensor } }, std::move(activationName),
        unitMetaData.NumericType);

    return activationUnit;
}

template <typename T>
void ActivationUnit<T>::Forward()
{
    const auto& activationFunc =
        ActivationWrapper<T>::GetFloatActivation(m_activationType);
    activationFunc->Apply(
        ActivationWrapper<T>::ForwardInputMap.at(m_sourceUnitId),
        ActivationWrapper<T>::ForwardOutput);
}

template <typename T>
void ActivationUnit<T>::AsyncForward(std::promise<bool> promise)
{
    const auto& activationFunc =
        ActivationWrapper<T>::GetFloatActivation(m_activationType);
    activationFunc->Apply(
        ActivationWrapper<T>::ForwardInputMap.at(m_sourceUnitId),
        ActivationWrapper<T>::ForwardOutput);
    promise.set_value(true);
}

template <typename T>
void ActivationUnit<T>::Backward()
{
    const auto& activationFunc =
        ActivationWrapper<T>::GetFloatActivation(m_activationType);
    const Zeros zeroInitializer;
    auto& backwardTemp = ActivationWrapper<T>::m_trainableTensorMap.at(
        "backwardTemp");
    auto& backwardOutput = ActivationWrapper<T>::BackwardOutputMap.at(
        m_sourceUnitId);

    zeroInitializer.Initialize(backwardTemp);
    for (const auto& [unitId, tensor] : ActivationWrapper<T>::BackwardInputMap)
        Compute::Add(backwardTemp, tensor);
    Compute::ScalarMul(backwardTemp,
                       1.0f / static_cast<float>(
                           ActivationWrapper<T>::BackwardInputMap.size()));
    activationFunc->ApplyDerivative(
        ActivationWrapper<T>::ForwardInputMap.at(m_sourceUnitId),
        backwardOutput);
    Compute::Dot(backwardTemp, backwardOutput, backwardOutput);
}

template <typename T>
void ActivationUnit<T>::AsyncBackward(std::promise<bool> promise)
{
    const auto& activationFunc =
        ActivationWrapper<T>::GetFloatActivation(m_activationType);
    const Zeros zeroInitializer;
    auto& backwardTemp = ActivationWrapper<T>::m_trainableTensorMap.at(
        "backwardTemp");
    auto& backwardOutput = ActivationWrapper<T>::BackwardOutputMap.at(
        m_sourceUnitId);

    zeroInitializer.Initialize(backwardTemp);
    for (const auto& [unitId, tensor] : ActivationWrapper<T>::BackwardInputMap)
        Compute::Add(backwardTemp, tensor);
    Compute::ScalarMul(backwardTemp,
                       1.0f / static_cast<float>(ActivationWrapper<T
                       >::BackwardInputMap.size()));
    activationFunc->ApplyDerivative(
        ActivationWrapper<T>::ForwardInputMap.at(m_sourceUnitId),
        backwardOutput);
    Compute::Dot(backwardTemp, backwardOutput, backwardOutput);

    promise.set_value(true);
}
} // namespace Takion::Graph

#endif
