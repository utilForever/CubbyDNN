// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Units/HiddenComputableUnits/ActivationUnits/ActivationUnit.hpp>
#include <cubbydnn/Computations/TensorOperations/Computations.hpp>
#include <cubbydnn/Computations/Activations/ActivationWrapper.hpp>

namespace CubbyDNN::Graph
{
ActivationUnit::ActivationUnit(
    const UnitId& unitId, const UnitId& sourceUnitId, Tensor forwardInput,
    std::unordered_map<UnitId, Tensor> backwardInputVector,
    Tensor forwardOutput, Tensor backwardOutput,
    std::unordered_map<std::string, Tensor> trainableUnit,
    std::string activationType, NumberSystem numberSystem)
    : ComputableUnit(unitId, numberSystem,
                     { { sourceUnitId, std::move(forwardInput) } },
                     std::move(backwardInputVector), std::move(forwardOutput),
                     { { sourceUnitId, std::move(backwardOutput) } }),
      TrainableUnit(std::move(trainableUnit)),
      m_activationType(std::move(activationType)),
      m_sourceUnitId(sourceUnitId)
{
}

ActivationUnit::ActivationUnit(ActivationUnit&& activationUnit) noexcept
    : ComputableUnit(std::move(activationUnit)),
      TrainableUnit(std::move(activationUnit)),
      m_activationType(std::move(activationUnit.m_activationType)),
      m_sourceUnitId(std::move(activationUnit.m_sourceUnitId))
{
}

ActivationUnit& ActivationUnit::operator=(
    ActivationUnit&& activationUnit) noexcept
{
    m_activationType = std::move(activationUnit.m_activationType);
    ComputableUnit::operator=(std::move(activationUnit));
    return *this;
}

ActivationUnit ActivationUnit::CreateUnit(const UnitMetaData& unitMetaData)
{
    std::string activationName =
        unitMetaData.Params.GetStringParam("activationName");

    auto sourceUnitId = unitMetaData.GetInputUnitId("input");

    Tensor forwardInputTensor(unitMetaData.GetInputShape("input"),
                              unitMetaData.Device, unitMetaData.NumericType);

    std::unordered_map<UnitId, Tensor> backwardInputMap;
    for (const auto& backwardInputUnitId : unitMetaData.OutputUnitVector())
    {
        Tensor tensor(unitMetaData.OutputShape(),
                      unitMetaData.Device, unitMetaData.NumericType);
        backwardInputMap[backwardInputUnitId] = std::move(tensor);
    }

    Tensor forwardOutputTensor(unitMetaData.OutputShape(),
                               unitMetaData.Device, unitMetaData.NumericType);

    Tensor backwardOutputTensor(unitMetaData.GetInputShape("input"),
                                unitMetaData.Device, unitMetaData.NumericType);

    Tensor backwardTempTensor(unitMetaData.OutputShape(),
                              unitMetaData.Device, unitMetaData.NumericType);

    auto activationUnit = ActivationUnit(
        unitMetaData.Id(), sourceUnitId,
        std::move(forwardInputTensor), std::move(backwardInputMap),
        std::move(forwardOutputTensor), std::move(backwardOutputTensor),
        { { "backwardTemp", backwardTempTensor } },
        std::move(activationName), unitMetaData.NumericType);

    return activationUnit;
}

void ActivationUnit::Forward()
{
    const auto& activationFunc = Compute::ActivationWrapper::GetFloatActivation(
        m_activationType);
    activationFunc->Apply(ForwardInputMap.at(m_sourceUnitId), ForwardOutput);
}

void ActivationUnit::AsyncForward(std::promise<bool> promise)
{
    const auto& activationFunc =
        Compute::ActivationWrapper::GetFloatActivation(m_activationType);
    activationFunc->Apply(ForwardInputMap.at(m_sourceUnitId), ForwardOutput);
    promise.set_value(true);
}

void ActivationUnit::Backward()
{
    const auto& activationFunc =
        Compute::ActivationWrapper::GetFloatActivation(m_activationType);
    const Zeros zeroInitializer;
    zeroInitializer.Initialize(m_trainableTensorMap["backwardTemp"]);
    for (const auto& [unitId, tensor] : BackwardInputMap)
        Compute::Add(m_trainableTensorMap["backwardTemp"], tensor);
    activationFunc->ApplyDerivative(ForwardInputMap.at(m_sourceUnitId),
                                    BackwardOutputMap.at(m_sourceUnitId));
    Compute::Dot(m_trainableTensorMap["backwardTemp"],
                 BackwardOutputMap.at(m_sourceUnitId),
                 BackwardOutputMap.at(m_sourceUnitId));
}

void ActivationUnit::AsyncBackward(std::promise<bool> promise)
{
    const auto& activationFunc =
        Compute::ActivationWrapper::GetFloatActivation(m_activationType);
    const Zeros zeroInitializer;
    zeroInitializer.Initialize(m_trainableTensorMap["backwardTemp"]);
    for (const auto& [unitId, tensor] : BackwardInputMap)
        Compute::Add(m_trainableTensorMap["backwardTemp"], tensor);
    activationFunc->ApplyDerivative(ForwardInputMap.at(m_sourceUnitId),
                                    BackwardOutputMap.at(m_sourceUnitId));
    Compute::Dot(m_trainableTensorMap["backwardTemp"],
                 BackwardOutputMap.at(m_sourceUnitId),
                 BackwardOutputMap.at(m_sourceUnitId));

    promise.set_value(true);
}
}
