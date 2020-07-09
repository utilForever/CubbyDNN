// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Units/HiddenComputableUnits/ActivationUnits/ActivationUnit.hpp>
#include <cubbydnn/Computations/TensorOperations/Computations.hpp>
#include <cubbydnn/Computations/Activations/ActivationWrapper.hpp>

namespace CubbyDNN::Graph
{
ActivationUnit::ActivationUnit(UnitId unitId, NumberSystem numberSystem,
                               Tensor forwardInput,
                               std::vector<Tensor> backwardInputVector,
                               Tensor forwardOutput,
                               Tensor backwardOutput,
                               std::unordered_map<std::string, Tensor>
                               trainableUnit,
                               std::string activationName)
    : ComputableUnit(unitId, numberSystem, { std::move(forwardInput) },
                     std::move(backwardInputVector), std::move(forwardOutput),
                     { std::move(backwardOutput) }),
      TrainableUnit(std::move(trainableUnit)),
      m_activationName(std::move(activationName))
{
}

ActivationUnit::ActivationUnit(ActivationUnit&& activationUnit) noexcept
    : ComputableUnit(std::move(activationUnit)),
      TrainableUnit(std::move(activationUnit)),
      m_activationName(std::move(activationUnit.m_activationName))
{
}

ActivationUnit& ActivationUnit::operator=(
    ActivationUnit&& activationUnit) noexcept
{
    m_activationName = std::move(activationUnit.m_activationName);
    ComputableUnit::operator=(std::move(activationUnit));
    return *this;
}

ActivationUnit ActivationUnit::CreateUnit(const UnitMetaData& unitMetaData)
{
    std::string activationName =
        unitMetaData.Parameters.GetStringParam("activationName");

    Tensor forwardInputTensor(unitMetaData.InputShapeVector().at(0),
                              unitMetaData.Device, unitMetaData.NumericType);

    std::vector<Tensor> backwardInputVector;
    backwardInputVector.reserve(unitMetaData.OutputUnitVector().size());
    for (std::size_t i = 0; i < unitMetaData.OutputUnitVector().size(); ++i)
    {
        Tensor tensor(unitMetaData.OutputShape(),
                      unitMetaData.Device, unitMetaData.NumericType);
        backwardInputVector.emplace_back(std::move(tensor));
    }

    Tensor forwardOutputTensor(unitMetaData.OutputShape(),
                               unitMetaData.Device, unitMetaData.NumericType);

    Tensor backwardOutputTensor(unitMetaData.InputShapeVector().at(0),
                                unitMetaData.Device, unitMetaData.NumericType);

    Tensor backwardTempTensor(unitMetaData.OutputShape(),
                              unitMetaData.Device, unitMetaData.NumericType);

    auto activationUnit = ActivationUnit(
        unitMetaData.Id(), unitMetaData.NumericType,
        std::move(forwardInputTensor), std::move(backwardInputVector),
        std::move(forwardOutputTensor), std::move(backwardOutputTensor),
        { { "backwardTemp", backwardTempTensor } }, std::move(activationName));

    return activationUnit;
}

void ActivationUnit::Forward()
{
    const auto& activationFunc = Compute::ActivationWrapper::GetFloatActivation(
        m_activationName);
    activationFunc->Apply(ForwardInputVector.at(0), ForwardOutput);
}

void ActivationUnit::AsyncForward(std::promise<bool> promise)
{
    const auto& activationFunc =
        Compute::ActivationWrapper::GetFloatActivation(m_activationName);
    activationFunc->Apply(ForwardInputVector.at(0), ForwardOutput);
    promise.set_value(true);
}

void ActivationUnit::Backward()
{
    const auto& activationFunc =
        Compute::ActivationWrapper::GetFloatActivation(m_activationName);
    const Zeros zeroInitializer;
    zeroInitializer.Initialize(m_trainableTensorMap["backwardTemp"]);
    for (const auto& tensor : BackwardInputVector)
        Compute::Add(m_trainableTensorMap["backwardTemp"], tensor);
    activationFunc->ApplyDerivative(ForwardInputVector.at(0),
                                    BackwardOutputVector.at(0));
    Compute::Dot(m_trainableTensorMap["backwardTemp"],
                 BackwardOutputVector.at(0),
                 BackwardOutputVector.at(0));
}

void ActivationUnit::AsyncBackward(std::promise<bool> promise)
{
    const auto& activationFunc =
        Compute::ActivationWrapper::GetFloatActivation(m_activationName);
    const Zeros zeroInitializer;
    zeroInitializer.Initialize(m_trainableTensorMap["backwardTemp"]);
    for (const auto& tensor : BackwardInputVector)
        Compute::Add(m_trainableTensorMap["backwardTemp"], tensor);
    activationFunc->ApplyDerivative(ForwardInputVector.at(0),
                                    BackwardOutputVector.at(0));
    Compute::Dot(m_trainableTensorMap["backwardTemp"],
                 BackwardOutputVector.at(0), BackwardOutputVector.at(0));

    promise.set_value(true);
}
}
