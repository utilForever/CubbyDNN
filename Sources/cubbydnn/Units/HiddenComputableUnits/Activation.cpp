// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Units/HiddenComputableUnits/ActivationUnit.hpp>
#include <cubbydnn/Computations/TensorOperations/Computations.hpp>

namespace CubbyDNN::Graph
{
ActivationUnit::ActivationUnit(UnitId unitId, NumberSystem numberSystem,
                               Tensor forwardInput,
                               std::vector<Tensor> backwardInputVector,
                               Tensor forwardOutput,
                               Tensor backwardOutput, Tensor backwardTemp,
                               std::unique_ptr<Compute::ActivationFunc>
                               activationFunc)
    : ComputableUnit(unitId, numberSystem, { std::move(forwardInput) },
                     std::move(backwardInputVector), std::move(forwardOutput),
                     { std::move(backwardOutput) }),
      m_activationFunc(std::move(activationFunc)),
      m_backwardTemp(std::move(backwardTemp))
{
}

ActivationUnit::ActivationUnit(ActivationUnit&& activationUnit) noexcept
    : ComputableUnit(std::move(activationUnit)),
      m_activationFunc(std::move(activationUnit.m_activationFunc)),
      m_backwardTemp(std::move(activationUnit.m_backwardTemp))
{
}

ActivationUnit& ActivationUnit::operator=(
    ActivationUnit&& activationUnit) noexcept
{
    m_activationFunc = std::move(activationUnit.m_activationFunc);
    ComputableUnit::operator=(std::move(activationUnit));
    return *this;
}

ActivationUnit ActivationUnit::CreateUnit(const UnitMetaData& unitMetaData,
                                          std::unique_ptr<Compute::
                                              ActivationFunc> activationFunc)
{
    auto forwardInputTensor = Tensor::CreateTensor(
        unitMetaData.InputShapeVector().at(0), unitMetaData.NumericType,
        unitMetaData.Device);

    std::vector<Tensor> backwardInputVector;
    backwardInputVector.reserve(unitMetaData.OutputUnitVector().size());
    for (std::size_t i = 0; i < unitMetaData.OutputUnitVector().size(); ++i)
    {
        auto tensor =
            Tensor::CreateTensor(unitMetaData.OutputShape(),
                                 unitMetaData.NumericType, unitMetaData.Device);
        backwardInputVector.emplace_back(std::move(tensor));
    }

    auto forwardOutputTensor = Tensor::CreateTensor(
        unitMetaData.OutputShape(), unitMetaData.NumericType,
        unitMetaData.Device);

    auto backwardOutputTensor = Tensor::CreateTensor(
        unitMetaData.InputShapeVector().at(0), unitMetaData.NumericType,
        unitMetaData.Device);

    auto backwardTempTensor =
        Tensor::CreateTensor(unitMetaData.OutputShape(),
                             unitMetaData.NumericType,
                             unitMetaData.Device);

    auto activationUnit = ActivationUnit(unitMetaData.Id(),
                                         unitMetaData.NumericType,
                                         std::move(forwardInputTensor),
                                         std::move(backwardInputVector),
                                         std::move(forwardOutputTensor),
                                         std::move(backwardOutputTensor),
                                         std::move(backwardTempTensor),
                                         std::move(activationFunc));

    return activationUnit;
}

void ActivationUnit::Forward()
{
    Compute::ActivationForward(ForwardInputVector.at(0), ForwardOutput,
                               m_activationFunc);
}

void ActivationUnit::AsyncForward(std::promise<bool> promise)
{
    Compute::ActivationForward(ForwardInputVector.at(0), ForwardOutput,
                               m_activationFunc);
    promise.set_value(true);
}

void ActivationUnit::Backward()
{
    const Zeros zeroInitializer;
    zeroInitializer.Initialize(m_backwardTemp);
    for (const auto& tensor : BackwardInputVector)
        Compute::Add(tensor, m_backwardTemp,
                     m_backwardTemp);
    Compute::ActivationBackward(ForwardInputVector.at(0),
                                BackwardOutputVector.at(0),
                                m_activationFunc);
    Compute::Dot(m_backwardTemp, BackwardOutputVector.at(0),
                 BackwardOutputVector.at(0));
}

void ActivationUnit::AsyncBackward(std::promise<bool> promise)
{
    const Zeros zeroInitializer;
    zeroInitializer.Initialize(m_backwardTemp);
    for (const auto& tensor : BackwardInputVector)
        Compute::Add(tensor, m_backwardTemp,
                     m_backwardTemp);
    Compute::ActivationBackward(
        ForwardInputVector.at(0), BackwardOutputVector.at(0), m_activationFunc);
    Compute::Dot(m_backwardTemp, BackwardOutputVector.at(0),
                 BackwardOutputVector.at(0));

    promise.set_value(true);
}
}
