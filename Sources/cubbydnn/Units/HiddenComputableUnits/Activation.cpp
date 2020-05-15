// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Units/HiddenComputableUnits/Activation.hpp>

namespace CubbyDNN::Graph
{
ActivationUnit::ActivationUnit(UnitId unitId, NumberSystem numberSystem,
                               Tensor forwardInput,
                               std::vector<Tensor> backwardInputVector,
                               Tensor forwardOutput, Tensor backwardOutput,
                               std::unique_ptr<Compute::ActivationFunc>
                               activationFunc)
    : ComputableUnit(unitId, numberSystem, { std::move(forwardInput) },
                     std::move(backwardInputVector), std::move(forwardOutput),
                     { std::move(backwardOutput) }),
      m_activationFunc(std::move(activationFunc))
{
}

ActivationUnit::ActivationUnit(ActivationUnit&& activationUnit) noexcept
    : ComputableUnit(std::move(activationUnit)),
      m_activationFunc(std::move(activationUnit.m_activationFunc))
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
        unitMetaData.Device, unitMetaData.PadSize);

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
        unitMetaData.Device, unitMetaData.PadSize);

    auto backwardOutputTensor = Tensor::CreateTensor(
        unitMetaData.InputShapeVector().at(0), unitMetaData.NumericType,
        unitMetaData.Device, unitMetaData.PadSize);

    auto activationUnit = ActivationUnit(unitMetaData.Id(),
                                         unitMetaData.NumericType,
                                         std::move(forwardInputTensor),
                                         std::move(backwardInputVector),
                                         std::move(forwardOutputTensor),
                                         std::move(backwardOutputTensor),
                                         std::move(activationFunc));

    return activationUnit;
}

void ActivationUnit::Forward()
{
    
}


}
