// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Units/SinkComputableUnits/LossUnit.hpp>

#include <cubbydnn/Computations/LossFunctions/LossFunctionWrapper.hpp>

namespace CubbyDNN::Graph
{
LossUnit::LossUnit(UnitId unitId, NumberSystem numberSystem,
                   Tensor forwardInput, Tensor label, Tensor delta,
                   std::string lossName)
    : ComputableUnit(unitId, numberSystem, { std::move(forwardInput) },
                     { std::move(label) },
                     Tensor(Shape({ 1, 1 }),
                            Compute::Device(0, Compute::DeviceType::Cpu,
                                            "lossOutput")),
                     { std::move(delta) }),
      m_lossName(std::move(lossName))
{
}

LossUnit::LossUnit(LossUnit&& lossUnit) noexcept
    : ComputableUnit(std::move(lossUnit)),
      m_lossName(std::move(lossUnit.m_lossName))
{
}

LossUnit& LossUnit::operator=(LossUnit&& lossUnit) noexcept
{
    ComputableUnit::operator=(std::move(lossUnit));
    m_lossName = std::move(lossUnit.m_lossName);
    return *this;
}

void LossUnit::Forward()
{
    Tensor& forwardInput = ForwardInputVector.at(0);
    const Tensor& label = BackwardInputVector.at(0);
    if (m_numericType == NumberSystem::Float)
    {
        const auto& lossFunc =
            Compute::LossFunctionWrapper::GetFloatLoss(m_lossName);
        const auto loss = lossFunc->Apply(forwardInput, label);

        Tensor& lossOutput = ForwardOutput;
        *(static_cast<float*>(lossOutput.DataPtr.Get())) = loss;
    }
    else
    {
        const auto& lossFunc =
            Compute::LossFunctionWrapper::GetIntegerLoss(m_lossName);
        const auto loss = lossFunc->Apply(forwardInput, label);

        Tensor& lossOutput = ForwardOutput;
        *(static_cast<float*>(lossOutput.DataPtr.Get())) = loss;
    }
}

void LossUnit::Backward()
{
    const Tensor& prevInput = ForwardInputVector.at(0);
    const Tensor& label = BackwardInputVector.at(0);
    Tensor& delta = BackwardOutputVector.at(0);

    if (m_numericType == NumberSystem::Float)
    {
        const auto& lossFunc =
            Compute::LossFunctionWrapper::GetFloatLoss(m_lossName);
        lossFunc->ApplyDerivative(label, prevInput, delta);
    }
    else
    {
        const auto& lossFunc =
            Compute::LossFunctionWrapper::GetIntegerLoss(m_lossName);
        lossFunc->ApplyDerivative(label, prevInput, delta);
    }
}
}
