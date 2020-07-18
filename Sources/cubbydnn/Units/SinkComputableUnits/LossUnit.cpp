// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <iostream>
#include <cubbydnn/Units/SinkComputableUnits/LossUnit.hpp>

#include <cubbydnn/Computations/LossFunctions/LossFunctionWrapper.hpp>

namespace CubbyDNN::Graph
{
LossUnit::LossUnit(const UnitId& unitId, const UnitId& predictionUnitId,
                   const UnitId& labelUnitId,
                   Tensor predictionTensor,
                   Tensor labelTensor,
                   Tensor backwardOutputTensor,
                   std::string lossType,
                   NumberSystem numberSystem)
    : ComputableUnit(unitId, numberSystem,
                     { { predictionUnitId, std::move(predictionTensor) },
                       { labelUnitId, std::move(labelTensor) } },
                     {},
                     Tensor(Shape({ 1, 1 }),
                            Compute::Device(0, Compute::DeviceType::Cpu,
                                            "none")),
                     { { predictionUnitId, std::move(backwardOutputTensor) } }),
      m_lossType(std::move(lossType)),
      m_predictionUnitId(predictionUnitId),
      m_labelUnitId(labelUnitId)
{
}

LossUnit::LossUnit(LossUnit&& lossUnit) noexcept
    : ComputableUnit(std::move(lossUnit)),
      m_lossType(std::move(lossUnit.m_lossType)),
      m_predictionUnitId(std::move(lossUnit.m_predictionUnitId)),
      m_labelUnitId(std::move(lossUnit.m_labelUnitId))
{
}

LossUnit& LossUnit::operator=(LossUnit&& lossUnit)
noexcept
{
    m_lossType = std::move(lossUnit.m_lossType);
    ComputableUnit::operator=(std::move(lossUnit));
    return *this;
}

LossUnit LossUnit::CreateUnit(const UnitMetaData& unitMetaData)
{
    auto predictionUnitId = unitMetaData.GetInputUnitId("prediction");
    auto labelUnitId = unitMetaData.GetInputUnitId("label");
    auto predictionTensor = Tensor(unitMetaData.GetInputShape("prediction"),
                                   unitMetaData.Device,
                                   unitMetaData.NumericType);
    auto labelTensor = Tensor(unitMetaData.GetInputShape("label"),
                              unitMetaData.Device, unitMetaData.NumericType);
    auto backwardOutputTensor =
        Tensor(unitMetaData.GetInputShape("prediction"), unitMetaData.Device,
               unitMetaData.NumericType);

    return LossUnit(unitMetaData.Id(), predictionUnitId, labelUnitId,
                    predictionTensor, labelTensor,
                    backwardOutputTensor,
                    unitMetaData.Params.GetStringParam("lossType"),
                    NumberSystem::Float);
}


void LossUnit::Forward()
{
    Tensor& prediction = ForwardInputMap.at(m_predictionUnitId);
    const Tensor& label = ForwardInputMap.at(m_labelUnitId);
    if (m_numericType == NumberSystem::Float)
    {
        const auto& lossFunc =
            Compute::LossFunctionWrapper::GetFloatLoss(m_lossType);
        const auto loss = lossFunc->Apply(prediction, label);

        Tensor& lossOutput = ForwardOutput;
        *(static_cast<float*>(lossOutput.DataPtr)) = loss;
    }
    else
    {
        const auto& lossFunc =
            Compute::LossFunctionWrapper::GetIntegerLoss(m_lossType);
        const auto loss = lossFunc->Apply(prediction, label);

        Tensor& lossOutput = ForwardOutput;
        *static_cast<int*>(lossOutput.DataPtr) = loss;
        std::cout << loss << std::endl;
    }
}

void LossUnit::AsyncForward(std::promise<bool> promise)
{
    Tensor& prediction = ForwardInputMap.at(m_predictionUnitId);
    const Tensor& label = ForwardInputMap.at(m_labelUnitId);
    if (m_numericType == NumberSystem::Float)
    {
        const auto& lossFunc =
            Compute::LossFunctionWrapper::GetFloatLoss(m_lossType);
        const auto loss = lossFunc->Apply(prediction, label);

        Tensor& lossOutput = ForwardOutput;
        *(static_cast<float*>(lossOutput.DataPtr)) = loss;
    }
    else
    {
        const auto& lossFunc =
            Compute::LossFunctionWrapper::GetIntegerLoss(m_lossType);
        const auto loss = lossFunc->Apply(prediction, label);

        Tensor& lossOutput = ForwardOutput;
        *static_cast<int*>(lossOutput.DataPtr) = loss;
    }
    promise.set_value(true);
}


void LossUnit::Backward()
{
    const auto& prediction = ForwardInputMap.at(m_predictionUnitId);
    const auto& label = ForwardInputMap.at(m_labelUnitId);
    auto& delta = BackwardOutputMap.at(m_predictionUnitId);

    if (m_numericType == NumberSystem::Float)
    {
        const auto& lossFunc =
            Compute::LossFunctionWrapper::GetFloatLoss(m_lossType);
        lossFunc->ApplyDerivative(
            label, prediction, delta);
    }
    else
    {
        const auto& lossFunc =
            Compute::LossFunctionWrapper::GetIntegerLoss(m_lossType);
        lossFunc->ApplyDerivative(
            label, prediction, delta);
    }
}

void LossUnit::AsyncBackward(std::promise<bool> promise)
{
    const auto& prevInput = ForwardInputMap.at(m_predictionUnitId);
    const auto& label = ForwardInputMap.at(m_labelUnitId);
    auto& delta = BackwardOutputMap.at(m_predictionUnitId);

    if (m_numericType == NumberSystem::Float)
    {
        const auto& lossFunc =
            Compute::LossFunctionWrapper::GetFloatLoss(m_lossType);
        lossFunc->ApplyDerivative(
            label, prevInput, delta);
    }
    else
    {
        const auto& lossFunc =
            Compute::LossFunctionWrapper::GetIntegerLoss(m_lossType);
        lossFunc->ApplyDerivative(
            label, prevInput, delta);
    }
    promise.set_value(true);
}
}
