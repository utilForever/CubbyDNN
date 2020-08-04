// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_GRAPH_LOSSUNIT_HPP
#define TAKION_GRAPH_LOSSuNIT_HPP

#include <Takion/Computations/LossFunctions/LossFunctionWrapper.hpp>
#include <Takion/Units/SinkComputableUnits/LossUnitDecl.hpp>
#include <Takion/Units/UnitType.hpp>
#include <iostream>

namespace Takion::Graph
{
template <typename T>
LossUnit<T>::LossUnit(const UnitId& unitId, const UnitId& predictionUnitId,
                      const UnitId& labelUnitId, Tensor predictionTensor,
                      Tensor labelTensor, Tensor backwardOutputTensor,
                      std::string lossType)
    : ComputableUnit(
          unitId,
          { { predictionUnitId, std::move(predictionTensor) },
            { labelUnitId, std::move(labelTensor) } },
          {},
          Tensor(Shape({ 1, 1 }),
                 Compute::Device(0, Compute::DeviceType::Cpu, "none")),
          { { predictionUnitId, std::move(backwardOutputTensor) } }),
      m_lossType(std::move(lossType)),
      m_predictionUnitId(predictionUnitId),
      m_labelUnitId(labelUnitId)
{
}

template <typename T>
LossUnit<T>::LossUnit(LossUnit<T>&& lossUnit) noexcept
    : ComputableUnit(std::move(lossUnit)),
      m_lossType(std::move(lossUnit.m_lossType)),
      m_predictionUnitId(std::move(lossUnit.m_predictionUnitId)),
      m_labelUnitId(std::move(lossUnit.m_labelUnitId))

{
}

template <typename T>
LossUnit<T>& LossUnit<T>::operator=(LossUnit<T>&& lossUnit) noexcept
{
    m_lossType = std::move(lossUnit.m_lossType);
    ComputableUnit<T>::operator=(std::move(lossUnit));
    return *this;
}

template <typename T>
LossUnit<T> LossUnit<T>::CreateUnit(const UnitMetaData<T>& unitMetaData)

{
    auto predictionUnitId = unitMetaData.GetInputUnitId("prediction");
    auto labelUnitId = unitMetaData.GetInputUnitId("label");
    auto predictionTensor =
        Tensor(unitMetaData.GetInputShape("prediction"), unitMetaData.Device,
               unitMetaData.NumericType);
    auto labelTensor = Tensor(unitMetaData.GetInputShape("label"),
                              unitMetaData.Device, unitMetaData.NumericType);
    auto backwardOutputTensor =
        Tensor(unitMetaData.GetInputShape("prediction"), unitMetaData.Device,
               unitMetaData.NumericType);

    return LossUnit(unitMetaData.Id(), predictionUnitId, labelUnitId,
                    predictionTensor, labelTensor, backwardOutputTensor,
                    unitMetaData.Params.GetStringParam("lossType"),
                    NumberSystem::Float);
}

template <typename T>
void LossUnit<T>::Forward()
{
    Tensor<T>& prediction =
        ComputableUnit<T>::ForwardInputMap.at(m_predictionUnitId);
    const Tensor& label = ComputableUnit<T>::ForwardInputMap.at(m_labelUnitId);

    const auto& lossFunc =
        Compute::LossFunctionWrapper<T>::GetFloatLoss(m_lossType);
    const auto loss = lossFunc->Apply(prediction, label);

    Tensor<T>& lossOutput = ForwardOutput;
    *(static_cast<float*>(lossOutput.DataPtr)) = loss;
    std::cout << "Loss: " << loss << std::endl;
}

template <typename T>
void LossUnit<T>::AsyncForward(std::promise<bool> promise)
{
    Tensor<T>& prediction =
        ComputableUnit<T>::ForwardInputMap.at(m_predictionUnitId);
    const Tensor& label = ComputableUnit<T>::ForwardInputMap.at(m_labelUnitId);

    const auto& lossFunc =
        Compute::LossFunctionWrapper<T>::GetFloatLoss(m_lossType);
    const auto loss = lossFunc->Apply(prediction, label);

    Tensor<T>& lossOutput = ComputableUnit<T>::ForwardOutput;
    *(static_cast<float*>(lossOutput.DataPtr)) = loss;

    promise.set_value(true);
}

template <typename T>
void LossUnit<T>::Backward()
{
    const auto& prediction =
        ComputableUnit<T>::ForwardInputMap.at(m_predictionUnitId);
    const auto& label = ComputableUnit<T>::ForwardInputMap.at(m_labelUnitId);
    auto& delta = ComputableUnit<T>::BackwardOutputMap.at(m_predictionUnitId);

    const auto& lossFunc =
        Compute::LossFunctionWrapper<T>::GetFloatLoss(m_lossType);
    lossFunc->ApplyDerivative(label, prediction, delta);
}

template <typename T>
void LossUnit<T>::AsyncBackward(std::promise<bool> promise)

{
    const auto& prevInput =
        ComputableUnit<T>::ForwardInputMap.at(m_predictionUnitId);
    const auto& label = ComputableUnit<T>::ForwardInputMap.at(m_labelUnitId);
    auto& delta = ComputableUnit<T>::BackwardOutputMap.at(m_predictionUnitId);

    const auto& lossFunc =
        Compute::LossFunctionWrapper<T>::GetFloatLoss(m_lossType);
    lossFunc->ApplyDerivative(label, prevInput, delta);
}
} // namespace Takion::Graph

#endif
