// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_GRAPH_LOSSUNIT_HPP
#define TAKION_GRAPH_LOSSUNIT_HPP

//#include <Takion/Computations/LossFunctions/LossFunctionWrapper.hpp>
#include <Takion/Units/SinkUnits/LossUnitDecl.hpp>
#include <Takion/Units/UnitType.hpp>

namespace Takion::Graph
{
template <typename T>
MSELoss<T>::MSELoss(const UnitId& unitId, const UnitId& predictionUnitId,
                    const UnitId& labelUnitId, Tensor<T> predictionTensor,
                    Tensor<T> labelTensor, Tensor<T> backwardOutputTensor,
                    std::size_t batchSize)
    : ComputableUnit<T>(
          unitId,
          { { predictionUnitId, std::move(predictionTensor) },
            { labelUnitId, std::move(labelTensor) }, batchSize },
          {},
          Tensor(Shape({ 1, 1 }),
                 Compute::Device(0, Compute::DeviceType::CPU, "none")),
          { { predictionUnitId, std::move(backwardOutputTensor) } }, batchSize),
      m_predictionUnitId(predictionUnitId),
      m_labelUnitId(labelUnitId)
{
}

template <typename T>
MSELoss<T>::MSELoss(MSELoss<T>&& lossUnit) noexcept
    : ComputableUnit(std::move(lossUnit)),
      m_predictionUnitId(std::move(lossUnit.m_predictionUnitId)),
      m_labelUnitId(std::move(lossUnit.m_labelUnitId))

{
}

template <typename T>
MSELoss<T>& MSELoss<T>::operator=(MSELoss<T>&& lossUnit) noexcept
{
    ComputableUnit<T>::operator=(std::move(lossUnit));
    return *this;
}

template <typename T>
MSELoss<T> MSELoss<T>::CreateUnit(const UnitMetaData<T>& unitMetaData)

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

    return MSELoss(unitMetaData.Id(), predictionUnitId, labelUnitId,
                   predictionTensor, labelTensor, backwardOutputTensor);
}

template <typename T>
void MSELoss<T>::Forward()
{
    using ComputableUnit<T>::ForwardOutput;
    using TrainableUnit<T>::TrainableTensorMap;

    Tensor<T>& prediction =
        ComputableUnit<T>::ForwardInputMap.at(m_predictionUnitId);
    const Tensor& label = ComputableUnit<T>::ForwardInputMap.at(m_labelUnitId);
    Tensor<T>& outputTensor = ForwardOutput;
    const auto batchSize = prediction.BatchSize;

    Compute::Sub(label, prediction, outputTensor);
    Compute::Dot(outputTensor, outputTensor);
    Compute::ScalarDiv(ForwardOutput, 2 * static_cast<T>(batchSize));
}

template <typename T>
void MSELoss<T>::AsyncForward(std::promise<bool> promise)
{
    using ComputableUnit<T>::ForwardOutput;

    Tensor<T>& prediction =
        ComputableUnit<T>::ForwardInputMap.at(m_predictionUnitId);
    const Tensor& label = ComputableUnit<T>::ForwardInputMap.at(m_labelUnitId);
    Tensor<T>& outputTensor = ForwardOutput;
    const auto batchSize = prediction.BatchSize;

    Compute::Sub(label, prediction, outputTensor);
    Compute::Dot(outputTensor, outputTensor);
    Compute::ScalarDiv(ForwardOutput, 2 * static_cast<T>(batchSize));

    promise.set_value(true);
}

template <typename T>
void MSELoss<T>::Backward()
{
    using ComputableUnit<T>::ForwardOutput;
    using TrainableUnit<T>::TrainableTensorMap;

    Tensor<T>& prediction =
        ComputableUnit<T>::ForwardInputMap.at(m_predictionUnitId);
    const Tensor& label = ComputableUnit<T>::ForwardInputMap.at(m_labelUnitId);
    Tensor<T>& outputTensor = ForwardOutput;
    const auto batchSize = prediction.BatchSize;

    Compute::Sub(label, prediction, outputTensor);
    Compute::ScalarDiv(ForwardOutput, static_cast<T>(batchSize));
}

template <typename T>
void MSELoss<T>::AsyncBackward(std::promise<bool> promise)

{
    using ComputableUnit<T>::ForwardOutput;
    using TrainableUnit<T>::TrainableTensorMap;

    Tensor<T>& prediction =
        ComputableUnit<T>::ForwardInputMap.at(m_predictionUnitId);
    const Tensor& label = ComputableUnit<T>::ForwardInputMap.at(m_labelUnitId);
    Tensor<T>& outputTensor = ForwardOutput;
    const auto batchSize = prediction.BatchSize;

    Compute::Sub(label, prediction, outputTensor);
    Compute::ScalarDiv(ForwardOutput, static_cast<T>(batchSize));

    promise.set_value(true);
}
} // namespace Takion::Graph

#endif
