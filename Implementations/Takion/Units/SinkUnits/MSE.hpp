// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_GRAPH_MSE_HPP
#define TAKION_GRAPH_MSE_HPP

#include <Takion/Units/SinkUnits/MSEDecl.hpp>
#include <Takion/Units/UnitType.hpp>

namespace Takion::Graph
{
template <typename T>
MSELoss<T>::MSELoss(const UnitId& unitId, const UnitId& predictionUnitId,
                    const UnitId& labelUnitId, Tensor<T> predictionTensor,
                    Tensor<T> labelTensor, Tensor<T> backwardOutputTensor,
                    Tensor<T> outputTensor,
                    std::size_t batchSize)
    : ComputableUnit<T>(
          unitId,
          { { predictionUnitId, predictionTensor },
            { labelUnitId, labelTensor } },
          {},
          outputTensor,
          { { predictionUnitId, backwardOutputTensor } },
          {}, batchSize),
      m_predictionUnitId(predictionUnitId),
      m_labelUnitId(labelUnitId)
{
}

template <typename T>
MSELoss<T>::MSELoss(MSELoss<T>&& lossUnit) noexcept
    : ComputableUnit<T>(std::move(lossUnit)),
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
MSELoss<T> MSELoss<T>::CreateUnit(const FrontEnd::UnitMetaData<T>& unitMetaData)

{
    const auto unitId = unitMetaData.Id();
    const auto predictionUnitId = unitMetaData.GetInputUnitId("prediction");
    const auto labelUnitId = unitMetaData.GetInputUnitId("label");

    const auto predictionShape = unitMetaData.GetInputShape("prediction");
    const auto labelShape = unitMetaData.GetInputShape("label");
    const auto batchSize = unitMetaData.BatchSize();
    const auto device = unitMetaData.Device;

    MSELoss<T>::m_checkArguments(predictionShape, labelShape, unitId.UnitName);

    auto predictionTensor =
        Tensor<T>(predictionShape, batchSize, device);
    auto labelTensor = Tensor<T>(labelShape, batchSize, device);
    auto backwardOutputTensor = Tensor<T>(predictionShape, batchSize, device);
    auto outputTensor = Tensor<T>(predictionShape, batchSize, device);

    return MSELoss<T>(unitMetaData.Id(), predictionUnitId, labelUnitId,
                      predictionTensor, labelTensor, backwardOutputTensor,
                      outputTensor,
                      batchSize);
}

template <typename T>
void MSELoss<T>::Forward()
{
    const auto batchSize = ComputableUnit<T>::BatchSize;
    Tensor<T>& prediction =
        ComputableUnit<T>::ForwardInputMap.at(m_predictionUnitId);
    const Tensor<T>& label =
        ComputableUnit<T>::ForwardInputMap.at(m_labelUnitId);
    Tensor<T>& outputTensor = ForwardOutput;

    Compute::Sub(label, prediction, outputTensor);
    Compute::Dot(outputTensor, outputTensor);
    Compute::ScalarDiv(outputTensor, static_cast<T>(2));

    const auto size = ForwardOutput.TensorShape.Size() * batchSize;
    T sum = static_cast<T>(0);
    for (std::size_t i = 0; i < size; ++i)
    {
        const auto output = outputTensor.At(i);
        sum += output;
    }

    m_loss = sum / static_cast<T>(batchSize);
}

template <typename T>
void MSELoss<T>::AsyncForward(std::promise<bool> promise)
{
    const auto batchSize = ComputableUnit<T>::BatchSize;
    Tensor<T>& prediction =
        ComputableUnit<T>::ForwardInputMap.at(m_predictionUnitId);
    const Tensor<T>& label =
        ComputableUnit<T>::ForwardInputMap.at(m_labelUnitId);
    Tensor<T>& outputTensor = ForwardOutput;

    Compute::Sub(label, prediction, outputTensor);
    Compute::Dot(outputTensor, outputTensor);
    Compute::ScalarDiv(outputTensor, static_cast<T>(2));

    const auto size = ForwardOutput.TensorShape.Size() * batchSize;
    T sum = static_cast<T>(0);
    for (std::size_t i = 0; i < size; ++i)
    {
        const auto output = outputTensor.At(i);
        sum += output;
    }

    m_loss = sum / static_cast<T>(batchSize);
    std::cout << "Loss : " << m_loss << std::endl;

    promise.set_value(true);
}

template <typename T>
void MSELoss<T>::Backward()
{
    Tensor<T>& prediction =
        ComputableUnit<T>::ForwardInputMap.at(m_predictionUnitId);
    const Tensor<T>& label = ComputableUnit<T>::ForwardInputMap.at(
        m_labelUnitId);
    Tensor<T>& outputTensor = BackwardOutputMap[m_predictionUnitId];

    Compute::Sub(label, prediction, outputTensor);
}

template <typename T>
void MSELoss<T>::AsyncBackward(std::promise<bool> promise)

{
    Tensor<T>& prediction =
        ComputableUnit<T>::ForwardInputMap.at(m_predictionUnitId);
    const Tensor<T>& label = ComputableUnit<T>::ForwardInputMap.at(
        m_labelUnitId);
    Tensor<T>& outputTensor = ForwardOutput;

    Compute::Sub(label, prediction, outputTensor);

    promise.set_value(true);
}

template <typename T>
void MSELoss<T>::m_checkArguments(const Shape& predictionShape,
                                  const Shape& labelShape,
                                  const std::string& unitName)
{
    if (predictionShape != labelShape)
    {
        const std::string errorMessage =
            std::string("MSELoss ") + unitName +
            " - prediction and label shape mismatch. " +
            "prediction : " + predictionShape.ToString() +
            " label : " + labelShape.ToString();

        throw std::runtime_error(errorMessage);
    }
}
} // namespace Takion::Graph

#endif
