// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_GRAPH_LOSSUNIT_HPP
#define TAKION_GRAPH_LOSSUNIT_HPP

#include <Takion/Units/SinkUnits/LossUnitDecl.hpp>
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
          { { predictionUnitId, std::move(predictionTensor) },
            { labelUnitId, std::move(labelTensor) } },
          {},
          std::move(outputTensor),
          { { predictionUnitId, std::move(backwardOutputTensor) } },
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
    Tensor<T>& prediction =
        ComputableUnit<T>::ForwardInputMap.at(m_predictionUnitId);
    const Tensor<T>& label = ComputableUnit<T>::ForwardInputMap.at(
        m_labelUnitId);
    Tensor<T>& outputTensor = ForwardOutput;
    const auto batchSize = prediction.BatchSize;

    Compute::Sub(label, prediction, outputTensor);
    Compute::Dot(outputTensor, outputTensor);
    Compute::ScalarDiv(outputTensor, static_cast<T>(2));

    const auto size = ForwardOutput.TensorShape.Size() * ForwardOutput.
                      BatchSize;
    T sum = static_cast<T>(0);
    for (std::size_t i = 0; i < size; ++i)
    {
        sum += outputTensor.At(i);
    }

    sum /= size;
    m_loss = sum;
    std::cout << "Loss : " << m_loss << std::endl;
}

template <typename T>
void MSELoss<T>::AsyncForward(std::promise<bool> promise)
{
    Tensor<T>& prediction =
        ComputableUnit<T>::ForwardInputMap.at(m_predictionUnitId);
    const Tensor<T>& label =
        ComputableUnit<T>::ForwardInputMap.at(m_labelUnitId);
    Tensor<T>& outputTensor = ForwardOutput;
    const auto batchSize = prediction.BatchSize;

    Compute::Sub(label, prediction, outputTensor);
    Compute::Dot(outputTensor, outputTensor);
    Compute::ScalarDiv(outputTensor, static_cast<T>(2));

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

#ifdef DEBUG

    const auto predictionSize =
        prediction.TensorShape.Size() * prediction.BatchSize;

    for (std::size_t i = 0; i < predictionSize; ++i)
    {
        std::cout << "prediction MSELoss : " << prediction.At(i) << std::endl;
    }

    const auto outputTensorSize =
        outputTensor.TensorShape.Size() * outputTensor.BatchSize;

    for (std::size_t i = 0; i < outputTensorSize; ++i)
    {
        std::cout << "output MSELoss: " << outputTensor.At(i) << std::endl;
    }
#endif
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
