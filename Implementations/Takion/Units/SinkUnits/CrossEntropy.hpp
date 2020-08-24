// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties

#ifndef TAKION_GRAPH_CROSSENTROPY_HPP
#define TAKION_GRAPH_CROSSENTROPY_HPP

#include <Takion/Units/SinkUnits/CrossEntropyDecl.hpp>
#include <Takion/Units/UnitType.hpp>
#include <iostream>

namespace Takion::Graph
{
template <typename T>
CrossEntropy<T>::CrossEntropy(const UnitId& unitId,
                              const UnitId& predictionUnitId,
                              const UnitId& labelUnitId,
                              Tensor<T> predictionTensor,
                              Tensor<T> labelTensor,
                              Tensor<T> backwardOutputTensor,
                              Tensor<T> outputTensor, Compute::Device device,
                              std::size_t batchSize)
    : ComputableUnit<T>(
          unitId,
          { { predictionUnitId, predictionTensor },
            { labelUnitId, labelTensor } },
          {}, outputTensor,
          { { predictionUnitId, backwardOutputTensor } }, {},
          batchSize),
      m_predictionUnitId(predictionUnitId),
      m_labelUnitId(labelUnitId),
      m_device(std::move(device))
{
}

template <typename T>
CrossEntropy<T>::CrossEntropy(CrossEntropy<T>&& lossUnit) noexcept
    : ComputableUnit<T>(std::move(lossUnit)),
      m_predictionUnitId(std::move(lossUnit.m_predictionUnitId)),
      m_labelUnitId(std::move(lossUnit.m_labelUnitId))

{
}

template <typename T>
CrossEntropy<T>& CrossEntropy<T>::operator=(CrossEntropy<T>&& lossUnit) noexcept
{
    ComputableUnit<T>::operator=(std::move(lossUnit));
    return *this;
}

template <typename T>
CrossEntropy<T> CrossEntropy<T>::CreateUnit(
    const FrontEnd::UnitMetaData<T>& unitMetaData)

{
    const auto unitId = unitMetaData.Id();
    const auto predictionUnitId = unitMetaData.GetInputUnitId("prediction");
    const auto labelUnitId = unitMetaData.GetInputUnitId("label");

    const auto predictionShape = unitMetaData.GetInputShape("prediction");
    const auto labelShape = unitMetaData.GetInputShape("label");
    const auto batchSize = unitMetaData.BatchSize();
    const auto device = unitMetaData.Device;

    CrossEntropy<T>::m_checkArguments(predictionShape, labelShape,
                                      unitId.UnitName);

    auto predictionTensor = Tensor<T>(predictionShape, batchSize, device);
    auto labelTensor = Tensor<T>(labelShape, batchSize, device);
    auto backwardOutputTensor = Tensor<T>(predictionShape, batchSize, device);
    auto outputTensor = Tensor<T>(predictionShape, batchSize, device);

    return CrossEntropy<T>(unitMetaData.Id(), predictionUnitId, labelUnitId,
                           predictionTensor, labelTensor, backwardOutputTensor,
                           outputTensor, device, batchSize);
}

template <typename T>
void CrossEntropy<T>::Forward()
{
    const auto batchSize = ComputableUnit<T>::BatchSize;
    Tensor<T>& prediction =
        ComputableUnit<T>::ForwardInputMap.at(m_predictionUnitId);
    const Tensor<T>& label =
        ComputableUnit<T>::ForwardInputMap.at(m_labelUnitId);

    if (m_device.Type() == Compute::DeviceType::CPU)
    {
        const auto lambda = [](T val)
        {
            return static_cast<T>(-std::log(val));
        };

        Compute::Apply(prediction, ForwardOutput, lambda);
        Compute::Dot(label, ForwardOutput, ForwardOutput);

        const auto size = ForwardOutput.TensorShape.Size() * batchSize;

        T sum = static_cast<T>(0);
        for (std::size_t i = 0; i < size; ++i)
        {
            const auto output = ForwardOutput.At(i);
            sum += output;
        }

        m_loss = sum / static_cast<T>(batchSize);
    }
}

template <typename T>
void CrossEntropy<T>::AsyncForward(std::promise<bool> promise)
{
    const auto batchSize = ComputableUnit<T>::BatchSize;
    Tensor<T>& prediction =
        ComputableUnit<T>::ForwardInputMap.at(m_predictionUnitId);
    const Tensor<T>& label =
        ComputableUnit<T>::ForwardInputMap.at(m_labelUnitId);

    if (m_device.Type() == Compute::DeviceType::CPU)
    {
        const auto lambda = [](T val)
        {
            return static_cast<T>(-std::log(val));
        };

        Compute::Apply(prediction, ForwardOutput, lambda);
        Compute::Dot(label, ForwardOutput, ForwardOutput);

        const auto size = ForwardOutput.TensorShape.Size() * batchSize;

        T sum = static_cast<T>(0);
        for (std::size_t i = 0; i < size; ++i)
        {
            const auto output = ForwardOutput.At(i);
            sum += output;
        }

        m_loss = sum / static_cast<T>(batchSize);

        promise.set_value(true);
    }
}

template <typename T>
void CrossEntropy<T>::Backward()
{
    Tensor<T>& prediction =
        ComputableUnit<T>::ForwardInputMap.at(m_predictionUnitId);
    const Tensor<T>& label =
        ComputableUnit<T>::ForwardInputMap.at(m_labelUnitId);
    Tensor<T>& backwardOutput = BackwardOutputMap[m_predictionUnitId];

    //Compute::ScalarMul(label, static_cast<T>(-1));
    Compute::Div(label, prediction, backwardOutput);
}

template <typename T>
void CrossEntropy<T>::AsyncBackward(std::promise<bool> promise)

{
    Tensor<T>& prediction =
        ComputableUnit<T>::ForwardInputMap.at(m_predictionUnitId);
    const Tensor<T>& label =
        ComputableUnit<T>::ForwardInputMap.at(m_labelUnitId);
    Tensor<T>& backwardOutput = BackwardOutputMap[m_predictionUnitId];

    //Compute::ScalarMul(label, static_cast<T>(-1));
    Compute::Div(label, prediction, backwardOutput);

    promise.set_value(true);
}

template <typename T>
void CrossEntropy<T>::m_checkArguments(const Shape& predictionShape,
                                       const Shape& labelShape,
                                       const std::string& unitName)
{
    if (predictionShape != labelShape)
    {
        const std::string errorMessage =
            std::string("CrossEntropy ") + unitName +
            " - prediction and label shape mismatch. " +
            "prediction : " + predictionShape.ToString() +
            " label : " + labelShape.ToString();

        throw std::runtime_error(errorMessage);
    }
}
}

#endif
