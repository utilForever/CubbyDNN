// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_GRAPH_DECL_HPP
#define TAKION_GRAPH_DECL_HPP

#include <Takion/Computations/GEMM/MathKernel.hpp>
#include <Takion/Units/HiddenUnits/Activations/SoftMaxDecl.hpp>
#include <algorithm>

namespace Takion::Graph
{
template <typename T>
SoftMax<T>::SoftMax(const UnitId& unitId, UnitId sourceUnitId,
                    Tensor<T> forwardInput,
                    std::unordered_map<UnitId, Tensor<T>> backwardInputVector,
                    Tensor<T> forwardOutput, Tensor<T> backwardOutput,
                    std::unordered_map<std::string, Tensor<T>>
                    internalTensorMap, Compute::Device device,
                    std::size_t batchSize)
    : ComputableUnit<T>(unitId, { { sourceUnitId, forwardInput } },
                        std::move(backwardInputVector),
                        forwardOutput,
                        { { sourceUnitId, std::move(backwardOutput) } },
                        std::move(internalTensorMap), batchSize),
      m_sourceUnitId(std::move(sourceUnitId)),
      m_device(std::move(device))
{
}

template <typename T>
SoftMax<T> SoftMax<T>::CreateUnit(const FrontEnd::UnitMetaData<T>& unitMetaData)
{
    const auto unitId = unitMetaData.Id();
    const auto batchSize = unitMetaData.BatchSize();
    const auto device = unitMetaData.Device;
    const auto inputShape = unitMetaData.GetInputShape("input");
    const auto outputShape = unitMetaData.GetOutputShape();

    SoftMax<T>::m_checkArguments(inputShape, outputShape, unitId.UnitName);

    auto sourceUnitId = unitMetaData.GetInputUnitId("input");

    Tensor<T> forwardInputTensor(inputShape, batchSize, device);

    std::unordered_map<UnitId, Tensor<T>> backwardInputMap;
    for (const auto& backwardInputUnitId : unitMetaData.OutputUnitVector())
    {
        Tensor<T> tensor(inputShape, batchSize, device);
        backwardInputMap[backwardInputUnitId] = tensor;
    }

    Tensor<T> forwardOutputTensor(outputShape, batchSize, device);
    Tensor<T> backwardOutputTensor(inputShape, batchSize, device);
    Tensor<T> backwardTempTensor(outputShape, batchSize, device);

    auto activationUnit =
        SoftMax<T>(unitMetaData.Id(), sourceUnitId,
                   forwardInputTensor,
                   backwardInputMap, forwardOutputTensor,
                   backwardOutputTensor,
                   { { "backwardTemp", backwardTempTensor } }, device,
                   batchSize);

    return activationUnit;
}

template <typename T>
void SoftMax<T>::Forward()
{
    const auto batchSize = ComputableUnit<T>::BatchSize;
    const auto shape = ForwardOutput.TensorShape;
    const auto size = shape.Size();
    const Tensor<T>& inputTensor = ForwardInputMap[m_sourceUnitId];

    if (m_device.Type() == Compute::DeviceType::CPU)
    {
#pragma omp parallel for schedule(static)
        for (long batchIdx = 0; batchIdx < static_cast<long>(batchSize);
             ++batchIdx)
        {
            T sum = static_cast<T>(0);
            for (std::size_t idx = 0; idx < size; ++idx)
            {
                const auto index = batchIdx * size + idx;
                const auto val = inputTensor.At(index);
                T toAdd = static_cast<T>(std::exp(val));
                sum = sum + toAdd;
            }
            for (std::size_t idx = 0; idx < size; ++idx)
            {
                const auto index = batchIdx * size + idx;
                const auto val = inputTensor.At(index);
                ForwardOutput.At(index) =
                    static_cast<T>(std::exp(val)) / sum;
            }
        }
    }
    else
    {
        throw std::runtime_error("Not implemented");
    }
}

template <typename T>
void SoftMax<T>::AsyncForward(std::promise<bool> promise)
{
    const auto batchSize = ComputableUnit<T>::BatchSize;
    const auto shape = ForwardOutput.TensorShape;
    const auto size = shape.Size();
    const Tensor<T>& inputTensor = ForwardInputMap[m_sourceUnitId];

    if (m_device.Type() == Compute::DeviceType::CPU)
    {
#pragma omp parallel for schedule(static)
        for (long batchIdx = 0; batchIdx < static_cast<long>(batchSize);
             ++batchIdx)
        {
            T sum = static_cast<T>(0);
            for (std::size_t idx = 0; idx < size; ++idx)
            {
                const auto index = batchIdx * size + idx;
                const auto val = inputTensor.At(index);
                T toAdd = std::min(static_cast<T>(std::exp(val)),
                                   std::numeric_limits<T>::max());
                sum = std::min(sum + toAdd, std::numeric_limits<T>::max());
            }
            for (std::size_t idx = 0; idx < size; ++idx)
            {
                const auto index = batchIdx * size + idx;
                const auto val = inputTensor.At(index);
                ForwardOutput.At(index) =
                    std::min(static_cast<T>(std::exp(val)),
                             std::numeric_limits<T>::max()) / sum;
            }
        }
    }
    else
    {
        throw std::runtime_error("Not implemented");
    }

    promise.set_value(true);
}

template <typename T>
void SoftMax<T>::Backward()
{
    const Zeros<T> zeroInitializer;

    const auto batchSize = ComputableUnit<T>::BatchSize;
    const auto shape = ForwardOutput.TensorShape;
    const auto size = shape.Size();
    Tensor<T>& backwardTemp = InternalTensorMap["backwardTemp"];
    Tensor<T>& backwardOutput = BackwardOutputMap[m_sourceUnitId];

    zeroInitializer.Initialize(backwardTemp);

    for (const auto& [unitId, tensor] : BackwardInputMap)
    {
        Compute::Add(tensor, backwardTemp);
    }

    if (m_device.Type() == Compute::DeviceType::CPU)
    {
        for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            for (std::size_t idxOut = 0; idxOut < size; ++idxOut)
            {
                const auto backwardOutputIdx = batchIdx * size + idxOut;
                T sum = static_cast<T>(0);

                for (std::size_t idxIn = 0; idxIn < size; ++idxIn)
                {
                    const auto backwardInputIdx = batchIdx * size + idxIn;

                    if (idxIn == idxOut)
                    {
                        const auto prevOutput =
                            ForwardOutput.At(backwardInputIdx);
                        const auto derivative = prevOutput * (1 - prevOutput);
                        const auto val =
                            backwardTemp.At(backwardInputIdx) * derivative;
                        sum += val;
                    }
                    else
                    {
                        const auto prevOutIn =
                            ForwardOutput.At(backwardInputIdx);
                        const auto prevOutOut =
                            ForwardOutput.At(backwardOutputIdx);

                        const auto derivative = -prevOutIn * prevOutOut;
                        const auto val =
                            backwardTemp.At(backwardInputIdx) * derivative;
                        sum += val;
                    }
                }
                backwardOutput.At(backwardOutputIdx) = sum;
            }
        }
    }
    else
    {
        throw std::runtime_error("Not implemented");
    }
}

template <typename T>
void SoftMax<T>::AsyncBackward(std::promise<bool> promise)
{
    const Zeros<T> zeroInitializer;

    const auto batchSize = ComputableUnit<T>::BatchSize;
    const auto shape = ForwardOutput.TensorShape;
    const auto size = shape.Size();
    Tensor<T>& backwardTemp = InternalTensorMap["backwardTemp"];
    Tensor<T>& backwardOutput = BackwardOutputMap[m_sourceUnitId];

    zeroInitializer.Initialize(backwardTemp);

    for (const auto& [unitId, tensor] : BackwardInputMap)
    {
        Compute::Add(tensor, backwardTemp);
    }

    if (m_device.Type() == Compute::DeviceType::CPU)
    {
        for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            for (std::size_t idxOut = 0; idxOut < size; ++idxOut)
            {
                const auto backwardOutputIdx = batchIdx * size + idxOut;
                T sum = static_cast<T>(0);

                for (std::size_t idxIn = 0; idxIn < size; ++idxIn)
                {
                    const auto backwardInputIdx = batchIdx * size + idxIn;

                    if (idxIn == idxOut)
                    {
                        const auto prevOutput =
                            ForwardOutput.At(backwardInputIdx);
                        const auto derivative = prevOutput * (1 - prevOutput);
                        const auto val =
                            backwardTemp.At(backwardInputIdx) * derivative;
                        sum += val;
                    }
                    else
                    {
                        const auto prevOutIn =
                            ForwardOutput.At(backwardInputIdx);
                        const auto prevOutOut =
                            ForwardOutput.At(backwardOutputIdx);

                        const auto derivative = -prevOutIn * prevOutOut;
                        const auto val =
                            backwardTemp.At(backwardInputIdx) * derivative;
                        sum += val;
                    }
                }
                backwardOutput.At(backwardOutputIdx) = sum;
            }
        }
    }
    else
    {
        throw std::runtime_error("Not implemented");
    }

    promise.set_value(true);
}

template <typename T>
void SoftMax<T>::ChangeBatchSize(std::size_t batchSize)
{
    ComputableUnit<T>::ChangeBatchSize(batchSize);
    Tensor<T>& backwardTemp = InternalTensorMap.at("backwardTemp");
    backwardTemp.ChangeBatchSize(batchSize);
}

template <typename T>
void SoftMax<T>::m_checkArguments(const Shape& inputShape,
                                  const Shape& outputShape,
                                  const std::string& unitName)
{
    if (inputShape != outputShape)
    {
        const std::string errorMessage =
            std::string("SoftMax " + unitName) +
            " - Shape mismatch between input and output." +
            " input : " + inputShape.ToString() +
            " output : " + outputShape.ToString();

        throw std::runtime_error(errorMessage);
    }
}
}

#endif
