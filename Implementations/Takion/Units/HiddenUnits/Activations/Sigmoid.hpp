// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_GRAPH_SIGMOID_HPP
#define TAKION_GRAPH_SIGMOID_HPP

#include <Takion/Computations/GEMM/MathKernel.hpp>
#include <Takion/Units/HiddenUnits/Activations/SigmoidDecl.hpp>

namespace Takion::Graph
{
using namespace Compute;

template <typename T>
Sigmoid<T>::Sigmoid(const UnitId& unitId, UnitId sourceUnitId,
                    Tensor<T> forwardInput,
                    std::unordered_map<UnitId, Tensor<T>> backwardInputVector,
                    Tensor<T> forwardOutput, Tensor<T> backwardOutput,
                    std::unordered_map<std::string, Tensor<T>>
                    internalTensorMap,
                    std::size_t batchSize)
    : ComputableUnit<T>(unitId, { { sourceUnitId, std::move(forwardInput) } },
                        std::move(backwardInputVector),
                        std::move(forwardOutput),
                        { { sourceUnitId, std::move(backwardOutput) } },
                        std::move(internalTensorMap), batchSize),
      m_sourceUnitId(std::move(sourceUnitId))
{
}

template <typename T>
Sigmoid<T>::Sigmoid(Sigmoid<T>&& activationUnit) noexcept
    : ComputableUnit<T>(std::move(activationUnit)),
      m_sourceUnitId(std::move(activationUnit.m_sourceUnitId))
{
}

template <typename T>
Sigmoid<T>& Sigmoid<T>::operator=(Sigmoid<T>&& activationUnit) noexcept
{
    ComputableUnit<T>::operator=(std::move(activationUnit));
    return *this;
}

template <typename T>
Sigmoid<T> Sigmoid<T>::CreateUnit(const FrontEnd::UnitMetaData<T>& unitMetaData)
{
    const auto unitId = unitMetaData.Id();
    const auto batchSize = unitMetaData.BatchSize();
    const auto device = unitMetaData.Device;
    const auto inputShape = unitMetaData.GetInputShape("input");
    const auto outputShape = unitMetaData.GetOutputShape();

    Sigmoid<T>::m_checkArguments(inputShape, outputShape, unitId.UnitName);

    auto sourceUnitId = unitMetaData.GetInputUnitId("input");

    Tensor<T> forwardInputTensor(inputShape, batchSize, device);

    std::unordered_map<UnitId, Tensor<T>> backwardInputMap;
    for (const auto& backwardInputUnitId : unitMetaData.OutputUnitVector())
    {
        Tensor<T> tensor(inputShape, batchSize, device);
        backwardInputMap[backwardInputUnitId] = std::move(tensor);
    }

    Tensor<T> forwardOutputTensor(outputShape, batchSize, device);
    Tensor<T> backwardOutputTensor(inputShape, batchSize, device);
    Tensor<T> backwardTempTensor(outputShape, batchSize, device);

    auto activationUnit =
        Sigmoid<T>(unitMetaData.Id(), sourceUnitId,
                   std::move(forwardInputTensor),
                   std::move(backwardInputMap), std::move(forwardOutputTensor),
                   std::move(backwardOutputTensor),
                   { { "backwardTemp", backwardTempTensor } }, batchSize);

    return activationUnit;
}

template <typename T>
void Sigmoid<T>::Forward()
{
    const Tensor<T>& inputTensor = ForwardInputMap.at(m_sourceUnitId);

    const auto lambdaForward = [](T val)
    {
        return static_cast<T>(static_cast<T>(1) / (1 + std::exp(-val)));
    };
    Compute::Apply(inputTensor, ForwardOutput, lambdaForward);
}

template <typename T>
void Sigmoid<T>::AsyncForward(std::promise<bool> promise)
{
    const Tensor<T>& inputTensor = ForwardInputMap.at(m_sourceUnitId);

    const auto lambdaForward = [](T val)
    {
        return static_cast<T>(static_cast<T>(1) / (1 + std::exp(-val)));
    };
    Compute::Apply(inputTensor, ForwardOutput, lambdaForward);

    promise.set_value(true);
}

template <typename T>
void Sigmoid<T>::Backward()
{
    const Zeros<T> zeroInitializer;

    Tensor<T>& backwardTemp = InternalTensorMap.at("backwardTemp");
    Tensor<T>& backwardOutput = BackwardOutputMap.at(m_sourceUnitId);
    const Tensor<T>& inputTensor = ForwardInputMap.at(m_sourceUnitId);

    zeroInitializer.Initialize(backwardTemp);

    for (const auto& [unitId, tensor] : BackwardInputMap)
    {
#ifdef DEBUG
        const auto tensorSize = tensor.TensorShape.Size() * tensor.BatchSize;
        for (std::size_t i = 0; i < tensorSize; ++i)
        {
            std::cout << "backwardInput Sigmoid : " << tensor.At(i) <<
                std::endl;
        }
#endif
        Compute::Add(tensor, backwardTemp);
    }

    const auto lambdaBackward = [](T val)
    {
        return static_cast<T>(std::exp(-val) / std::pow(1 + std::exp(-val), 2));
    };

    Compute::ScalarDiv(backwardTemp, static_cast<T>(BackwardInputMap.size()));
    Compute::Apply(backwardTemp, backwardOutput, lambdaBackward);

#ifdef DEBUG
    const auto backwardTempSize =
        backwardTemp.TensorShape.Size() * backwardTemp.BatchSize;
    for (std::size_t i = 0; i < backwardTempSize; ++i)
    {
        std::cout << "backwardTemp Sigmoid : " << backwardTemp.At(i)
                  << std::endl;
    }
    const auto backwardOutputSize =
        backwardOutput.TensorShape.Size() * backwardOutput.BatchSize;
    for (std::size_t i = 0; i < backwardOutputSize; ++i)
    {
        std::cout << "backwardOutput Sigmoid : " << backwardOutput.At(i)
            << std::endl;
    }
#endif
    Compute::Dot(inputTensor, backwardOutput, backwardOutput);
}

template <typename T>
void Sigmoid<T>::AsyncBackward(std::promise<bool> promise)
{
    const Zeros<T> zeroInitializer;

    Tensor<T>& backwardTemp = InternalTensorMap.at("backwardTemp");
    Tensor<T>& backwardOutput = BackwardOutputMap.at(m_sourceUnitId);
    const Tensor<T>& inputTensor = ForwardInputMap.at(m_sourceUnitId);

    zeroInitializer.Initialize(backwardTemp);

    for (const auto& [unitId, tensor] : BackwardInputMap)
        Compute::Add(tensor, backwardTemp);

    const auto lambdaBackward = [](T val)
    {
        return static_cast<T>(std::exp(val) / (1 + exp(val)) -
                              (std::exp(val) / std::pow(1 + exp(val), 2)));
    };

    Compute::ScalarDiv(backwardTemp, static_cast<T>(BackwardInputMap.size()));
    Compute::Apply(backwardTemp, backwardOutput, lambdaBackward);
    Compute::Dot(inputTensor, backwardOutput, backwardOutput);

    promise.set_value(true);
}

template <typename T>
void Sigmoid<T>::m_checkArguments(const Shape& inputShape,
                                  const Shape& outputShape,
                                  const std::string& unitName)
{
    if (inputShape != outputShape)
    {
        const std::string errorMessage =
            std::string("Sigmoid " + unitName) +
            " - Shape mismatch between input and output." +
            " input : " + inputShape.ToString() +
            " output : " + outputShape.ToString();

        throw std::runtime_error(errorMessage);
    }
}
} // namespace Takion::Graph

#endif
