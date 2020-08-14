// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_GRAPH_ACTIVATIONUNIT_HPP
#define TAKION_GRAPH_ACTIVATIONUNIT_HPP

#include <Takion/Computations/GEMM/MathKernel.hpp>
#include <Takion/Units/HiddenUnits/Activations/ActivationUnitDecl.hpp>

namespace Takion::Graph
{
using namespace Compute;

template <typename T>
ReLU<T>::ReLU(
    const UnitId& unitId, const UnitId& sourceUnitId, Tensor<T> forwardInput,
    std::unordered_map<UnitId, Tensor<T>> backwardInputVector,
    Tensor<T> forwardOutput, Tensor<T> backwardOutput,
    std::unordered_map<std::string, Tensor<T>> trainableUnit,
    std::size_t batchSize)
    : ComputableUnit(unitId,
                     { { sourceUnitId, std::move(forwardInput) } },
                     std::move(backwardInputVector), std::move(forwardOutput),
                     { { sourceUnitId, std::move(backwardOutput) } },
                     batchSize),
      TrainableUnit(std::move(trainableUnit)),
      m_sourceUnitId(sourceUnitId)
{
}

template <typename T>
ReLU<T>::ReLU(ReLU<T>&& activationUnit) noexcept
    : ComputableUnit(std::move(activationUnit)),
      TrainableUnit(std::move(activationUnit)),
      m_sourceUnitId(std::move(activationUnit.m_sourceUnitId))
{
}

template <typename T>
ReLU<T>& ReLU<T>::operator=(
    ReLU<T>&& activationUnit) noexcept
{
    ComputableUnit<T>::operator=(std::move(activationUnit));
    return *this;
}

template <typename T>
ReLU ReLU<T>::CreateUnit(
    const FrontEnd::UnitMetaData<T>& unitMetaData)
{
    const auto unitId = unitMetaData.Id();
    const auto batchSize = unitMetaData.BatchSize();
    const auto device = unitMetaData.Device;
    const auto inputShape = unitMetaData.GetInputShape("input");
    const auto outputShape = unitMetaData.GetOutputShape();

    ReLU<T>::m_checkArguments(inputShape, outputShape, unitId.UnitName);

    auto sourceUnitId = unitMetaData.GetInputUnitId("input");

    Tensor<T> forwardInputTensor(unitMetaData.GetInputShape("input"),
                                 unitMetaData.Device, unitMetaData.NumericType);

    std::unordered_map<UnitId, Tensor<T>> backwardInputMap;
    for (const auto& backwardInputUnitId : unitMetaData.OutputUnitVector())
    {
        Tensor<T> tensor(inputShape, batchSize, device);
        backwardInputMap[backwardInputUnitId] = std::move(tensor);
    }

    Tensor<T> forwardOutputTensor(outputShape, batchSize, device);
    Tensor<T> backwardOutputTensor(inputShape, batchSize, device);
    Tensor<T> backwardTempTensor(outputShape, batchSize, device);

    auto activationUnit = ReLU<T>(
        unitMetaData.Id(), sourceUnitId, std::move(forwardInputTensor),
        std::move(backwardInputMap), std::move(forwardOutputTensor),
        std::move(backwardOutputTensor),
        { { "backwardTemp", backwardTempTensor } }, batchSize);

    return activationUnit;
}

template <typename T>
void ReLU<T>::Forward()
{
    using ComputableUnit<T>::ForwardInputMap;
    using ComputableUnit<T>::ForwardOutput;
    const Tensor<T>& inputTensor = ForwardInputMap.at(m_sourceUnitId);

    const auto lambdaForward = [](T val) { return val > 0 ? val : 0; };
    Compute::Apply(inputTensor, ForwardOutput, lambdaForward);
}

template <typename T>
void ReLU<T>::AsyncForward(std::promise<bool> promise)
{
    using ComputableUnit<T>::ForwardInputMap;
    using ComputableUnit<T>::ForwardOutput;
    const Tensor<T>& inputTensor = ForwardInputMap.at(m_sourceUnitId);

    const auto lambdaForward = [](T val) { return val > 0 ? val : 0; };
    Compute::Apply(inputTensor, ForwardOutput, lambdaForward);

    promise.set_value(true);
}

template <typename T>
void ReLU<T>::Backward()
{
    using ComputableUnit<T>::BackwardInputMap;
    using ComputableUnit<T>::BackwardOutputMap;
    using ComputableUnit<T>::ForwardInputMap;
    using TrainableUnit<T>::m_trainableTensorMap;
    const Zeros zeroInitializer;

    Tensor<T>& backwardTemp =
        m_trainableTensorMap.at("backwardTemp");
    Tensor<T>& backwardOutput =
        BackwardOutputMap.at(m_sourceUnitId);
    const Tensor<T>& inputTensor = ForwardInputMap.at(m_sourceUnitId);

    const auto batchSize = backwardTemp.BatchSize;

    zeroInitializer.Initialize(backwardTemp);

    for (const auto& [unitId, tensor] : BackwardInputMap)
        Compute::Add(tensor, backwardTemp);

    const auto lambdaBackward = [](T val) { return val > 0 ? 1 : 0; };

    Compute::ScalarDiv(backwardTemp, static_cast<T>(BackwardInputMap.size()));
    Compute::Apply(backwardTemp, backwardOutput, lambdaBackward);
    Compute::Dot(inputTensor, backwardOutput, backwardOutput);
}

template <typename T>
void ReLU<T>::AsyncBackward(std::promise<bool> promise)
{
    using ComputableUnit<T>::BackwardInputMap;
    using ComputableUnit<T>::BackwardOutputMap;
    using ComputableUnit<T>::ForwardInputMap;
    using TrainableUnit<T>::m_trainableTensorMap;
    const Zeros zeroInitializer;

    Tensor<T>& backwardTemp = m_trainableTensorMap.at("backwardTemp");
    Tensor<T>& backwardOutput = BackwardOutputMap.at(m_sourceUnitId);
    const Tensor<T>& inputTensor = ForwardInputMap.at(m_sourceUnitId);

    const auto batchSize = backwardTemp.BatchSize;

    zeroInitializer.Initialize(backwardTemp);

    for (const auto& [unitId, tensor] : BackwardInputMap)
        Compute::Add(tensor, backwardTemp);

    const auto lambdaBackward = [](T val) { return val > 0 ? 1 : 0; };

    Compute::ScalarDiv(backwardTemp, static_cast<T>(BackwardInputMap.size()));
    Compute::Apply(backwardTemp, backwardOutput, lambdaBackward);
    Compute::Dot(inputTensor, backwardOutput, backwardOutput);

    promise.set_value(true);
}

template <typename T>
void ReLU<T>::m_checkArguments(const Shape& inputShape,
                               const Shape& outputShape,
                               const std::string& unitName)
{
    if (inputShape != outputShape)
    {
        const std::string errorMessage =
            std::string("ReLU " + unitName) +
            " - Shape mismatch between input and output." +
            " input : " + inputShape.ToString() +
            " output : " + outputShape.ToString();

        throw std::runtime_error(errorMessage);
    }
}
} // namespace Takion::Graph

#endif
