// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.


#ifndef TAKION_GRAPH_DENSE_HPP
#define TAKION_GRAPH_DENSE_HPP

#include <Takion/Units/HiddenUnits/DenseDecl.hpp>
#include <Takion/Computations/GEMM/MathKernel.hpp>

namespace Takion::Graph
{
template <typename T>
DenseUnit<T>::DenseUnit(
    const UnitId& unitId, const UnitId& sourceUnitId, Tensor<T> forwardInput,
    std::unordered_map<UnitId, Tensor<T>> backwardInputMap,
    Tensor<T> forwardOutput, Tensor<T> backwardOutput,
    std::unordered_map<std::string, Tensor<T>> trainableUnit,
    std::unique_ptr<Compute::Optimizer<T>> optimizer, std::size_t batchSize)
    : ComputableUnit<T>(unitId, { { sourceUnitId, std::move(forwardInput) } },
                        std::move(backwardInputMap), std::move(forwardOutput),
                        { { sourceUnitId, std::move(backwardOutput) } }),
      TrainableUnit<T>(std::move(trainableUnit), std::move(optimizer),
                       batchSize),
      m_sourceUnitId(sourceUnitId)
{
}

template <typename T>
DenseUnit<T>::DenseUnit(DenseUnit<T>&& denseUnit) noexcept
    : ComputableUnit(std::move(denseUnit)),
      TrainableUnit(std::move(denseUnit)),
      m_sourceUnitId(std::move(denseUnit.m_sourceUnitId))
{
}


template <typename T>
DenseUnit<T>& DenseUnit<T>::operator=(DenseUnit<T>&& denseUnit) noexcept
{
    ComputableUnit<T>::operator=(std::move(denseUnit));
    TrainableUnit<T>::operator=(std::move(denseUnit));

    return *this;
}

template <typename T>
DenseUnit<T> DenseUnit<T>::CreateUnit(
    const FrontEnd::UnitMetaData<T>& unitMetaData,
    std::unique_ptr<Compute::Optimizer<T>>
    optimizer)
{
    const auto unitId = unitMetaData.Id();
    auto sourceUnitId = unitMetaData.GetInputUnitId("input");
    const auto batchSize = unitMetaData.BatchSize();
    const auto weightShape = unitMetaData.InternalVariableShape("weight");
    const auto biasShape = unitMetaData.InternalVariableShape("bias");
    const auto inputShape = unitMetaData.GetInputShape("input");
    const auto outputShape = unitMetaData.OutputShape();
    const auto weightTransposeShape = weightShape.GetTransposedShape();

    DenseUnit<T>::m_checkShape(inputShape, outputShape, weightShape, biasShape);

    const auto& weightInitializer = unitMetaData.GetInitializer("weight");
    const auto& biasInitializer = unitMetaData.GetInitializer("bias");

    Tensor<T> forwardInputTensor(unitMetaData.GetInputShape("input"),
                                 unitMetaData.BatchSize(),
                                 unitMetaData.Device);

    std::unordered_map<UnitId, Tensor<T>> backwardInputMap;

    for (const auto& outputUnitId : unitMetaData.OutputUnitVector())
    {
        Tensor<T> tensor(unitMetaData.OutputShape(), unitMetaData.BatchSize(),
                         unitMetaData.Device);
        backwardInputMap[outputUnitId] = std::move(tensor);
    }

    Tensor<T> forwardOutputTensor(outputShape,
                                  batchSize, unitMetaData.Device);

    Tensor<T> backwardOutputTensor(inputShape,
                                   batchSize,
                                   unitMetaData.Device);

    Tensor<T> weight(weightShape, unitMetaData.Device);
    Tensor<T> weightTranspose(weightTransposeShape, unitMetaData.Device);

    Tensor<T> weightUpdate(weightShape, batchSize, unitMetaData.Device);
    Tensor<T> weightUpdateMean(weightUpdate, unitMetaData.Device);

    Tensor<T> bias(biasShape, unitMetaData.Device);
    Tensor<T> biasUpdate(biasShape, batchSize, unitMetaData.Device);
    Tensor<T> biasUpdateMean(biasShape, unitMetaData.Device);

    Tensor<T> delta(unitMetaData.OutputShape(), batchSize, unitMetaData.Device);

    Tensor<T> previousInputTranspose(
        inputShape.GetTransposedShape(),
        unitMetaData.BatchSize(),
        unitMetaData.Device);

    weightInitializer->Initialize(weight);
    biasInitializer->Initialize(bias);

    auto denseUnit = DenseUnit(
        unitId, sourceUnitId, std::move(forwardInputTensor),
        { std::move(backwardInputMap) }, std::move(forwardOutputTensor),
        std::move(backwardOutputTensor),
        { { "weight", std::move(weight) },
          { "weightTranspose", std::move(weightTranspose) },
          { "weightUpdate", std::move(weightUpdate) },
          { "weightUpdateMean:", std::move(weightUpdateMean) },
          { "bias", std::move(bias) },
          { "biasUpdate", std::move(biasUpdate) },
          { "biasUpdateMean", std::move(biasUpdateMean) },
          { "delta", std::move(delta) },
          { "previousInputTranspose", std::move(previousInputTranspose) }
        },
        std::move(optimizer), batchSize);

    return denseUnit;
}

template <typename T>
void DenseUnit<T>::Forward()
{
    using ComputableUnit<T>::ForwardInputMap;
    using ComputableUnit<T>::TrainableTensorMap;
    using ComputableUnit<T>::ForwardOutput;

    const Tensor<T>& input = ForwardInputMap.at(m_sourceUnitId);
    const Tensor<T>& weight = TrainableTensorMap.at("weight");
    const Tensor<T>& bias = TrainableTensorMap.at("bias");
    Tensor<T>& output = ForwardOutput;

    Compute::Multiply(input, weight, output);
    Compute::Add(output, bias, output);
}

template <typename T>
void DenseUnit<T>::AsyncForward(std::promise<bool> promise)
{
    using ComputableUnit<T>::ForwardInputMap;
    using ComputableUnit<T>::TrainableTensorMap;
    using ComputableUnit<T>::ForwardOutput;

    const Tensor<T>& input = ForwardInputMap.at(m_sourceUnitId);
    const Tensor<T>& weight = TrainableTensorMap.at("weight");
    const Tensor<T>& bias = TrainableTensorMap.at("bias");
    Tensor<T>& output = ForwardOutput;

    Compute::Multiply(input, weight, output);
    Compute::Add(output, bias, output);

    promise.set_value(true);
}

template <typename T>
void DenseUnit<T>::Backward()
{
    using ComputableUnit<T>::ForwardInputMap;
    using ComputableUnit<T>::TrainableTensorMap;
    using ComputableUnit<T>::ForwardOutput;
    using ComputableUnit<T>::BackwardInputMap;
    using ComputableUnit<T>::BackwardOutputMap;

    Tensor<T>& weight = TrainableTensorMap.at("weight");
    Tensor<T>& weightTranspose = TrainableTensorMap.at("weightTranspose");
    Tensor<T>& weightUpdate = TrainableTensorMap.at("weightUpdate");
    Tensor<T>& weightUpdateMean = TrainableTensorMap.at("weightUpdateMean");

    Tensor<T>& bias = TrainableTensorMap.at("bias");
    Tensor<T>& biasUpdateMean = TrainableTensorMap.at("biasUpdateMean");

    Tensor<T>& delta = TrainableTensorMap.at("delta");

    Tensor<T>& previousInputTranspose =
        TrainableTensorMap.at("PreviousInputTranspose");

    Tensor<T>& previousForwardInput = ForwardInputMap.at(m_sourceUnitId);
    Tensor<T>& backwardOutput = BackwardOutputMap.at(m_sourceUnitId);

    const auto batchSize = previousForwardInput.BatchSize;

    const Compute::Zeros<T> zeroInitializer;
    zeroInitializer.Initialize(delta);

    for (auto& [unitId, gradient] : BackwardInputMap)
        Compute::Add(delta, gradient);

    Compute::ScalarDiv(delta, static_cast<T>(BackwardInputMap.size()), delta);
    Compute::Transpose(delta, weightTranspose);
    Compute::Multiply(delta, weightTranspose, backwardOutput);

    Compute::Transpose(previousForwardInput, previousInputTranspose);
    Compute::Multiply(previousInputTranspose, delta, weightUpdate);

    Compute::Shrink(weightUpdate, weightUpdateMean);
    Compute::Shrink(delta, biasUpdateMean);

    ComputableUnit<T>::m_optimizer->Optimize(weight, weightUpdateMean);
    ComputableUnit<T>::m_optimizer->Optimize(bias, biasUpdateMean);
}

template <typename T>
void DenseUnit<T>::AsyncBackward(std::promise<bool> promise)
{
    using ComputableUnit<T>::ForwardInputMap;
    using ComputableUnit<T>::TrainableTensorMap;
    using ComputableUnit<T>::ForwardOutput;
    using ComputableUnit<T>::BackwardInputMap;
    using ComputableUnit<T>::BackwardOutputMap;

    Tensor<T>& weight = TrainableTensorMap.at("weight");
    Tensor<T>& weightTranspose = TrainableTensorMap.at("weightTranspose");
    Tensor<T>& weightUpdate = TrainableTensorMap.at("weightUpdate");
    Tensor<T>& weightUpdateMean = TrainableTensorMap.at("weightUpdateMean");

    Tensor<T>& bias = TrainableTensorMap.at("bias");
    Tensor<T>& biasUpdateMean = TrainableTensorMap.at("biasUpdateMean");

    Tensor<T>& delta = TrainableTensorMap.at("delta");

    Tensor<T>& previousInputTranspose =
        TrainableTensorMap.at("PreviousInputTranspose");

    Tensor<T>& previousForwardInput = ForwardInputMap.at(m_sourceUnitId);
    Tensor<T>& backwardOutput = BackwardOutputMap.at(m_sourceUnitId);

    const auto batchSize = previousForwardInput.BatchSize;

    const Compute::Zeros<T> zeroInitializer;
    zeroInitializer.Initialize(delta);

    for (auto& [unitId, gradient] : BackwardInputMap)
        Compute::Add(delta, gradient);

    Compute::ScalarDiv(delta, static_cast<T>(BackwardInputMap.size()), delta);
    Compute::Transpose(delta, weightTranspose);
    Compute::Multiply(delta, weightTranspose, backwardOutput);

    Compute::Transpose(previousForwardInput, previousInputTranspose);
    Compute::Multiply(previousInputTranspose, delta, weightUpdate);

    Compute::Shrink(weightUpdate, weightUpdateMean);
    Compute::Shrink(delta, biasUpdateMean);

    ComputableUnit<T>::m_optimizer->Optimize(weight, weightUpdateMean);
    ComputableUnit<T>::m_optimizer->Optimize(bias, biasUpdateMean);

    promise.set_value(true);
}

template <typename T>
void DenseUnit<T>::m_checkShape(Shape inputShape, Shape outputShape,
                                Shape weightShape, Shape biasShape)
{
    if (inputShape.Dim() != 1 || outputShape.Dim() != 1)
    {
        const std::string errorMessage =
            std::string(
                "Dense - input and output shape should be 1 dimensional "
                "tensor") +
            "Given input shape : " + inputShape.ToString() +
            "Given output shape : " + outputShape.ToString();

        throw std::runtime_error(errorMessage);
    }

    if (weightShape.Dim() != 2)
    {
        const std::string errorMessage =
            std::string("Dense - Weight should be 2 dimensional tensor") +
            "Given weight shape : " + weightShape.ToString();
        throw std::runtime_error(errorMessage);
    }

    if (biasShape.Dim() != 1)
    {
        const std::string errorMessage =
            std::string("Dense - Bias should be 1 dimensional tensor") +
            "Given bias shape : " + biasShape.ToString();
        throw std::runtime_error(errorMessage);
    }

    if (weightShape.NumRow() != inputShape.NumCol())
        throw std::runtime_error("Shape mismatch");

    if (weightShape.NumCol() != biasShape.NumCol())
    {
        const std::string errorMessage =
            std::string("Dense - Shape mismatch between weight and bias") +
            "Given bias shape : " + biasShape.ToString() +
            "Given weight shape : " + weightShape.ToString();

        throw std::runtime_error(errorMessage);
    }

    if (weightShape.NumCol() != outputShape.NumCol())
    {
        const std::string errorMessage =
            std::string("Dense - output shape mismatch") +
            "Given weight shape : " + weightShape.ToString() +
            " Given output shape" + outputShape.ToString();

        throw std::runtime_error(errorMessage);
    }
}
} // namespace Takion::Graph

#endif
