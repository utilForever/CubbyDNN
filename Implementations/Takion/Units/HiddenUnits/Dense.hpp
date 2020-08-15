// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.


#ifndef TAKION_GRAPH_DENSE_HPP
#define TAKION_GRAPH_DENSE_HPP

#include <Takion/Units/HiddenUnits/DenseDecl.hpp>
#include <Takion/Computations/GEMM/MathKernel.hpp>
#include <unordered_map>


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
                        { { sourceUnitId, std::move(backwardOutput) } }, {},
                        batchSize),
      TrainableUnit<T>(std::move(trainableUnit), std::move(optimizer)),
      m_sourceUnitId(sourceUnitId)
{
}

template <typename T>
DenseUnit<T>::DenseUnit(DenseUnit<T>&& denseUnit) noexcept
    : ComputableUnit<T>(std::move(denseUnit)),
      TrainableUnit<T>(std::move(denseUnit)),
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
    std::unique_ptr<Compute::Optimizer<T>> optimizer)
{
    const auto unitId = unitMetaData.Id();
    auto sourceUnitId = unitMetaData.GetInputUnitId("input");
    const auto batchSize = unitMetaData.BatchSize();
    const auto weightShape = unitMetaData.InternalVariableShape("weight");
    const auto biasShape = unitMetaData.InternalVariableShape("bias");
    const auto inputShape = unitMetaData.GetInputShape("input");
    const auto outputShape = unitMetaData.GetOutputShape();
    const auto weightTransposeShape = weightShape.GetTransposedShape();

    DenseUnit<T>::m_checkShape(inputShape, outputShape, weightShape, biasShape,
                               unitId.UnitName);

    const auto& weightInitializer = unitMetaData.GetInitializer("weight");
    const auto& biasInitializer = unitMetaData.GetInitializer("bias");

    Tensor<T> forwardInputTensor(unitMetaData.GetInputShape("input"),
                                 unitMetaData.BatchSize(),
                                 unitMetaData.Device);

    std::unordered_map<UnitId, Tensor<T>> backwardInputMap;

    for (const auto& outputUnitId : unitMetaData.OutputUnitVector())
    {
        Tensor<T> tensor(unitMetaData.GetOutputShape(),
                         unitMetaData.BatchSize(),
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
    Tensor<T> weightUpdateMean(weightShape, unitMetaData.Device);

    Tensor<T> bias(biasShape, unitMetaData.Device);
    Tensor<T> biasUpdate(biasShape, batchSize, unitMetaData.Device);
    Tensor<T> biasUpdateMean(biasShape, unitMetaData.Device);

    Tensor<T> delta(unitMetaData.GetOutputShape(), batchSize,
                    unitMetaData.Device);

    Tensor<T> previousInputTranspose(
        inputShape.GetTransposedShape(),
        unitMetaData.BatchSize(),
        unitMetaData.Device);

    weightInitializer->Initialize(weight);
    biasInitializer->Initialize(bias);

    std::unordered_map<std::string, Tensor<T>> trainableUnitMap = {
        { "weight", weight },
        { "weightTranspose", weightTranspose },
        { "weightUpdate", weightUpdate },
        { "weightUpdateMean", weightUpdateMean },
        { "bias", bias },
        { "biasUpdate", biasUpdate },
        { "biasUpdateMean", biasUpdateMean },
        { "delta", delta },
        { "previousInputTranspose", previousInputTranspose }
    };

    auto denseUnit = DenseUnit<T>(
        unitId, sourceUnitId, std::move(forwardInputTensor),
        std::move(backwardInputMap), std::move(forwardOutputTensor),
        std::move(backwardOutputTensor),
        std::move(trainableUnitMap),
        std::move(optimizer), batchSize);

    return denseUnit;
}

template <typename T>
void DenseUnit<T>::Forward()
{
    const Tensor<T>& input = ForwardInputMap.at(m_sourceUnitId);
    const Tensor<T>& weight = TrainableTensorMap.at("weight");
    const Tensor<T>& bias = TrainableTensorMap.at("bias");
    Tensor<T>& output = ForwardOutput;

    Compute::Multiply(input, weight, output);
    Compute::Add(bias, output, output);
}

template <typename T>
void DenseUnit<T>::AsyncForward(std::promise<bool> promise)
{
    const Tensor<T>& input = ForwardInputMap.at(m_sourceUnitId);
    const Tensor<T>& weight = TrainableTensorMap.at("weight");
    const Tensor<T>& bias = TrainableTensorMap.at("bias");
    Tensor<T>& output = ForwardOutput;

    Compute::Multiply(input, weight, output);
    Compute::Add(bias, output, output);

    promise.set_value(true);
}

template <typename T>
void DenseUnit<T>::Backward()
{
    Tensor<T>& weight = TrainableTensorMap.at("weight");
    Tensor<T>& weightTranspose = TrainableTensorMap.at("weightTranspose");
    Tensor<T>& weightUpdate = TrainableTensorMap.at("weightUpdate");
    Tensor<T>& weightUpdateMean = TrainableTensorMap.at("weightUpdateMean");

    Tensor<T>& bias = TrainableTensorMap.at("bias");
    Tensor<T>& biasUpdateMean = TrainableTensorMap.at("biasUpdateMean");

    Tensor<T>& delta = TrainableTensorMap.at("delta");

    Tensor<T>& previousInputTranspose =
        TrainableTensorMap.at("previousInputTranspose");

    Tensor<T>& previousForwardInput = ForwardInputMap.at(m_sourceUnitId);
    Tensor<T>& backwardOutput = BackwardOutputMap.at(m_sourceUnitId);

    const Compute::Zeros<T> zeroInitializer;
    zeroInitializer.Initialize(delta);

    for (auto& [unitId, gradient] : BackwardInputMap)
        Compute::Add(delta, gradient);

    Compute::ScalarDiv(delta, static_cast<T>(BackwardInputMap.size()));
    Compute::Transpose(weight, weightTranspose);
    Compute::Multiply(delta, weightTranspose, backwardOutput);

    Compute::Transpose(previousForwardInput, previousInputTranspose);
    Compute::Multiply(previousInputTranspose, delta, weightUpdate);

    Compute::Shrink(weightUpdate, weightUpdateMean);
    Compute::Shrink(delta, biasUpdateMean);

    m_optimizer->Optimize(weight, weightUpdateMean);
    m_optimizer->Optimize(bias, biasUpdateMean);
}

template <typename T>
void DenseUnit<T>::AsyncBackward(std::promise<bool> promise)
{
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

    const Compute::Zeros<T> zeroInitializer;
    zeroInitializer.Initialize(delta);

    for (auto& [unitId, gradient] : BackwardInputMap)
        Compute::Add(delta, gradient);

    Compute::ScalarDiv(delta, static_cast<T>(BackwardInputMap.size()));
    Compute::Transpose(weight, weightTranspose);
    Compute::Multiply(delta, weightTranspose, backwardOutput);

    Compute::Transpose(previousForwardInput, previousInputTranspose);
    Compute::Multiply(previousInputTranspose, delta, weightUpdate);

    Compute::Shrink(weightUpdate, weightUpdateMean);
    Compute::Shrink(delta, biasUpdateMean);

    m_optimizer->Optimize(weight, weightUpdateMean);
    m_optimizer->Optimize(bias, biasUpdateMean);

    promise.set_value(true);
}

template <typename T>
void DenseUnit<T>::m_checkShape(const Shape& inputShape,
                                const Shape& outputShape,
                                const Shape& weightShape,
                                const Shape& biasShape,
                                const std::string& unitName)
{
    if (inputShape.Dim() != 1 || outputShape.Dim() != 1)
    {
        const std::string errorMessage =
            std::string("Dense ") + unitName +
            " - input and output shape should be 1 dimensional "
            "tensor" +
            "Given input shape : " + inputShape.ToString() +
            "Given output shape : " + outputShape.ToString();

        throw std::runtime_error(errorMessage);
    }

    if (weightShape.Dim() != 2)
    {
        const std::string errorMessage =
            std::string("Dense ") + unitName +
            " - Weight should be 2 dimensional tensor" +
            "Given weight shape : " + weightShape.ToString();
        throw std::runtime_error(errorMessage);
    }

    if (biasShape.Dim() != 1)
    {
        const std::string errorMessage =
            std::string("Dense ") + unitName +
            " - Bias should be 1 dimensional tensor" +
            "Given bias shape : " + biasShape.ToString();
        throw std::runtime_error(errorMessage);
    }

    if (weightShape.NumRow() != inputShape.NumCol())
        throw std::runtime_error("Shape mismatch");

    if (weightShape.NumCol() != biasShape.NumCol())
    {
        const std::string errorMessage =
            std::string("Dense ") + unitName +
            " - Shape mismatch between weight and bias" +
            "Given bias shape : " + biasShape.ToString() +
            "Given weight shape : " + weightShape.ToString();

        throw std::runtime_error(errorMessage);
    }

    if (weightShape.NumCol() != outputShape.NumCol())
    {
        const std::string errorMessage =
            std::string("Dense ") + unitName + " - output shape mismatch" +
            "Given weight shape : " + weightShape.ToString() +
            " Given output shape" + outputShape.ToString();

        throw std::runtime_error(errorMessage);
    }
}
} // namespace Takion::Graph

#endif
