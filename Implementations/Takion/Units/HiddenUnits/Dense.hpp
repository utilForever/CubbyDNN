// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.


#ifndef TAKION_GRAPH_DENSE_HPP
#define TAKION_GRAPH_DENSE_HPP

#include <Takion/Computations/Computations.hpp>
#include <Takion/Units/HiddenComputableUnits/DenseDecl.hpp>

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
DenseUnit<T> DenseUnit<T>::CreateUnit(const UnitMetaData& unitMetaData,
                                      std::unique_ptr<Compute::Optimizer<T>>
                                      optimizer)
{
    const auto unitId = unitMetaData.Id();
    auto sourceUnitId = unitMetaData.GetInputUnitId("input");
    const auto batchSize = unitMetaData.BatchSize();
    const auto weightShape = unitMetaData.InternalVariableShape("weight");
    const auto transposedWeightShape = weightShape.Transpose();
    const auto biasShape = unitMetaData.InternalVariableShape("bias");

    const auto& weightInitializer = unitMetaData.GetInitializer("weight");
    const auto& biasInitializer = unitMetaData.GetInitializer("bias");

    Tensor forwardInputTensor(unitMetaData.GetInputShape("input"),
                              unitMetaData.BatchSize(),
                              unitMetaData.Device);

    std::unordered_map<UnitId, Tensor<T>> backwardInputMap;
    for (const auto& outputUnitId : unitMetaData.OutputUnitVector())
    {
        Tensor tensor(unitMetaData.OutputShape(), unitMetaData.BatchSize(),
                      unitMetaData.Device);
        backwardInputMap[outputUnitId] = std::move(tensor);
    }

    Tensor<T> forwardOutputTensor(unitMetaData.OutputShape(),
                                  batchSize, unitMetaData.Device);

    Tensor<T> backwardOutputTensor(unitMetaData.GetInputShape("input"),
                                   batchSize,
                                   unitMetaData.Device);

    Tensor<T> weight(weightShape, unitMetaData.Device);
    Tensor<T> weightTranspose(transposedWeightShape, unitMetaData.Device);

    Tensor<T> weightUpdate(weightShape, batchSize, unitMetaData.Device);
    Tensor<T> weightUpdateMean(weightUpdate, unitMetaData.Device);

    Tensor<T> bias(biasShape, unitMetaData.Device);
    Tensor<T> biasUpdate(biasShape, batchSize, unitMetaData.Device);
    Tensor<T> biasUpdateMean(biasShape, unitMetaData.Device);

    Tensor<T> delta(unitMetaData.OutputShape(), batchSize, unitMetaData.Device);

    Tensor<T> previousInputTranspose(
        unitMetaData.GetInputShape("input").Transpose(),
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
    using ComputableUnit<T>::m_trainableTensorMap;
    using ComputableUnit<T>::ForwardOutput;

    Tensor<T>& input = ForwardInputMap.at(m_sourceUnitId);

    Compute::Multiply(m_trainableTensorMap.at("weight"),
                      input, ForwardOutput);
    Compute::Add(ForwardOutput, m_trainableTensorMap.at("bias"),
                 ForwardOutput);
}

template <typename T>
void DenseUnit<T>::AsyncForward(std::promise<bool> promise)
{
    using ComputableUnit<T>::ForwardInputMap;
    using ComputableUnit<T>::m_trainableTensorMap;
    using ComputableUnit<T>::ForwardOutput;

    Tensor<T>& input = ForwardInputMap.at(m_sourceUnitId);

    Compute::Multiply(m_trainableTensorMap.at("weight"), input, ForwardOutput);
    Compute::Add(ForwardOutput, m_trainableTensorMap.at("bias"), ForwardOutput);

    promise.set_value(true);
}

template <typename T>
void DenseUnit<T>::Backward()
{
    using ComputableUnit<T>::ForwardInputMap;
    using ComputableUnit<T>::m_trainableTensorMap;
    using ComputableUnit<T>::ForwardOutput;
    using ComputableUnit<T>::BackwardInputMap;
    using ComputableUnit<T>::BackwardOutputMap;

    auto& weight = m_trainableTensorMap.at("weight");
    auto& weightTranspose = m_trainableTensorMap.at("weightTranspose");
    auto& weightUpdate = m_trainableTensorMap.at("weightUpdate");
    auto& weightUpdateMean = m_trainableTensorMap.at("weightUpdateMean");

    auto& bias = m_trainableTensorMap.at("bias");
    auto& biasUpdate = m_trainableTensorMap.at("biasUpdate");
    auto& biasUpdateMean = m_trainableTensorMap.at("biasUpdateMean");

    auto& delta = m_trainableTensorMap.at("delta");

    auto& previousInputTranspose =
        m_trainableTensorMap.at("PreviousInputTranspose");

    auto& previousForwardInput = ForwardInputMap.at(m_sourceUnitId);
    auto& backwardOutput = BackwardOutputMap.at(m_sourceUnitId);

    const auto batchSize = previousForwardInput.BatchSize;

    const Zeros zeroInitializer;
    zeroInitializer.Initialize(delta);

    for (auto& [unitId, gradient] : BackwardInputMap)
        Compute::Add(delta, gradient);

    Compute::ScalarDiv(delta, static_cast<T>(BackwardInputMap.size()), delta);
    Compute::Transpose(weight, weightTranspose);
    Compute::Multiply(delta, weightTranspose, backwardOutput);

    Compute::Transpose(previousForwardInput, previousInputTranspose);
    Compute::Multiply(previousInputTranspose, delta, weightUpdate);

    Compute::ScalarDiv(weightUpdate, static_cast<T>(batchSize), weightUpdate);
    Compute::ScalarDiv(biasUpdate, static_cast<T>(batchSize), biasUpdate);

    ComputableUnit<T>::m_optimizer->Optimize(weight, weightUpdate);
    ComputableUnit<T>::m_optimizer->Optimize(bias, biasUpdate);
}

template <typename T>
void DenseUnit<T>::AsyncBackward(std::promise<bool> promise)
{
    using ComputableUnit<T>::ForwardInputMap;
    using ComputableUnit<T>::m_trainableTensorMap;
    using ComputableUnit<T>::ForwardOutput;
    using ComputableUnit<T>::BackwardInputMap;
    using ComputableUnit<T>::BackwardOutputMap;

    auto& weight = m_trainableTensorMap.at("weight");
    auto& weightTranspose = m_trainableTensorMap.at("weightTranspose");
    auto& weightUpdate = m_trainableTensorMap.at("weightUpdate");
    auto& weightUpdateMean = m_trainableTensorMap.at("weightUpdateMean");

    auto& bias = m_trainableTensorMap.at("bias");
    auto& biasUpdate = m_trainableTensorMap.at("biasUpdate");
    auto& biasUpdateMean = m_trainableTensorMap.at("biasUpdateMean");

    auto& delta = m_trainableTensorMap.at("delta");

    auto& previousInputTranspose =
        m_trainableTensorMap.at("PreviousInputTranspose");

    auto& previousForwardInput = ForwardInputMap.at(m_sourceUnitId);
    auto& backwardOutput = BackwardOutputMap.at(m_sourceUnitId);

    const auto batchSize = previousForwardInput.BatchSize;

    const Zeros zeroInitializer;
    zeroInitializer.Initialize(delta);

    for (auto& [unitId, gradient] : BackwardInputMap)
        Compute::Add(delta, gradient);

    Compute::ScalarDiv(delta, static_cast<T>(BackwardInputMap.size()), delta);
    Compute::Transpose(weight, weightTranspose);
    Compute::Multiply(delta, weightTranspose, backwardOutput);

    Compute::Transpose(previousForwardInput, previousInputTranspose);
    Compute::Multiply(previousInputTranspose, delta, weightUpdate);

    Compute::ScalarDiv(weightUpdate, static_cast<T>(batchSize), weightUpdate);
    Compute::ScalarDiv(biasUpdate, static_cast<T>(batchSize), biasUpdate);

    ComputableUnit<T>::m_optimizer->Optimize(weight, weightUpdate);
    ComputableUnit<T>::m_optimizer->Optimize(bias, biasUpdate);
}
} // namespace Takion::Graph

#endif
