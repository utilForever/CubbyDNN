// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.


#ifndef TAKION_GRAPH_DENSE_HPP
#define TAKION_GRAPH_DENSE_HPP

#include <Takion/Computations/TensorOperations/Computations.hpp>
#include <Takion/Units/HiddenComputableUnits/DenseDecl.hpp>

namespace Takion::Graph
{
template <typename T>
DenseUnit<T>::DenseUnit(
    const UnitId& unitId, const UnitId& sourceUnitId, Tensor<T> forwardInput,
    std::unordered_map<UnitId, Tensor<T>> backwardInputMap,
    Tensor<T> forwardOutput, Tensor<T> backwardOutput,
    std::unordered_map<std::string, Tensor<T>> trainableUnit,
    std::unique_ptr<Compute::Optimizer<T>> optimizer)
    : ComputableUnit<T>(unitId, { { sourceUnitId, std::move(forwardInput) } },
                        std::move(backwardInputMap), std::move(forwardOutput),
                        { { sourceUnitId, std::move(backwardOutput) } }),
      TrainableUnit<T>(std::move(trainableUnit), std::move(optimizer)),
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
    const auto batchSize = unitMetaData.OutputShape().NumCols();

    Tensor forwardInputTensor(unitMetaData.GetInputShape("input"),
                              unitMetaData.Device, unitMetaData.NumericType);

    std::unordered_map<UnitId, Tensor<T>> backwardInputMap;
    for (const auto& outputUnitId : unitMetaData.OutputUnitVector())
    {
        Tensor tensor(unitMetaData.OutputShape(), unitMetaData.Device,
                      unitMetaData.NumericType);
        backwardInputMap[outputUnitId] = std::move(tensor);
    }

    Tensor forwardOutputTensor(unitMetaData.OutputShape(), unitMetaData.Device,
                               unitMetaData.NumericType);

    Tensor backwardOutputTensor(unitMetaData.GetInputShape("input"),
                                unitMetaData.Device, unitMetaData.NumericType);

    auto weightShape = unitMetaData.GetInternalVariableShape("weight");
    auto biasShape = unitMetaData.GetInternalVariableShape("bias");

    Tensor weightTensor(weightShape, unitMetaData.Device,
                        unitMetaData.NumericType);
    const auto& weightInitializer = unitMetaData.GetInitializer("weight");
    weightInitializer->Initialize(weightTensor);

    Tensor biasTensor(biasShape, unitMetaData.Device, unitMetaData.NumericType);
    Tensor biasMatrix(Shape({ biasShape.NumRows(), batchSize }),
                      unitMetaData.Device, unitMetaData.NumericType);

    const auto& biasInitializer = unitMetaData.GetInitializer("bias");
    biasInitializer->Initialize(biasTensor);

    Shape weightUpdateShape = forwardInputTensor.TensorShape;
    weightUpdateShape.SetNumRows(weightShape.NumRows());
    weightUpdateShape.SetNumCols(weightShape.NumCols());
    Tensor weightUpdate(weightUpdateShape, unitMetaData.Device,
                        unitMetaData.NumericType);

    std::vector<float> oneVector(unitMetaData.OutputShape().NumCols(), 1);

    Tensor oneVect(Shape({ batchSize, 1 }), unitMetaData.Device, oneVector);

    Tensor biasUpdate(biasShape, unitMetaData.Device, unitMetaData.NumericType);

    Tensor delta(unitMetaData.OutputShape(), unitMetaData.Device,
                 unitMetaData.NumericType);

    auto denseUnit = DenseUnit(
        unitId, sourceUnitId, std::move(forwardInputTensor),
        { std::move(backwardInputMap) }, std::move(forwardOutputTensor),
        std::move(backwardOutputTensor),
        { { "weight", std::move(weightTensor) },
          { "bias", std::move(biasTensor) },
          { "biasMatrix", std::move(biasMatrix) },
          { "weightUpdate", std::move(weightUpdate) },
          { "oneVector", std::move(oneVect) },
          { "biasUpdate", std::move(biasUpdate) },
          { "delta", delta } },
        std::move(optimizer), unitMetaData.NumericType);

    return denseUnit;
}

template <typename T>
void DenseUnit<T>::Forward()
{
    Tensor<T>& input = ComputableUnit<T>::ForwardInputMap.at(m_sourceUnitId);

    Compute::Multiply(ComputableUnit<T>::m_trainableTensorMap.at("weight"),
                      input, ComputableUnit<T>::ForwardOutput,
                      false, false, true);
    Compute::Multiply(
        m_trainableTensorMap.at("bias"), m_trainableTensorMap.at("oneVector"),
        m_trainableTensorMap.at("biasMatrix"), false, true, false);
    Compute::Add(ForwardOutput, m_trainableTensorMap.at("biasMatrix"), true);
}

template <typename T>
void DenseUnit<T>::AsyncForward(std::promise<bool> promise)
{
    const Zeros<T> zeroInitializer;
    zeroInitializer.Initialize(ForwardOutput);
    Tensor<T>& input = ForwardInputMap.at(m_sourceUnitId);

    Compute::Multiply(m_trainableTensorMap.at("weight"), input, ForwardOutput,
                      false, false, true);
    Compute::Add(ForwardOutput, m_trainableTensorMap.at("bias"), true);
    promise.set_value(true);
}

template <typename T>
void DenseUnit<T>::Backward()
{
    auto& weight = m_trainableTensorMap["weight"];
    auto& bias = m_trainableTensorMap["bias"];
    auto& weightUpdate = m_trainableTensorMap["weightUpdate"];
    auto& biasUpdate = m_trainableTensorMap["biasUpdate"];
    auto& previousForwardInput = ForwardInputMap.at(m_sourceUnitId);
    auto& backwardOutput = BackwardOutputMap.at(m_sourceUnitId);
    auto& delta = m_trainableTensorMap["delta"];
    auto& oneVector = m_trainableTensorMap["oneVector"];
    const auto batchSize = previousForwardInput.TensorShape.NumRows();

    const Zeros zeroInitializer;
    zeroInitializer.Initialize(delta);

    for (auto& [unitId, gradient] : BackwardInputMap)
        Compute::Add(delta, gradient);

    Compute::ScalarMul(delta, 1.0f / BackwardInputMap.size());

    Compute::Multiply(weight, delta, backwardOutput, true, false, true);
    Compute::Multiply(delta, previousForwardInput, weightUpdate, false, true,
                      false);
    Compute::Multiply(delta, oneVector, biasUpdate, false, false, false);

    Compute::ScalarMul(weightUpdate, 1.0f / batchSize);
    Compute::ScalarMul(biasUpdate, 1.0f / batchSize);

    m_optimizer->Optimize(weight, weightUpdate);
    m_optimizer->Optimize(bias, biasUpdate);
}

template <typename T>
void DenseUnit<T>::AsyncBackward(std::promise<bool> promise)
{
    auto& weight = m_trainableTensorMap["weight"];
    auto& bias = m_trainableTensorMap["bias"];
    auto& weightUpdate = m_trainableTensorMap["weightUpdate"];
    auto& biasUpdate = m_trainableTensorMap["biasUpdate"];
    const Zeros zeroInitializer;

    auto& previousForwardInput = ForwardInputMap.at(m_sourceUnitId);
    auto& backwardOutput = BackwardOutputMap.at(m_sourceUnitId);

    Tensor<T>& delta = m_trainableTensorMap["delta"];
    zeroInitializer.Initialize(delta);
    for (auto& [unitId, gradient] : BackwardInputMap)
    {
        Compute::Add(delta, gradient);
    }
    Compute::ScalarMul(delta, 1.0f / BackwardInputMap.size());

    Compute::Multiply(weight, delta, backwardOutput, true, false, true);
    Compute::Multiply(delta, previousForwardInput, weightUpdate, false, true,
                      false);
    Compute::Shrink(delta, weightUpdate);
    Compute::Shrink(delta, biasUpdate, 0);

    m_optimizer->Optimize(weight, weightUpdate);
    m_optimizer->Optimize(bias, biasUpdate);

    promise.set_value(true);
}
} // namespace Takion::Graph

#endif
