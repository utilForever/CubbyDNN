// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.


#include <cubbydnn/Units/HiddenComputableUnits/Dense.hpp>
#include <cubbydnn/Computations/TensorOperations/Computations.hpp>

namespace CubbyDNN::Graph
{
DenseUnit::DenseUnit(const UnitId& unitId, const UnitId& sourceUnitId,
                     Tensor forwardInput,
                     std::unordered_map<UnitId, Tensor> backwardInputMap,
                     Tensor forwardOutput, Tensor backwardOutput,
                     std::unordered_map<std::string, Tensor> trainableUnit,
                     std::unique_ptr<Compute::Optimizer> optimizer,
                     NumberSystem numberSystem)
    : ComputableUnit(unitId, numberSystem,
                     { { sourceUnitId, std::move(forwardInput) } },
                     std::move(backwardInputMap), std::move(forwardOutput),
                     { { sourceUnitId, std::move(backwardOutput) } }),
      TrainableUnit(std::move(trainableUnit), std::move(optimizer)),
      m_sourceUnitId(sourceUnitId)
{
}

DenseUnit::DenseUnit(DenseUnit&& denseUnit) noexcept
    : ComputableUnit(std::move(denseUnit)),
      TrainableUnit(std::move(denseUnit)),
      m_sourceUnitId(std::move(denseUnit.m_sourceUnitId))
{
}

DenseUnit& DenseUnit::operator=(DenseUnit&& denseUnit) noexcept
{
    ComputableUnit::operator=(std::move(denseUnit));
    TrainableUnit::operator=(std::move(denseUnit));

    return *this;
}

DenseUnit DenseUnit::CreateUnit(const UnitMetaData& unitMetaData,
                                std::unique_ptr<Compute::Optimizer>
                                optimizer)
{
    const auto unitId = unitMetaData.Id();
    auto sourceUnitId = unitMetaData.GetInputUnitId("input");

    Tensor forwardInputTensor(unitMetaData.GetInputShape("input"),
                              unitMetaData.Device,
                              unitMetaData.NumericType);

    std::unordered_map<UnitId, Tensor> backwardInputVector;
    backwardInputVector.reserve(unitMetaData.OutputUnitVector().size());
    for (const auto& outputUnitId : unitMetaData.OutputUnitVector())
    {
        Tensor tensor(unitMetaData.OutputShape(),
                      unitMetaData.Device, unitMetaData.NumericType);
        backwardInputVector[outputUnitId] = std::move(tensor);
    }

    Tensor forwardOutputTensor(unitMetaData.OutputShape(),
                               unitMetaData.Device, unitMetaData.NumericType);

    Tensor backwardOutputTensor(unitMetaData.GetInputShape("input"),
                                unitMetaData.Device, unitMetaData.NumericType);

    auto weightShape = unitMetaData.GetInternalVariableShape("weight");
    auto biasShape = unitMetaData.GetInternalVariableShape("bias");

    Tensor weightTensor(weightShape,
                        unitMetaData.Device, unitMetaData.NumericType);
    const auto& weightInitializer = unitMetaData.GetInitializer("weight");
    weightInitializer->Initialize(weightTensor);

    Tensor weightDelta(weightShape, unitMetaData.Device,
                       unitMetaData.NumericType);

    Tensor biasTensor(biasShape, unitMetaData.Device,
                      unitMetaData.NumericType);
    const auto& biasInitializer = unitMetaData.GetInitializer("bias");
    biasInitializer->Initialize(biasTensor);

    Tensor transposedWeight(weightShape.Transpose(),
                            unitMetaData.Device, unitMetaData.NumericType);

    Shape batchMeanDeltaShape{
        backwardInputVector.at(sourceUnitId).TensorShape[0],
        backwardInputVector.at(sourceUnitId).TensorShape[1] };

    Tensor batchMeanDelta(batchMeanDeltaShape, unitMetaData.Device,
                          unitMetaData.NumericType);

    auto denseUnit = DenseUnit(
        unitId, sourceUnitId,
        std::move(forwardInputTensor),
        { std::move(backwardInputVector) }, std::move(forwardOutputTensor),
        std::move(backwardOutputTensor),
        { { "weight", std::move(weightTensor) },
          { "bias", std::move(biasTensor) },
          { "weightTranspose", std::move(transposedWeight) },
          { "batchMeanDelta", std::move(batchMeanDelta) } },
        std::move(optimizer), unitMetaData.NumericType);

    return denseUnit;
}


void DenseUnit::Forward()
{
    Tensor& input = ForwardInputMap.at(m_sourceUnitId);

    Compute::Multiply(m_trainableTensorMap["weight"], input,
                      ForwardOutput);
    Compute::Add(ForwardOutput, m_trainableTensorMap["bias"]);
}

void DenseUnit::AsyncForward(std::promise<bool> promise)
{
    Tensor& input = ForwardInputMap.at(m_sourceUnitId);

    Compute::Multiply(m_trainableTensorMap["weight"], input,
                      ForwardOutput);
    Compute::Add(ForwardOutput, m_trainableTensorMap["bias"]);
    promise.set_value(true);
}

//TODO : Make it receive multiple backward inputs
void DenseUnit::Backward()
{
    auto& weight = m_trainableTensorMap["weight"];
    auto& bias = m_trainableTensorMap["bias"];
    auto& weightUpdate = m_trainableTensorMap["weightUpdate"];
    auto& biasUpdate = m_trainableTensorMap["biasUpdate"];
    const auto batchSize = weight.TensorShape.NumCols();

    auto& previousForwardInput = ForwardInputMap.at(m_sourceUnitId);
    auto& backwardOutput = BackwardOutputMap.at(m_sourceUnitId);

    Tensor& delta = m_trainableTensorMap["delta"];
    for (auto& [unitId, gradient] : BackwardInputMap)
    {
        Compute::Add(delta, gradient);
    }

    Compute::ScalarMul(delta, 1.0f / BackwardInputMap.size());
    Compute::Multiply(weight, delta,
                      backwardOutput, true, false);

    Compute::Multiply(delta, previousForwardInput, weightUpdate, false, true);
    Compute::ScalarMul(weightUpdate, 1.0f / batchSize);

    Compute::BatchMean(delta, biasUpdate, delta.TensorShape.Dim() - 2);

    m_optimizer->Optimize(weight, weightUpdate);
    m_optimizer->Optimize(bias, biasUpdate);
}

void DenseUnit::AsyncBackward(std::promise<bool> promise)
{
    auto& weight = m_trainableTensorMap["weight"];
    auto& bias = m_trainableTensorMap["bias"];
    auto& weightUpdate = m_trainableTensorMap["weightUpdate"];
    auto& biasUpdate = m_trainableTensorMap["biasUpdate"];
    const auto batchSize = weight.TensorShape.NumCols();

    auto& previousForwardInput = ForwardInputMap.at(m_sourceUnitId);
    auto& backwardOutput = BackwardOutputMap.at(m_sourceUnitId);

    Tensor& delta = m_trainableTensorMap["delta"];
    for (auto& [unitId, gradient] : BackwardInputMap)
    {
        Compute::Add(delta, gradient);
    }

    Compute::ScalarMul(delta, 1.0f / BackwardInputMap.size());
    Compute::Multiply(weight, delta, backwardOutput, true, false);

    Compute::Multiply(delta, previousForwardInput, weightUpdate, false, true);
    Compute::ScalarMul(weightUpdate, 1.0f / batchSize);

    Compute::BatchMean(delta, biasUpdate, delta.TensorShape.Dim() - 2);

    m_optimizer->Optimize(weight, weightUpdate);
    m_optimizer->Optimize(bias, biasUpdate);

    promise.set_value(true);
}
} // namespace CubbyDNN
