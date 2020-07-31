// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.


#include <iostream>
#include <cubbydnn/Units/HiddenComputableUnits/Dense.hpp>
#include <cubbydnn/Computations/TensorOperations/Computations.hpp>

namespace Takion::Graph
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
    const auto batchSize = unitMetaData.OutputShape().NumCols();

    Tensor forwardInputTensor(unitMetaData.GetInputShape("input"),
                              unitMetaData.Device,
                              unitMetaData.NumericType);

    std::unordered_map<UnitId, Tensor> backwardInputMap;
    for (const auto& outputUnitId : unitMetaData.OutputUnitVector())
    {
        Tensor tensor(unitMetaData.OutputShape(),
                      unitMetaData.Device, unitMetaData.NumericType);
        backwardInputMap[outputUnitId] = std::move(tensor);
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

    Tensor biasTensor(biasShape, unitMetaData.Device,
                      unitMetaData.NumericType);
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

    Tensor oneVect(Shape({ batchSize, 1 }), unitMetaData.Device,
                   oneVector);

    Tensor biasUpdate(biasShape, unitMetaData.Device,
                      unitMetaData.NumericType);

    Tensor delta(unitMetaData.OutputShape(), unitMetaData.Device,
                 unitMetaData.NumericType);

    auto denseUnit = DenseUnit(
        unitId, sourceUnitId,
        std::move(forwardInputTensor),
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


void DenseUnit::Forward()
{
    Tensor& input = ForwardInputMap.at(m_sourceUnitId);

    Compute::Multiply(m_trainableTensorMap.at("weight"), input,
                      ForwardOutput, false, false, true);
    Compute::Multiply(
        m_trainableTensorMap.at("bias"), m_trainableTensorMap.at("oneVector"),
        m_trainableTensorMap.at("biasMatrix"), false, true, false);
    Compute::Add(ForwardOutput, m_trainableTensorMap.at("biasMatrix"), true);
}

void DenseUnit::AsyncForward(std::promise<bool> promise)
{
    const Zeros zeroInitializer;
    zeroInitializer.Initialize(ForwardOutput);
    Tensor& input = ForwardInputMap.at(m_sourceUnitId);

    Compute::Multiply(m_trainableTensorMap.at("weight"), input, ForwardOutput,
                      false, false, true);
    Compute::Add(ForwardOutput, m_trainableTensorMap.at("bias"), true);
    promise.set_value(true);
}

void DenseUnit::Backward()
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

    Compute::Multiply(weight, delta,
                      backwardOutput, true, false, true);
    Compute::Multiply(delta, previousForwardInput, weightUpdate, false,
                      true, false);
    Compute::Multiply(delta, oneVector, biasUpdate, false, false, false);

    Compute::ScalarMul(weightUpdate, 1.0f / batchSize);
    Compute::ScalarMul(biasUpdate, 1.0f / batchSize);

    m_optimizer->Optimize(weight, weightUpdate);
    m_optimizer->Optimize(bias, biasUpdate);

}

void DenseUnit::AsyncBackward(std::promise<bool> promise)
{
    auto& weight = m_trainableTensorMap["weight"];
    auto& bias = m_trainableTensorMap["bias"];
    auto& weightUpdate = m_trainableTensorMap["weightUpdate"];
    auto& biasUpdate = m_trainableTensorMap["biasUpdate"];
    const Zeros zeroInitializer;

    auto& previousForwardInput = ForwardInputMap.at(m_sourceUnitId);
    auto& backwardOutput = BackwardOutputMap.at(m_sourceUnitId);

    Tensor& delta = m_trainableTensorMap["delta"];
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
} // namespace Takion
